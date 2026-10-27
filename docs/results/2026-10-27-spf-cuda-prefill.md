# spf: CUDA prefill on the Spark, Qwen3-8B dense (2026-10-27)

Lane `spf`, branch `lane-spf/cuda-prefill`, base `origin/main` `e0a042208` (after PR #185 and #187). DGX Spark **spark1** (spark-4e77, GB10 sm_121),
`Qwen/Qwen3-8B` BF16, `serve` with the recipe default (`prefill chunk 4096 (recipe qwen3, cuda)`), `--max-seq 20480 --mem-budget 40`. The box's
vLLM (`flight` / `guidance` / `capcom`, 34 GB) stayed up and idle. Every GPU job ran in a fleetq slot (`--quiet gpu,cpu` for timing, `--load cpu`
for builds). Raw logs: `2026-10-27-spf-cuda-prefill/raw/`, scripts: `scripts/spf/`.

**Result: one fix, default on. `rope_bf16` on a prefill chunk was 6.6 % of the kernel time (62 ms of 943 ms at 4k) and 6x slower than its memory traffic
allows, because it recomputed double-precision `pow`/`sincos` eight times per value. A kernel that computes them once per (token, frequency) is bitwise
the old one and lifts 8B prefill by +5.7 % at 1k, +5.9 % at 4k and +3.1 % at 16k (three interleaved rounds, same binary pair), decode unchanged.**

| | 1k (990 tok) | 4k (4087 tok) | 16k (16.6k tok) | decode |
|---|---|---|---|---|
| prefill tok/s before (`origin/main`), median of 3 interleaved rounds (min-max) | 4630 (4561-4646) | 4238 (4227-4279) | 3561 (3518-3565) | 15.1 / 14.7 / 13.2 |
| prefill tok/s with the fix | 4895 (4892-4906) | 4486 (4436-4515) | 3671 (3669-3675) | 15.1 / 14.7 / 13.2 |
| change (paired per round) | **+5.7 %** (+5.7 +5.3 +7.6) | **+5.9 %** (+5.5 +4.7 +6.1) | **+3.1 %** (+3.0 +4.4 +3.1) | x1.000 / 0.999 / 1.000 |


## 1. Measurement on current main (`origin/main` `e0a042208`, Qwen3-8B, spark1)

`localeval speed --sizes 1024,4096,16384 --gen 128 --reps 3` (+1 warmup), thinking off, one fresh `serve` per round, 3 rounds, `scripts/spf/le-window.sh`
(`raw/base8b/`, `raw/ab8b/`). Prefill tok/s, median of the 3 round medians (min-max):

| window | 1k | 4k | 16k | decode 1k / 4k / 16k |
|---|---|---|---|---|
| 23:20 EDT (`raw/base8b`, first measurement) | 4507 (4461-4554) | 3999 (3937-4031) | 3473 (3467-3510) | 15.3 / 14.8 / 13.3 |
| 00:19 EDT (`raw/ab8b`, base arm of the A/B) | 4630 (4561-4646) | 4238 (4227-4279) | 3561 (3518-3565) | 15.1 / 14.7 / 13.2 |
| records quoted in the brief (fx3, **spark2**, vLLM up) | 4265 at 4k | | | 15.6 |

The same binary moves 4-6 % between two windows an hour apart (GB10 clocks / temperature; the SM clock sampled during a 16k prefill went 2418 -> 2320 MHz at
about 94 W). Only paired, interleaved arms are compared below. The records of the brief were measured on spark2; spark1's number is its own, no cross-box claim
is made. The brief's vLLM / SGLang figures for the 8B (4763 / 4317 tok/s at 4k) were not re-measured here: no 8B vLLM was started next to the box's production
server.

## 2. Profile of the 8B 4k prefill

`scripts/spf/prof-serve.sh`: `intarsia serve` under `nsys profile -t cuda`, four identical 4092-token requests (`max_tokens 1`), the last (warm) request ranked
(`raw/serve-base4087/rank.txt`). A single `generate` under nsys is not a steady state (cuBLAS init, 350 ms; weights touched for the first time), so the serve
profile is the one used. The request's wall time (949 ms on the completions API) equals its kernel time (943 ms): the GPU is the bottleneck, idle gaps total
0.85 ms in 363 gaps (largest 25 us), there is no launch-overhead lever on a 4k chunk.

| op | kernel | ms | % | calls | avg us | note |
|---|---|---|---|---|---|---|
| q, o, gate, up, down GEMMs | `nvjet_sm121_tst_mma_128x208x64` (cuBLAS) | 668.3 | 70.9 | 180 | 3713 | 81 TFLOP/s sustained over the 7 GEMMs of a layer; 92.5 for the same shapes in a 20-iteration micro-benchmark (finding 2) |
| k, v GEMMs | `nvjet_sm121_tst_mma_128x176x64` | 31.8 | 3.4 | 72 | 441 | |
| attention | `flash_kernel<128, 64>` (mma.sync flash) | 74.0 | 7.8 | 36 | 2056 | 62 TFLOP/s of causal flops |
| **rope** | `rope_kernel` | **61.9** | **6.6** | 72 | **860** | **0.24 ms of traffic per call: 3.6x off the memory floor** |
| rmsnorm (2 hidden norms + q/k norm per layer) | `rmsnorm_kernel` | 52.3 | 5.5 | 145 | 361 | hidden norm 0.54 ms for 128 MB = 240 GB/s, at the floor |
| silu * up | `silu_glu_kernel` | 45.0 | 4.8 | 36 | 1249 | 300 MB per call = 240 GB/s, at the floor |
| lm head (last row) + kv append + rest | `gemv_kernel`, `kv_append_kernel` | 9.9 | 1.0 | | | |

At 16k the picture is the same with attention at 24 % (`raw/prof-base/55000/rank.txt`, a cold single-shot `generate`: flash 1076 ms of 4418, rope 243 ms).
GEMMs are 77 % of the 4k time and run at cuBLAS's rate; the elementwise kernels are at the memory floor; attention is at 62-70 TFLOP/s. **Rope is the one
op that is far from its floor.**

## 3. Findings

1. **`rope_kernel` recomputes a double-precision `pow` and `sincos` per head group.** `kernels/cuda/elementwise.cu`: one thread per (token, frequency, group of
   4 heads); the angle `pos * theta^(-2k/RD)` is computed in `double` (positions reach tens of thousands, so f32 would drift). GB10 has a consumer-class
   FP64 rate (1/64 of FP32), so a 32-head q recomputes it 8x and a 8-head k 2x. 0.86 ms per call on a 4k chunk against 0.24 ms of traffic.
2. **cuBLAS GEMMs in the engine run 11-20 % slower than the same shapes in a micro-benchmark** (`scripts/spf/lt_sweep.cu`: 94 TFLOP/s at M = 4087 for
   q/o, 95 gate/up, 89 down, 92.5 weighted over a layer; engine 81; same kernel name, same grid). Excluded: operand alignment (pointers printed for the first 24 GEMMs of a
   4091-token prefill: all 4096-byte aligned, `raw/dbg.txt`; a +16 byte offset costs 20-30 % in the micro-benchmark, so a misaligned pointer would show),
   cold weights (24 distinct weight copies: no change), data distribution (uniform vs gaussian: no change), the cuBLAS workspace (32 MB in both). SM clock
   during the engine's 16k prefill 2320-2418 MHz against 2405 in the micro-benchmark: 3 %. What remains is sustained-power behaviour of the part (a cold single-shot nsys trace shows
   the same layer's GEMM durations growing by 5-10 % over one 4k pass); not closable from the engine.
3. **cuBLAS's heuristic picks the wrong kernel for some M, and swapping it is not always bitwise.** `raw/lt_sweep_M.txt` (13 values of M x the four shapes,
   `cublasGemmEx` as the engine calls it against every `cublasLt` heuristic algorithm): 8 of 52 cases are 3-51 % slower than the best Lt algorithm (M = 3961
   gate/up picks `cutlass_80_tensorop_bf16_s16816gemm_relu 128x256` at 83 TFLOP/s where an nvjet tile does 91; M = 3000 k/v 0.38 vs 0.25 ms; M = 2500 -9 % to
   -13 %). The benchmark prompts (M = 991, 4087, 4096 x 4 + 293) are mostly not hit. One of the 52 best-Lt outputs differs bitwise from `cublasGemmEx`
   (M = 293: 331 of 300032 elements), so an algorithm override would need its own bitwise proof per shape. Not pursued.

## 4. The fix: `rope_multi_kernel`

`kernels/cuda/elementwise.cu`: for calls of 5 or more rows with one position per row (`pos0 + t`; not the batched-decode row table, not one-row decode), one thread
per (token, frequency) computes the angle once and rotates all `N` heads (loop over groups of 4, the same loads, the same expressions, the same stores as
`rope_kernel`; `x` and `out` may still alias). Calls below 5 rows (decode, verify passes of 2-4 rows) keep `rope_kernel`, so decode and MTP verify are untouched.
`INTARSIA_CUDA_ROPE_MULTI=0` restores the old kernel (read per launch; `docs/tuning.md` row). 5 rows is the smallest row count the kernel admission sweep
(`rows = 1, 5, 32`) exercises, so the new path is inside the admission cases.

Kernel level (steady-state serve profile of the same request, `raw/serve-rope4087/rank.txt`): rope 61.9 -> 18.3 ms (860 -> 254 us per call). The remaining 18 ms
are the double-precision math still done once per call plus the memory traffic (not split further). Whole-profile kernel total 943 -> 864 ms; 33 ms of that is the GEMM
getting faster between the two windows (see section 1), so the A/B below, not this table, is the speed claim.

### A/B (`scripts/spf/le-window.sh`, `raw/ab8b/`, 3 rounds, round 2 reversed, one fresh `serve` per arm and round)

| | 1k (990) | 4k (4087 / 4090) | 16k (16653) |
|---|---|---|---|
| base | 4630 (4561-4646) | 4238 (4227-4279) | 3561 (3518-3565) |
| rope fix | 4895 (4892-4906) | 4486 (4436-4515) | 3671 (3669-3675) |
| paired change per round | +5.7, +5.3, +7.6 % | +5.5, +4.7, +6.1 % | +3.0, +4.4, +3.1 % |
| marginal prefill (delta tokens / delta TTFT) base -> fix | | 4129 -> 4368 (+5.8 %) | 3378 -> 3472 (+2.8 %) |
| decode tok/s base -> fix | 15.11 -> 15.12 | 14.70 -> 14.69 | 13.17 -> 13.16 |

Every round of every size is above the 3 % bar; decode x1.000 / 0.999 / 1.000.

### Gates

| gate | result | evidence |
|---|---|---|
| CPU tests | PASS: `cargo test --workspace` on the Mac, 1014 passed, 0 failed; CI on PR #191: cargo check (linux, cuda cfgs, hip cfgs, macOS metal), cargo test (linux), reproducible build all green | `raw/cargo-test-workspace-mac.log` |
| CUDA greedy ids, 8B | PASS: 256 ids + first-token top-20 logits (4 decimals) identical, base vs fix and base vs `INTARSIA_CUDA_ROPE_MULTI=0`, 4 prompts (short, 4k x2, 16k), 12 comparisons. Note: the 4k / 16k comparisons run the prompt through `--show-logits` first, so the ids come from decode on a prefix-cached prompt; the prefill itself is covered by the logits of that pass | `raw/ids8.log` |
| CUDA greedy ids, 35B (affected: partial-rotary rope, prefill and macro segments) | PASS: short, 4k, 16k, base vs fix, ids + top-20 logits identical | `raw/ids35.log` |
| kernel admission | PASS: `kernels test --backend cuda` full 77 kernels / 1514 cases, quick 77 / 531, `--op rope` with the 8B's shapes 36 cases (fix and `ROPE_MULTI=0` give identical max_abs 8.503e-4 / rel_l2 1.078e-5), no REJECTED. Run on the tree before the env knob was made per-launch (see below) | `raw/ktest.log` |
| `plan show` byte-identical, CUDA | PASS: 8B, 35B, Gemma-4-26B, json and text, base vs fix | `raw/plan.log` |
| `plan show` byte-identical, Metal | PASS: 8B, 35B, json and text (two Metal builds, origin/main and this tree) | `raw/metal-plan.log` |
| `plan show`, HIP | not run. The change touches `kernels/cuda/elementwise.cu`, `docs/tuning.md`, a CUDA test and docs only; no Rust source of the HIP path changed, and the Metal result above shows the planner output does not depend on the diff | |
| prefill >= 3 % median over 3 interleaved rounds, decode >= 0.99x | PASS (section 4 table) | `raw/ab8b/` |
| >= 3 planted mutants caught | PASS, 5 of 5 CAUGHT (REJECTED by `kernels test --op rope -m <8B>`): sign flip, position off by one, skipped head group, wrong frequency denominator, swapped output lane. Clean tree: 90 cases pass | `raw/mutants.log`, `scripts/spf/mutants.py` |
| `tests/rope_multi.rs` (bitwise, in-process, 8 head layouts x 9 row counts x 3 positions x knob on/off) | PASS on spark1 GPU (`INTARSIA_REQUIRE_GPU=1`, 2.96 s), tree `rope2` = the committed code; quick admission on the same tree: 77 kernels, 531 cases, no REJECTED | `raw/rope2-gpu.log` |

The environment variable was read once per process in the tree that produced every number above and is read per launch in the committed code (so that
`tests/rope_multi.rs` can flip it); the kernels and arithmetic are the same.

**Gap in the night.** spark1 dropped out of fleetq from about 01:00 to 03:35 EDT (the server marked it offline; jobs waited, nothing was bypassed). The mutants
and the GPU test were run when it came back (03:37).

## 5. Qwen3.6-35B-A3B NVFP4

Correctness only (this change reaches the 35B through the partial-rotary rope of its 10 attention layers): ids identical in section 4. **The speed windows
the brief asked for (35B at 1k against the box's vLLM, MTP default against `--spec off`) were not run**: they were queued behind the gates when spark1 went
offline for fleetq. `scripts/spf/chain2.sh` (vLLM vs intarsia MTP default vs `--spec off`, 3 rounds, 1k / 4k / 16k) and `scripts/spf/chain5.sh` (steady-state nsys
of a 1k request, kernel total against wall) are ready. Reading the code before measuring: the MTP head's prompt rows are queued during prefill and run at the
first draft step (`mtp_run_pending`), so MTP should cost the 1k TTFT only a hidden-state copy; the 1k gap to vLLM recorded by lane sk (-3.9 %, taken before the
MTP default) is therefore unlikely to be MTP, and is still unexplained.

## 6. Not done, and what is left

* GEMM: the sustained 83 vs 94 TFLOP/s gap (finding 2) and the per-M heuristic misses (finding 3) are the only large levers left on the 8B, and neither is an
  engine bug. Flash attention (62 TFLOP/s causal) and the two memory-bound elementwise ops are at 70-100 % of what the part allows.
* `qk_norm` + `rope` + `kv_append` as one op (13 + 4 ms at 4k, about 1.5 %) was not built: it would need a new registered op and change the 8B / 35B `plan show`.
* The remaining 4.6 ms/call of double-precision `pow`/`sincos` in `rope_multi_kernel` could come from a per-chunk table shared by the 72 calls (up to about 1 % at 4k);
  not built (stateful launcher, graph-capture rules).

## 7. Fleet log

All jobs owner `claude-f31ccf91`, spark1 (vLLM `flight/guidance/capcom` up, nothing taken down). Builds (cpu): #5848, #5882, #5894, #5944 (trees `base`, `dev`, `rope`, `rope2`). Quiet GPU/CPU: #5850 (8B baseline x3), #5857 / #5864 (single-shot nsys),
#5860 / #5863 / #5866 and the `lt_sweep` jobs (GEMM micro-benchmarks, clocks, nsys of the micro-benchmark), #5869 / #5870 (steady-state serve nsys, base and fix),
#5890 (alignment print + knob A/B), the A/B window, `plan show`, ids 8B, ids 35B, kernels test. Cancelled unrun: #5954 (mutants), #5956 (rope2 GPU test). Endor (cpu):
#5893 (Metal `plan show`). The spark1 trees `~/intarsiaml-spf/{base,dev,rope,rope2,lt}` and the local worktree `.worktrees/spf` are kept for the open gates (the brief
asked for their removal at the end).
