# scc: multi-stream serving of Qwen3.6-35B-A3B NVFP4 on one GB10 (CUDA), and a faster row-exact fp8 / nvfp4 GEMV

Lane scc, 2026-10-27, origin intarsiaml. spark2 (spark-fa00, one GB10), Qwen3.6-35B-A3B NVFP4 (`nvidia/Qwen3.6-35B-A3B-NVFP4`), `serve --backend cuda --max-batch 16 --max-seq 8192`.
Base `origin/main` e7e0845ec, branch `lane-scc/cuda-35b-concurrency`. Raw output, localeval result files, nsys rankings and logs: `2026-10-27-scc-cuda-35b-concurrency/raw/`; scripts: `scripts/scc/`.

## 1. Result

Aggregate output tok/s, localeval `speed --sizes 1024 --gen 256 --reps 3 --concurrency 1,4,8,16` (a burst of N identical 1k-token prompts, 256 forced output tokens each, so the
aggregate includes the serialized prefill of the burst), three interleaved rounds in one fleetq `--quiet gpu,cpu` window (#6295; round 2 reversed), median of the rounds.
TTFT p95 is the maximum of a burst (the nearest-rank p95 of n <= 16 samples is the maximum; localeval keeps only median / min / max per burst), median over the rounds.

| streams | ours main (e7e0845ec) | ours branch | change (median of the per-round ratios) | vLLM, spark2 service, `--max-num-seqs 2` |
|---|---|---|---|---|
| 1 | 78.6 | 78.0 | -0.8 % (rounds -0.8, -0.5, -0.9) | 74.7 |
| 4 | 161.0 | 159.9 | +0.1 % (+0.1, +0.5, -0.7) | 113.8 |
| 8 | 203.4 | 208.6 | +3.2 % (+4.2, +3.2, +2.0) | 114.1 |
| 16 | 235.2 | **258.1** | **+9.7 %** (+8.4, +10.7, +9.7) | 113.7 |

| TTFT p50 / p95 (s) | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| ours main | 0.19 / 0.19 | 0.48 / 0.78 | 0.87 / 1.60 | 1.69 / 3.39 |
| ours branch | 0.19 / 0.19 | 0.48 / 0.78 | 0.88 / 1.61 | 1.70 / 3.38 |
| vLLM (2 seqs) | 0.18 / 0.18 | 2.54 / 4.83 | 7.01 / 13.77 | 16.07 / 31.87 |

**The vLLM column is not a ceiling for vLLM.** The only vLLM serving this model on a Spark is the spark2 `retro` service (`--max-num-seqs 2 --max-num-batched-tokens 8192 --kv-cache-dtype fp8
--gpu-memory-utilization 0.32`, role `guidance`), so its aggregate is flat at about 114 tok/s from 4 streams on (a queue of two), and its TTFT is queue time. Every earlier 35B vLLM concurrency record
(fx3, `docs/results/2026-10-22-fx3-spark-standings.md`; `/opt/code/model-benchmarks/data/records/localeval.jsonl`) was taken with the same `max_num_seqs` of 2, and the brief forbids standing up another vLLM.
**So the gap to vLLM at 4 / 8 / 16 streams is not measured.** One stream is measured: ours 78.6 against 74.7 (-5 % for vLLM). What a `--max-num-seqs 16` vLLM does here is the open question (section 8).

What changed (section 4): the row-exact multi-row GEMVs the batched decode step runs, `matmul_fp8_gemv_rows` (fp8 projections of the 35B) and `matmul_nvfp4_gemv_rows` (shared experts, lm_head), re-staged and re-read their
activations for every 8 outputs. The fp8 kernel is now a persistent kernel that stages x once per block (when `M * K * 2` fits 96 KB), the nvfp4 kernel lets a warp own 2 or 4 outputs. Per (row, output) the arithmetic is
unchanged, so the batched step is still bitwise the solo step. Decode kernel time of a 16-row step: 55.9 ms -> 50.3 ms (-10 %). Knobs: `INTARSIA_CUDA_FP8_ROWS=0` and `INTARSIA_CUDA_NVFP4_ROWS=0` give the old kernels.

## 2. Baseline (before the change)

Window #6212, main only, vs the vLLM service, three rounds (`raw/base-vs-vllm/`, `raw/base-vs-vllm.agg.json`): ours 78.5 / 159.8 / 203.5 / 235.3 tok/s at 1 / 4 / 8 / 16 streams, vLLM 74.8 / 114.3 / 114.1 / 114.7.
Round-to-round spread of ours is under 1 % at 16 streams (234.6 - 235.3), 2 % at 4 (159.5 - 161.3). The gate window (section 1) reproduced it (235.2 at 16). For scale, rb2 (2026-10-23, spark1, 128 output tokens) had 123.3 / 166.6 at 4 / 16 streams. Single stream in the same server: 83.0 decode tok/s, 5310 tok/s prefill at 1k (vLLM 78.7 / 5327).

## 3. Profile of main at 16 streams

nsys (`-t cuda --cuda-graph-trace=node`), `serve --max-batch 16`, a warm-up burst, then 16 streams of ~1k-token prompts with 256 forced tokens (`scripts/scc/prof.sh`, `burst.py`, `rank.py`; #6221, ranking from #6230;
`raw/prof16-base/`). The decode window is the 14.6 s after the last prefill (232 steps of 16 rows):

| rank | kernel | ms / step | share | calls / step | note |
|---|---|---|---|---|---|
| 1 | `moe_dec_gate_up_kernel` | 14.86 | 26.6 % | 40 | 371 us a layer; expert-weight stream, 128 (token, expert) pairs, shares weight loads only through L2 |
| 2 | `fp8_gemv_rows_kernel` | 13.75 | 24.6 % | 130 | 105.7 us a call, 2.4x the 5.8 ms the same projections cost in the solo step: the **closable** sink |
| 3 | `moe_dec_down_v2_kernel` | 8.60 | 15.4 % | 40 | |
| 4 | `gdn_delta_split_kernel` | 8.40 | 15.0 % | 30 | 280 us a layer for 16 rows (solo 19 us a row), state read + write at about the bandwidth floor |
| 5 | `nvfp4_gemv_kernel` (rows) | 4.56 | 8.2 % | 121 | solo 1.4 ms |
| 6 | `splitkv_kernel` (attention) | 2.13 | 3.8 % | 10 | |
| 7 | `gemv_kernel` (bf16) | 1.94 | 3.5 % | 140 | |
| | rest | 1.5 | 2.7 % | | |

Kernel time per step 55.9 ms; the unprofiled step (host profile below) takes 55.2 ms. Under nsys the burst takes 20.3 s: about 4.5 s of serialized prefill (TTFT max 4.48 s), then the 232 decode steps.

Sinks that are not levers, by measurement:

* **Scheduler / launch gaps.** nsys shows 11.4 % GPU idle, 232 gaps of 7.1 ms, one before every step's `embedding_kernel`; inside the step (graph replay, `--cuda-graph-trace=node`) gaps are under 5 us (0.1 ms a step in total).
  That gap is the profiler's graph launch cost: `INTARSIA_HOST_PROFILE=1` on the same burst without nsys (`raw/hostprof-base/`, the server's stderr is on spark2) reads, per 16-row decode-only iteration,
  `wall 55.218 ms, sync 55.106, launch 0.021, upload 0.009, step-setup 0.053, accept 0.000, gpu-idle-gap 0.030 ms (253 gaps)`. The host is not on the critical path; graph replay is already on.
* **Device-side top-k sampling** (`INTARSIA_DEVICE_SAMPLING`, default off by the 2026-10-23 decision) exists on CUDA already, and applies to sampled rows with `top_k` in 1..=64 only; localeval's forced-token requests are
  greedy, whose argmax is on the device already (`argmax_kernel` 0.04 ms a step). Not a lever for this load, not touched.
* **MoE grouping** (`moe_dec_gate_up`, `moe_dec_down_v2`, 42 % together): the kernels work per (token, expert) pair, so rows routed to the same expert re-read its weights from L2. localeval sends one prompt to all streams, so
  the routing of its rows overlaps more than real traffic and any gain measured with it would be inflated; the gate_up kernel already runs at about 366 GB/s of effective weight bandwidth (above the 273 GB/s DRAM figure, so it
  is L2-fed). Not attempted.
* `gdn_delta_split` is linear in rows and near the state-bandwidth floor.

## 4. The lever: fp8 and nvfp4 row-exact GEMVs

`fp8_gemv_rows_kernel` (kernels/cuda/fp8.cu): a block of 8 warps owns 8 outputs, stages all M activation rows through shared memory in K-chunks (and, for W8A8, requantises every element with a correctly rounded
division and an e4m3 round trip), then every warp walks its weight row once per chunk. At M = 16 the staging is about as many instructions as the dot products and costs a barrier per chunk; each x vector is unpacked from shared memory
once per output. 16-row time was 2.4x the 1-row total although it reads the same weights. `nvfp4_gemv_kernel` at M >= 4 has no staging but re-reads and re-unpacks x from global for every output.

Both stay row-exact (batched == solo) by keeping the arithmetic of every (row, output): the lane-strided 16-element blocks, the 16-long fmaf chain, `acc += s` / `acc = fmaf(s, scale, acc)` and the xor warp tree.

* fp8, **persistent kernel** (`fp8_gemv_rows_full_kernel`): when `M * K * 2 <= 96 KB` (M = 16: K <= 3072, all the 2048-wide projections: GDN in-qkv 8192x2048, in-z / attention q 4096x2048, k / v 512x2048) a block
  stages x once and its 16 warps walk 2-output tiles of the matrix with no barrier; one grid of at most one block per SM. Otherwise (K = 4096 at M >= 13: the GDN out and attention o projections) the chunked kernel runs with 1, 2 or 4 outputs per warp
  (default: the largest leaving 96 blocks). `INTARSIA_CUDA_FP8_ROWS=0` is the original kernel, `=tile` skips the persistent kernel; `_NT` and `_FULL` pick the shapes for tests and sweeps.
* nvfp4, **output tile** (`nvfp4_gemv_tile_kernel`, rows launcher only, M >= 4): a warp owns 2 or 4 consecutive outputs and unpacks each x block once for them (4 at 96 blocks, else 2 at 32, else 1).
  `INTARSIA_CUDA_NVFP4_ROWS=0` is the kernel of before; solo and MTP-verify calls (M <= 3, the non-rows launcher) never take it.

Micro-timing, 16 rows, microseconds a call (`rows_exact -- --ignored timing`; first column from the quiet window #6249, the rest from a normal slot, so indicative; `raw/micro-fp8-nt.txt`, `raw/micro-rows-v2.txt`; weights stay L2-resident, so the real
saving per call is larger where DRAM binds):

| fp8 N x K | original | chunked, 2 out / warp | chunked, 4 | persistent 2x16 (default) | persistent 4x8 |
|---|---|---|---|---|---|
| 8192 x 2048 | 159.5 | 116.7 | 114.7 | **70.4** | 66.8 |
| 4096 x 2048 | 82.0 | 65.0 | 57.5 | **39.9** | 39.7 |
| 2048 x 2048 | 47.8 | 36.9 | 40.0 | **29.0** | 30.9 |
| 512 x 2048 | 20.5 | 17.1 | 21.1 | 19.4 | 20.5 |
| 2048 x 4096 (K too big, chunked) | 89.6 | 67.8 | 71.4 | | |

| nvfp4 N x K | original | 2 out / warp | 4 out / warp |
|---|---|---|---|
| 248320 x 2048 (lm_head, f32 out) | 2980 | 2763 | **2071** |
| 512 x 2048 | 14.5 | **10.3** | 14.3 |
| 2048 x 512 | 14.3 | 13.5 | 14.4 |

Small batches (same micro-timing, `raw/micro-rows-v2.txt`, auto against the original): M = 2 / 4 / 8 are faster on every 35B shape (8192x2048 0.66 / 0.49 / 0.47x the time, 4096x2048 0.62 / 0.58 / 0.51, 2048x2048 0.71 / 0.72 / 0.65, 2048x4096 0.87 / 0.67 / 0.60) except 512x2048 at M = 2 (7.1 against 6.1 us, +16 %: 16 blocks only); M = 4 / 8 there 0.86 / 0.84. The served 2 and 3 stream loads were not measured.

Served, an earlier tile-only build (window #6249, 3 rounds, 16 streams, change against main): 1 output / warp (the original kernel) -0.5 %, 2 +4.3 %, 4 +3.6 %, launcher's pick +3.5 %; at 8 streams +0.7 / +2.1 / +2.3 / +2.3 %.
The persistent kernel is what takes 16 streams to +9.7 %. After the change (nsys, same burst, `raw/prof16-dev3/`; #6308): fp8 rows 13.75 ms a step -> 8.98 ms (`fp8_gemv_rows_full_kernel` 5.7 ms in 90 calls + the chunked K = 4096 calls 3.3 ms in 40),
nvfp4 4.56 -> 3.60 ms, total kernel time 55.9 -> 50.3 ms a step.

## 5. Gates

| gate | result |
|---|---|
| aggregate tok/s at 8 or 16 streams >= +5 %, median of >= 3 interleaved rounds | **PASS**: 16 streams +9.7 % (rounds +8.4 / +10.7 / +9.7; 258.1 vs 235.2 tok/s). 8 streams +3.2 %. #6295 |
| single-stream decode and prefill >= 0.99x | **PASS**, with noise caveats. Recipe default (no `--max-batch`, MTP on), 3 interleaved rounds, ratio of medians dev / main: prefill 1k 1.000 / 4k 1.000 / 16k 0.999 (per-round ratios no lower than 0.982 at 1k), decode 1k 1.058 / 4k 0.999 / 16k 0.994 (MTP decode swings +-4 % between rounds of the same binary, 96.7 - 100.8 tok/s at 1k). That path does not call the changed kernels (rows kernels are `--max-batch` only). In the `--max-batch 16` server the one-stream burst reads -0.8 % (rounds -0.8 / -0.5 / -0.9), also -0.6 % for the knob-off arm of the earlier window: a consistent 0.6 - 0.8 % offset of the new binary on the solo step that I could not attribute (its kernels are untouched). Still >= 0.99. #6295 |
| batched == solo on the probe set | **PASS**: `batch_cuda_real` on the 35B, `real_35b_batched_decode_is_bitwise_the_solo_step` 4290 (row, step) pairs bitwise identical (84.6 s), `real_35b_batched_engine_equals_the_local_engine` ok (`raw/batch-real.log`, #6283). `rows_exact` (G-B): each kernel form against the one-row kernel with no tolerance, fp8: 10 forms (original, chunked tile 1 / 2 / 4 / auto, persistent 1x16 / 2x16 / 2x8 / 4x8 / auto) x 11 shapes (35B projections and dead-output / ragged-K edges) x 9 weight and quantisation settings x M 1..16 x both output dtypes; nvfp4: 5 forms x 8 shapes x 2; `rows_registry` 3 passed (#6275) |
| fixed-seed served-token identity | **PASS**: `scripts/htk/equal.py` against main and branch (`serve --max-batch 16`): 7 sampling settings (6 sampled with seeds 100+i, 1 greedy) x 1 / 4 / 16 streams x 64 tokens, **21/21 cells identical, 9408 token ids** (`raw/ident/`, #6293) |
| `kernels test` admits, >= 3 planted mutants caught | **PASS, with a scope note**: no new kernel is registered (the two rows kernels keep their names and registrations), and `intarsia kernels test --backend cuda --quick` on the branch binary admits all 77 kernels, 531 cases, none rejected (#6292). The mutants were run against the suite that gates these kernels, the bitwise `rows_exact` tests, not against `kernels test` (the same split as lane htk's `tests/topk.rs` mutants). **Mutants (`scripts/scc/mutants.py`, `raw/mutants-run1.log`): 17 planted, 16 caught, 1 survived.** Caught: fp8 chunked kernel fma order, store column, dropped tail, wrong weight row, partial-warp skip; fp8 persistent kernel fma order, store column, dropped tail, accumulator alias, tile stride, staged half swap, W8A8 requantisation skipped; nvfp4 tile fma order, store column, wrong scale row, accumulator alias, partial-warp exit. **Survived: `fp8-full-lane-skew`** (lane l takes the blocks of lane l+1, a rotation of the warp tree's inputs). Not an equivalent mutant in general, but the f32 sums of fp8 x fp8 products are exact on every data class of the test (a pure-python emulation of the rotation over six data classes, 360 outputs, never changed a bit), so no `rows_exact` input can see a cross-lane association; the kernels' lane loop is the unchanged `for (u = lane; u < KV; u += 32)`. Test sensitivity gap, pre-existing |
| `plan show` byte-identical, 8B and Gemma-4 on CUDA | **PASS**: Qwen3-8B (json 8758 B, text 1478 B), Qwen3.6-35B (15095 / 3866 B) on spark2 (#6277), Gemma-4-26B-A4B-NVFP4 (8631 / 1865 B) on spark1 with the same two binaries (#6281); main binary vs branch binary, `--max-seq 8192 --mem-budget 40` |
| Mac `cargo test --workspace` and CI | Mac `cargo test --workspace --locked --no-fail-fast` on the final tree: 966 passed, 0 failed (`raw/mac-cargo-test2.log`; the first run of the day failed `docs_env_vars` on the three new tuning variables without a row, fixed in the second commit). CI: on the PR |
| `INTARSIA_*` knob that turns it off | `INTARSIA_CUDA_FP8_ROWS=0`, `INTARSIA_CUDA_NVFP4_ROWS=0` (rows_exact covers the `0` form bitwise; window #6249's knob-off-equivalent arm `dev1`, the original chunked kernel at 1 output / warp, read -0.5 % against main at 16 streams). Rows in `docs/tuning.md` for all five variables (`docs_env_vars` test passes) |

## 6. Method

* Binaries built on spark2 (`scripts/scc/build.sh`, `--load cpu`) from the working tree (`INTARSIA_BUILD_STAMP` = tree stamp), main as `bin/base`, final branch as `bin/dev3`. Served `--max-seq 8192 --mem-budget 40` with `--max-batch 16` (35B: weights 20.8 GB, KV for 16 slots fits next to the 38 GB vLLM service).
* Timing only through localeval 0.4.0 (`speed`, `--thinking off`, fleetq slot `--quiet gpu,cpu`, `scripts/scc/le-window.sh`): per arm a fresh server, rounds interleaved (round 2 reversed). localeval runs one burst per stream count per server visit, so a "round" is one burst; the 1-stream row of the burst
  is not the single-stream decode figure (it includes the prefill; the decode figures are in the gate row above and come from the `--sizes 1024,4096,16384 --gen 128` runs of the default-config arms).
* Windows used (fleetq `--quiet`): #6212 baseline, #6221 nsys 16 streams, #6232 host profile, #6235 nsys solo, #6249 tile sweep, #6295 gate window = 6 of 6. Non-timing GPU jobs (tests, mutants, identity, nsys of the branch #6308, micro-timing) ran in normal slots.
* vLLM: spark2's `spark2-vllm` (role `retro`, `missioncntrl/vllm-openai:65b7662d`, vLLM 0.26.1rc1.dev602) was up and idle throughout and never taken down.

## 7. Reproduce

```
scripts/scc/build.sh base|dev3                 # fleetq run --host spark2 --load cpu ...
scripts/scc/win6.sh                            # fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 42 ...  (gate window)
scripts/scc/prof.sh <bin> <name>               # nsys profile of a 16-stream burst; RANK_ARGS="5" ranks the decode window
scripts/scc/test.sh build|run --test rows_exact --test rows_registry -- --test-threads=1
ssh spark2 'python3 -' < scripts/scc/mutants.py [name ...]
scripts/scc/ident.sh base dev3 ; scripts/scc/plan.sh ; scripts/scc/plan-g.sh ; scripts/scc/batch-real.sh build|run
scripts/scc/agg.py docs/results/2026-10-27-scc-cuda-35b-concurrency/raw/gateA --base base
```

## 8. Not done / open

* **vLLM at `--max-num-seqs` 16 on the 35B**: no record exists and the service on spark2 is capped at 2. One window (restart of the retro service with `--takes-down retro`, three rounds of `--concurrency 1,4,8,16`, restore) would give the real gap at 4 - 16 streams. Needs a user decision (it stops a role).
* **Persistent kernel for K = 4096 at M >= 13** (GDN out, attention o: 3.3 ms of the 50.3 ms step is the chunked path): stage x as e4m3 bytes when W8A8 (half the shared memory); W8A16 rows would still need the chunked kernel.
* **MoE grouping** (gate_up + down = 23 ms a step, 42 %): share an expert's weight load across the rows routed to it. Gain depends on routing overlap; measure with distinct prompts per stream, not localeval's one prompt.
* Streams' 4-stream aggregate did not move (+0.1 %): the 4-row steps are dominated by the MoE and the serialized prefill of the burst, where this change saves little.
* The 512x2048 fp8 projection at M = 2 is 1 us slower with the persistent kernel (grid of 16 blocks); no served 2-3 stream run exists.
* `~/intarsiaml-scc/out` on spark2 (99 % full disk) holds nsys reports and ~76 MB trace CSVs; delete when the lane is closed.
* The lane-skew survivor above (test sensitivity of the fp8 rows suite to cross-lane association); a data class with products that round (f32 activations) would close it.
* The unexplained 0.6 - 0.8 % solo-step offset of the new binary (the kernels of the solo path are unchanged).
* Work folders kept on request: `/opt/code/intarsiaml/.worktrees/scc` and `~/intarsiaml-scc` on spark2 (binaries `bin/base`, `bin/dev`, `bin/dev2`, `bin/dev3`, the synced tree, `out/` logs and nsys reports).
