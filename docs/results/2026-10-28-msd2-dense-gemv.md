# msd2: dense GEMVs at 2..16 rows, Qwen3.6-35B-A3B NVFP4 multi-stream decode on one GB10 (2026-10-28)

Lane `msd2`, branch `lane-msd2/dense-gemv`, base `origin/main` `fec300c65` (#221). spark1 (spark-4e77, GB10 sm_121), `serve --backend cuda --max-batch 16 --max-seq 8192
--mem-budget 40`. Raw: `2026-10-28-msd2-dense-gemv/raw/`. Scripts: `scripts/msd/` (from lane msd, #222, retargeted; `w1.sh`, `w2.sh`, `cpu-build.sh`, `evt.py` new).
GPU windows used: **2 of 2** (W1 #6553 profile + micro + tests, W2 #6573 identity + batch-real + A/B + dev profile); everything else was `--load cpu` builds or sub-3-minute
correctness / micro jobs. Status: **draft PR, gates not met** (+3.4 % at 16 streams, +1.8 % at 8 against +5 / +3 %). The four new paths are bitwise exact and stay in the
tree **opt-in** (default off).

## Result

1. **The corrected profile (burst with `ignore_eos`, base) shows no host gap**: 0.41 ms per step at 16 streams (0.8 %), 0.39 ms at 8 (1.2 %). #222's 5.8 / 3.0 ms was entirely
   its f32-logits copy. Kernel time is the whole step, so the lane went to the dense GEMVs.
2. The fp8 projections at K = 2048 are already near the solo kernel in an evicted micro-timing (16 rows 8192x2048: 88.7 us vs 71.9 us for one row, 189 vs 233 GB/s), so the
   room was in the small and odd calls: the K = 4096 GDN out / o_proj at 13..16 rows (chunked kernel), the latency-bound small-N bf16 and nvfp4 calls, and the lm_head at 8..16
   rows. Four exact changes (section 3) take **1.95 ms off a 48.9 ms step at 16 streams and about 1.1 ms of GEMV time off a 33.3 ms step at 8** (nsys, section 2).
3. Serve A/B, 3 interleaved rounds, localeval `speed`, 1k prompt, 256 tokens (the #213 / #222 harness): **16 streams 260.2 -> 268.2 tok/s (+3.4 %), 8 streams 212.4 -> 216.9
   (+1.8 %), 4 streams +1.2 %, 1 stream -0.6 %** (79.4 -> 78.9; inside the 1 % band). Misses both speed gates.
4. Exactness holds everywhere: `rows_exact` (3 new tests, every form x M 1..16), served-token identity base vs dev 21/21 cells (9408 ids), `batch_cuda_real` on the 35B
   (4290 (row, step) pairs bitwise), `plan show` byte-identical for 8B, 35B and Gemma-4.

## 1. Corrected profile (base, W1)

nsys `--cuda-graph-trace=node`, `scripts/msd/prof2.sh base base`, burst of N identical 3.7k-char prompts, 256 tokens, `ignore_eos` (`raw/base-b16`, `raw/base-b8`; `gaps.py`,
`shapes.py` over a decode window).

| per decode step | B=16 | B=8 |
|---|---:|---:|
| step span (graph start to next) | **48.87 ms** | **33.29 ms** |
| host / idle gap between steps | **0.41 ms (0.8 %)**, p50 396 us | **0.39 ms (1.2 %)**, p50 388 us |
| D2H per step | 0.000 MB (48 us) | 0.000 MB (2 us) |
| routed MoE (`moe_dec_gate_up` + `moe_dec_down_v2`) | 21.96 | 13.24 |
| fp8 rows (GDN in_qkv / in_z, attn q: persistent; out / o_proj: chunked; k / v) | 5.18 + 3.14 + 0.40 = 8.72 | 6.68 + 0.27 = 6.95 |
| GDN (`gdn_delta_split`, `gdn_conv_rows`) | 8.88 | 4.35 |
| lm_head (nvfp4 248320x2048) | 2.09 | 1.65 |
| shared expert nvfp4 (gate / up, down) | 0.92 + 0.47 | 0.79 + 0.28 |
| bf16 router / GDN b,a / shared-expert gate | 0.65 + 0.76 + 0.48 = 1.89 | 0.46 + 0.46 + 0.29 = 1.21 |
| attention (`splitkv`, rope, kv append, combine) | 2.4 | 1.4 |
| norms, top-k, glue | 0.7 | 0.6 |

The host gap is far under the 4 ms threshold of the brief; nothing host-side to fix. (The p50 gap of about 0.39 ms is the graph launch plus sampling turnaround per step.)

## 2. Where the change lands (nsys, base vs dev with the knobs on, ms per step; `raw/dev-b16`, `raw/dev-b8`, W2)

| kernel group | B=16 base -> dev | B=8 base -> dev |
|---|---:|---:|
| fp8 K=2048 calls (70 / step) | 5.18 -> 5.11 (`fp8_gemv_rows_q`, 2x8) | 6.68 -> 6.76 (unchanged path) |
| fp8 K=4096 out / o_proj (40) | **3.14 -> 2.64** (chunked -> e4m3-staged persistent) | (persistent already) |
| fp8 k / v 512x2048 (20) | 0.40 -> 0.35 | 0.27 -> 0.26 |
| lm_head | **2.09 -> 1.85** | **1.65 -> 1.34** |
| nvfp4 shared expert gate / up (80) | 0.92 -> 0.85 | **0.79 -> 0.52** |
| bf16 small-N (140) | **1.89 -> 0.69** | **1.21 -> 0.68** |
| routed MoE (untouched) | 21.96 -> 22.16 | 13.24 -> 13.84 |
| **step span** | **48.87 -> 46.92 (-4.0 %)** | **33.29 -> 32.90 (-1.2 %)** |

At 8 streams the GEMV changes save 1.1 ms, but the untouched MoE kernels came out 0.6 ms slower in the dev profile (run-to-run spread of the expert stream; the A/B rounds
show the same scatter, 8-stream deltas +1.0 / +1.8 / +4.5 %).

## 3. What changed (all in `kernels/cuda`, selection inside the launchers, no planner / registry change)

| knob (opt-in) | what | exact because |
|---|---|---|
| `INTARSIA_CUDA_FP8_ROWS_Q=1` | `fp8_gemv_rows_q_kernel`: persistent fp8 rows kernel that stages W8A8 activations as e4m3 **codes** (M x K bytes, half the bf16 staging), so 16 x 4096 fits; used for K = 4096 at M >= 13, every call at M >= 12, and N <= 1024 at M >= 4 (shape 1x16). W8A16 (decided on the device) stages bf16 row groups of M / 2. | e4m3 -> f16 -> f32 gives the float the bf16 image gives; same lane-strided blocks, fmaf chain, `acc += s`, warp tree per (m, n) |
| `INTARSIA_CUDA_BF16_ROWS_SPLIT=auto` | row-group split (`gemv_split_kernel`): block (bx, g) runs rows g*R.. through `gemv_body<rows>` itself; R = largest of 8 / 4 / 2 leaving 96 blocks, else 1 | `gemv_body` per row is unchanged; only which block runs which rows |
| `INTARSIA_CUDA_NVFP4_ROWS_SPLIT=auto` | the same split for the nvfp4 rows launcher (`nvfp4_gemv_split_kernel`, body refactored out of `nvfp4_gemv_tile_kernel`) | same body per (m, n) |
| `INTARSIA_CUDA_NVFP4_ROWS_FULL=auto` | `nvfp4_gemv_full_kernel`: persistent, x staged once in shared memory, for M >= 5 and N >= 16384 (the lm_head); 2x16 below 12 rows, 4x8 from 12 | same arithmetic as the tile body; x read from shared memory instead of L2 |

Also: the dynamic shared-memory cap of the persistent kernels (`fp8_gemv_rows_full_kernel` on main too) is now raised once per kernel and device to 96 KB instead of to each
call's size. Setting it per call is a real race between host threads launching the same kernel at different K (one thread can shrink the cap between another's set
and launch), and the likely cause of the one launch error W1's `rows_exact` run hit in `nvfp4_rows_are_bitwise_the_one_row_gemv` (4 threads). Its message was not captured,
and the same test passed on that build before the fix too (#6570), so the cause is not proven; the parallel tests also set and read environment variables concurrently.

Evicted micro-timing (`rows_exact` `timing_rows_evict`: each launch reads a different weight copy from a pool of about 192 MB; `raw/micro-evict-1.tab`, W1, quiet GPU), 16 rows, us:

| call | base | new | solo (1 row) |
|---|---:|---:|---:|
| fp8 2048x4096 | 76.8 | 54.4 .. 57.8 | 41.5 |
| fp8 8192x2048 | 88.7 | 82.5 | 71.9 |
| fp8 512x2048 | 20.5 | 15.8 | 9.4 |
| nvfp4 lm_head 248320x2048 (8 / 16 rows) | 1593 / 2080 | 1282 / 1922 | 1176 (2 rows) |
| nvfp4 512x2048 | 13.1 | 10.8 | |
| bf16 32x2048 / 1x2048 / 256x2048 | 14.3 / 11.2 / 14.3 | 4.1 / 4.1 / 9.9 | |

(Host-timed back-to-back launches: the small-kernel figures sit at a 4 us launch floor; the in-graph nsys durations in section 2 are the reliable ones for them.)

## 4. Serve A/B (W2, `scripts/msd/le-window.sh w2 3 base dev`, `raw/w2/`)

localeval `speed --sizes 1024 --gen 256 --reps 3 --concurrency 1,4,8,16 --thinking off`, a fresh server per arm, arms interleaved (even rounds reversed), same harness as #213 / #222.
Dev ran with the four knobs on (they were the default in that build; the opt-in flip after the A/B changes only the unset value).

| streams | base tok/s (r1, r2, r3) | dev tok/s (r1, r2, r3) | median of paired ratios | gate |
|---|---|---|---:|---|
| 1 | 79.3, 79.5, 79.4 | 79.3, 78.9, 78.9 | **-0.6 %** (0.0 / -0.8 / -0.6) | not slower beyond 1 %: pass |
| 4 | 162.3, 164.7, 163.4 | 164.3, 167.2, 165.1 | +1.2 % | (none) |
| 8 | 212.4, 212.4, 214.8 | 216.2, 221.9, 216.9 | **+1.8 %** (1.8 / 4.5 / 1.0) | >= +3 %: **fail** |
| 16 | 260.2, 260.9, 259.5 | 269.4, 267.4, 268.2 | **+3.4 %** (3.5 / 2.5 / 3.4) | >= +5 %: **fail** |

Single-stream 1k: decode 84.0 -> 83.8 tok/s (0.998), prefill 5322 -> 5342 (1.004). TTFT unchanged at every concurrency. vLLM at 16 streams is 340.1 (#213): 0.76x -> 0.79x.

## Gates

| gate | result |
|---|---|
| >= +5 % at 16 streams, >= +3 % at 8, 1 stream not slower (3 interleaved rounds) | **not met**: +3.4 % / +1.8 % / -0.6 % |
| bitwise equal to main; batched == solo on the 35B (greedy ids) | pass: served tokens base vs dev 21/21 cells, 9408 ids (`raw/ident/`); `batch_cuda_real` 35B 2 passed, 4290 (row, step) pairs bitwise (`raw/batch-real.txt`) |
| `plan show` byte-identical, 8B / 35B / Gemma-4 | pass (`raw/plan.txt`, `raw/plan-final.txt`) |
| `rows_exact` | pass, 9 tests incl. 3 new (`fp8_rows_q_...`, `rows_split_...`, `nvfp4_rows_full_...`), repeated 4x at 4 threads (`raw/pre-w2.txt`, `raw/optin-check.txt`) |
| Mac `cargo test --workspace --release` | pass (`raw/mac-cargo-test.log`, `raw/mac-cargo-test-final.log`) |
| CI | see the PR |

## Not done, next

1. What is left of the dense GEMVs is small: the fp8 K = 2048 calls run at about 1.07..1.23x their one-row time (0.5 ms at 16 streams) and the remaining small calls are at
   their launch floor. Fusing the same-input small calls (router + shared-expert gate, GDN b + a) into one rows launch would save about 0.3 ms per step; that needs a rows
   form of the pair ops (an engine change).
2. The 16-stream gap to vLLM is now mostly the routed experts (22 ms, 45 %) and GDN (8.9 ms, at its f32-state ceiling). #222's row-grouped experts did not pay off
   (independent routing); bf16 GDN state is the only large lever left and changes numerics.
3. The knobs can be turned on as a set (`INTARSIA_CUDA_FP8_ROWS_Q=1 INTARSIA_CUDA_BF16_ROWS_SPLIT=auto INTARSIA_CUDA_NVFP4_ROWS_SPLIT=auto INTARSIA_CUDA_NVFP4_ROWS_FULL=auto`)
   for +3 % at 16 streams with identical tokens, should a later lane stack a further gain on them.
