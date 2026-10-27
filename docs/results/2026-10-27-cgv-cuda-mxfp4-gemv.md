# cgv: CUDA MXFP4 GEMV, Qwen3.8-27B decode on the Spark (2026-10-27)

Lane `cgv`, branch `lane-cgv/cuda-mxfp4-gemv`, base `origin/main` `8f93d2397`. DGX Spark **spark1** (GB10 sm_121), `Qwen3.8-27B` MXFP4 (Quark, `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`),
recipe defaults (W4A16, tp=1, CUDA graphs, prefill chunk 256), `intarsia serve --backend cuda --max-seq 20480 --mem-budget 45`. Raw logs: `2026-10-27-cgv-cuda-mxfp4-gemv/raw/`.
Scripts: `scripts/cgv/`. Status: **draft PR, not merged**: the lane's gate "MTP-off decode >= +5 % at 1k and 4k" cannot be met by any GEMV change (section 1) and the change
misses it (1.000x). The change itself is a bitwise, knob-guarded MTP-on speedup of **+7 to +14 %** (section 4).

## Result

1. **The MXFP4 GEMV is 80 % of one MTP-off decode step, and it is already at about 95 % of what this box's DRAM streams.** The "255-263 GB/s achievable" of the s27d microbench came from
   rotating only 4 weight copies (190 MB). With 16 or 64 copies (what a token does: 15.5 GB) the same kernel streams 237 GB/s, and a plain coalesced read kernel 239 GB/s. In the engine
   the M=1 GEMV runs at 217-229 GB/s per launch. Ceiling for the whole MTP-off step at 239 GB/s: 64.7 ms (15.5 tok/s); measured 74.8 ms (13.4 tok/s). If every GEMV launch reached the
   plain-read rate the step would gain 5.5 ms (8 %), which no kernel change gets, because the rest is launch ramp and tail per kernel (about 500 GEMV launches per token).
2. **The real gap is at M = 2..4, the MTP verify pass** (67 % of an MTP-on pass): the one-row-at-a-time GEMV costs 1.13x its M=1 time at M=3 (189 GB/s in the pass, s27d's number).
   A multi-row kernel (`matmul_mxfp4_gemv_mr`, bitwise the old one per element) brings M=3 to 1.01-1.06x the M=1 time (207 GB/s in the pass).
3. End to end (4 paired rounds, `INTARSIA_CUDA_MXFP4_GEMV_MR=1` vs the same tree at `main`): **MTP on 1.088x at 1k, 1.074x at 4k, 1.139x at 16k**; MTP off 1.000x / 0.999x / 1.000x.
4. #209's open item (MTP-on decode at 1k 0.984x) re-measured with 5 interleaved rounds: **1.013x** (1.004-1.051), MTP off 1.003x. It was noise.

## 1. GEMV share of one MTP-off step, and the ceiling

nsys (`scripts/cgv/profs.sh`, `raw/prof/`), `main`, 4210-token prompt, 192 tokens, `--spec off`: decode 13.38 tok/s, 191 steps in 14287.6 ms (74.8 ms), kernel time 14179 ms.

| kernel | ms per step | share of the step | GB per step | GB/s |
|---|---:|---:|---:|---:|
| `mxfp4_gemv_kernel` (496 launches) | 59.6 | 79.6 % | 12.94 | 217 |
| lm_head `gemv_kernel` (bf16, 248320 rows) | 10.8 | 14.4 % | 2.54 | 235 |
| attention, GDN, norms, glue | 3.8 | 5.1 % | | |
| gaps | 0.6 | 0.8 % | | |

Bytes per step 15.48 GB (E2M1 + e8m0 scales: 128 gate/up, 64 down, 64 out/o, 48 qkv, 48 z, 16 q, 32 k/v, 96 a/b launches + the head). Ceilings: at the s27d figure of 260 GB/s 59.5 ms (16.8 tok/s);
at the measured DRAM streaming rate (239 GB/s, below) 64.7 ms (15.5 tok/s). The measured step is 86 % of the second.

**The ceiling is 239 GB/s, not 260.** `scripts/cgv/mxfp4bench/bench.cu`, `STREAM=1`: a plain coalesced 16-byte read of a 47.3 MB buffer, rotated over NB buffers:

| NB (buffers) | 2 | 4 | 16 | 64 |
|---|---:|---:|---:|---:|
| plain read GB/s | 255-265 | 247-255 | 239 | 239 |
| the baseline M=1 GEMV, gate/up | | 263 | 236 | 236 |

Rotating 4 copies (190 MB) lets part of the traffic come from somewhere other than DRAM (a cache or the memory controller's page state: not investigated); the engine streams 18 GB per token and
sees the NB=16/64 rate. In the engine the M=1 gate/up GEMV takes 207 us (47.35 MB, 229 GB/s, 96 % of 239); qkv 124.5 us (224 GB/s); z 77.0 us (217); q 147.8 us (226); down about 214 us (221);
out/o about 77 us (217); k/v 17.6 us (158); a/b 5.9 us. The shortfall against 239 is fixed cost per launch plus wave quantisation: the N=5120 shapes (down, out/o) launch 640 blocks on 96
resident slots (6.7 waves). Splitting those two shapes finer would save about 1.2 ms per step (1.6 %); it is not in this PR.

So the first-listed lever of the lane (close the in-pass M=1 gap to 255-263 GB/s) is not there: the whole step cannot gain more than 8 % even with every launch at the plain-read rate, and
the lane gate of +5 % MTP-off is out of reach. What the profile does show is the M=3 gap (next sections).

## 2. Microbench (`scripts/cgv/mxfp4bench`, `raw/microbench/bench-final.txt`)

Baseline `mxfp4_gemv_kernel` against `mxfp4_gemv_mr_kernel<M, R>` (R output rows per warp), the 27B's six shapes, M = 1..4, best of 5 blocks of 60 launches, weights rotated over **64** copies
(the engine's regime; NB=4 is in the raw file, same ordering). `prod` is the production launcher (R = 2 at M = 2 and 3, R = 4 at M = 4). Every output of every variant is bitwise equal to the baseline's (NB=4 and 64).

| shape (N x K) | M | base us (GB/s) | R1 | R2 | R4 | R8 | prod us (GB/s) | prod vs base |
|---|---:|---|---:|---:|---:|---:|---|---:|
| gate/up 17408 x 5120 | 1 | 200.7 (236) | 205.4 | 206.0 | 204.9 | 207.1 | 204.6 (231) | -1.9 % |
| | 2 | 206.8 (229) | 203.7 | 203.2 | 205.7 | 212.3 | 202.2 (234) | +2.3 % |
| | 3 | 220.9 (214) | 200.3 | 204.2 | 209.5 | 211.7 | 203.5 (233) | +8.6 % |
| | 4 | 252.9 (187) | 221.2 | 229.1 | 207.6 | 216.1 | 207.5 (228) | +21.9 % |
| down 5120 x 17408 | 1 | 209.7 (226) | 214.3 | 212.0 | 212.4 | 214.8 | 212.5 (223) | -1.3 % |
| | 2 | 212.4 (223) | 213.3 | 214.3 | 213.3 | 225.6 | 214.1 (221) | -0.8 % |
| | 3 | 223.2 (212) | 209.9 | 217.0 | 226.9 | 222.7 | 217.1 (218) | +2.8 % |
| | 4 | 276.2 (171) | 224.3 | 235.9 | 225.5 | 229.9 | 225.6 (210) | +22.4 % |
| out/o 5120 x 6144 | 1 | 75.6 (221) | 75.1 | 75.5 | 75.5 | 77.4 | 75.1 (223) | +0.7 % |
| | 2 | 78.8 (212) | 74.8 | 75.8 | 75.8 | 82.1 | 76.0 (220) | +3.7 % |
| | 3 | 83.0 (202) | 75.4 | 77.6 | 81.6 | 81.6 | 77.0 (217) | +7.8 % |
| | 4 | 94.1 (178) | 81.9 | 85.7 | 84.6 | 84.3 | 81.8 (204) | +15.0 % |
| qkv 10240 x 5120 | 1 | 121.2 (230) | 122.3 | 122.7 | 122.9 | 124.2 | 122.0 (228) | -0.7 % |
| | 2 | 125.0 (223) | 120.8 | 120.4 | 123.9 | 132.6 | 121.1 (230) | +3.2 % |
| | 3 | 135.2 (206) | 121.1 | 122.4 | 126.2 | 132.3 | 122.0 (228) | +10.8 % |
| | 4 | 156.0 (179) | 131.1 | 138.4 | 125.2 | 136.8 | 125.1 (223) | +24.7 % |
| z 6144 x 5120 | 1 | 75.5 (222) | 74.6 | 75.8 | 75.3 | 76.5 | 74.8 (223) | +0.9 % |
| | 2 | 77.9 (214) | 74.8 | 74.7 | 75.8 | 79.8 | 74.4 (225) | +4.7 % |
| | 3 | 82.9 (202) | 75.2 | 75.5 | 78.7 | 79.2 | 75.3 (222) | +10.1 % |
| | 4 | 93.5 (179) | 80.3 | 85.6 | 78.0 | 81.8 | 78.2 (214) | +19.6 % |
| q 12288 x 5120 | 1 | 143.7 (233) | 146.3 | 145.6 | 147.1 | 147.9 | 145.7 (230) | -1.4 % |
| | 2 | 147.9 (226) | 144.4 | 144.2 | 147.1 | 152.7 | 145.3 (230) | +1.8 % |
| | 3 | 158.2 (211) | 146.1 | 146.9 | 150.4 | 152.1 | 146.6 (228) | +7.9 % |
| | 4 | 181.1 (184) | 156.5 | 164.7 | 148.9 | 156.6 | 149.6 (223) | +21.1 % |

M=1 is flat (the old kernel stays there: the registry routes only M = 2..4 to the new one). At M=3 and 4 the new kernel streams at the M=1 rate (218-233 GB/s). Two parts: a compile-time row
count alone (R1) is most of the M=3 gain (the old kernel loops to a run-time row count over an 8-entry accumulator array); R > 1 shares the activation loads across rows
and keeps several rows' weights in flight (best at M=4: R4 -12 % against R1 on gate/up). R8 loses (too few blocks). What was not tried: the L1/occupancy counters (no ncu run), so the
reason R1 beats the old code at M >= 2 is not pinned down; the kernel comment states only what was measured.

## 3. In-pass effect (nsys, 4210-token prompt, MTP on, `raw/prof/*-on.shapes.txt`)

77 verify passes in both runs (acceptance 74.5 %, 2.48 tokens per pass, same ids). GEMV time per pass: `main` 68.5 ms (188.9 GB/s), branch 62.6 ms (206.5 GB/s); the M=1 step is 59.6 ms (217).

| launch (N) | `main` median us (M=3) | branch median us (M=3) | M=1 median us (`main`) |
|---|---:|---:|---:|
| gate/up 17408 | 234.3 | 210.0 | 207.0 |
| down + out/o (5120, two populations: mean) | 168.8 | 161.5 | 149.7 |
| qkv 10240 | 141.7 | 130.6 | 124.5 |
| z 6144 | 88.9 | 81.5 | 77.0 |
| q 12288 | 169.4 | 156.5 | 147.8 |
| k/v 1024 | 23.4 | 17.3 | 17.6 |

Decode of that run: 24.20 tok/s on `main`, 25.71 on the branch (+6.2 %); `--spec off`: 13.38 / 13.34 (identical kernels). Remaining M=3 excess over M=1: 0-5 %.

## 4. Served A/B (localeval speed, interleaved, `scripts/cgv/le-window.sh`)

Same box, same window, a fresh `intarsia serve` per arm and round, round r reversed against r-1, 3 timed reps after a warmup, 128 generated tokens, thinking off. Prompts 992-1000 /
4090 / 16.6k tokens. Arms: `main` (tree `8f93d2397`) and the branch (`INTARSIA_CUDA_MXFP4_GEMV_MR` default on). The off arms were checked in their serve logs (no `--spec mtp` line).
n = 4 rounds. Spread = (max - min) / median of the arm over the rounds. `raw/gate/`, `raw/gate-pairs.md`.

| size | mode | main tok/s (rounds) | branch tok/s (rounds) | paired ratios | median | spread main / branch |
|---|---|---|---|---|---:|---|
| 1k | MTP off | 13.69 13.67 13.68 13.68 | 13.68 13.71 13.67 13.69 | 1.000 1.003 0.999 1.001 | **1.000** | 0.1 % / 0.3 % |
| 4k | MTP off | 13.46 13.44 13.43 13.43 | 13.45 13.45 13.41 13.42 | 0.999 1.000 0.998 0.999 | **0.999** | 0.2 % / 0.3 % |
| 16k | MTP off | 12.82 12.82 12.81 12.76 | 12.80 12.82 12.80 12.80 | 0.999 1.000 0.999 1.003 | **1.000** | 0.4 % / 0.2 % |
| 1k | MTP on | 21.80 22.57 23.09 21.89 | 24.15 24.10 24.23 24.58 | 1.107 1.068 1.049 1.123 | **1.088** | 5.8 % / 2.0 % |
| 4k | MTP on | 23.69 22.92 22.54 23.69 | 24.44 25.30 25.50 24.74 | 1.032 1.104 1.132 1.044 | **1.074** | 5.0 % / 4.3 % |
| 16k | MTP on | 20.73 21.94 20.70 19.58 | 24.08 23.57 24.09 21.86 | 1.161 1.074 1.163 1.117 | **1.139** | 11.4 % / 9.3 % |

MTP on varies by round with the acceptance of that round's drafts (the spread columns); every one of the 12 paired MTP-on ratios is above 1.03, and the rounds-median ratios sit above
the spread of the baseline arm. The MTP-off ratios are 1.000 within 0.3 %: the M=1 kernel is the old one.

### #209 re-check (the open item of `2026-10-27-cmx-cuda-mxfp4-prefill.md`)

`385ff6ae1` (#209 merged) has first parent `8c562765b` (#208). Arms: `pre209` = `8c562765b`, `main` = `8f93d2397`. The diff between `385ff6ae1` and `8f93d2397` in kernels, crates and recipes is #210
(LFM2: `gdn.metal`, `crates/intarsia-models`, tests), which does not touch the 27B CUDA path. 1k, 256 generated tokens, 5 interleaved rounds (`raw/a209/`, `raw/a209-pairs.md`);
prefill (`main` 786, `pre209` 64 tok/s) is the known #209 effect.

| mode | pre209 tok/s (rounds) | main tok/s (rounds) | paired ratios | median | min-max |
|---|---|---|---|---:|---|
| MTP off | 13.71 13.62 13.62 13.64 13.61 | 13.69 13.70 13.67 13.67 13.65 | 0.998 1.006 1.003 1.002 1.003 | **1.003** | 0.998-1.006 |
| MTP on | 22.60 23.61 23.19 23.66 22.58 | 23.76 23.72 23.50 23.80 23.53 | 1.051 1.004 1.013 1.006 1.042 | **1.013** | 1.004-1.051 |

No round is below 1.00. The 0.984x of the CMX doc (3 rounds, per-round spread up to 3 % at 1k, here 4.6 % for `pre209` MTP-on) was noise; #209 does not slow MTP-on decode.

## What changed

- `kernels/cuda/mxfp4.cu`: `mxfp4_gemv_mr_kernel<OutT, M, R>` and `ik_matmul_mxfp4_gemv_mr` / `_mr_rows` (declared in `ik_cuda_abi.h`). One warp owns R output rows (R = 2 for M = 2, 3; R = 4 for
  M = 4), loads each activation group once for all R rows, keeps the weights of the R rows in flight together, and runs for each (row, activation row) the same chain as `mxfp4_gemv_kernel`:
  32 `fmaf` over a group, one `fmaf` with the group scale, groups in the order lane, lane + 32, ..., then `ik_warp_sum`. Does not read `act` (W4A16 only). N not a multiple of R: the tail warp
  re-reads the last row and does not store the extra rows.
- Registry (`crates/intarsia-backend-cuda/src/quantmoe.rs`, `kernels.rs`, `device_quantmoe.rs`): `matmul_mxfp4_gemv_mr` (ordinary calls, priority 20) and `matmul_mxfp4_gemv_mr_rows`
  (`exact_rows` calls: verify passes and batched steps, priority 5 over the GEMV rows twin's -1), both for W4A16 calls of M = 2..=4 with K % 32 == 0. M = 1, M >= 5, W4A4 and W4A8 calls are
  unchanged. Knob `INTARSIA_CUDA_MXFP4_GEMV_MR=0` (default on, `docs/tuning.md`) restores the old kernels (`kernels.rs:283`).
- Tests: `crates/intarsia-backend-cuda/tests/mxfp4_gemv_mr.rs` (new, GPU), `tests/rows_registry.rs` pins, the W4A16 sweep cases for M = 2..4 and N tails in `crates/intarsia-core/src/quant/cases.rs`.

## Gates

| Gate | Result |
|---|---|
| MTP-off decode >= +5 % at 1k and 4k, not slower at 16k, >= 4 interleaved rounds | **FAIL**: 1.000x / 0.999x / 1.000x (n = 4). Not reachable: M=1 is the old kernel at 95 % of the DRAM streaming rate (section 1) |
| (not a lane gate) MTP-on decode | 1.088x / 1.074x / 1.139x at 1k / 4k / 16k, n = 4, every paired round above 1.03 |
| 256 greedy ids bitwise vs `main` | **PASS**: 4 prompts (41, 34, 556, 3870 tokens), `--spec off`: branch, branch with knob 0 and `main` identical; `--spec mtp --spec-k 2`: branch == `main` == `--spec off` ids (`raw/ids/ids.txt`) |
| batched == solo | **PASS**: `real_35b_batched_decode_is_bitwise_the_solo_step` on the 27B, 4290 (row, step) pairs bitwise identical, N = 2..16 (these steps run `matmul_mxfp4_gemv_mr_rows` at N = 2..4) |
| verify rows == single-row steps (`verify_rows_real`, q <= 4, ctx 17..4000) | **PASS** with the engine's per-row verify attention (`INTARSIA_CUDA_VERIFY_ATTN_PER_ROW=1`, what `--spec mtp` runs): 0 windows with diffs, and with knob 0. Without it 4 windows differ (max |dlogit| 0.115), the **same 4 windows and values with knob 0**: the multi-row attention, not this change (as in the CMX doc) |
| `kernels test --backend cuda` admits the kernel | **PASS**: `matmul_mxfp4_gemv_mr` and `_mr_rows` ADMITTED, 16 cases each (M = 2..4, N = 1, 3, 135, 136, 137, 1000, 5120, K up to 5120, bf16 and f32 out), rel_l2 2.975e-5 (the same as the GEMV's); the whole CUDA quick sweep 81 kernels / 585 cases, nothing rejected |
| >= 3 planted mutants caught | **PASS**: 8 planted (`scripts/cgv/mutants.py`): x row stride, scale row of row r, weight row of row r, store column, group scale, e8m0 of the wrong row, tail store guard: 7 caught by `kernels test` (the tail store guard only after N = 1 cases were added to the sweep; before that `tests/mxfp4_gemv_mr.rs` caught it); the tail row clamp (an out-of-bounds read whose result is discarded) changes no output and is caught by compute-sanitizer memcheck (clean tree 0 errors, mutant 15) |
| `plan show` 8B, 35B (and Gemma-4) unchanged | **PASS**: json and text byte-identical (8B md5 `b1a1150e`/`7742cef8`, 35B `ca8cba7e`/`2c7deccd`, Gemma-4 `2a659412`/`ec4ef23e`). The 27B plan differs by exactly the two new kernel rows |
| GPU unit tests (`INTARSIA_REQUIRE_GPU=1`) | **PASS**: `mxfp4_gemv_mr` 2/2 (M = 1..4 x 10 shapes x 2 output dtypes, both kernel names, byte for byte the one-row GEMV; the first m rows of a 4-row call equal the m-row call), `mxfp4_mma` 3/3, `rows_exact` 6/6, `rows_registry` 3/3 |
| Mac `cargo test --workspace --release` | **PASS**: 1075 passed, 0 failed (`raw/cargo-test-workspace-mac.log`; a first run found `kernel_list_is_unchanged_by_core_op_fusion_p1` and `every_cuda_kernel_has_a_quick_case` red until the two kernel names and the sweep cases were added) |
| CI green | CI_RESULT |

## Not done, open

- The MTP-off step cannot gain 5 % from the GEMV. Remaining headroom measured here: finer blocks for the N=5120 shapes (about 1.2 ms of 74.8 per step), fusing gate and up (one launch less
  in 64 layers, about 0.3 ms), the lm_head at 235 of 239 GB/s. None reaches 5 %.
- MTP-on, remaining excess of M=3 over M=1: 0-5 % per shape. A fused W4A16 draft+verify path would be a different change.
- No ncu counters (L1 wavefronts, occupancy) were collected; the reasons are the measured ones only.
- 16k MTP-on rounds spread 9-11 %; the paired medians are what to read.
- Numbers are one box (spark1) with spark1-vllm up and idle during the windows (quiet slots).
- The commit trailer names Opus 5.5 as the lane brief requires; the session that did the work was Sonnet 5.5.

## Fleet log

Owner `claude-f31ccf91`, spark1 only. fleetq: #6370/#6371 (checkpoint copy 19 GB from endor, net+disk), builds #6372 (main), #6374 (pre209), #6381 (dev), #6376-#6378 (nvcc), #6384 (tests), quiet GPU windows
#6379 (microbench), #6383 (profiles, clock sample, bench probes), #6385 (profiles, plan, GPU tests, ids, admission, mutants, microbench), #6387 (A/B); 4 GPU windows. `~/intarsiaml-cgv` on spark1 and
`.worktrees/cgv` removed at the end of the lane. spark1-vllm (flight/guidance/capcom) stayed up.
