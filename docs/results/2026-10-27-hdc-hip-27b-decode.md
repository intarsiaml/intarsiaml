# Lane HDC: Qwen3.8-27B MXFP4 tp=2 decode on the two R9700s (HIP), attribution and one fix (2026-10-27)

Branch `lane-hdc/hip-27b-decode`, built on origin/main `8f93d2397`. Host tatooine, 2x R9700 (gfx1201), `--tp 2 --devices 0,1`, W4A8 (`mxfp4_act = fp8`, the recipe default), the Quark MXFP4 checkpoint, `--max-seq 20480`, MTP k2 default in the "MTP on" rows, `--spec off` in the "MTP off" rows. Every GPU number comes from a fleetq slot that takes radiance down and brings it back (4 windows: #6375, #6380, #6386, #6388; each ended with `/v1/models` answering, `raw/*/guard.log`). Raw data, logs and the rocprofv3 summaries: `docs/results/2026-10-27-hdc-hip-27b-decode/raw/`, scripts `scripts/hdc/` (`drive.sh` runs the arms, `attr.sh` the profiles, `gates.sh` the gates, `analyze.py` / `gemvshapes.py` read the kernel traces, `speedtab2.py` pools the localeval runs).

## Headline

| | |
|---|---|
| Asked | Attribute one MTP-off decode step at tp=2, A/B the knobs on main, make one targeted fix if the attribution shows a clear item. |
| Where the step goes | 18.35 ms per step at 1k (54.5 tok/s). Weight bytes per rank and token are 7.93 GB, so the floor at 640 GB/s is 12.39 ms (80.7 tok/s): main runs at **67.5 %** of the bandwidth ceiling at 1k (67.4 % at 4k, 66.9 % at 16k). The MXFP4 GEMVs themselves are already at 84-95 % of 640 GB/s and the logits GEMV is at 100 %; the gap is **many small launches around them**: 320 activation-quantisation launches (0.68 ms), 128 all-reduces (1.2-1.6 ms), 177 norm launches (0.59 ms) and about 1.5 ms of idle between 1128 launches. |
| Knobs on main | `INTARSIA_TP_ARNORM=1`: **+0.6 / +0.7 / +0.7 %** at 1k / 4k / 16k, bitwise equal, below the +3 % bar, stays opt-in. `INTARSIA_HIP_WMMA_ROWS=1` (any `WMMA_TABLE`): **a no-op on the 27B** (plan byte-identical, speed 1.000 / 1.000 / 1.000), because the 27B decode has no fp8 dense weight and no bf16 router. Both together = ARNORM alone. |
| The fix | `INTARSIA_HIP_MXFP4_A8_FUSE` (quantise the activation inside the GEMV block, an existing knob, off) becomes **the default for single-row calls (M = 1)**, and the fused kernel now loads the first weight group of each warp before it quantises. No new numerics (the output hash of 512 launches is identical in the pre-pass, fused-all-M and new-default modes). MTP-off decode **+4.4 / +4.3 / +4.5 %** (n = 4 per arm, spread at most 1.2 %) at 1k / 4k / 16k: 54.5 / 53.8 / 50.8 to 56.9 / 56.1 / 53.0 tok/s. |
| Gates | All met (table below). Merged. |

## 1. Attribution: one MTP-off decode step, tp=2, 1k context, main `8f93d2397`

Method. `generate --spec off --profile-every 0 --ignore-eos --max-tokens 130` on a 997-token prompt under `rocprofv3 --kernel-trace --stats`, both ranks in one process, decode under hipGraph (`graph replays=130 (captures 1)`, the kernels of the graph appear in the trace). Steps are cut at the logits GEMV (one per step per rank); the table is the median over 109 steps (`scripts/hdc/analyze.py`, `raw/analysis/w1-base.txt`, same numbers in `w2-base.txt`). The unprofiled step is 18.35 ms (54.25-54.66 tok/s in 4 runs, `raw/w1/attr-1/gen-*.err`); the profiler adds dispatch cost per launch, so its period (20.8 ms) and its idle (3.98 ms) are inflated and not used. "Idle" below is the unprofiled step minus the union of the kernel intervals (the kernel durations are the profiler's).

Rank 0, per step (rank 1 differs only in the all-reduce, 1.21 ms, because it waits less):

| item | us/step | % of 18.35 ms | launches | note |
|---|---:|---:|---:|---|
| MXFP4 W4A8 GEMV (`mxfp4_a8_rows_kernel<1>`) | 11091 | 60.4 | 320 | 6.47 GB of weights + scales per rank: 583 GB/s average = 91 % of 640 |
| activation quantisation pre-pass (`mx_a8_quant_kernel`) | 676 | 3.7 | 320 | 2.1 us each, latency only |
| logits GEMV (bf16, vocab-parallel shard 1.27 GB) | 1985 | 10.8 | 1 | 640 GB/s: at the ceiling |
| all-reduce (`ar2_kernel`) | 1595 | 8.7 | 128 | 12.5 us each on rank 0 (9.4 on rank 1): wait for the peer included |
| norms (`rmsnorm_add` 486, gated rmsnorm 103) | 592 | 3.2 | 177 | 3.8 us per rmsnorm launch |
| GDN (`gdn_conv_delta`) | 340 | 1.9 | 48 | state read+write 151 MB/rank is 0.24 ms of that |
| attention (`splitkv`, qk-norm-rope, gate) | 214 | 1.2 | 32 | 1k context |
| SiLU-GLU, argmax, embedding, other | 321 | 1.7 | 102 | |
| **sum of kernels (union)** | 16792 | 91.5 | 1128 | |
| idle between launches (18.35 - 16.79) | about 1560 | 8.5 | | graph replay of 1128 kernels, about 1.4 us per launch |

GEMV efficiency by op (`raw/analysis/w1-base-gemv-shapes.txt`; ideal = (K/2 + K/32) bytes per row at 640 GB/s):

| op (rows x K per rank) | calls/step | median us | ideal us | % of 640 GB/s |
|---|---:|---:|---:|---:|
| gate+up pair (17408 x 5120) | 64 | 77.6 | 74.0 | 95 |
| down_proj (5120 x 8704) | 64 | 40.6 | 37.0 | 91 |
| GDN in_proj qkv+z (8192 x 5120) | 48 | 37.6 | 34.8 | 92 |
| attention q+gate (6144 x 5120) | 16 | 28.3 | 26.1 | 92 |
| GDN out / attention o (5120 x 3072) | 64 | 15.5 | 13.1 | 84 |
| GDN in_proj a,b (48 x 5120) | 48 | 3.4 | 0.2 | 6 (latency bound, 0.17 ms in all) |
| attention k,v pair (1024 x 5120) | 16 | 6.1 | 4.4 | 72 |

Bandwidth ceiling. Per rank and token: text layers 6.471 GB (MXFP4 weights and scales; 12.94 GB over two ranks, read from the safetensors header: MLP 9.09, GDN 2.96, attention 0.89), vocab-parallel lm_head shard 1.271 GB (the embedding table is one row), GDN recurrent state read+write 0.151 GB (48 layers x 24 value heads/rank x 128 x 128 x f32, twice), KV for the 16 attention layers 0.034 / 0.134 / 0.537 GB at 1k / 4k / 16k (2 KV heads per rank, head dim 256, bf16); MTP and vision weights are not read. At 640 GB/s (T2 measured 627 GB/s on this card):

| context | bytes/rank | floor ms | ceiling tok/s | main MTP off | % of ceiling | with this change | % of ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1k | 7.93 GB | 12.39 | 80.7 | 54.5 | 67.5 | 56.9 | 70.5 |
| 4k | 8.03 GB | 12.55 | 79.7 | 53.8 | 67.4 | 56.1 | 70.3 |
| 16k | 8.43 GB | 13.18 | 75.9 | 50.8 | 66.9 | 53.0 | 69.9 |

The 5.96 ms between the floor and the step at 1k: GEMV below 640 GB/s about 1.0 ms, quantisation pre-pass 0.68, all-reduce 1.6 (of which at most 0.6 ms is unavoidable latency at about 5 us), norms 0.59, GDN and attention beyond their bytes about 0.3, small kernels 0.3, idle about 1.5. No single item is large; the launch count (1128 kernels per step) is the common factor. The per-op table (`--op-table`, synchronised, `raw/w1/attr-1/op.err`) agrees on the ranking (`matmul_mxfp4` 199 ms, `all_reduce` 80 ms, `matmul_mxfp4_pair` 69 ms over 49 steps) but inflates every launch; it is a cross-check only. Host profile (`INTARSIA_HOST_PROFILE=1`) prints nothing for `--max-batch 1` (no batched scheduler), so the host gap comes from the trace, not from it.

T2's "rank 1 idles 8 ms per step" (before the vocab-parallel head) is gone: both ranks run the logits GEMV (1985 us each), rank 1's busy time is 16.37 ms against rank 0's 16.79.

## 2. The knobs on main, MTP off (A/B, interleaved)

`localeval speed --sizes 1024,4096,16384 --gen 128 --reps 3`, one fresh `serve` per item, 4 rounds in a rotated order (window W2, `raw/w2/steps.log`; the `both` arm failed there because localeval stats the `--recipe` label as a path and it was too long, rerun in W3 with a short label, `raw/w3`). Decode tok/s, median of the 4 run medians, (min-max); ratio to main.

| arm | 1k | ratio | 4k | ratio | 16k | ratio |
|---|---|---:|---|---:|---|---:|
| main | 54.49 (54.5-55.8) | 1.000 | 53.72 (53.6-53.7) | 1.000 | 50.77 (50.3-50.8) | 1.000 |
| `INTARSIA_TP_ARNORM=1` | 54.81 (54.7-54.9) | 1.006 | 54.08 (54.0-54.1) | 1.007 | 51.13 (51.1-51.1) | 1.007 |
| `INTARSIA_HIP_WMMA_ROWS=1` + `WMMA_TABLE=fp8a8=all,fp8a16=all,bf16=all` | 54.49 (54.5-54.5) | 1.000 | 53.74 (53.7-53.8) | 1.000 | 50.78 (50.7-51.0) | 1.000 |
| both (W3, n = 4) | 54.80 (54.5-54.8) | 1.006 | 54.06 (54.0-55.2) | 1.005 | 51.13 (51.1-51.4) | 1.007 |
| `INTARSIA_HIP_MXFP4_A8_FUSE=1` (old fused kernel, every M) | 55.99 (55.9-56.2) | 1.028 | 55.34 (55.3-55.4) | 1.030 | 52.43 (52.3-52.5) | 1.033 |
| ARNORM + FUSE=1 | 56.39 (56.2-56.5) | 1.035 | 55.71 (55.7-55.7) | 1.037 | 52.92 (52.8-54.2) | 1.042 |

* **WMMA small-row GEMV is a no-op for this model.** `plan show` of the 27B at tp2 is byte-identical (3331 B) under `INTARSIA_HIP_WMMA_ROWS=1` with the default table, with `bf16=all,fp8a16=1-16`, and with `fp8a8=all,fp8a16=all,bf16=all` (`raw/w1/attr-1/plan-*.txt`): the decode ops are `matmul_mxfp4_a8_gemv`, `matmul_mxfp4_pair_a8_gemv` and `logits_bf16_gemv`; the family covers dense fp8 and the bf16 router/gate GEMVs the 27B does not have. The timing arm is 1.000 at all three sizes with a spread of 0.1-0.4 %, which also measures the run-to-run noise of this method: MTP-off readings repeat within about 0.5 %.
* **ARNORM** fuses 128 of the 129 all-reduces with the following residual+rmsnorm (plan line `fused all-reduce + residual add + rmsnorm: 128 of the 129`); the trace shows 128 `rmsnorm_add` launches (0.49 ms) gone and the fused kernel taking 2.08 ms against 1.57 + 0.49 before. Launches 1128 to 1000, step 18.35 to 18.25 ms. It is bitwise (texts below) and is a small, real gain, but under the +3 % bar, so `TP_ARNORM_DEFAULT_ON` stays false.
* MTP on is secondary (5-10 % run spread, n = 3 per arm, window W2): main 88.2 / 78.4 / 74.9 tok/s; ARNORM 92.5 / 80.1 / 76.1 (x1.048 / 1.021 / 1.016); FUSE=1 88.0 / 80.7 / 73.9 (x0.998 / 1.029 / 0.987). All inside the spread.

## 3. The fix: fused activation quantisation for M = 1 decode, weights prefetched ahead of it

What the trace says. The pre-pass `mx_a8_quant_kernel` is 320 launches of 2.1 us per step (0.68 ms) and each one is a launch gap as well. The existing `INTARSIA_HIP_MXFP4_A8_FUSE=1` removes them (every GEMV block quantises the row into LDS) but was left off because the micro-timings of the GEMV alone looked level and the multi-row kernels got slower (2.5x at M = 8 on the big shapes). The step-level reading is different: it saves 0.5-0.6 ms per step (+2.8 to +3.3 %) because the 320 launches and their gaps go.

Changes (`kernels/hip/mxfp4_a8.hip`):

1. `a8_fuse_mode()`: unset (the default) fuses only M = 1 calls (plain decode, the MTP draft pass); `INTARSIA_HIP_MXFP4_A8_FUSE=1` keeps the old "every M the LDS fits" behaviour; `0` (or anything else) is the pre-pass kill switch. M >= 2 (verify, tail peels) is unchanged, so the multi-row slowdown is not exposed (kernel micro-timing below). A null or misaligned `x` now falls back to the pre-pass instead of failing.
2. The fused kernel issues the first weight group of both rows of each warp (`a8_first_load`) before the block quantises, so the DRAM latency of the first weight loads overlaps the quantisation and the barrier instead of following it. The body is shared with the pre-pass kernel (`a8_rows_body<M, OutT, PRE>`); the arithmetic and its order are untouched.

Kernel level (`matmul_mxfp4_a8_gemv_rows`, `timing_mxfp4_rows`, `raw/w3/gates-1/time-*.txt`, table in `raw/analysis/w3-kernel-timing-a8.txt`; microseconds; off = pre-pass + kernel, default = the new default, all-M = `INTARSIA_HIP_MXFP4_A8_FUSE=1`):

| shape (N x K) | M = 1 off | M = 1 default | ratio | default / off, M = 2..16 | all-M / off, worst of M = 3..16 |
|---|---:|---:|---:|---|---|
| 1024 x 5120 | 11.69 | 8.89 | 0.760 | 1.002-1.007 | 1.37 at M = 8 |
| 12288 x 5120 | 57.90 | 55.55 | 0.959 | 0.997-1.011 | 2.34 at M = 8 |
| 17408 x 5120 | 77.76 | 75.26 | 0.968 | 0.999-1.010 | 2.50 at M = 8 |
| 5120 x 17408 | 80.34 | 77.39 | 0.963 | 0.993-1.010 | 1.45 at M = 3 |
| 5120 x 6144 | 38.06 | 36.33 | 0.955 | 0.997-1.010 | 1.51 at M = 8 |

The M = 1 call gets 3-5 % faster on the big shapes (24 % on the small one); the M >= 2 columns are the same code path as before (the +-1 % is run noise), while fusing every M would make the verify and tail calls up to 2.5x slower, which is why the default stops at M = 1.

Trace of the new default (`raw/analysis/w4-dev.txt`, `w4-dev-gemv-shapes.txt`): 808 launches per step (from 1128); the rows kernel (now fused) takes 11441 us against 11091 + 676 = 11767 us before, so the kernel time falls by 0.33 ms and the idle by about 0.4 ms; the unprofiled step falls from 18.35 to 17.59 ms (generate, 3 interleaved rounds: 54.30 vs 56.80 tok/s, `raw/w4/attr-1.txt`). The old fused kernel without the prefetch measured 11668 us, so the prefetch is worth another 0.23 ms (+1.2 %).

### A/B on main vs the change, MTP off (the gate)

W3, `raw/w3/steps.log`, 4 interleaved rounds per arm (order rotated and reversed in alternate rounds), decode tok/s, median (min-max):

| arm | 1k | ratio | 4k | ratio | 16k | ratio |
|---|---|---:|---|---:|---|---:|
| main | 54.48 (54.3-55.3) | 1.000 | 53.77 (53.7-53.8) | 1.000 | 50.77 (50.8-50.8) | 1.000 |
| **this change** | **56.90 (56.6-57.0)** | **1.044** | **56.06 (56.0-56.5)** | **1.043** | **53.04 (53.0-53.6)** | **1.045** |
| this change + `INTARSIA_TP_ARNORM=1` | 57.17 (57.2-57.2) | 1.049 | 56.39 (56.4-56.4) | 1.049 | 53.39 (52.6-53.4) | 1.052 |
| main + ARNORM + WMMA knob | 54.80 (54.5-54.8) | 1.006 | 54.06 (54.0-55.2) | 1.005 | 51.13 (51.1-51.4) | 1.007 |

Prefill is unchanged (3553 / 3881 / 3571 tok/s on main against 3566 / 3778 / 3563; the 4k column swings by +-3 % between runs of the same arm, `both` and `dev` rows in `raw/w3`; the prefill GEMMs and the M >= 2 paths are the same code).

### MTP on (secondary)

W3 (3 rounds) and W4 (6 rounds), `raw/w3`, `raw/w4`, pooled n = 9 per arm, decode tok/s, median (min-max):

| arm | 1k | ratio | 4k | ratio | 16k | ratio |
|---|---|---:|---|---:|---|---:|
| main | 91.72 (86.3-93.6) | 1.000 | 84.28 (75.5-90.7) | 1.000 | 74.11 (72.3-78.2) | 1.000 |
| this change | 88.63 (84.3-92.8) | 0.966 | 82.90 (81.0-88.9) | 0.984 | 74.99 (72.5-79.4) | 1.012 |

The run spread is 8-18 %, so none of the three ratios is distinguishable from 1.000 (per-run values in `raw/w3/steps.log`, `raw/w4/steps.log`). The change should not move this number: in an MTP step the main model runs the verify pass on k + 1 = 3 rows, which keeps the pre-pass (the M = 2..16 kernel columns above are 0.99-1.01), and the only M = 1 calls are the draft passes of the one-layer MTP head. I cannot show it with the localeval spread; a clean MTP-on reading would need fixed-token `generate --spec mtp` runs (about 0.3 % spread), which the 4-window limit did not leave room for. The default MTP-on configuration is therefore unmeasured-but-unchanged, and the +4.4 % is the plain decode rate (MTP off, and the draft-free steps of any `--spec off` request).

## Gates

| gate | result |
|---|---|
| MTP-off decode >= +3 % at 1k and 4k, not slower at 16k | **met**: x1.044 / x1.043 / x1.045 (n = 4 per arm, spread <= 1.2 %) |
| bitwise-equal output | **met**: 256 greedy tokens of the 18 gate prompts (6 short, 6 about 4k, 6 about 16k), md5 of the text per prompt, `--spec off`: main = this change = this change + ARNORM = main + ARNORM = main + old fused (`raw/w2/texts-*`, `raw/w3/texts-*`); this change with MTP on equals main with MTP off 18/18; output hash of the 512-launch kernel test `a8_fused_is_bitwise_the_unfused` is `fac47447935a8383` with the knob 0, 1 and unset (the same value the W4A8 lane recorded for the original kernel) |
| batched == solo | **met**: `mixids.py` (a 48-token stream B prefilled beside a running 200-token stream A, three B prompts), `--max-batch 16` and `--max-batch 1`, MTP off: A `915f1c0953`, B `dd42cb5b78` / `e74f590855` / `537fb01934` for main and for this change in both, equal to the values LDF recorded for main |
| `plan show` diff explained | **no diff**: byte-identical for the 8B (1346 B), the 35B (3184 B) and the 27B at tp2 (3331 B), main CLI against the change; the plan names kernels, not their internal fusion |
| kernel tests on the GPU | `rows_exact` 13 passed (every M equals the one-row kernel, the default path), `mxfp4_a8` 6 passed, `mxfp4_decode` 2, `a8_served` 1, `mxfp4_a8_fuse` passed in three modes; `kernels test --backend hip --quick`: 74 rows ADMITTED, none rejected (`raw/w3/gates-1/`) |
| Mac `cargo test --workspace --release` | **met**: 198 `test result: ok`, 0 failed (before the test-file edit below, which is `cfg(ik_hip)`; `docs_env_vars` re-run after it) |
| CI | see the PR |

## What did not work or was not done

* The ARNORM fusion is real but small here (+0.6 %): its fused kernel is slower per call (16 us against 12.5 + 3.8) and the saving is the 128 launch gaps. The r9700-engine note "no gain at TP2 decode" holds for this engine too. A faster fused kernel (a single handshake per layer instead of per block) is the open item for Lane 2 (c); it is worth at most the 128 x 1.4 us of gaps plus whatever latency the handshake loses.
* The quantisation of the next GEMV's input could also be folded into the producers (rmsnorm, SiLU-GLU, gated rmsnorm), which would remove the 320 per-block redundant quantisations (each of the 1088 blocks of the gate+up GEMV quantises the same 5120 values); the fused kernel is still 0.35 ms slower than the rows kernel alone. Not attempted: it needs a new fused op in the plan (`fusion.rs`) and bitwise-equal quantisation in three kernels.
* Lane 1 (WMMA small-row GEMV) has nothing to offer the 27B decode; Lane 6 item 2 (zero-copy GDN rollback) concerns MTP verify, whose rollback cost is not in this MTP-off step. Not measured here.
* GEMV below the ceiling: the K = 3072 ops (GDN out, attention o) run at 84 % and the 48 x 5120 a/b projection is latency bound (3.4 us x 48); together 0.4 ms of potential, not pursued.

## Windows (fleetq, tatooine, all took radiance down and restored it)

| window | job | what |
|---|---|---|
| W1 | #6375 | plan show under the knob arms, `--op-table`, rocprofv3 traces of main and ARNORM, generate timings, a host-profile server (printed nothing) |
| W2 | #6380 | knob A/B, 6 arms x 4 rounds MTP off, 3 arms x 3 rounds MTP on, texts for 4 arms, traces of ARNORM / FUSE / both |
| W3 | #6386 | the change: GPU tests, admission, plan show, kernel timing, A/B x 4 MTP off, x 3 MTP on, texts, mixids |
| W4 | #6388 | MTP on, main against the change, 6 rounds; trace of the change |

CPU jobs (builds): #6373 (main CLI), #6382 (change CLI and the HIP test binaries).
