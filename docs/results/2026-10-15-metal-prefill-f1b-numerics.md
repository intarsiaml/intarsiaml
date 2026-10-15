# Metal prefill F1b: numerics of the NAX flash prefill attention (2026-10-15)

## Outcome

**Merge-ready: YES.** Branch `metal-f1b-nax-attn` = F1 (`metal-f1-nax-attn` 08426ff) + `git merge spike/metal` (6271eec, no conflicts) + one kernel change: the softmax normaliser `l` is now summed over the **bf16-rounded P that the P.V product applies** (F1, and the P2 kernel, sum the unrounded float P). No type change, no extra matmul, no new register state.

| | spike/metal 6271eec (P2) | F1 + merge (5c30da0) | **F1b (this branch)** | guardrail |
|---|---:|---:|---:|---:|
| long golden, prefill top-1 | 99.65% (1429/1434) | 99.58% (1428/1434) | **99.72% (1430/1434)** | >= 99.6% |
| long golden, prefill KL | 4.342e-4 | 3.993e-4 | **3.699e-4** | <= 4.34e-4 (and <= 5e-4) |
| short golden, prefill top-1 / KL | 99.77% (433/434) / 2.686e-4 | 99.54% (432/434) / 2.602e-4 | **100.00% (434/434) / 2.832e-4** | KL <= 5e-4 |
| decode (teacher forced) KL, short / long | 3.739e-4 / 3.731e-4 | 3.089e-4 / 3.041e-4 | 3.496e-4 / 3.517e-4 | <= 5e-4 |
| attention kernel TFLOPS in the engine, 4k, chunk 256 (GPU trace) | 12.1 | 37.6 | **38.4** | >= 30 |
| 4k, chunk 2048 | 13.9 | 45.4 | **50.5** | |
| 16k, chunk 256 / 2048 | 12.3 / 13.4 | 40.7 / 39.8 | **41.4 / 43.4** | |
| plain prefill 4k, chunk 256 / 2048 (median of 8 ABBA runs) | 1603 / 1414 ms | 1353 / 1182 ms | **1331 / 1182 ms** (-17.0% / -16.4% vs 6271eec) | |
| plain prefill 16k, chunk 256 / 2048 | 11273 / 10354 ms | 7053 / 6467 ms | **6786 / 6150 ms** (-39.8% / -40.6% vs 6271eec) | |

Read the accuracy row with this caveat: **the top-1 count is noise at this sample size, and the P rounding is not what drives it.** The experiments below replace bf16 P by half, by float (relaxed matmul), and by a bf16 hi + lo split, all of which carry far less P rounding error, and the long top-1 lands at 1426, 1428 and 1428 of 1434. Equivalent-maths variants spread over 1426-1430 (P2's two threadgroup shapes, F1, and every variant of the sweep). The hypothesis in the F1 and P2 docs (the bf16 rounding of P is the likely source of the one-position gap) is **not supported**: removing the rounding does not move top-1 up. What I ship is the variant that (a) removes a real, if small, inconsistency (numerator weights rounded, denominator not), (b) costs nothing (it is slightly faster, section 3), and (c) lands at 1430. Choosing among variants by their measured count is a selection on noise, and I say so; the principled part is (a). The guardrail margin is two positions, which a different but equally correct rounding order could take back.

## 1. What was tried (P-mode sweep)

`INTARSIA_METAL_FLASH_NAX=<variant>` selects an instantiation; the sweep kernel, launcher and test are kept in `docs/results/raw/2026-10-15-metal-prefill-f1b/experiments/` (`attention_flash_nax_pmodes.metal`, `flash_nax_pmodes.rs`, `flash_prefill_nax_pmodes.rs`). One fleetq slot (#1958), script `exp1.sh`, logs `exp1/`, accuracy runs `acc/` (`acc.sh`: `intarsia accuracy --gate-class bf16` on both fp32 goldens, default chunk 256, prefill + decode teacher forced). The accuracy run takes ~10 s per golden.

| mode | P (A operand) | V (B operand) | l summed over | f64 check (worst, of tolerance) | rel. rms error, scale 1.0 | microbench TFLOPS (chunk 256, pos0 3840) | long top-1 | long KL | short top-1 | short KL |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| P2 (spike/metal) | bf16 | bf16 | unrounded P | | | 12.3 | 1429 | 4.342e-4 | 433 | 2.686e-4 |
| p0 = F1 | bf16 | bf16 | unrounded P | 0.746 | 1.723e-3 | 40.65 | 1428 | 3.993e-4 | 432 | 2.602e-4 |
| **p8 = F1b** | bf16 | bf16 | **bf16-rounded P** | 0.746 | 1.714e-3 | **41.48** | **1430** | **3.699e-4** | **434** | 2.832e-4 |
| p1 | half | half (converted from bf16) | unrounded P | 0.654 | 1.658e-3 | 28.92 | 1430 | 4.038e-4 | 434 | 2.996e-4 |
| p7 | half, P and l scaled by 2^12 | half | scaled P | 0.654 | 1.658e-3 | 29.11 | 1428 | 3.692e-4 | 433 | 2.561e-4 |
| p4 | bf16 hi + bf16 lo (two P.V products) | bf16 | unrounded P | 0.654 | 1.657e-3 | 29.35 | 1426 | 4.285e-4 | 433 | 2.785e-4 |
| p5 | float, `relaxed_precision = true` | bf16 | unrounded P | 0.685 | 1.664e-3 | 35.63 | 1428 | 3.920e-4 | 433 | 2.391e-4 |
| p6 | float, relaxed | float (converted) | unrounded P | 0.685 | 1.664e-3 | 28.14 | 1428 | 3.920e-4 | 433 | 2.391e-4 |
| p2 (strict float) | float | bf16 | | fails | 1.38 | 18.44 | not run (garbage) | | | |
| p3 (strict float) | float | float | | fails | 1.35 | 17.83 | not run (garbage) | | | |

* Rel. rms error = `sqrt(sum (got - ref)^2 / sum ref^2)` against the f64 reference on the 4k-context chunk (T 256, pos0 3840, 32/8 heads, uniform random q, k, v, `flash_numerics_report`); the bf16 rounding of the output alone puts a floor near 1.657e-3 (p4 sits on it). p0 is 0.47e-3 above the floor in quadrature, p8 0.44e-3, p1 / p5 / p7 / p4 within 0.1e-3 of it. So rounding P to bf16 costs about 0.5e-3 of relative error on a 1.66e-3 floor, and the F1b normaliser takes about 7% of that. On this uniformly random data the whole effect is small; real attention is peakier and has a mean-V component, which is where the common-mode normaliser error would act, but I have no measurement beyond the two goldens.
* Top-1 counts vs the spread: P2 1429 and 1429 (two threadgroup shapes), F1 1428, p4 1426, p5 / p6 1428, p7 1428, p1 1430, p8 1430. The most precise variants (p4, p5, p6) are not better than the least precise (p0). KL is a smoother metric and moves in the same noise band (3.69e-4 to 4.34e-4 on the long golden).
* **API findings.** (1) `matmul2d` accepts `float`, `half` and `bfloat` independently for the left and right operand (static_asserts in `MetalPerformancePrimitives.framework/Headers/__impl/MPPTensorOpsMatMul2dImpl.h` lines 4385-4446, Xcode MacOSX.sdk under `/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/`). (2) `cooperative_tensor::load` only takes a tensor of the same element type (Metal toolchain header `metal/metal_cooperative_tensor`, the `load` overloads at lines ~486-503: `is_same_v<element_type, tensor::value_type>`). The V cache is bf16, so half / float V has to be loaded as bf16 and widened element by element (`ct_b[i] = BT(ct_l[i])`), which is why p1, p6 and p7 are 29% slower than p0. (3) A float left operand under the default strict precision gives wrong results (p2, p3: relative error ~1.3, 18 TFLOPS), as the F1 doc found ("garbage"). With `relaxed_precision = true`, the descriptor MLX uses (`mlx/backend/metal/kernels/steel/attn/nax.h` lines 387-456, mlx 0.32.3 at `/Users/josef/Models/kolibri-ref/.venv/lib/python3.12/site-packages/mlx/include/`; its `steel_attention_nax.h` lines 396-445 feeds the float S tile straight into `mma` against a `T` V tile), the same element-wise operand copy is correct (p5, p6) and at 35.6 TFLOPS in the microbench. MPP documents the flag as "allow implementation to sacrifice accuracy for performance" (`MPPTensorOpsMatMul2d.h` line 120); the measured error of p5 (1.664e-3) sits between bf16 P and half P, so the float operand is probably narrowed inside the primitive. I did not find out why strict float fails. (4) p1 was checked with and without P scaled by 2^12 (p7, so weights below half's normal range, 6.1e-5, stay normal): no accuracy difference beyond noise.
* The cheap variants that keep full speed are p0 and p8 only. p5 (35.6 microbench, ~33 expected in the engine) would also clear 30 TFLOPS but loses 12% of attention speed for a gain that does not show in top-1 (1428).

## 2. What shipped

`kernels/metal/attention_flash_nax.metal`: in the P loop, `Pf[f][e] = bfloat(exp2(S - m))` and `ls += float(Pf[f][e])` instead of `ls += pv`; header comment updated. Nothing else in the kernel, launcher, registry, tests (other than the cleanups below) or any other kernel changed. Chunk-split bitwise invariance is untouched (the arithmetic of a row is the same function of that row and the key tiles from key 0) and verified by the existing tests and the engine ids (section 4). The final tree's metallib is bit-identical to the one the gates ran on (`intarsia` binary sha1 93f47cc4339b both for the gated tree and for the commit tip: the last commit only removed a test and edited comments).

Tests: `tests/flash_prefill_nax.rs` gains `flash_numerics_report` (informational, prints the relative rms error per variant at three sharpnesses, asserts finiteness only). I also wrote a "constant V comes out unchanged" test meant to pin the new normaliser; it passes on the F1 kernel too (an unrounded-l row deviates by well under half a bf16 ulp of the output, so bf16 output rounding hides the difference) and was therefore removed. The F1b kernel is pinned by the unchanged f64-reference tests, which it passes with the same worst error as F1 (0.746 of tolerance).

## 3. Speed (fleetq #1962 microbench and decode, #1963 traces and plain prefill; `speed/`, `speed.sh`, `speed2.sh`, `trace.sh`, `plain-abba.sh`)

pre = binary of 6271eec, f1 = binary of 5c30da0, new = branch tip. Endor idle 0-2% before each run, no `zoom.us`. All inside exclusive slots.

Attention kernel TFLOPS, kernel-mode GPU trace (`speed/trace/`, `trace-summary.txt`, FLOP convention of `analyze_prefill.py`; attention ms in brackets):

| prompt | chunk | pre (P2) | f1 | **new** |
|---|---:|---:|---:|---:|
| 4k | 256 | 12.10 (402 ms) | 37.60 (129 ms) | **38.39 (127 ms)** |
| 4k | 2048 | 13.88 (351 ms) | 45.39 (107 ms) | **50.53 (96 ms)** |
| 16k | 256 | 12.29 (6596 ms) | 40.74 (1990 ms) | **41.40 (1958 ms)** |
| 16k | 2048 | 13.40 (6047 ms) | 39.76 (2039 ms) | **43.40 (1868 ms)** |

Microbench (`flash_bench`, one chunk, GPU time per call, 36 layers): alternating F1 / F1b, three rounds each (`speed/slot-part2.log`): chunk 256, pos0 3840: F1 41.29 / 40.81 / 40.59, F1b 41.88 / 41.27 / 41.10 TFLOPS (+1.4%); chunk 2048, pos0 4096: F1 44.74 / 44.83 / 44.23, F1b 50.21 / 48.96 / 48.09 (+10%); pos0 14336: F1 43.37 / 43.32 / 42.59, F1b 49.06 / 47.71 / 47.17 (+11%). The first burst's chunk-2048 F1 numbers were noisier (p10-p90 spread 7%) than the alternating ones. Why F1b is faster at long chunks I did not investigate (the summed value is converted from the register that already holds the bf16 weight instead of keeping the float exp2 live next to it; this is a guess).

Plain prefill, `generate --max-tokens 1 --prefill-chunk C`, ABBA, 4 rounds = 8 runs per binary and config, first run discarded (`speed/plain/`, `plain-summary.txt`; raw `.err` files per run):

| prompt | chunk | pre ms (mean, median) | new ms (mean, median) | new vs pre (median) |
|---|---:|---:|---:|---:|
| 4k | 256 | 1606, 1603 | 1340, 1331 | -17.0% |
| 4k | 2048 | 1413, 1414 | 1183, 1182 | -16.4% |
| 16k | 256 | 11294, 11273 | 6802, 6786 | -39.8% |
| 16k | 2048 | 10338, 10354 | 6161, 6150 | -40.6% |

F1 vs F1b, 4 runs per binary (`speed/plain-f1/`): 4k 256 +0.1%, 4k 2048 +1.4% (within the 1-2% scatter), 16k 256 -3.6%, 16k 2048 -2.1% (F1b faster).

Decode is unchanged: 364 dispatches per step for both binaries; step period (cb trace, 129 tokens) pre short 27.52 / 27.25 ms, new 27.20 / 27.19 (36.3-36.8 tok/s both); 4k pre 28.17 / 28.06, new 28.22 / 28.07 (35.5-35.6 tok/s) (`speed/slot-part1.log` section 4; `gates/slot.log` section 3 agrees: 27.12 vs 27.25 short, 28.70 vs 28.65 4k).

## 4. Gates (final tree; `gates/slot.log`, fleetq #1960; `gates.sh`)

| gate | result |
|---|---|
| `tests/flash_prefill_nax.rs` vs f64 reference (GQA 1/2/4/8, partial tiles, chunk offsets pos0 > 0, cap = pos0 + T, NaN cache tail), bitwise chunk-split invariance (256+44, 253+47, 288+12, 298+2, 7+293, ...) and agreement across threadgroup shapes, misaligned fallback | pass; both variants worst error 0.746 of tolerance (`gates/tests/flash_prefill_nax-final.txt`) |
| `cargo test -p intarsia-backend-metal --release -- --test-threads=1` | pass, final tree (`gates/tests/cargo-test-metal-final.txt`; the gated tree ran the same set plus the removed test, `cargo-test-metal.txt`) |
| `cargo test --workspace --release --exclude intarsia-backend-metal` | 269 passed, 0 failed (`gates/tests/cargo-test-workspace.txt`) |
| `cargo check --workspace` | pass (`gates/tests/cargo-check-workspace.txt`) |
| `intarsia kernels test --backend metal -v` | 23 kernels, 272 cases, all ADMITTED; `attn_prefill_bf16_flash_nax` 7 cases, max_abs 5.025e-3, max_rel 2.408e-3 (P2: 14 cases, 5.025e-3, 2.431e-3) |
| engine selection, 1k prompt | new: `attn_prefill_bf16_flash_nax` x 144; `INTARSIA_METAL_NO_METAL4=1`: `attn_prefill_bf16_flash` x 144; pre: `attn_prefill_bf16_flash` x 144; no misaligned fallback |
| decode dispatches | 364 per step, both |
| accuracy, both goldens (table above) | all KL <= 5e-4 (worst 4.34e-4, the base), long top-1 99.72% |
| greedy ids 48 and 256 tokens, prompt-short and prompt-4k, vs 6271eec | **identical in all four** (the token-49 near-tie of the short prompt did not flip) |
| chunk invariance, 4k prompt, 48 ids, chunk 128 / 256 / 1024 / 2048 / 4050 | ids identical for all, new and pre; short-golden prefill KL identical at chunk 128 / 256 / 512 / 2048 (new 2.832e-4, pre 2.686e-4) |
| fresh `intarsia qualify --backend metal --bandwidth-gbs 614 --golden-dir /opt/code/intarsiaml/golden` (new output dir) | PASS: 12 stages, **10 PASS, 2 SKIP** (`ab`, `tp`); decode 36.8 / 36.6 / 35.1 tok/s at 128 / 1024 / 4096 (91 / 90 / 86% of floor); prefill 2380 / 3821 / 3114 tok/s (`gates/qualify/`) |

Guardrail: all KL <= 5e-4 (yes), long top-1 >= 99.6% (99.72%), no greedy-id difference (none). **Merge-ready: yes.**

## 5. Limits

* The two goldens are 434 + 1434 positions. A one-position top-1 difference is ~0.07%; the sweep shows a +-2 position spread between kernels of equal maths. The F1b top-1 (1430) should be read as "no longer below the guardrail", not as measurably better than F1 or P2.
* Only head_dim 128 NAX path, as in F1. The shipped change is a normaliser fix that applies to every group size and both threadgroup shapes (s4 / s8 are bitwise equal).
* Strict-precision float operands for matmul2d are broken or unsupported in the layout assumed here; relaxed precision is only needed if P is ever kept in float. Not diagnosed.
* Exact higher-precision P (p4, hi + lo) costs 28% attention speed for no measured benefit; half P costs the same through the V conversion. If a future V cache is fp16, p1 / p7 would drop that conversion.

## 6. Reproduce / files

* fleetq jobs: #1958 (sweep: tests, numerics, microbench, accuracy of every variant), #1960 (gates), #1962 (speed burst part 1: microbench and decode; its trace and plain-prefill sections did not run, helper scripts were not executable), #1963 (part 2: alternating microbench, GPU traces, plain prefill ABBA, final-tree tests). All exclusive endor slots, released.
* Binaries for the comparison: `git archive 6271eec` / `5c30da0` built with `cargo build --release -p intarsia-cli --features metal`; branch tip built the same way.
* Kernel-level: `cargo test -p intarsia-backend-metal --release --test flash_prefill_nax -- --nocapture`; sweep: `docs/results/raw/2026-10-15-metal-prefill-f1b/exp1.sh` against the experiments sources.
* Raw logs and scripts: `docs/results/raw/2026-10-15-metal-prefill-f1b/` (`exp1/`, `acc/`, `gates/`, `speed/`, `experiments/`, `*.sh`, `summarize_plain.py`).
