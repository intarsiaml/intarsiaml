# Metal prefill P1: NAX bf16 GEMM in the engine (2026-10-14)

Note on provenance: the coordinator's earlier "P1 spike is GO, proceed to full P1 integration" message (scope, gates, speed method, report format) never reached this agent; only the later authorization message did. Gates and method below come from that later message and from `docs/research/metal-prefill-plan.md` (P1 and section 3), the report format from the spike doc. Nothing was added beyond that list except the extra checks named as such.

## Outcome with P2 merged (final state of the branch)

The branch is merged with `spike/metal` at `05f0b8b` (P2 flash prefill attention). P1 + P2 together, same slot (#1908), ABBA, pre = `spike/metal` `05f0b8b` binary (P2 only), new = this branch:

| | pre (P2 only) | new (P1 + P2) | factor |
|---|---:|---:|---:|
| plain `generate` prefill, 1k prompt (973 tokens) | 2588 / 2566 ms (376 / 379 tok/s) | 460 / 463 ms (2116 / 2101 tok/s) | 5.6x |
| plain `generate` prefill, 4k prompt (4062 tokens) | 10669 / 10689 ms (381 / 380 tok/s) | 1674 / 1660 ms (2427 / 2447 tok/s) | 6.4x |
| localeval TTFT, ~990-token prompt | 2.65 s / 2.52 s | 0.32 s / 0.31 s | 8x |
| localeval TTFT, ~4088-token prompt | 11.17 s / 10.66 s | 1.91 s / 1.71 s | 6x |
| localeval marginal prefill, 114 to 991 tokens (1k) | 384 / 405 tok/s | 3309 / 3548 tok/s | 8.6x |
| localeval marginal prefill, 991 to 4088 tokens (4k) | 363 / 380 tok/s | 1944 / 2202 tok/s | 5.4x |
| localeval decode tok/s (128 / 1k / 4k) | 36.4, 35.9, 35.1 / 36.3, 36.1, 35.1 | 36.2, 36.2, 35.1 / 36.3, 36.1, 35.1 | unchanged |

(pre-1 / pre-2 and new-1 / new-2 in order; `full/p1p2/speed/`.) Against the plan's exit criterion (>= 50% of mlx-lm marginal prefill, mlx-lm 4423 tok/s at 1k and 3134 at 4k): **1k 75 to 80% (3309 to 3548 tok/s; the 50% bar is 2210), 4k 62 to 70% (1944 to 2202 tok/s; the 50% bar is 1570)**. The plan's 65% target is met at 1k and straddled at 4k (62% in the first new run, 70% in the second). The 85% stretch is not met.

4k GPU-trace split, new binary, kernel mode, chunk 256 (`full/p1p2/speed/trace/analysis-4k-c256.txt`): sum of kernel time 1519 ms = **matmul 1049 ms (69.0%, 53.8 TFLOPS aggregate)**, **attention 403 ms (26.5%, `attn_prefill_bf16_flash`, 12.1 TFLOPS)**, elementwise 68 ms (4.4%), plain prefill 1660 to 1674 ms. At 1k: kernel sum 312 ms = matmul 268 ms (86.1%, 50.4 TFLOPS), attention 26 ms (8.4%, 10.6 TFLOPS), elementwise 17 ms (5.4%); plain prefill 460 ms. Before P1 and P2 (plan, section 2.2): 4k matmul 10280 ms / attention 7744 ms, 1k matmul 2548 ms / attention 400 ms. Matmul is the bigger share again (69% at 4k), attention second; the 1519 ms kernel sum is about 0.37 ms per prompt token.

Gates on the merged tree (#1907, `full/p1p2/`): `cargo test -p intarsia-backend-metal --release` pass; `kernels test` 22 kernels, 265 cases all ADMITTED (`attn_prefill_bf16_flash` 14 cases max_abs 5.0e-3, `matmul_bf16_nax`, `matmul_pair_bf16_nax` as before); engine selection at 1k: `matmul_bf16_nax` x 432, `matmul_pair_bf16_nax` x 288, `attn_prefill_bf16_flash` x 144 (pre: `matmul_bf16` x 1008 and the flash kernel); decode 364 dispatches per step on both binaries, step period short 27.5 ms (pre 27.6), 4k 28.6 ms (pre 28.4); **accuracy** (bf16 gate class):

| golden | stage | pre (P2 only) top-1 / KL | new (P1 + P2) top-1 / KL | gate |
|---|---|---|---|---|
| short | prefill | 99.77% / 2.590e-4 | 99.77% / **2.686e-4** | PASS |
| short | decode | 100% / 3.765e-4 | 100% / 3.739e-4 | PASS |
| long | prefill | 99.65% / 4.119e-4 | 99.65% / **4.342e-4** | PASS |
| long | decode | 100% / 4.043e-4 | 100% / 3.731e-4 | PASS |

(Prefill KL moves by +1.0e-5 and +2.2e-5 with P1 on top of P2, within the gate of 1e-3 and with the same top-1.) **Greedy ids, pre against new**, short and 4k prompts: 48 tokens identical in both, 4k 256 tokens identical, **short 256 tokens differ from token 49 on** (207 of 256 positions differ afterwards, a different continuation). This is the same near-tie the coordinator named: at token 49 pre (P2 only) picks id 26471 and new picks id 29994, which is also what `495a447` (before P2) and the P1-only binary pick (`full/identity/` against `full/p1p2/identity/`: P1-only, `495a447` and P1 + P2 all give `56411 7341 29994 264 2802 315` at positions 47 to 52, P2-only gives `56411 7341 26471 279 6301 41938`). So P1 + P2 reproduces the pre-P2 ids; it is P2 alone that flipped the tie, and P1's different last-bit rounding flips it back. Not a regression signal by itself (both tokens are valid continuations of the same near-tie), but it is a difference in greedy output between the two binaries and is reported as such. **Chunk invariance** with attention flash: 4k prompt ids identical for chunk 128 / 512 / 1024 / 4050 against 256 (new and pre); short-golden prefill KL identical at chunk 128 / 256 / 512 (new 2.686e-4, pre 2.590e-4). The 4050 chunk leaves a 12-row final chunk (tiled matmul, not NAX). **`qualify`** (full, with goldens, `full/p1p2/qualify/full-report.md`): verdict PASS, 12 stages, 10 pass, 2 skip (`ab`, `tp`); decode 36.3 / 36.1 / 35.0 tok/s at 128 / 1024 / 4096 (89 / 89 / 86% of the byte floor), prefill 2315 / 3316 / 2491 tok/s at 128 / 1024 / 4096 tokens, `acc-short` and `acc-long` PASS (prefill KL 2.69e-4 / 4.34e-4), `envelope` PASS. `intarsia-core`: still no change.

The sections below describe the P1-only measurements (pre = `spike/metal` `495a447`, before P2) and remain valid for what they measure.

**Outcome (P1 alone, measured before P2 landed): P1 is in and passes its gates.** The NAX GEMM (`matmul_bf16_nax`, plus `matmul_pair_bf16_nax` for gate+up and k+v) is registered at priority 5 for `Op::Matmul` / `Op::MatmulPair` with M >= 16 and is what the engine runs in prefill. Prefill speed on Qwen3-8B BF16, endor, same slot, pre = `spike/metal` and new = this branch, ABBA order:

| | pre | new | factor |
|---|---:|---:|---:|
| plain `generate` prefill, 1k prompt (973 tokens), chunk 256 | 2918 / 3197 ms (333 / 304 tok/s) | 806 / 786 ms (1207 / 1237 tok/s) | 3.8x |
| plain `generate` prefill, 4k prompt (4062 tokens), chunk 256 | 17981 / 17946 ms (226 tok/s) | 8930 / 8944 ms (454 tok/s) | 2.0x |
| localeval TTFT, 991-token prompt | 2.86 s / 2.86 s | 0.66 s / 0.66 s | 4.3x |
| localeval TTFT, 4087-token prompt | 17.98 s / 18.03 s | 9.18 s / 8.92 s | 2.0x |
| localeval marginal prefill, 1k to 4k | 205 / 204 tok/s | 364 / 375 tok/s | 1.8x |
| decode tok/s (128 / 1k / 4k context, localeval) | 36.5, 36.2, 35.2 / 35.9, 36.0, 35.1 | 36.3, 36.2, 35.2 / 36.3, 36.1, 35.2 | unchanged |

All gates pass: kernel admission, f64 unit tests, row independence, accuracy against both fp32 goldens, greedy ids (identical to pre), chunk invariance, full `qualify` (10 PASS, 2 SKIP that are not applicable), decode still 364 dispatches per step. **No `intarsia-core` change** (`git diff spike/metal --stat -- crates/intarsia-core` is empty).

**The plan's exit criterion (>= 50% of mlx-lm marginal prefill at 1k and 4k, that is >= 2210 and >= 1570 tok/s) is not met by P1 alone, and was not expected to be.** New marginal prefill is 1454 to 1457 tok/s at 1k (33% of mlx-lm's 4423) and 364 to 375 tok/s at 4k (12% of mlx-lm's 3134). At 4k the matmul share fell from 10.3 s to 1.04 s of kernel time and attention (`attn_prefill_bf16`, 7.7 s, 87% of the kernel time) is now the whole story: that is P2 (flash prefill attention).

Host endor (Apple M5 Max, `applegpu_g17s`, macOS 27.0, Metal compiler 32023.921), model `/Users/josef/models/Qwen3-8B` BF16. Branch `metal-p1-gemm`, merged with `spike/metal` at `495a447` (B5 decode attention) and again at the end (see "Merges"). Spike doc: `docs/results/2026-10-14-metal-prefill-p1-spike.md`. Raw logs: `docs/results/raw/2026-10-14-metal-prefill-p1/full/`. fleetq (all exclusive, all released): #1887 (unit tests, pair and pick microbench), #1889 (P1 correctness gates), #1892 (P1 speed), #1895 (cargo test, kernels test, qualify without goldens), #1900 (qualify resume, a no-op, see 3.9), #1903 (qualify with goldens), #1906 (B6 re-check), #1907 (P1 + P2 correctness and qualify), #1908 (P1 + P2 speed).

## 1. What changed

* `kernels/metal/gemm_nax_bf16.metal`: two entry-point families, `gemm_nax_t<BM,BN,NSG>` (Matmul) and `gemm_nax_pair_t` (MatmulPair: one grid of `tiles_n0 + tiles_n1` by `tiles_m` threadgroups, first column groups on W0/C0, rest on W1/C1; bitwise two single launches). Tiles 64x64 and 128x64, 4 simdgroups, strict precision only. The spike's relaxed variants, 64x128 and 128x128 tiles are gone (relaxed was bitwise equal and no faster).
* `src/gemm_nax.rs`: launchers `launch_nax` / `launch_nax_pair`, `pick_variant(m, n)`. The tile is chosen per call: 128x64 when `M >= 128` and the 128x64 grid has at least 192 threadgroups (40 GPU cores), else 64x64. The rule comes from `full/bench-pick.txt` (per shape x M, both tiles; gate/up M = 16..64 prefer 64x64 by 4 to 6%, k/v 1024x4096 at M 256 and 1024 prefers 64x64 by 30% and 5%, q/o at M 256 prefers 64x64 by 5%, everything large prefers 128x64 by 3 to 12%). Because every tile gives the same bits, the pick cannot change a result, only speed.
* `src/kernels.rs`: `matmul_bf16_nax` (Op::Matmul) and `matmul_pair_bf16_nax` (Op::MatmulPair), both `m >= 16 && k >= 16 && out == bf16`, bf16 activations and weights, `BF16_TOL`, `Exactness::Deterministic`, **priority 5**: above the tiled `matmul_bf16` (0), below the GEMV class (10, `m == 1` only). `NAX_MIN_M = 16`. `src/lib.rs`: dispatch of the two names, `has_metal4()`, the metal4.0 library loading and `metal4:0|1` in the backend fingerprint (these were in the spike already).
* Tests: `tests/gemm_nax.rs` (9 tests), `tests/metal4_fallback.rs`, unit tests in `kernels.rs`. `examples/gemm_nax.rs` has `bench`, `err`, `pair`, `pick` modes.
* No change to `dense.rs`, `attn.rs`, the model, the engine, recipes, chunk size (256), or any attention code. No `intarsia-core` change: the model already calls `try_fused(Op::MatmulPair, ...)` for gate+up and for k+v, `Registry::select` previously found no kernel for M > 1 and the model ran the two matmuls; now it finds the NAX pair kernel. This is the only graph-level difference in prefill (432 matmul + 288 pair launches instead of 1008 matmuls per 1k prompt, `full/select/`).

## 2. Pair kernel measured before it was kept

`full/bench-pair.txt`: gate/up (2 x 12288x4096) and k/v (2 x 1024x4096), one pair dispatch against two single dispatches (3 GiB pool of distinct weights, 15 rounds, interleaved), TFLOPS over both weights:

| Shape | M | tile 64x64 pair / 2 singles | tile 128x64 pair / 2 singles |
|---|---:|---:|---:|
| gate+up | 256 | 58.4 / 57.3 | 61.7 / 58.6 |
| gate+up | 1024 | 62.9 / 62.3 | 63.7 / 63.7 |
| gate+up | 2048 | 56.0 / 54.0 | 57.2 / 54.4 |
| k+v | 64 | 13.4 / 7.6 | 11.0 / 5.8 |
| k+v | 256 | 50.2 / 30.2 | 39.9 / 23.5 |
| k+v | 1024 | 55.5 / 54.8 | 58.6 / 51.9 |
| k+v | 2048 | 57.2 / 55.7 | 60.4 / 57.9 |

The pair is never slower, +2 to +8% for gate+up and +66% for k+v at the 256-row chunk (k/v alone is 1024 wide: 16 x 4 threadgroups cannot fill 40 cores, two of them in one grid can). It also halves the launches of those four ops. Kept.

## 3. Gates

Every gate was run on the binary built from this tree against the pre binary built from `git archive spike/metal` (`495a447`, extracted and built in a separate target directory; `diff -r` of `kernels/` and `crates/` against this tree shows only the P1 files).

1. **Engine really selects the NAX kernels** (checked first in the burst, which aborts otherwise): kernel-mode trace of the 1k prompt, `full/select/`. pre: `matmul_bf16` x 1008. new: `matmul_bf16_nax` x 432, `matmul_pair_bf16_nax` x 288 (144 gate+up, 144 k+v: 432 + 2 x 288 = 1008). `kernels list` shows the two NAX kernels on new, none on pre, none with `INTARSIA_METAL_NO_METAL4=1`.
2. **Kernel admission** (`intarsia kernels test --backend metal -m Qwen3-8B -v`, `full/kernels-test.txt`, repeated in `full/final/` after the last merge): 21 kernels, 251 cases, all ADMITTED. `matmul_bf16_nax` 7 cases max_abs 3.185e-3 rel_l2 8.1e-5; `matmul_pair_bf16_nax` 6 cases max_abs 8.6e-5 rel_l2 3.8e-6; tolerance `BF16_TOL` unchanged. The admission sweep only covers M in {32, 40} for these, so the M tails are covered by item 3.
3. **f64 unit tests** (`tests/gemm_nax.rs`, 9 tests): the Qwen3-8B shapes at M in {16, 17, 205, 222, 256, 973, 1024, 2048}, N and K tails, odd (2, 6 byte) and 8-byte weight offsets, K tails 17 to 100, every tile; worst error 0.97 of the tolerance (half a bf16 ulp + 2e-6 of the L1 mass, the existing `gemm_matches_f64_reference` budget), guard bytes around the output untouched. Pair kernel bitwise equal to two singles including the guard bytes (M 16, 17, 205, 222, 1024; N0 != N1; offsets 0 to 8). Plus the spike's f64 table (max error 0.50 bf16 ulp on the four Qwen3-8B shapes).
4. **Row independence**: rows of a 1024-row call equal, bit for bit, the same rows computed in calls of 5, 17, 64, 205, 512, 1021 rows at other offsets with either tile; through `Backend::launch` with the shipped tile pick, a 1024x12288 call (128x64 tile) equals calls of 256x1024, 205x256, 17x12288 (rows 1007..1024), 1024x1024 (64x64 tile) bitwise. Run-to-run deterministic.
5. **Accuracy against the fp32 goldens**, `--gate-class bf16` (`full/accuracy/`, `qualify` agrees):

| golden | stage | pre top-1 | pre KL | new top-1 | new KL | gate |
|---|---|---:|---:|---:|---:|---|
| short (`qwen3-8b-fp32`) | prefill | 100% | 2.608e-4 | 100% | **3.167e-4** | KL <= 1e-3, top-1 >= 99.5%: PASS |
| short | decode (teacher forced) | 100% | 4.042e-4 | 100% | 3.737e-4 | PASS |
| long (8192) | prefill | 99.86% | 3.738e-4 | **99.79%** | 3.762e-4 | PASS |
| long | decode | 100% | 4.071e-4 | 100% | 3.708e-4 | PASS |

   No non-finite rows. The short prefill KL moves up by 6e-5 (2.6e-4 to 3.2e-4) and the long top-1 down by one position of 1400 (99.86 to 99.79%); both stay far inside the gate (1e-3, 99.5%). The cause was not isolated: the NAX GEMM accumulates differently from the tiled f32 GEMM, the bf16 outputs differ in the last place, and the golden is fp32. `qualify` (item 9) re-reports the same numbers (prefill KL 3.17e-4 / 3.76e-4, decode 3.74e-4 / 3.16e-4 with top-1 99.60% on the long decode).
6. **Greedy ids** (`full/identity/summary.txt`): short prompt (122 tokens) and 4k prompt (4062 tokens), 48 and 256 forced tokens, pre against new: **ids identical in all four cases**.
7. **Chunk invariance** (`full/chunk/`): 4k prompt, 48 tokens, `--prefill-chunk` 128 / 256 / 512 / 1024 / 4050: ids identical to chunk 256 for new (and for pre). Chunks 128 to 1024 leave final chunks of 94, 222, 478, 990 rows (>= 16: NAX path, so row-independence makes the matmuls bitwise invariant); **4050 leaves a 12-row final chunk, which runs on `matmul_bf16`, not NAX** (M < 16), so that case is only invariant at the level of the greedy ids, not bitwise. Short golden `accuracy --mode prefill` at chunk 128 / 256 / 512: new KL 3.167e-4 at all three, pre 2.608e-4 at all three, top-1 100%. The default chunk (256) is unchanged. `qualify`'s own chunk noise envelope stage ran and PASSed (item 9).
8. **Decode unchanged** (cb-mode GPU trace of a 129-token decode, `period.py` with 364 expected, `full/decode/`): pre and new both **364 dispatches per step** at short and 4k context; step period short 28.9 ms (pre) / 28.6 (new), 4k 33.2 / 29.0 (the pre 4k run had 1.7 ms of idle per step, the new run 0.3: run-to-run noise on a single pair, not a change; the localeval decode figures above are the same to within 0.1 tok/s).
9. **`intarsia qualify --backend metal --bandwidth-gbs 614`** (`full/final/qualify-golden/full-report.md`): verdict PASS, 12 stages: 10 pass, 0 fail, 0 error, 2 skipped (`ab`: no reference server, `tp`: tp = 1; both not applicable), including `kernels`, `gate`, `acc-short`, `acc-long`, `bench`, `bench-long`, `envelope` (chunk envelope and baseline written and self-checked). Decode 35.9 / 35.9 / 33.2 tok/s at 128 / 1024 / 4096 (89 / 89 / 82% of the byte floor), prefill 2150 / 1495 / 432 tok/s at 128 / 1024 / 4096 tokens. Two earlier runs are kept in `full/final/`: `qualify/` (run from the worktree, which has no `golden/`, so `acc-short`, `acc-long`, `envelope` were SKIP; the other stages PASS) and a `--resume` of it (a no-op: resume treats SKIP as finished). The repository's baseline files (`golden/baselines`) were **not** re-recorded or changed; the envelope stage wrote its own into the report directory.
10. `cargo test -p intarsia-backend-metal --release` and `cargo check --workspace`: pass (`full/final/cargo-test-metal.txt`, re-run after the last merge, see "Merges").
11. **Fallback**: `tests/metal4_fallback.rs` (passes): with the library the NAX kernels are selected by priority for a prefill matmul and a prefill `MatmulPair`; with `INTARSIA_METAL_NO_METAL4=1` (and with `MetalBackend::new_without_metal4()`) neither is registered nor launchable, the registry selects `matmul_bf16`, the matmul matches the f64 reference, the `MatmulPair` has no kernel (the model then runs the two matmuls, as before), and launching a NAX kernel by name is an `Err`. On a real macOS < 26 host this was not exercised (the fleet has none).

## 4. Speed

### 4.1 Method

One fleetq slot (#1892), ABBA order pre, new, new, pre everywhere so a drift shows as pre-1 against pre-2. GPU idle (`ioreg` Device Utilization %, 3 samples before each run; the first sample after a run reads 92 to 99 because it averages the previous run, the next two were 0 to 12; `full/speed/plain/idle.txt`, `full/speed/localeval/idle.txt`, `gpu-idle-*.txt`). Plain prefill: `generate --max-tokens 1 --profile-every 0`, chunk 256, prompts of the phase C profile (1k = 973 tokens, 4k = 4062 tokens), one discarded warm-up. localeval: `localeval speed --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off` against `intarsia serve` started per run, run inside the slot with `FLEETQ_JOB` set (the same call as the phase B `leval.sh`, so the `local-model-bench` skill's rules hold: harness is localeval, its slot logic is bypassed only because the slot is held; unanimis capture was left on by localeval). Result files: `full/speed/localeval/res-p1-{pre-1,new-1,new-2,pre-2}/` (copies of `/opt/code/deepeval/results/*-adhoc-p1-*`).

### 4.2 localeval speed sweep (prompt tokens: TTFT s, prefill tok/s, marginal prefill, decode tok/s)

| run | 114-134 | ~990 | ~4087 |
|---|---|---|---|
| p1-pre-1 | 0.43 s, 302 | 2.86 s, 345 (352) , dec 36.2 | 17.98 s, 227 (205), dec 35.2 |
| p1-new-1 | 0.06 s, 1905 | 0.66 s, 1496 (1457), dec 36.2 | 9.18 s, 445 (364), dec 35.2 |
| p1-new-2 | 0.06 s, 1921 | 0.66 s, 1495 (1454), dec 36.1 | 8.92 s, 458 (375), dec 35.2 |
| p1-pre-2 | 0.43 s, 300 | 2.86 s, 344 (353), dec 36.0 | 18.03 s, 226 (204), dec 35.1 |

pre-1 against pre-2 and new-1 against new-2 agree within 3%, so the ABBA bracket shows no drift; the two-sided effect is 4.3x TTFT at 1k and 2.0x at 4k. (Rows under 1024 uncached tokens are request overhead, not prefill throughput; the marginal column is the clean figure.)

### 4.3 Per-shape TFLOPS inside the engine (kernel-mode GPU trace, new binary, chunk 256, `full/speed/trace/analysis-*.txt`)

| kernel | M | n x k | 4k: us/call | 4k TFLOPS | 1k TFLOPS | spike microbench (distinct weights) |
|---|---:|---|---:|---:|---:|---:|
| matmul_pair (gate+up) | 256 | 24576 x 4096 | 874 | 58.9 | 52.9 | ~57 to 62 |
| matmul (down) | 256 | 4096 x 12288 | 505 | 51.0 | 46.4 | ~50 |
| matmul (q, o) | 256 | 4096 x 4096 | 169 | 50.9 | 44.3 | ~52 |
| matmul_pair (k+v) | 256 | 2048 x 4096 | 89 | 48.1 | 41.1 | 50 (pair) |
| tail chunk (222 / 205 rows), all shapes | | | | 42.6 to 51.9 / 39.3 to 48.1 | | |

(For the pair the trace's `n` is `n0 + n1`, so its FLOP count and TFLOPS are for both weights.) Matmul aggregate **54.1 TFLOPS at 4k** (5.5 before), 4k matmul time 1042 ms (10280 ms before), 1k matmul 285 ms (2548 ms before). The 1k numbers are 5 to 15% lower than the 4k ones (the 1k prompt is only 4 chunks; a cold weight stream is a possible reason, not tested).

Where the time is now (kernel sums): 4k: attention 7699 ms (87.4%), matmul 1042 ms (11.8%), elementwise 72 ms (0.8%), sum 8814 ms against 8930 ms plain prefill. 1k: attention 384 ms (55.8%), matmul 285 ms (41.4%), elementwise 19 ms (2.8%), sum 688 ms against 796 ms. Attention per chunk grows from 29 ms (chunk 0) to 946 ms (chunk 14) at 4k, as in the plan. Predicted in the plan for P1 alone at 25 TFLOPS: 4k 10.07 s; at the 54 TFLOPS measured here the arithmetic gives ~8.8 s, measured 8.93 s.

### 4.4 Tile pick sweep

`full/speed/bench-pick.txt` (rerun with the shipped rule in the speed slot) and `full/bench-pick.txt` (first sweep, before the `M >= 128` condition was added; its `picked=` column disagrees with the shipped rule for gate/up at M < 128). The rule was chosen from it; microbench only, GPU time of back-to-back dispatches.

## 5. Merges

* `spike/metal` was at `25127d9` when the branch was created. During the work it moved three times, B5 (decode attention, `495a447`), B6 (decode small kernels, `9390beb`) and P2 (flash prefill attention, `05f0b8b`); the P2 merge had conflicts in `build.rs` (`LAUNCHER_SOURCES`) and `lib.rs` (the `launch` dispatch chain), both resolved by keeping both sides (`src/flash.rs` and `src/gemm_nax.rs` in the launcher hash, the NAX and flash dispatch arms). B5 and B6 merged without conflicts (all P1-only gates in sections 3 and 4 ran on the B5 merge, with the "pre" binary at `495a447`; B6 touches `dense_ops.metal`, `qk_rope_kv.metal`, `examples/small_ops.rs`, none of the P1 files). After B6 the re-check in section 6 was run, after P2 the final one in the P1 + P2 outcome section.

## 6. After the B6 merge (before P2 was merged)

#1906 (`full/post-merge/`, binary built from the tree merged with `spike/metal` `9390beb`, pre = `495a447`): `cargo test -p intarsia-backend-metal --release` and `kernels test` pass; the 1k prefill selects `matmul_bf16_nax` x 432 and `matmul_pair_bf16_nax` x 288; decode 364 dispatches per step (36.6 tok/s); accuracy short prefill 3.167e-4, decode 3.737e-4, long prefill 3.762e-4, decode 3.708e-4, all PASS and numerically identical to the pre-B6 run; greedy ids short and 4k 256 tokens identical to `495a447`. The final re-check on the tree with P2 is in the "Outcome with P2 merged" section above.

## 7. Not done, risks, limits

* With P2 merged, matmul is again the larger share (69% at 4k, 86% at 1k) and sits at 50 to 59 TFLOPS per shape; the remaining prefill levers are P3 (chunk size), P4 (fused qkv, gate+up beyond the pair) and the 5 to 8% of elementwise. P3 (chunk size) was not touched, as instructed: the `prefill_chunk` baseline identity is unchanged.
* Prefill KL against the fp32 golden moved from 2.6e-4 to 3.2e-4 on the short golden; inside the gate, cause not isolated.
* The 12-row-tail case of the chunk test (chunk 4050) runs on the tiled GEMM, so the result is invariant in greedy ids only; the prompt tail < 16 rows is a deliberate choice of `NAX_MIN_M = 16` from the plan, not re-measured. A prompt of 1 to 15 tokens (or a last chunk of that size) uses the old kernel; this is also the only way a result can differ between two chunkings of the same prompt in the matmuls.
* Admission covers M = 32 and 40 only for these kernels; M tails are covered by `tests/gemm_nax.rs`, not by `intarsia kernels test`. Adding M = 17 and 205 to the core sweep would close that (core change, not made).
* Start-up: the metal4.0 library adds 4 pipelines (2 tiles x single and pair) to `MetalBackend::new`; the cost was not isolated (the `plan` stage loads in 31 to 36 s, dominated by the weights).
* NaN / Inf propagation and subnormal inputs were not tested; K above 12288 not tested.
* Wall-clock noise: plain prefill pre/new runs repeat within 1 to 3%, localeval pairs within 3%; the mid-length (~1k) row can swing more (phase B recorded up to +-39% on marginal prefill in fleet history), the 4k and TTFT numbers are the ones to quote.
* Thermal state was not sampled; `pmset -g therm` reported no warning in the spike run only.
* The qualify baseline files and `golden/baselines` were not re-recorded, nothing in `golden/` was touched.

## 8. Reproduce

```sh
cargo build --release -p intarsia-cli --features metal && cargo build --release -p intarsia-backend-metal --example gemm_nax
# pre binary: git archive spike/metal (495a447) into a scratch dir, cargo build --release -p intarsia-cli --features metal
# copy both to $S/intarsia-pre, $S/intarsia-new, then in fleetq slots:
FLEETQ_JOB=N docs/results/raw/2026-10-14-metal-prefill-p1/full/checks_p1.sh     # selection, decode 364, ids, chunks, accuracy
FLEETQ_JOB=N docs/results/raw/2026-10-14-metal-prefill-p1/full/speed_p1.sh      # plain prefill ABBA, traces, localeval ABBA
FLEETQ_JOB=N docs/results/raw/2026-10-14-metal-prefill-p1/full/final_gates.sh   # cargo test, kernels test, qualify (no goldens here)
FLEETQ_JOB=N docs/results/raw/2026-10-14-metal-prefill-p1/full/qualify_golden.sh  # qualify with --golden-dir of the main checkout
FLEETQ_JOB=N docs/results/raw/2026-10-14-metal-prefill-p1/full/p1p2_checks.sh      # P1+P2 merged tree: tests, selection, decode, accuracy, ids, chunks, qualify
FLEETQ_JOB=N docs/results/raw/2026-10-14-metal-prefill-p1/full/p1p2_speed.sh       # P1+P2: plain prefill ABBA, traces, localeval ABBA
```
