# Metal prefill F1: NAX (Metal 4) flash prefill attention (2026-10-15)

## Outcome

**Merge-ready: NO, by the letter of the guardrail, by one token position.** Everything else passes. The long-golden prefill top-1 is 99.58% (1428 of 1434 positions), the guardrail is 99.6%, and the base measured 99.65% (1429 of 1434). Long prefill KL improves (3.99e-4 against 4.34e-4, limit 5e-4). A P2 kernel variant with another threadgroup shape gives the same 99.65% top-1 and a KL of 3.18e-4 on the same golden, so one position and 1e-4 of KL are inside the spread of equivalent kernels (section 4). I did not tune the kernel to the metric. The call is the coordinator's.

`attn_prefill_bf16_flash_nax` (`kernels/metal/attention_flash_nax.metal`, launcher `crates/intarsia-backend-metal/src/flash_nax.rs`) runs prefill attention (head_dim 128, group 1/2/4/8, any T and pos0) on the M5 neural accelerators. Registered at priority 12 above P2's 10, for every prefill chunk, only with the metal4.0 library; other head dims, a missing metal4.0 library and misaligned operands run the P2 kernel.

| | pre (spike/metal `766c723`, P2 attention) | new | change |
|---|---:|---:|---:|
| attention kernel TFLOPS in the engine, 4k, chunk 256 (GPU trace) | 11.9 | **37.4** | 3.1x |
| 4k, chunk 1024 | 10.7 | 33.9 | 3.2x |
| 16k, chunk 256 | 10.9 | 33.8 | 3.1x |
| 16k, chunk 1024 | 11.8 | 32.6 | 2.8x |
| attention GPU time, 4k, chunk 256 | 408.6 ms (26.4%) | 130.1 ms | -279 ms |
| kernel sum, 4k, chunk 256 | 1548.8 ms | 1273.9 ms | -275 ms (-18%) |
| plain prefill, prompt-4k, chunk 256 (4 runs each, ABBA-ish) | 1692 / 1804 / 1798 / 1677 ms (median 1745) | 1457 / 1433 / 1426 / 1424 ms (median 1430) | -18% |
| plain prefill, prompt-4k, chunk 1024 | 1497 / 1483 / 1480 / 1686 ms (median 1490) | 1263 / 1261 / 1295 / 1244 ms (median 1262) | -15% |
| plain prefill, prompt-16k (~16k tokens), chunk 256 | 12578 / 12673 ms | 7866 / 7684 ms | 1.62x |
| plain prefill, prompt-16k, chunk 1024 | 11534 / 11569 ms | 6956 / 6955 ms | 1.66x |
| localeval marginal prefill 1k to 4k (tok/s) | 2556 / 2554 | 2722 / 3215 | +7% / +26% |
| localeval decode tok/s at 128 / 1k / 4k | 36.7 / 36.6 / 35.7 and 36.7 / 36.6 / 35.6 | 36.7 / 36.6 / 35.6 and 36.5 / 36.6 / 35.0 | unchanged |

The 4k saving (about 280 ms of attention, 18% of prefill GPU time) matches the estimate in the brief (250 ms, 15%). Branch `metal-f1-nax-attn` from `spike/metal` `766c723` (2e1fdb1 plus the qualify fixes, no kernel change). Host endor (Apple M5 Max), Qwen3-8B BF16. `spike/metal` had no new commits when I finished, so `git merge spike/metal` is a no-op.

## 1. What changed

* `kernels/metal/attention_flash_nax.metal` (new, in `METAL4_SOURCES`, `-std=metal4.0`): the kernel. A simdgroup owns 16 query rows of one query head; a threadgroup is G heads of a kv head times `NSG / G` 16-row slabs (`attn_nax_d128_s4` serves groups 1/2/4, `attn_nax_d128_s8` group 8; BK = 32 keys per tile). Fragment layout, `matmul2d` 16x32x16 on register cooperative tensors, S = Q.K^T with a transposed B, and S fragments reused as the A operand of P.V follow MLX's NAX attention (`mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h` and `steel/attn/nax.h`, installed package `mlx 0.31.2`, read only). Online softmax in float (log2 domain), P rounded to bf16 for the P.V product (the l sum uses the unrounded P), `-inf` masking, key tiles anchored at absolute key 0, strict precision. Same bindings and `OpP` fields as the P2 kernel.
* `src/flash_nax.rs` (launcher, variants, `fallbacks()`), `src/kernels.rs` (`ATTN_PREFILL_BF16_FLASH_NAX`, `metal4_shaders()`, `NAX_KERNELS` so it is hidden without the metal4.0 library, priority 12, `.when(nax_flash_ok)`), `src/lib.rs` (module, dispatch arm), `build.rs` (`METAL4_SOURCES`, `LAUNCHER_SOURCES`). Not touched: the P2 kernel and `flash.rs`, decode attention (`attention_splitkv.metal`, `attn.rs`), `intarsia-core`.
* Operands it cannot take go to the P2 kernel (counted in `flash_nax::fallbacks()`, reported once on stderr): a leading element skip on q / out, q / out / K / V not 16-byte aligned (q and out are read and written in 8-byte pieces, the K / V tensors need 16-byte rows).
* `examples/flash_bench.rs`: `nax` and `attn_nax_*` kernels, `--total N` (whole-prompt aggregate over all chunks of an N-token prompt, FLOP convention `4*D*H*(T*pos0 + T*(T+1)/2)` as before).
* Tests: `tests/flash_prefill_nax.rs` (new), `tests/metal4_fallback.rs` (extended).

## 2. How the kernel got to 37 TFLOPS: three attempts at the K / V feed

The matmul side is not the problem (ablation below: 49-52 TFLOPS with no memory traffic). The loads are. Microbench `examples/flash_bench`, 256-row chunk at 3840 cached positions, GPU time:

| attempt | us / call | TFLOPS |
|---|---:|---:|
| P2 `attn_prefill_bf16_flash` (simdgroup_matrix, threadgroup staging) | 1353 | 12.3 |
| 1. NAX, K / V fragments by 16-byte lane loads straight from device (MLX's scheme) | 1317 | 12.6 |
| 2. NAX, tiles staged through threadgroup memory (P2's scheme: coalesced loads, register prefetch, padded rows), 8 simdgroups | 1130 | 14.7 |
| 3. NAX, K / V fragments by the primitive's `cooperative_tensor::load` from device tensor slices | **404** | **41.2** |

Ablations on attempt 1 / 2 that pointed to the loads (`docs/results/raw/2026-10-15-metal-prefill-f1/experiments/`, the experimental kernel and launcher are kept there): no softmax work 12.0 TFLOPS, no mask 12.9, no K / V memory traffic (synthesized fragments) 48.8, no `mma` 30.8 (loads and softmax only), loads from a 16 KB L1-resident window 31.9, no `mma` and a hot window 36. In the threadgroup-staged kernel the staging and barriers cost ~200 us and the fragment reads from threadgroup memory ~550 us. A lane load instruction touches 8 rows x 64 bytes, which neither the device path nor the threadgroup path serves well; `cooperative_tensor::load` is the path the NAX GEMM (55-65 TFLOPS) uses and is 3.3x faster. After that, softmax and mask cost 3-4% (ablation on the final kernel), and a BK = 64 tile is 2.7x slower (register pressure), so BK stays 32. Two things I tried that do not work: P kept in float as the A operand of P.V (the mixed float x bf16 operand layout differs, the result is garbage and it is 2.2x slower), and a 16x64x16 descriptor for P.V (the API requires N = 16 or 32 when both operands are cooperative tensors).

`cooperative_tensor::load` does not bounds-check its tensor slice, so only tiles lying entirely inside the visible keys use it. The last, partial tile (and rows past `pos0 + T`, which must read as zero: the cache there may hold NaN and 0 x NaN poisons P.V) goes through bounds-checked lane loads feeding the same `matmul2d`.

### Kernel microbench, final kernel (GPU time per call, 36 layers with one KV cache each, `speed/bench-*.txt`)

| chunk | pos0 | P2 us | NAX us | NAX TFLOPS | factor |
|---:|---:|---:|---:|---:|---:|
| 256 | 0 | 53.2 | 26.0 | 20.7 | 2.0x |
| 256 | 1024 | 402.0 | 129.1 | 37.5 | 3.1x |
| 256 | 3840 | 1352.9 | 404.5 | 41.2 | 3.3x |
| 256 | 15872 | 6592.9 | 2110.2 | 31.8 | 3.1x |
| 1024 | 0 | 719.5 | 217.4 | 39.6 | 3.3x |
| 1024 | 1024 | 1960.6 | 664.2 | 38.8 | 3.0x |
| 1024 | 3840 | 5605.5 | 2273.6 | 32.1 | 2.5x |
| 1024 | 15360 | 21625.3 | 8487.1 | 31.4 | 2.5x |

Whole prompt, summed over all chunks (`--total`, `speed/bench-total.txt`): 4096 tokens at chunk 256 P2 452.8 ms (10.9 TFLOPS) against NAX 141.2 ms (35.1); chunk 1024 395.9 ms (12.5) against 131.0 ms (37.8); 16384 tokens at chunk 256 7160.6 ms (11.1) against 2354.2 ms (33.6) ; chunk 1024 6646.5 ms (11.9) against 2510.8 ms (31.5) (`bench-total.txt`; the earlier burst's `experiments`-era run gave 6436 against 2437). The aggregate at 16k is lower than the single-chunk numbers because the K / V streams (33 MB per layer and tensor) no longer sit in the SLC.

Short chunks (`bench-small-chunk.txt`): at 4k context a chunk of T <= 32 rows runs 380 us per layer on NAX against 316 on P2 (about 20% slower: 16-row slabs against 8-row ones, one threadgroup per kv head); T = 48 is level; T = 64 and up NAX is faster (10% at 64, 2x at 128, 3x at 256). Only a prompt's last chunk can be that short.

### Chunk-size choice (registration for every T)

I first registered the kernel for T >= 48 only, as the brief says "only where it is faster". The first correctness run (`checks-run1-min48/`, `checks_f1-run1-min48.log`) showed the cost: with a 4050-token chunk the 4062-token prompt has a 12-row tail on the P2 kernel, and its greedy ids flipped at position 22 of 48 (22/48 differ); every other chunking (128, 256, 512, 1024, 4030) and the base binary at every chunking were identical. So the kernel is now registered for every prefill chunk (`NAX_FLASH_MIN_M = 2`; T = 1 is the decode op). The price is about 2 ms per prompt at 4k (9 ms at 16k) for a short tail, against 280 ms and 4.7 s saved. With it, chunk invariance of attention is bitwise (`registry_splits_are_bitwise_the_single_call`: splits 256+44, 253+47, 288+12, 298+2, 7+293 and others give the bits of the single call), and chunk 4050 is identical again.

## 3. Gates (final tree; raw logs `docs/results/raw/2026-10-15-metal-prefill-f1/checks/`, `checks_f1.log`)

| gate | result |
|---|---|
| `tests/flash_prefill_nax.rs` vs f64 reference: GQA 1/2/4/8, ragged T, partial key tiles, pos0 > 0 (chunk offsets), cap = pos0 + T, NaN cache tail, head_dim 128; 17 cases x 2 variants | pass, worst error 0.746 of tolerance (the P2 test's tolerance: half a bf16 ulp + 5 sigma of bf16 probability rounding) |
| bitwise: both variants (threadgroup shapes) equal; single call equal to every chunk split (256+44 ... 7+293) by kernel name and through the registry | pass |
| misaligned q / out / k / v (offsets 2, 4, 8, 16, 32, 48) fall back to P2 and are correct; fallback counter | pass |
| predicate: head_dim 64 / 256, group 3 / 16, f32 / fp8 caches, T = 1 not taken; `INTARSIA_METAL_NO_METAL4=1` and `new_without_metal4()` register no NAX attention, the registry picks P2, launching the NAX kernel by name is an `Err` (`tests/metal4_fallback.rs`) | pass |
| `intarsia kernels test --backend metal -v` | 23 kernels, 272 cases, all ADMITTED; `attn_prefill_bf16_flash_nax` 7 cases max_abs 5.025e-3, max_rel 2.413e-3 (P2: 14 cases, 5.025e-3, 2.431e-3) |
| `cargo test -p intarsia-backend-metal --release -- --test-threads=1` | pass (all binaries, `tests/cargo-test-metal.txt`) |
| `cargo test --workspace --release --exclude intarsia-backend-metal` | 269 passed, 0 failed (`cargo-test-workspace.txt`) |
| `cargo check --workspace` | pass |
| engine selection, 1k prompt (973 tokens, chunk 256): new `attn_prefill_bf16_flash_nax` x 144, `matmul_bf16_nax` x 432, `matmul_pair_bf16_nax` x 288; with `INTARSIA_METAL_NO_METAL4=1`: `attn_prefill_bf16_flash` x 144, `matmul_bf16` x 1008; no "misaligned operands" fallback message | pass |
| decode unchanged: 364 dispatches per step (cb-mode trace), step period short 27.3 ms (pre 27.2), 4k 28.1 ms (pre 28.2) | pass |
| greedy ids, prompt-short and prompt-4k, 48 and 256 tokens, vs the `766c723` binary | **identical in all four** (the short-prompt token-49 near-tie did not flip here) |
| chunk invariance, 4k prompt, 48 tokens, `--prefill-chunk` 128 / 256 / 512 / 1024 / 4030 / 4050 | ids identical for all (new and pre); short-golden prefill KL identical at chunk 128 / 256 / 512 (new 2.602e-4, pre 2.686e-4) |
| fresh `intarsia qualify --backend metal --bandwidth-gbs 614 --golden-dir /opt/code/intarsiaml/golden` (new output dir, no `--resume`) | verdict PASS, 12 stages: 10 PASS, 2 SKIP (`ab`, `tp`); decode 36.6 / 36.6 / 35.4 tok/s at 128 / 1024 / 4096 (90 / 90 / 87% of floor), prefill 2332 / 3524 / 2985 tok/s |
| accuracy vs fp32 goldens, merge guardrail (prefill KL long <= 5e-4, long top-1 >= 99.6%) | KL **pass**, top-1 **99.58% fails by one position** (section 4) |

### 4. Accuracy

`intarsia accuracy --gate-class bf16 --golden-dir`, default chunk 256, prefill mode:

| golden | binary | prefill top-1 | prefill KL | decode (teacher forced) top-1 / KL |
|---|---|---:|---:|---:|
| short (`qwen3-8b-fp32`) | pre `766c723` | 99.77% | 2.686e-4 | 100% / 3.739e-4 |
| short | **new** | 99.54% | 2.602e-4 | 100% / 3.089e-4 |
| short | pre with the other P2 variant (`INTARSIA_METAL_FLASH=attn_flash_d128_s8`) | 100.00% | 2.811e-4 | 100% / 2.885e-4 |
| long (1434 positions) | pre `766c723` | 99.65% (1429) | 4.342e-4 | 100% / 3.731e-4 |
| long | **new** | **99.58% (1428)** | 3.993e-4 | 100% / 3.041e-4 (qualify run: 99.80% / 3.02e-4) |
| long | pre with the other P2 variant | 99.65% | 3.177e-4 | 100% / 2.798e-4 |

The NAX kernel and both P2 variants compute the same maths (bf16 products, float accumulation, bf16 P, float softmax) and differ in summation order. Between the two P2 variants alone the long KL moves from 4.34e-4 to 3.18e-4 and the short top-1 from 99.77% to 100%; between P2 and NAX the long top-1 moves by one position out of 1434. The gates in the tool (`>= 99.5%`, KL `<= 1e-3`) all pass. Against the stricter merge guardrail it is one position short, and I report it as not merge-ready rather than choose another rounding order until the number clears 99.6.

## 5. Speed method and caveats

* One fleetq slot per burst, exclusive, endor. GPU idle sampled 3x before each run (first sample reads the previous run, then 0-2%), no `zoom.us` (`speed/zoom.txt` empty). pre = binary of `766c723`, new = branch tip; ABBA order for the plain-prefill and localeval pairs; one discarded warm-up. Plain prefill = `generate --max-tokens 1 --profile-every 0 --prefill-chunk C`, chunk explicit so the P3 default change does not matter. Kernel TFLOPS = `analyze_prefill.py` over a kernel-mode GPU trace (`speed/trace-*/analysis-*.txt`, FLOP convention above). localeval: `localeval speed --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off` against `intarsia serve`, the command of `docs/results/raw/2026-10-14-metal-phase-b/sameslot/`.
* The first 4k plain-prefill pairs scattered more than 25% (new 1451 / 1908 ms; `speed/plain/`), so I repeated the 4k runs with 8 runs per binary and chunk (`speed/plain4k/`, quoted above). The 16k runs repeat within 1-2%. localeval 4k (4086-4090 tokens): TTFT pre 1.52 / 1.52 s, new 1.43 / 1.25 s; the new-1 run has a wide rep range (2468-3294 tok/s), so quote the plain-prefill rows for the 4k saving.
* Decode is unchanged: localeval decode and the cb-mode dispatch count agree within noise.
* Tooling note: one `cargo test` of the NAX test files ran on endor outside a fleetq slot while another job held it (the 5-second unit tests after the registration change); all benchmarks and the other tests ran inside slots.
* fleetq jobs: #1915 (first correctness + bench), #1916 (ablations, attempt comparisons, chunk sweeps, totals), #1920 (gate run 1, threshold 48), #1922 (P-float experiment), #1924 (gate run 2, final), #1926 (speed burst), #1928 (4k repeats).

## 6. Not done / limits

* Only head_dim 128 has a NAX kernel (64 and 256 stay on P2 by design of the brief). The prefix-reuse path was exercised by `qualify` and `acc-long` (mode all) only.
* The merge guardrail is missed by one position (section 4); no attempt was made to retune numerics to pass it.
* Possible further gains not pursued: a 32-row (TQ = 2) slab per simdgroup is blocked by registers (O alone would be 128 floats); the matmul2d with both operands cooperative accepts only N = 16 or 32.
* Kernel-level reproduce: `cargo run --release -p intarsia-backend-metal --example flash_bench -- --total 4096 --chunk 256 flash nax` inside a fleetq slot. Gate scripts: `checks_f1.sh`, `speed_f1.sh`, `speed_plain4k.sh`, `slot.sh` in `docs/results/raw/2026-10-15-metal-prefill-f1/`.
