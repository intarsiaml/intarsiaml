# MPRE: Metal prefill toward the hardware limit (2026-10-29)

Lane MPRE (`lane-mpre/metal-prefill`), endor (Apple M5 Max, 128 GB, macOS 27.0), 2026-10-29 11:30 to 14:00 EDT. Models
`~/models/Qwen3.6-35B-A3B-NVFP4` and `~/Models/Qwen3-8B` (BF16). Base: `main` 867b97145 (binary `intarsia-base`; main moved to
7aebefb83 during the lane, #250 is HIP-only, so the Metal code is the same). Candidate c1: 60caa4deb (both kernels below; the
next commit only adds the new kernel to a pinned test list and the tuning rows). Every GPU run was a fleetq slot on endor:
4 windows (#7026, #7063, #7081, #7096) and 6 short checks under 3 minutes; builds `--load cpu`. Raw data and scripts:
[raw/](2026-10-29-mpre-metal-prefill/raw/) (the scripts name the lane's scratch paths under `.worktrees/mpre/.lane`).

## Result

| 35B NVFP4, localeval, 3 interleaved rounds (medians) | 1k | 4k | 16k |
|---|---:|---:|---:|
| prefill tok/s, base | 4117 (4106-4120) | 4580 (4458-4597) | 3120 (3110-3124) |
| prefill tok/s, **c1** | **4583** (4566-4599) | **5414** (5403-5444) | **4179** (4163-4295) |
| c1 / base | **1.11x** | **1.18x** | **1.34x** |
| decode tok/s, base / c1 | 124.7 / 124.4 | 121.5 / 120.2 | 109.7 / 108.6 |
| decode c1 / base | 0.998 | 0.989 | 0.990 |

The 8B BF16 plan is byte-identical (neither kernel serves it); its A/B (2 rounds) is noise: prefill 3962 / 3898 / 2725 base
against 3946 / 3872 / 2714 c1 (-0.4 / -0.7 / -0.4 %), decode 35.8 / 34.7 / 31.5 against 36.0 / 35.0 / 31.0.

Two changes, both prefill-only, both KL-gated (not bitwise the old kernels), each with an off switch that restores the previous
build bitwise (`plan show` with both knobs off is byte-identical to base on every model):

| change | what | kernel time, 35B (kernel-mode trace) | knob |
|---|---|---|---|
| `attn_nax_x`: NAX flash attention for head_dim 256 | the 35B's 10 attention layers ran the P2 `simdgroup_matrix` flash kernel (head_dim 128 has had a NAX kernel since F1b) | 4k 132 -> 49 ms, 16k 2020 -> 800 ms (11 -> 27 TFLOPS) | `INTARSIA_METAL_FLASH_NAX_D256=0` |
| `matmul_nvfp4_nax`: NAX GEMM for the NVFP4 projections at M > 8 | the shared experts (3 NVFP4 GEMMs per layer) ran the tiled `simdgroup_matrix` `matmul_nvfp4_mma` at 11 TFLOPS | 4k 93 -> 23 ms, 16k 333 -> 94 ms | `INTARSIA_METAL_NVFP4_NAX=0` |

The task's PR bar (+5% at 35B 4k or 16k, no cell below -1%): prefill +18% / +34%, every prefill cell faster. Decode is at
0.989 / 0.990 at 4k / 16k: the decode kernels and every decode pick are unchanged (`plan show`: the only differences are the
`attn_prefill` pick and the new `matmul_nvfp4_nax` row with its 2 MB scratch), round 1's 4k decode 114.9 is a single slow
cell (rounds 2 and 3: 120.2 / 121.3 against base 121.4 / 122.0), and decode noise between rounds is ~5% on this box (NXP:
0.992 for the same reason). Treat decode as unchanged within noise, not as measured-equal.

## 1. Attribution (window 1, #7026)

Kernel-mode GPU timestamps (`INTARSIA_METAL_PROFILE=kernel:`; every launch waited on, so use the shares), `generate` of one
prompt per process, chunk 2048 (35B recipe) / 2048 (8B recipe). Peaks measured in the same slot: NAX bf16 GEMM 64.2 TFLOPS
(`examples/gemm_nax bench`, 2048 x 4096 x 4096, best tile; 62-64 at M 4096), flash attention P2 d256 11.5-11.8 TFLOPS, NAX
d128 41-52 TFLOPS (`examples/flash_bench`, chunk 2048 at pos0 0 / 2048 / 14336). TFLOPS below are real FLOPs (MoE: 8 experts
x 3 x 2 x 512 x 2048 per token per layer; shared expert the same for one expert; attention 4 D H T^2 / 2 causal).

### Qwen3.6-35B-A3B NVFP4, base

| class | 4k: ms (share), TFLOPS | 16k: ms (share), TFLOPS | vs peak |
|---|---|---|---|
| MoE expert GEMMs (`moe_experts_bf16`, grouped NAX, NVFP4 decoded per tile) | 304 (30.5%), 27 | 1100 (21.7%), 30 | 42-47% of 64 |
| dense fp8 GEMMs (`matmul_fp8_nax`: attention/GDN projections, incl. x QDQ + weight decode per call) | 247 (24.8%), 44 | 849 (16.7%), 49 | 68-77% |
| GDN delta rule (`gdn_delta_rule_bf16_pre`, bitwise serial form) | 160 (16.0%), 3.3 | 557 (11.0%), 3.3 | fp32 vector, reduction-latency bound (MGD) |
| attention (`attn_prefill_bf16_flash`, P2, head_dim 256) | 133 (13.3%), 10.4 | **2020 (39.9%)**, 10.9 | 17% |
| dense NVFP4 GEMMs (shared experts, `matmul_nvfp4_mma`) | 95 (9.5%), 11 | 335 (6.6%), 12 | 17% |
| norms / elementwise / route / combine / gdn conv / bf16 GEMMs | 59 (6%) | 208 (4%) | bandwidth-bound small ops |
| GPU idle between command buffers | ~0 warm (1k warm run: 2.4 ms of 444 ms); 19% in a fresh process at 4k, 57% at 1k (weight page-in) | 4% fresh | |

Why 16k falls (2868 on the scoreboard, 3120 today): attention is quadratic and was on the 11 TFLOPS P2 kernel, so at 16k it
was 40% of the GPU time. With c1 it is 22% at 27 TFLOPS.

Why 1k to 4k is flat: a warm 1k prompt is one 973-row pass (235 ms, 4132 tok/s in the window 2 cb trace); at 1k the MoE has
about 30 rows per expert (32-row tiles; 20 TFLOPS in the 1k trace against 27 at 4k) and the fp8 GEMMs run at M 973, while at
4k the passes are 2048 rows but attention (13%) and GDN grow with context. The gains and losses roughly cancel. (The 1k
kernel trace is from a fresh process, so its absolute times include page-in; the warm wall time is the cb run.)

Matrix FLOPs of a 4k prompt (MoE 8.3 + shared 1.0 + fp8 projections 10.5 + attention 1.4 TFLOP) are ~21 TFLOP: at the 64 TFLOPS
NAX peak that is 0.33 s, about 12,000 tok/s, against ~0.78 s of GPU time with c1 (kernel mode), so the 35B is at ~45% of the
matrix limit at 4k; GDN (0.45 TFLOP of fp32 recurrence) is the part with no matrix form today.

### Qwen3-8B BF16, base

| class | 4k: ms (share), TFLOPS | 16k: ms (share), TFLOPS |
|---|---|---|
| bf16 GEMMs (`matmul_pair_bf16_nax`, `matmul_bf16_nax`) | 979 (86.9%), 58 (90% of peak) | 4075 (67.6%), 56 |
| attention (`attn_prefill_bf16_flash_nax`, d128) | 105 (9.3%) | 1793 (29.7%), 44 |
| norms / elementwise | 40 (3.6%) | 161 (2.7%) |
| GPU idle | 4% fresh process | 0.8% |

The 8B is at the hardware limit for its GEMMs (56-60 of 64 TFLOPS); what is left is attention at 16k (44 TFLOPS, MLX's NAX
attention is in the same range) and ~3% of elementwise that a GEMM-epilogue fusion could take. No 8B change was made: a +5% at
4k would need the GEMMs above the measured peak.

## 2. The changes

### 2a. head_dim 256 NAX flash attention (`attn_nax_x`, kernels/metal/attention_flash_nax.metal)

A plain `attn_nax<256>` instantiation spills: 16 Q fragments and 16 accumulator fragments per simdgroup do not fit, and it ran
at 5.5-9 TFLOPS, slower than P2. Re-reading Q from cache per key tile (`QR`), and splitting V's columns over two threadgroups
that each recompute the scores, reached 9-15 TFLOPS (short check 2). The kernel kept: two simdgroups share each (query head,
16-row slab); half h holds Q columns and accumulates O columns 128h..128h+127, which is the head_dim 128 kernel's register
footprint. Per 32-key tile each half computes its partial score tile over its 128 dims, the halves trade them through
threadgroup memory (each lane with the same lane of its partner: same fragment coordinates; slots double-buffered, one barrier
per tile) and both form S = S_0 + S_1, so both run the same softmax. The softmax, bf16 P with the normaliser summed over the
rounded weights, masking, and key tiles from absolute key 0 are F1b's, so a row's bits do not depend on how the prompt is cut
into chunks or on the threadgroup shape.

| flash_bench, 35B shape (16 / 2 heads, d256), chunk 2048 | pos0 0 | pos0 2048 | pos0 14336 |
|---|---:|---:|---:|
| P2 `attn_prefill_bf16_flash` (TFLOPS) | 10.0-12.2 | 9.3-12.3 | 10.5-11.3 |
| `attn_nax_d256_h4` (default, 4 heads x 2 halves per threadgroup) | 21.7 | 23.4 | **27.0** |
| `attn_nax_d256_h2` (groups 2 and 6) | 23.2 | 23.0 | 25.5 |
| 64-key tile (not kept: half the speed, other bits) | 11.7 | 13.1 | 13.7 |

Registered for head_dim 256 with an even group, so it also serves Qwen3.8-27B (24 / 4 heads, `h2`; `plan show` changes there,
the model itself was not gated in this lane) and would serve MiniCPM-V-4.7 (16 / 2), which does not load on Metal in either
build. Tests (`tests/flash_prefill_nax.rs`): f64 reference on 8 head_dim-256 shapes (groups 2, 4, 6, 8; partial slabs and
tiles, continuation chunks, NaN cache tails, a 1024-row chunk; worst error 0.735 of the tolerance), chunk invariance bitwise
over 5 splits and h2 == h4 bitwise, predicate. `kernels test --op attn_prefill` on the 35B: 19 cases, 0 failed, ADMITTED.

### 2b. NVFP4 GEMM on the neural accelerators (`matmul_nvfp4_nax`, src/nvfp4.rs)

`nv_w_to_bf16` decodes code x block scale to bf16 into the plan scratch (exactly the B tile `matmul_nvfp4_mma` stages: at most
5 significant bits, exact), and writes the scale pair [gs, 0]; the GEMM is the fp8 NAX one (`gemm_nax_fp8_*`, float
accumulate, scaled by gs in registers, rounded once; tile by `gemm_nax::pick_variant`). x is read in place, so an x with a
leading element skip or not 16-byte aligned runs `matmul_nvfp4_mma` (counted, reported once; none in the 35B). Priority 5 above
`mma`, M > 8 only, so decode (M <= 8: `matmul_nvfp4_gemv`, `_gemv_rows`) is untouched. Tests (`tests/quant.rs`): the oracle
at 9 shapes x bf16/f32 out x weight offsets x x offsets (2: fallback, 16: NAX) and at the 35B shapes; new
`nvfp4_nax_rows_do_not_depend_on_the_call` (a row's bits do not depend on M, its position, or the tile, for 3 shapes and all 5
tiles; run-to-run equal). `kernels test --op matmul_nvfp4`: 14 cases, 0 failed, max_abs 2.4e-3 (as `mma`), ADMITTED.

## 3. Gates (window 2, #7063; short check 6, #7078; window 4, #7096)

| gate | result |
|---|---|
| KL vs vLLM goldens (`accuracy --mode both`, M35P / NXP commands; prefill / decode 21-way KL, scored rows) | short 5.589e-3 / 6.946e-3 -> 5.868e-3 / 1.325e-2 (16 pos); long 1.461e-2 / 1.568e-2 -> 1.464e-2 / 1.631e-2; 4k 1.209e-2 / 8.958e-3 -> 1.254e-2 / 9.001e-3. Class floor (<= 3e-2): PASS on every cell. All-rows prefill KL improved on long (4.621e-2 -> 4.509e-2) and 4k (9.978e-2 -> 9.456e-2) |
| short decode at 64 positions per prompt (the 16-position cell moved by 6.3e-3, above M35P's 3.0e-3 floor, so re-run as NXP did) | off 5.257e-3, d256 only 5.285e-3, nvfp4 only 4.133e-3, both 5.723e-3 (+4.7e-4, inside the noise floor). The 16-position move was window sampling (MGD: that window spreads 1.7x between equivalent builds) |
| per-knob prefill, short | off 5.589e-3, d256 only 5.036e-3, nvfp4 only 5.908e-3, both 5.868e-3: the attention kernel moves prefill KL down, the NVFP4 GEMM up, both within +-6e-4 |
| self-golden, c1 against the knobs-off build (reported, not graded: different K order on 40 layers) | short prefill mean KL21 4.43e-3, 15 / 1441 top-1 flips; decode 8.65e-3, 2 / 160 flips; 4k prefill 7.14e-2 over all 12759 rows (700 flips, 5.5%), decode 6.58e-3 (5 / 192); long prefill 3.09e-2 (1574 / 36887 flips, 4.3%), decode 8.64e-3 (2 / 80). The tool's tight self budgets FAIL, as expected for a K-order change; the large all-rows prefill figures sit on the unscored rows of the 4k / long sets (the vLLM all-rows KL of those sets is 9.5e-2 / 4.5e-2 for base and c1 alike), while the scored rows moved by under 5e-4 against vLLM |
| MMLU n=1425 (localeval, NXP's command, `--limit 25`; bar 0.830) | **PASS, 0.8428** (NXP base 0.840, NXP / main 0.842); `raw/w4/` |
| greedy ids, 256 tokens, short-0 / 4k-0 / 4k-1 / 16k-0 | `--spec mtp --spec-k 2` == `--spec off`, `--prefill-chunk 512` == default chunk, run-to-run (`--repeat 2`) equal: 4 / 4 each. Chunk invariance holds at whole-model level |
| exactness tests (real 35B / 8B) | `batch_recurrent` 14 passed (packed prefill, prefix reuse, hybrid decode beside prefill chunks); `batch_metal_real` 35B tests pass (batched decode bitwise solo, engine equals local engine; the LFM2 test fails only because `INTARSIA_LFM2_DIR` is not set, no LFM2 checkpoint on endor), long prompts 2.6k / 4k / 16k equal the local engine (`--ignored`), `batch_metal` (8B) passed; `rows_exact` 7, `row_tables` 8 passed |
| PRC prefix-reuse gates (#236, 8B, one test per process) | `reuse_rule_resumes_with_cold_bits`, `multi_turn_resumes_with_cold_bits`, `engine_repeats_equal_the_cold_reference`, `engine_multi_turn_equals_cold_reference`: all pass. The Metal 2-row minimum-pass rule is untouched: the 8B plan is byte-identical, NAX flash still takes every pass of >= 2 rows, and both new kernels keep the M boundaries of the kernels they replace (NVFP4: M > 8 as `mma`/GEMV before) |
| `plan show --backend metal` (base / c1 / c1 knobs off) | Qwen3-8B, Llama-3.1-8B, Qwen2.5-7B, Kolibri-1: identical. 35B: `attn_prefill` -> `_flash_nax`, `matmul_nvfp4_nax` row (2 MB scratch); Qwen3.8-27B-Quark: `attn_prefill` -> `_flash_nax`. Knobs off: identical to base on all of them. MiniCPM-V-4.7 does not load in either build |
| full `kernels test` on the 35B shapes | 48 kernels ADMITTED, 0 rejected |
| `cargo test --workspace --release` (endor) | 1104 passed, 1 failed, 28 ignored: the failure was `kernel_list_is_unchanged_by_core_op_fusion_p1`, a pinned kernel-name list that needed `matmul_nvfp4_nax`; fixed (2nd commit), `-p intarsia-backend-metal --lib` 13 / 13 |
| CUDA / HIP | not touched: every change is in `crates/intarsia-backend-metal`, `kernels/metal` and docs |

## 4. Prefill chunk 4096 (measured, not changed)

c1 served with `--prefill-chunk 4096` (third arm of window 3): 4578 / 5749 / 4439 tok/s, +6.2% at 4k and +6.2% at 16k over
c1 at the recipe's 2048, decode equal. The 35B's prefill is chunk-invariant bitwise (ids above), so the change is likely
bitwise, but the recorded Metal 35B baselines carry `prefill_chunk` and would refuse a run at another chunk, so it needs the
baselines re-recorded and its own ids check; not done here.

## 5. What is left between c1 and the limit (35B, 4k, c1 kernel trace: 780 ms)

| item | share at 4k / 16k | where it stands | lever |
|---|---|---|---|
| MoE expert GEMMs | 35.5% / 30.5% | 27-30 TFLOPS: 64-row tiles at most (the mean at chunk 2048), NVFP4 decoded per tile per threadgroup, staged through threadgroup memory with 2 barriers per K step, 128 threads | 128-row tiles (chunk 4096 gives 128 rows per expert) with in-register epilogue; more simdgroups per tile |
| fp8 projections | 29.5% / 23.5% | 47-49 TFLOPS incl. the per-call x QDQ and weight decode (the decode is repeated every chunk) | larger chunk (amortises the decode), or keep decoded weights resident (memory) |
| GDN delta rule | 18.3% / 15.5% | bitwise serial form, 3.3 TFLOPS fp32 | chunked / WY form on the matrix units (not bitwise; leaves chunk invariance to be re-proved) |
| attention | 6.4% / 22.0% | 27 TFLOPS (d128 NAX: 41-52) | fewer barriers (2 tiles per exchange), K/V shared by more heads |

## Reproduce

Scripts in `raw/scripts/` (`w1.sh` attribution, `w2.sh` gates, `w3.sh` A/B, `w4.sh` MMLU and the 4k / long self-golden, `sc6.sh` per-knob KL, `attr.py`
trace classes, `spd.py` localeval tables). fleetq: #7026 (window 1), six short checks under 3 min (#7033 to #7078: flash microbenches, kernel tests, per-knob KL;
see `raw/sc/`), #7063 (window 2), #7081 (window 3), #7096 (window 4).
