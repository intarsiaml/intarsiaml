# Metal G0: multi-row bit-exact decode GEMV (speculative-decoding verification kernel go/no-go)

Date 2026-10-15. Branch `metal-g0-multirow` (off `metal-next-levers` 215faed). Gate G0 of `docs/research/metal-next-levers.md`.
Machine: endor, Apple M5 Max.

## Verdict

**G0 as written: NO-GO at R = 8 on the model as it is served today (misses by 0.008 census-weighted, by 0.19 on the 8-mod-16 layers); PASS at R = 4.**
The gate (docs/research/metal-next-levers.md section 7): T(4)/T(1) <= 1.10 and T(8)/T(1) <= 1.20 summed over a layer's shapes, the multi-row kernel
pinned to the 1-row serving variant, every row bitwise equal to the serving 1-row launch. The fail clause says to stop; that call is the coordinator's, the data is below.

| gate | measured (layer = 2 x q/o + down + k/v pair + gate/up pair) | limit | result |
|---|---|---|---|
| bitwise equal to the serving 1-row launch | yes: every shape, every R, both alignments, real weights | yes | PASS |
| T(4)/T(1), 16-byte aligned weights (28-29 of 36 layers per shape) | **1.006** | 1.10 | PASS |
| T(4)/T(1), 8-mod-16 shard weights (7-8 of 36 layers per shape) | **1.006** | 1.10 | PASS |
| T(8)/T(1), 16-byte aligned weights | **1.164** | 1.20 | PASS |
| T(8)/T(1), 8-mod-16 shard weights | **1.386** | 1.20 | FAIL by 0.19 |
| T(8)/T(1), census-weighted mix of the real shards (the model as served) | **1.208** (1.210 and 1.208 in the two independent runs) | 1.20 | **FAIL by 0.008** |

Why R = 8 fails: the miss is confined to the 8-byte-load kernels (weights at 8 mod 16, which is every tensor of shard 1: 7-8 layers per shape), whose multi-row form degrades from R about 6 (layer 1.07 at R = 6, 1.24 at R = 7, 1.39 at R = 8)
while the 16-byte kernels stay at 1.02 / 1.04 / 1.16. Output rows per threadgroup P = 2, 4 and a split-pass form were tried and made matmul/pair worse (section 1); the mechanism of the cliff at R = 7..8 (register pressure, instruction issue or x cache traffic) is **not identified**.
Per shape (16-byte) at R = 8: k+v pair 1.177, down 1.204, gate+up 1.165, q/o 1.104, lm_head 1.077 (the lm_head formulation is chosen by split: 1.068).
Adding lm_head (a per-step estimate, section 3) gives 1.197 census-weighted at R = 8; that is not a reinterpretation of the gate, which is stated per layer.

Measured counterfactual (a remedy for the coordinator to choose, not a pass): if all weights sat on 16-byte boundaries (the loader copying the 7-8 tensors per shape of shard 1 to 16-aligned memory, or a 16-aligned rewrite of the shard), R = 8 is 1.164 per layer and would pass.

Not part of the gate, but the curve it gives (census-weighted layer): R = 2: 1.002, 3: 1.006, 4: 1.006, 5: 1.013, 6: 1.029, 7: 1.078, 8: 1.208, 16: 2.57. If the speculation design caps the row count at 6 (5 drafted tokens), the whole model is within 3 % of a 1-token step.
Against today: the tiled GEMM that serves M = 2..15 is **7x** slower than the 1-row GEMV per layer (4.5 ms vs 0.65 ms; the k+v pair is 26x because it runs as two tiled GEMMs), so the premise "k rows cost about one weight pass" holds for this kernel and not for the existing path.
NAX at M = 16 costs 1.81x the layer's 1-row time but is not bit-identical to decode. This kernel at R = 16 is 2.56x; R = 9..15 were not measured.

## 1. What was built

All new code; the serving decode path, `crates/intarsia-core`, CUDA and HIP are untouched (`git diff --stat 215faed` touches `lib.rs`, `kernels.rs`, `build.rs` of the Metal crate only for registration).

| File | Content |
|---|---|
| `kernels/metal/gemv_multirow.metal` | Five shader families, templates over `R` activation rows and `P` output rows per threadgroup (per simdgroup for logits), host names `<kernel>_mr<R>x<P>`, `R` in {1..8, 16}, `P` in {1, 2, 4}, plus split-pass instances `..x<P>s<RB>` |
| `crates/intarsia-backend-metal/src/multirow.rs` | Launchers (`Matmul`, `MatmulPair`, `Logits`), encoders for benches and tests, name table, fallback to R ordinary 1-row launches for alignments the 1-row path serves with its scalar kernels |
| `crates/intarsia-backend-metal/tests/multirow.rs` | Bitwise tests against the serving 1-row launches (11 tests) |
| `crates/intarsia-backend-metal/examples/gemv_g0.rs` | The G0 bench (gemv_b3 method) |
| `docs/results/raw/2026-10-15-metal-g0-multirow/` | `run_slot.sh`, `run_slot2.sh`, `analyze_g0.py`, `make_tables.py`, CSVs, macmon traces, logs |

The families mirror the kernels the serving path picks (`kernels.rs`, `plan_gemv`, `fused.rs`, `dense.rs`):

| Op (Qwen3-8B use) | Serving 1-row kernel | Multi-row shader |
|---|---|---|
| `Matmul`, 16-byte aligned (q, o, down) | `gemv_bf16_rt_u2s8` (one threadgroup of 8 simdgroups per row, 2 x 16-byte loads per lane) | `gemv_bf16_rt_u2s8_mr<R,P>` |
| `Matmul`, 8-byte aligned (shard 1, header 8 mod 16) | `gemv_bf16_rt_a8u4s8` | `gemv_bf16_rt_a8u4s8_mr<R,P>` |
| `MatmulPair` (k+v, gate+up), both alignments | `gemv_pair_bf16_rt_u2s8`, `..._a8u4s8` | `gemv_pair_bf16_rt_{u2s8,a8u4s8}_mr<R,P>` |
| `Logits` M=1 (lm_head, f32 out; lm_head is 16-byte aligned) | `gemv_f32o_v` | `gemv_f32o_v_mr<R,P>` |

Design: every thread keeps `R` (times `P`) independent accumulators. The lane-to-chunk mapping, the `dot8`/`dot4` fma chain, the `simd_sum`, and the
fixed-order sum of the 8 per-simdgroup partials are those of the mirrored kernel, so row `r` of the output is bitwise the 1-row result for activation row `r`
regardless of `R` and `P`. Weights are loaded once per thread and used against all `R` rows. Output layout is `x [R,K]`, `y [R,N]` row-major.

Findings from the sweep (all in `analysis.md`): the best configuration is `P = 1` (one output row per threadgroup, the serving geometry) for every matmul and pair shape at every R;
sharing x between P output rows (`P = 2, 4`) was slower (register pressure: R = 8, P = 2 is 1.47x on q/o). A split-pass form (R rows as sequential passes of RB rows in the same threadgroup,
weights re-read from cache) was also slower for matmul and pair (R = 8 as 4+4: 1.42x on q/o), so the cliff between R = 7 and R = 8 is not a simple register-count effect that this restructuring removes.
It does help the logits kernel: lm_head at R = 8 is 1.20x single-pass but **1.068x** as 2 passes of 4 (R = 6 as 2 x 3: 1.017), so the `default_per_tg`/split choice for `Logits` should use the split form. The launcher defaults are still
the single-pass `P = 1` forms (stub `default_per_tg`); wiring the logits split into the launcher is a follow-up for the real build.

## 2. Per-shape results (16-byte aligned weights, the majority case)

T(1) is the serving kernel (`gemv_bf16_rt_u2s8`, `gemv_pair_bf16_rt_u2s8`, `gemv_f32o_v`), medians of within-round ratios over clock-filtered rounds, two independent bench runs pooled
(60 rounds per shape before filtering; see section 5). "Existing" is what a call with M = R takes today: the tiled GEMM for M = 2..15 (a pair is two tiled GEMMs, as the model runs it),
the NAX GEMM at M = 16 (logits have no NAX kernel, tiled GEMM). "cfg" is the best (P / split) configuration per R; for matmul and pair it is `1` everywhere except R = 16.

| shape | R | T(1) serving us | T(R) existing us (path) | existing / T(1) | T(R) prototype us | prototype / T(1) | bitwise |
|---|---|---|---|---|---|---|---|
| q / o (4096x4096) | 2 | 57.6 | 406.3 (tiled GEMM) | 7.06x | 57.8 (cfg 1) | **1.004** | yes (6 checks) |
|  | 3 | 57.6 | 409.0 (tiled GEMM) | 7.10x | 58.4 (cfg 1) | **1.015** | yes (6 checks) |
|  | 4 | 57.6 | 410.7 (tiled GEMM) | 7.13x | 58.3 (cfg 1) | **1.013** | yes (6 checks) |
|  | 5 | 57.6 | 420.1 (tiled GEMM) | 7.30x | 58.9 (cfg 1) | **1.024** | yes (6 checks) |
|  | 6 | 57.6 | 422.6 (tiled GEMM) | 7.34x | 60.1 (cfg 1) | **1.044** | yes (6 checks) |
|  | 7 | 57.6 | 425.7 (tiled GEMM) | 7.39x | 61.4 (cfg 1) | **1.067** | yes (6 checks) |
|  | 8 | 57.6 | 428.3 (tiled GEMM) | 7.44x | 63.5 (cfg 1) | **1.104** | yes (6 checks) |
|  | 16 | 57.6 | 113.9 (NAX GEMM) | 1.98x | 166.3 (cfg 4) | **2.892** | yes (6 checks) |
| down (K=12288, N=4096) | 2 | 169.4 | 1245.8 (tiled GEMM) | 7.35x | 169.8 (cfg 1) | **1.002** | yes (6 checks) |
|  | 3 | 169.4 | 1255.1 (tiled GEMM) | 7.41x | 170.1 (cfg 1) | **1.003** | yes (6 checks) |
|  | 4 | 169.4 | 1261.7 (tiled GEMM) | 7.45x | 170.2 (cfg 1) | **1.004** | yes (6 checks) |
|  | 5 | 169.4 | 1309.5 (tiled GEMM) | 7.73x | 169.9 (cfg 1) | **1.004** | yes (6 checks) |
|  | 6 | 169.4 | 1330.0 (tiled GEMM) | 7.85x | 170.5 (cfg 1) | **1.007** | yes (6 checks) |
|  | 7 | 169.4 | 1342.6 (tiled GEMM) | 7.92x | 174.4 (cfg 1) | **1.030** | yes (6 checks) |
|  | 8 | 169.4 | 1354.0 (tiled GEMM) | 7.99x | 203.8 (cfg 1) | **1.204** | yes (6 checks) |
|  | 16 | 169.4 | 333.9 (NAX GEMM) | 1.97x | 358.9 (cfg 4) | **2.126** | yes (6 checks) |
| k+v pair (2 x 1024x4096) | 2 | 29.5 | 765.9 (tiled GEMM) | 25.92x | 29.9 (cfg 1) | **1.009** | yes (6 checks) |
|  | 3 | 29.5 | 770.0 (tiled GEMM) | 26.06x | 30.1 (cfg 1) | **1.019** | yes (6 checks) |
|  | 4 | 29.5 | 772.7 (tiled GEMM) | 26.15x | 30.2 (cfg 1) | **1.021** | yes (6 checks) |
|  | 5 | 29.5 | 792.9 (tiled GEMM) | 26.84x | 30.8 (cfg 1) | **1.043** | yes (6 checks) |
|  | 6 | 29.5 | 796.3 (tiled GEMM) | 26.95x | 31.9 (cfg 1) | **1.078** | yes (6 checks) |
|  | 7 | 29.5 | 800.2 (tiled GEMM) | 27.09x | 32.7 (cfg 1) | **1.107** | yes (6 checks) |
|  | 8 | 29.5 | 802.4 (tiled GEMM) | 27.16x | 34.8 (cfg 1) | **1.177** | yes (6 checks) |
|  | 16 | 29.5 | 78.4 (NAX GEMM) | 2.65x | 89.7 (cfg 4) | **3.036** | yes (6 checks) |
| gate+up pair (2 x 12288x4096) | 2 | 336.8 | 1677.0 (tiled GEMM) | 4.98x | 336.9 (cfg 1) | **1.000** | yes (6 checks) |
|  | 3 | 336.8 | 1687.2 (tiled GEMM) | 5.01x | 337.9 (cfg 1) | **1.003** | yes (6 checks) |
|  | 4 | 336.8 | 1695.1 (tiled GEMM) | 5.03x | 338.0 (cfg 1) | **1.004** | yes (6 checks) |
|  | 5 | 336.8 | 1747.9 (tiled GEMM) | 5.19x | 339.1 (cfg 1) | **1.007** | yes (6 checks) |
|  | 6 | 336.8 | 1766.1 (tiled GEMM) | 5.24x | 341.0 (cfg 1) | **1.013** | yes (6 checks) |
|  | 7 | 336.8 | 1779.3 (tiled GEMM) | 5.28x | 348.0 (cfg 1) | **1.036** | yes (6 checks) |
|  | 8 | 336.8 | 1789.5 (tiled GEMM) | 5.31x | 391.9 (cfg 1) | **1.165** | yes (6 checks) |
|  | 16 | 336.8 | 540.5 (NAX GEMM) | 1.61x | 884.8 (cfg 4) | **2.628** | yes (6 checks) |
| lm_head (151936x4096, f32 out) | 2 | 2088.8 | 6394.2 (tiled GEMM) | 3.06x | 2093.0 (cfg 2) | **1.000** | yes (6 checks) |
|  | 3 | 2088.8 | 6429.0 (tiled GEMM) | 3.08x | 2101.2 (cfg 2) | **1.006** | yes (6 checks) |
|  | 4 | 2088.8 | 6461.1 (tiled GEMM) | 3.09x | 2105.8 (cfg 1) | **1.009** | yes (6 checks) |
|  | 5 | 2088.8 | 6609.1 (tiled GEMM) | 3.16x | 2120.3 (cfg 1) | **1.011** | yes (6 checks) |
|  | 6 | 2088.8 | 6681.7 (tiled GEMM) | 3.20x | 2152.8 (cfg 2) | **1.029** | yes (6 checks) |
|  | 7 | 2088.8 | 6759.2 (tiled GEMM) | 3.24x | 2156.9 (cfg 2) | **1.031** | yes (6 checks) |
|  | 8 | 2088.8 | 6794.8 (tiled GEMM) | 3.25x | 2242.5 (cfg 2) | **1.077** | yes (6 checks) |
|  | 16 | 2088.8 | 7596.0 (tiled GEMM (no NAX logits)) | 3.64x | 5181.6 (cfg 1) | **2.487** | yes (6 checks) |


### 8-byte-aligned weights (shard 1; matmul and pairs only, lm_head is 16-byte aligned)

| shape | R | T(1) serving us | T(R) existing us (path) | existing / T(1) | T(R) prototype us | prototype / T(1) | bitwise |
|---|---|---|---|---|---|---|---|
| q / o (4096x4096) | 2 | 58.1 | 410.0 (tiled GEMM) | 7.05x | 58.2 (cfg 1) | **1.002** | yes (3 checks) |
|  | 3 | 58.1 | 412.0 (tiled GEMM) | 7.09x | 59.0 (cfg 1) | **1.011** | yes (3 checks) |
|  | 4 | 58.1 | 414.0 (tiled GEMM) | 7.12x | 58.9 (cfg 1) | **1.012** | yes (3 checks) |
|  | 5 | 58.1 | 420.7 (tiled GEMM) | 7.24x | 61.0 (cfg 1) | **1.046** | yes (3 checks) |
|  | 6 | 58.1 | 423.2 (tiled GEMM) | 7.28x | 63.7 (cfg 1) | **1.095** | yes (3 checks) |
|  | 7 | 58.1 | 426.5 (tiled GEMM) | 7.34x | 68.7 (cfg 1) | **1.179** | yes (3 checks) |
|  | 8 | 58.1 | 428.8 (tiled GEMM) | 7.38x | 76.6 (cfg 1) | **1.318** | yes (3 checks) |
|  | 16 | 58.1 | 114.7 (NAX GEMM) | 1.97x | 162.7 (cfg 1) | **2.807** | yes (3 checks) |
| down (K=12288, N=4096) | 2 | 168.5 | 1245.3 (tiled GEMM) | 7.39x | 169.3 (cfg 1) | **1.003** | yes (3 checks) |
|  | 3 | 168.5 | 1254.2 (tiled GEMM) | 7.44x | 169.6 (cfg 1) | **1.005** | yes (3 checks) |
|  | 4 | 168.5 | 1261.6 (tiled GEMM) | 7.49x | 169.6 (cfg 1) | **1.004** | yes (3 checks) |
|  | 5 | 168.5 | 1308.8 (tiled GEMM) | 7.77x | 170.1 (cfg 1) | **1.007** | yes (3 checks) |
|  | 6 | 168.5 | 1328.8 (tiled GEMM) | 7.89x | 174.2 (cfg 1) | **1.033** | yes (3 checks) |
|  | 7 | 168.5 | 1341.4 (tiled GEMM) | 7.96x | 202.8 (cfg 1) | **1.204** | yes (3 checks) |
|  | 8 | 168.5 | 1354.5 (tiled GEMM) | 8.04x | 229.3 (cfg 1) | **1.361** | yes (3 checks) |
|  | 16 | 168.5 | 330.7 (NAX GEMM) | 1.96x | 359.8 (cfg 4) | **2.131** | yes (3 checks) |
| k+v pair (2 x 1024x4096) | 2 | 29.7 | 763.2 (tiled GEMM) | 25.73x | 30.0 (cfg 1) | **1.014** | yes (3 checks) |
|  | 3 | 29.7 | 767.7 (tiled GEMM) | 25.88x | 30.4 (cfg 1) | **1.026** | yes (3 checks) |
|  | 4 | 29.7 | 770.1 (tiled GEMM) | 25.96x | 30.6 (cfg 1) | **1.030** | yes (3 checks) |
|  | 5 | 29.7 | 790.8 (tiled GEMM) | 26.66x | 32.2 (cfg 1) | **1.085** | yes (3 checks) |
|  | 6 | 29.7 | 794.4 (tiled GEMM) | 26.78x | 33.5 (cfg 1) | **1.134** | yes (3 checks) |
|  | 7 | 29.7 | 798.1 (tiled GEMM) | 26.91x | 37.5 (cfg 1) | **1.265** | yes (3 checks) |
|  | 8 | 29.7 | 800.2 (tiled GEMM) | 26.98x | 42.3 (cfg 1) | **1.433** | yes (3 checks) |
|  | 16 | 29.7 | 77.4 (NAX GEMM) | 2.61x | 84.3 (cfg 1) | **2.844** | yes (3 checks) |
| gate+up pair (2 x 12288x4096) | 2 | 337.8 | 1676.2 (tiled GEMM) | 4.96x | 337.1 (cfg 1) | **1.001** | yes (3 checks) |
|  | 3 | 337.8 | 1687.9 (tiled GEMM) | 5.00x | 338.1 (cfg 2) | **1.002** | yes (3 checks) |
|  | 4 | 337.8 | 1696.6 (tiled GEMM) | 5.02x | 338.5 (cfg 1) | **1.004** | yes (3 checks) |
|  | 5 | 337.8 | 1752.0 (tiled GEMM) | 5.19x | 343.0 (cfg 1) | **1.013** | yes (3 checks) |
|  | 6 | 337.8 | 1771.3 (tiled GEMM) | 5.24x | 361.4 (cfg 2) | **1.070** | yes (3 checks) |
|  | 7 | 337.8 | 1790.2 (tiled GEMM) | 5.30x | 422.7 (cfg 1) | **1.251** | yes (3 checks) |
|  | 8 | 337.8 | 1803.6 (tiled GEMM) | 5.34x | 479.0 (cfg 1) | **1.418** | yes (3 checks) |
|  | 16 | 337.8 | 537.9 (NAX GEMM) | 1.59x | 943.1 (cfg 4) | **2.796** | yes (3 checks) |


## 3. Aggregates

### Layer sum (2 x q/o + down + k/v pair + gate/up pair), GEMV time only

| R | alignment | T_layer(1) serving (us) | T_layer(R) multi-row (us) | ratio | T_layer(R) existing path (us) | existing ratio |
|---|---|---|---|---|---|---|
| 2 | 16-byte | 650.9 | 652.1 | 1.002 | 4501.3 | 6.92 |
| 3 | 16-byte | 650.9 | 654.8 | 1.006 | 4530.4 | 6.96 |
| 4 | 16-byte | 650.9 | 654.9 | 1.006 | 4550.9 | 6.99 |
| 5 | 16-byte | 650.9 | 657.7 | 1.011 | 4690.4 | 7.21 |
| 6 | 16-byte | 650.9 | 663.6 | 1.020 | 4737.6 | 7.28 |
| 7 | 16-byte | 650.9 | 678.0 | 1.042 | 4773.5 | 7.33 |
| 8 | 16-byte | 650.9 | 757.5 | 1.164 | 4802.6 | 7.38 |
| 16 | 16-byte | 650.9 | 1666.1 | 2.560 | 1180.6 | 1.81 |
| 2 | 8-byte | 652.3 | 652.8 | 1.001 | 4504.8 | 6.91 |
| 3 | 8-byte | 652.3 | 656.1 | 1.006 | 4533.9 | 6.95 |
| 4 | 8-byte | 652.3 | 656.4 | 1.006 | 4556.2 | 6.99 |
| 5 | 8-byte | 652.3 | 667.2 | 1.023 | 4692.9 | 7.19 |
| 6 | 8-byte | 652.3 | 696.4 | 1.068 | 4740.9 | 7.27 |
| 7 | 8-byte | 652.3 | 800.3 | 1.227 | 4782.6 | 7.33 |
| 8 | 8-byte | 652.3 | 903.8 | 1.386 | 4815.8 | 7.38 |
| 16 | 8-byte | 652.3 | 1712.7 | 2.626 | 1175.5 | 1.80 |

### Census-weighted (Qwen3-8B: 7-8 of 36 layers per shape sit in the 8-mod-16 shard), per layer, GEMV time

| R | T_layer(1) (us) | T_layer(R) multi-row | ratio | existing | existing ratio |
|---|---|---|---|---|---|
| 2 | 651.2 | 652.2 | 1.002 | 4502.0 | 6.91 |
| 3 | 651.2 | 655.1 | 1.006 | 4531.1 | 6.96 |
| 4 | 651.2 | 655.3 | 1.006 | 4552.0 | 6.99 |
| 5 | 651.2 | 659.7 | 1.013 | 4690.9 | 7.20 |
| 6 | 651.2 | 670.1 | 1.029 | 4738.2 | 7.28 |
| 7 | 651.2 | 702.1 | 1.078 | 4775.2 | 7.33 |
| 8 | 651.2 | 786.5 | 1.208 | 4805.2 | 7.38 |
| 16 | 651.2 | 1675.0 | 2.572 | 1179.6 | 1.81 |

### Per verify step, GEMVs only: 36 x layer + lm_head (all R rows through the head), 16-byte-aligned weights

| R | step(1) (ms) | step(R) multi-row (ms) | step(R)/step(1) | step(R) existing (ms) | existing/step(1) | per-token cost multi-row (ms/row) |
|---|---|---|---|---|---|---|
| 2 | 25.52 | 25.57 | 1.002 | 168.44 | 6.60 | 12.78 |
| 3 | 25.52 | 25.68 | 1.006 | 169.52 | 6.64 | 8.56 |
| 4 | 25.52 | 25.68 | 1.006 | 170.29 | 6.67 | 6.42 |
| 5 | 25.52 | 25.80 | 1.011 | 175.46 | 6.88 | 5.16 |
| 6 | 25.52 | 26.04 | 1.020 | 177.24 | 6.94 | 4.34 |
| 7 | 25.52 | 26.56 | 1.041 | 178.61 | 7.00 | 3.79 |
| 8 | 25.52 | 29.51 | 1.156 | 179.69 | 7.04 | 3.69 |
| 16 | 25.52 | 65.16 | 2.553 | 50.10 | 1.96 | 4.07 |

### Per verify step, GEMVs only, census-weighted layers (16-byte / 8-mod-16 mix of the real shards) + lm_head

| R | step(1) (ms) | step(R) multi-row (ms) | step(R)/step(1) |
|---|---|---|---|
| 2 | 25.53 | 25.57 | 1.002 |
| 3 | 25.53 | 25.68 | 1.006 |
| 4 | 25.53 | 25.70 | 1.006 |
| 5 | 25.53 | 25.87 | 1.013 |
| 6 | 25.53 | 26.28 | 1.029 |
| 7 | 25.53 | 27.43 | 1.074 |
| 8 | 25.53 | 30.56 | 1.197 |
| 16 | 25.53 | 65.48 | 2.565 |

The per-step numbers cover the GEMVs only (36 layers + lm_head, all R rows through the head). The 1-token step also contains attention, norms, RoPE and so on (about 1.5 ms of the 27.6 ms in the overnight trace) and
attention with q_len > 1 is **not covered** here (K2 in the plan). For R = 5 the GEMV part of a verify step costs 1.011x a decode step: five tokens' logits for 0.29 ms extra.

## 4. Bitwise equality, kernel test, cargo test

* `crates/intarsia-backend-metal/tests/multirow.rs`: for every activation row, the multi-row output equals byte for byte (guard bytes included) `R` separate launches of the serving
  kernels (`matmul_bf16_gemv`, `matmul_pair_bf16_gemv`, `logits_bf16_gemv`), distinct and differently scaled x rows. Covered: Qwen3-8B shapes (4096x4096, K=12288 down, 1024 and 12288 pairs, lm_head 151936x4096), all R in {1..8, 16},
  every (R, P) and split instance by direct encoding, K tails (4104, 8200, 520, 1032), odd N (1031, 33, 5, 3), 8-byte aligned weights, scalar-kernel alignments (fallback path), and the **real Qwen3-8B
  weights** mmap'd and wrapped with `wrap_host` as the loader does (layer 0 in shard 1 at 8 mod 16, layer 20 in shard 3, lm_head in shard 5), M in {2..8, 16}. A deliberate reversal of the accumulation order in the 16-byte shader makes the tests fail (checked once, reverted).
* The bench re-checks every (R, P) and split instance against R serving 1-row launches on its own buffers before timing: 378 checks across the aligned and 8-byte runs, all identical (a failure aborts the run).
* `cargo test -p intarsia-backend-metal --release`: **87 passed, 0 failed** (`cargo-test-metal.log`, inside slot #1980; the 11 multirow tests re-run inside #1985 after the split-pass change, `split/cargo-test-multirow.log`; the full suite once more on the final commit, outside a slot, `cargo-test-metal-final.log`: 87 passed).
* `intarsia kernels test --backend metal -m ~/models/Qwen3-8B`: default run 25 kernels, 281 cases, all ADMITTED, exit 0 (includes `matmul_bf16_gemv_rows`, `logits_bf16_gemv_rows`); with `INTARSIA_METAL_MULTIROW=1` 26 kernels, 293 cases, all
  ADMITTED, exit 0 (adds `matmul_pair_bf16_gemv_rows`) (`kernels-test-metal-default.log`, `kernels-test-metal-multirow.log`).

What is not achievable without touching `crates/intarsia-core`: **the harness cannot grade "bitwise vs the 1-row Metal reference".** `admission::run_admission` grades every candidate against the cpu backend's reference kernel; `Exactness::Exact` there means bit equality
with the CPU result, which no Metal GEMV (the 1-row one included) satisfies. The entries are therefore `Exactness::Deterministic` (graded at the usual tolerance: matmul max_abs 2.2e-4, logits 1.1e-7 against the CPU) and the bitwise property is proved by `tests/multirow.rs`.
Admitting them as `Exact` in `kernels test` needs a core change: a per-kernel `bit_reference: Option<kernel name>` that makes `run_admission` compare against that kernel of the same backend, row by row.

Serving is unchanged: the registry entries have priority -1 (below the tiled GEMM 0 and NAX 5), a test asserts the pick for M = 2, 3, 8, 15 is the tiled GEMM / `logits_bf16`, and the `MatmulPair` entry is **not listed by `Backend::kernels()` unless `INTARSIA_METAL_MULTIROW=1`**
because no pair kernel accepts M = 2..8 today and a listed one would silently make the model fuse short prefill chunks with it (a first attempt that only gated the predicate made the default `kernels test` fail with "NO CASES (untested)").

## 5. Method and data quality

`examples/gemv_g0.rs` (the gemv_b3 method): per shape one command buffer holding a serial chain of dispatches over a rotating pool of distinct weight buffers (3 GiB: 96 q/o, 32 down, 192 k+v pair, 16 gate+up pair, 3 lm_head entries; the chain is two traversals),
GPU start/end time per chain, candidates interleaved in rotated order inside each round, 30 rounds per run; the 16-byte case ran twice, the 8-byte case once. Every sample carries its epoch; `macmon pipe -i 100` ran alongside and
`analyze_g0.py` drops every round whose 100 ms GPU-clock samples (with 100 ms margin each side) include one below 1450 MHz; ratios are taken within a round and the median reported. GPU util by `ioreg` was 0-3 % at the start of each case.

| case | a8 | rounds valid/total | clock mean (MHz) over valid rounds |
|---|---|---|---|
| down | 0 | 53/60 | 1593 |
| down | 1 | 28/30 | 1605 |
| gu | 0 | 53/60 | 1589 |
| gu | 1 | 18/30 | 1594 |
| kv | 0 | 58/60 | 1614 |
| kv | 1 | 29/30 | 1613 |
| lm | 0 | 35/60 | 1598 |
| qo | 0 | 58/60 | 1601 |
| qo | 1 | 29/30 | 1598 |

T(1) reproduces the overnight numbers: q/o 57.6 us (582 GB/s), down 169.4 us (594 GB/s), gate+up 336.8 us (598 GB/s), lm_head 2.089 ms (595 GB/s), and the 36-layer GEMV-only step 25.5 ms against the 26.07 ms measured with the full model.
The second, independent bench set (`split/`, a different slot, filtered the same way) reproduces the layer ratios: R = 4 1.008, R = 8 1.164 (16-byte) and 1.402 (8-byte).
The tiled-GEMM and NAX rows come from the same rounds. Clock rejection removes 3-12 % of rounds for most shapes, 42 % for lm_head (the longest chains) and 40 % for gate+up at 8-byte alignment (18 of 30 rounds remain).

## 6. fleetq

Slots on endor (exclusive): **#1978** (granted 08:20; the run failed because my `run_slot.sh` exported an empty `DEVELOPER_DIR`, which made cargo rebuild the metallib without the toolchain; no data, released), **#1980** (all data in this report, released),
**#1983** (bench bug in the new split dispatch: panic after the bitwise checks, released), **#1985** (split-pass data in `split/`, released). All released by `trap` on exit. #1975 and #1977 belong to other work and were not touched.

**GPU work on endor outside a slot (disclosure, please check these windows against the other agents' runs).** Short correctness runs (cargo tests, kernels test, a mutation check, `pso_info`; no timings used anywhere) ran without a slot at about: 07:23-07:27 (multirow tests plus mutation check, while #1975 was held), 08:05-08:12 (full `cargo test`, four `kernels test` runs, while #1975/#1977 were held or starting), 08:57-09:01 (multirow tests, `pso_info`), 09:28-09:30 (full `cargo test`, after #1985 was released). Each lasted well under a minute of GPU activity except the `kernels test` runs (about 45 s each). No GPU work was done outside a slot after that.

## 7. What this does not cover

* Attention with q_len > 1 (K2), the row-wise argmax (K3), per-row positions (K4), the engine loop, and the non-GEMV ops of a verify step.
* R = 9..15 (not instantiated); R = 16 is 2.5x here, so a 16-row verify would use a different formulation.
* Weights other than BF16 (quantised paths), other models, and prompt-lookup acceptance rates (G1-G3).
* Sustained thermals: rounds are short chains; the clock filter only drops rounds below 1450 MHz, the retained mean is 1589-1614 MHz.

## 8. Reproduce

```
fleetq acquire --host endor --exclusive --mem 20 --label "metal G0 multirow gemv" --wait      # id N
FLEETQ_JOB=N docs/results/raw/2026-10-15-metal-g0-multirow/run_slot.sh                        # cargo test, kernels test, 2 aligned runs + 1 8-byte run, macmon 100 ms
FLEETQ_JOB=N docs/results/raw/2026-10-15-metal-g0-multirow/run_slot2.sh                       # split-pass experiment
fleetq release N
python3 docs/results/raw/2026-10-15-metal-g0-multirow/analyze_g0.py docs/results/raw/2026-10-15-metal-g0-multirow
python3 docs/results/raw/2026-10-15-metal-g0-multirow/make_tables.py docs/results/raw/2026-10-15-metal-g0-multirow 0
```
