# Metal 35B decode: bitwise fused-kernel folds (lane mdl, levers L2 / L3 / L4)

Lane mdl, 2026-10-22, endor (Apple M5 Max). Base: origin/main `db66c31a` + PR #94. Head: branch `lane-mdl/decode-levers`.
Qwen3.6-35B-A3B NVFP4, `--features metal`. Scripts and raw logs: `docs/results/raw/2026-10-22-mdl-decode-levers/`.

## What changed

The core crate already had fused decode ops that the model uses when a backend registers a kernel for them. Metal had none.
Registered now (all bitwise the launches they replace; same row grouping, K order, simd_sum and rounding points):

| lever | op | Metal kernel | replaces |
|---|---|---|---|
| L2 | `MatmulFp8Pair` | `matmul_fp8_pair_gemv` | 2 fp8 GEMVs on one input (k/v, qkv+z) |
| L3 | `RmsNormAddMatmulFp8Pair` | `rmsnorm_add_matmul_fp8_pair_gemv` | rmsnorm_add + 2 fp8 GEMVs |
| L3 | `SigmoidMulMatmulFp8` | `sigmoid_mul_matmul_fp8_gemv` | sigmoid_mul + o_proj GEMV |
| L3 | `GatedRmsnormMatmulFp8` | `gated_rmsnorm_matmul_fp8_gemv` | gated norm + GDN out_proj GEMV |
| L3 | `RmsNormAddMatmulPair` | `rmsnorm_add_matmul_pair_bf16_gemv` | rmsnorm_add + bf16 pair (router, b/a gates) |
| L4 (part) | `MatmulNvfp4Glu`, `SharedDownCombine` | `matmul_nvfp4_glu_gemv`, `shared_down_combine_gemv` | shared-expert gate/up/silu and down/combine |

`MoeBlock` (routed + shared in one dispatch) is not done. Knob: `INTARSIA_METAL_NO_FOLD=<exact kernel names>` leaves folds out
(docs/tuning.md), for A/B.

## Bitwise checks (all PASS)

- Kernel tests, byte-identical to the unfused launches: `tests/fp8_fused.rs` (4), `tests/fused_norm_pair.rs` (1), `tests/nvfp4_fused.rs` (4). `cargo test -p intarsia-backend-metal --release --locked` passes (fleetq #4117).
- 256 greedy ids, main vs head, same slot (fleetq #4128, `raw/.../slotA/summary.txt`): 35B short IDENTICAL, 35B 4408-token prompt IDENTICAL, Qwen3-8B (dense no-op) IDENTICAL.
- `qualify` quick on the 35B (fleetq #4134, #4177): doctor, recipe, plan, kernels (845 cases), gate PASS; acc-oracle prefill top-1 96.26 %, KL 1.09e-2 (floor 2e-2, same as the committed Metal dump's ~1.1e-2).

## Speed (exclusive slot, fleetq #4156): the folds are SLOWER

Decode, 1k context (973 prompt tokens), 129 tokens, interleaved, n=3:

| | run 1 | run 2 | run 3 | mean tok/s |
|---|---|---|---|---|
| main | 123.03 | 123.81 | 124.00 | 123.6 |
| head (all folds) | 119.74 | 117.27 | 120.87 | 119.3 |

-3.5 %. Dispatches per token (command-buffer profile, 65 vs 129 tokens): main 764.0, head 455.0 (-40 %).

So fewer dispatches did not buy time: a fold that stages its input in threadgroup memory and repeats the norm in every
threadgroup costs more GPU time than the launch it removes. The per-fold bisect is below (if it is empty, it did not run before the hard stop).

## Bisect (INTARSIA_METAL_NO_FOLD, exclusive, n=2-3)

(see `raw/2026-10-22-mdl-decode-levers/slotC/summary.txt`)
