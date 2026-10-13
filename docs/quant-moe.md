# Quantised weights and the MoE block

Code: `crates/intarsia-core/src/quant/` (formats, op validation, oracle math, admission cases), cpu kernels in
`crates/intarsia-backend-cpu/src/quant_ref.rs`, CUDA in `kernels/cuda/{fp8,nvfp4,moe}.cu` registered by
`crates/intarsia-backend-cuda/src/quantmoe.rs`. ARCHITECTURE 3.4 (`QuantFormat`).

## Formats (`quant::QuantFormat`: `detect`, `load`, `dequant_ref`, `quantize`)

| id | checkpoint tensors | meaning |
|---|---|---|
| `bf16` | `weight` | dense |
| `fp8_e4m3` | `weight` e4m3 `[N,K]`, `weight_scale` f32 scalar, `input_scale` f32 scalar | `W = float(weight) * weight_scale` |
| `nvfp4` | `weight` u8 `[N,K/2]`, `weight_scale` e4m3 `[N,K/16]`, `weight_scale_2` f32 scalar, `input_scale` | `W = e2m1(code) * e4m3(weight_scale[n,k/16]) * weight_scale_2` |

NVFP4 details, checked against real tensors of the checkpoint and TensorFold's numpy dequantiser: element `2i` is
the low nibble of byte `i`, element `2i+1` the high nibble; the e2m1 magnitudes are `0 .5 1 1.5 2 3 4 6` with
the sign in bit 3; block scales are plain row-major `[N,K/16]` (no swizzle). The Qwen3.6 checkpoint is
`W4A16_NVFP4` (activations stay 16-bit), so the NVFP4 `input_scale` is ignored.

`dequant_ref` is the meaning of every kernel: a quantised matmul equals the dense matmul of the dequantised weight.
Tensors keep the checkpoint layout (no repacking at load; the routed experts of a layer are stacked into one
buffer per matrix kind, gate rows then up rows, `quant::moe::MoeWeights::load`).

## FP8 activations: W8A8 or W8A16 (per recipe)

`[defaults] fp8_act = "static" | "none"` (`recipe::Defaults::fp8_act`, `quant::Fp8Act`, default `static`).

* `static` (W8A8): `xq = e4m3(x / input_scale)` (RNE, saturating at 448), `y = input_scale * weight_scale * xq . wq`.
  This is the scheme the checkpoint declares and calibrated, and what vLLM runs for it.
* `none` (W8A16): bf16 activations, `y = weight_scale * x . wq`.

The mode travels in the op's `scales` tensor `[weight_scale, input_scale]` (`input_scale <= 0` = W8A16); kernels read
it on the device, so one kernel serves both. Which mode tracks the vLLM logits better is NOT measured yet: it needs
the assembled 35B model (no layer loop for linear attention / MoE existed when this was written). Compare both with
`intarsia accuracy --golden qwen36-35b-a3b-nvfp4-vllm-top20.safetensors` once the model loads. The two modes do differ
on real slices (relative L2 of the outputs 2e-2 for `in_proj_z`, 0.28 for `o_proj` on unit-variance random inputs,
where the tiny `o_proj` input_scale 0.0029 saturates; real activations are smaller), so the choice matters.

## MoE block (HF `Qwen3_5MoeSparseMoeBlock`)

```
logits = router(x)                         Matmul (bf16 dense gate, [E,H])
(w, id) = topk(softmax(logits)), w /= sum  Op::MoeTopk   attrs top_k, norm_topk
routed  = sum_k w_k * down_e(silu(gate_e x) * up_e x)    Op::MoeExperts (NVFP4 experts, stacked)
shared  = shared_down(silu(shared_gate x) * shared_up x) MatmulNvfp4 x3 + SiluGlu
out     = routed + sigmoid(shared_expert_gate x) * shared   Op::MoeCombine
```

The generic chain is `MoeWeights::forward(&dyn OpRunner, ..)`: the model loader supplies a runner that goes through
`Model::exec` (profiling, plan recording, scratch). Ops: `matmul_fp8`, `matmul_nvfp4`, `moe_topk`, `moe_experts`,
`moe_combine` (signatures in `ops.rs`, ABI in `kernels/abi/ik_abi.h`).

Rounding follows the HF bf16 chain (linear outputs, `silu(gate)`, `silu(gate)*up`, expert output, `y*w`, `sigmoid`, the
final add are rounded to the activation dtype); with f32 activations the cpu oracle equals the HF float32 module.

Shared expert width, the router dtype and the shared-expert-gate dtype were taken from the checkpoint: shared
expert projections are NVFP4 (`W4A16_NVFP4`), the router `gate.weight` and `shared_expert_gate.weight` are bf16.

## CUDA kernels

| kernel | when | notes |
|---|---|---|
| `matmul_fp8_gemv` | M <= 4 | warp per output row, x staged (and quantised) once per block in shared memory |
| `matmul_fp8_mma` | any M | 64-column tiles, e4m3 -> bf16 in shared memory (exact), bf16 `mma.sync`, f32 accumulate |
| `matmul_nvfp4_gemv` | M <= 8 | warp per row, one 16-block per lane-iteration, fp4 decode with `prmt` |
| `matmul_nvfp4_mma` | any M | dequantise-to-bf16 tiles, bf16 MMA (prefill, correct path) |
| `moe_topk_bf16` | E <= 512, K <= 32 | warp per token, ties: lowest index |
| `moe_experts_bf16` | E <= 1024 | group pairs by expert (1 block), gate/up and down launches over the *active* experts only (decode T=1 launches 8 expert slots), each weight row streamed once and applied to up to 8 pairs, ordered reduce (no atomics on the output) |
| `moe_combine_bf16` | | elementwise |

All are `Deterministic` (the counting sort uses atomics, but each pair's arithmetic does not depend on its slot).
`moe_experts` needs scratch (`quant::ops::moe_experts_scratch`, plan-reserved).

Not done (v1): FP4 tensor-core MMA (`mma ... e2m1`), cuBLASLt FP8 for prefill, a fused whole-block decode op,
GEMV variants with the router/top-k fused. The prefill path (`*_mma`) re-dequantises a weight tile for every
M-tile of 64 rows, so very long chunks are slower than a dequantise-once-then-cuBLAS scheme would be.

## Tests

* Mac: `cargo test -p intarsia-core quant`, `cargo test -p intarsia-backend-cpu --test quant_hf`. The latter checks
  dequantisation, FP8 (both modes), NVFP4 matmul, top-k and a whole 8-expert MoE block against values computed by
  `tools/gen_quantmoe_fixture.py` (torch float8 + e2m1 table + the HF module) from real tensor slices
  (`tests/fixtures/qwen36_quant_slices.safetensors`, 0.5 MB, cut by `tools/extract_quant_slices.py`).
* spark1: `intarsia kernels test --backend cuda --op matmul_fp8` (and `matmul_nvfp4`, `moe_topk`, `moe_experts`,
  `moe_combine`) runs the admission sweep incl. the real 35B-A3B shapes (FP8 8192x2048 ... NVFP4 248320-row head,
  64-expert H=2048 I=512 top-8); `cargo test -p intarsia-backend-cuda --test quantmoe -- --include-ignored --nocapture`
  adds skewed/invalid routing and a microbenchmark.
