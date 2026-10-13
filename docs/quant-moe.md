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

**Batch invariance of `moe_experts`.** On CUDA a token's routed-expert output does not depend on how many other tokens share the call. On the HIP backend
it does from 16 tokens on: T < 16 (and P = T*K <= 16) use GEMV-style kernels with one accumulation order (bit-identical to T = 1), T >= 16 uses a bf16-WMMA
path with a different order (same results within the admission tolerance, not the same bits). So on HIP `moe_experts` is `Deterministic` (run-to-run identical for a
given shape and batch) but not batch-invariant; `INTARSIA_HIP_MOE_WMMA=0` restores invariance at a prefill cost (MoE 3x slower at T = 256). Details and the test: `docs/backends/hip.md`, "MoE and batch size".

Not done (v1): FP4 tensor-core MMA (`mma ... e2m1`), cuBLASLt FP8 for prefill, a fused whole-block decode op,
GEMV variants with the router/top-k fused. The `*_mma` prefill path re-dequantises a weight tile for every M-tile
of 64 rows; the `*_gemm` kernels below (phase 1b) are 1.3-2x faster with a better memory schedule but still dequantise per
M-tile.

## Prefill and decode kernels added in phase 1b (key `moe`)

Standalone numbers below are from a GB10 (sm_121, same part as spark1) with `kernels/cuda/*.cu` built by nvcc 13.0 and
driven by a small C++ harness (random weights at the real layer shape: 256 experts, H=2048, I=512, top-8). The
end-to-end spark1 numbers are not in yet (spark1 was offline for the whole session).

| kernel | selected for | what it does |
|---|---|---|
| `moe_experts_mma_bf16` (`moe_grouped.cu`) | T >= 32, H % 128, I % 128 | counting-sort (token, slot) pairs by expert and cut them into tiles of BM = 16/32/64 pairs of one expert; gate_up block = tile x 64 intermediate columns (64 gate + 64 up rows): weights are copied with cp.async, dequantised once per tile and 64-wide k step into bf16 shared memory, `mma.sync` m16n8k16 bf16, each k16 partial sum formed from zero and added to the f32 accumulator with an fp32 add (accumulating inside the mma truncates and put about 1 % of the bf16 outputs one step off), SiLU-GLU in the epilogue; down block = tile x 128 output rows; fixed-order reduce over k. Only active experts get blocks. |
| `moe_experts_decode_bf16` (`moe_decode.cu`) | T = 1 | gate_up+silu in one launch, down + per-token reduce in the second (4 launches -> 2). Bit-identical to `moe_experts_bf16`. |
| `matmul_nvfp4_gemm`, `matmul_fp8_gemm` (`nvfp4_gemm.cu`, `fp8_gemm.cu`, `wq_gemm.cuh`) | M >= 33, K % 128 | same schedule as the grouped MoE kernel for dense weights. W8A8 pre-quantises the activations once into scratch. |

Memory schedule (the thing that decided the speed, measured with a no-compute read microbenchmark on the same box): reading
a weight row in 32-byte pieces at a 1 KB row stride reached 120-130 GB/s, runs of 64 B/row 209 GB/s, 128 B/row 231 GB/s,
fully linear 243 GB/s. The first version of the grouped kernel (32 B/row per thread) was bound by that; staging 64-byte
runs through cp.async (units of 128 fp4 k-values per row, two 64-wide steps per unit) removed it. Other things tried
and kept: 8 warps per block for BM >= 32 (more overlap between dequantise, copy and mma), `ldmatrix.x4` for both
operands, a 3-slot A ring, full-row output stores through shared memory, tile index on `blockIdx.y` so the weight
column tiles of one pair tile are adjacent.

Standalone results, microseconds per call (old `moe_experts_bf16` -> `moe_experts_mma_bf16`, BM chosen from pairs/expert):

| tokens | old | new | speed-up |
|---|---|---|---|
| 32 | 1528 | 1425 | 1.07x |
| 64 | 2216 | 2079 | 1.07x |
| 128 | 3219 | 2307 | 1.4x |
| 256 | 5767 | 2515 | 2.3x |
| 512 | 11006 | 2635 | 4.2x |
| 1024 | 21712 | 3075 | 7.1x |
| 2048 | 43270 | 4190 | 10.3x |
| 4096 | 93833 | 6985 | 13.4x |

Below 32 tokens the old kernel is as fast or faster (a 16-token batch touches almost every expert once, so there is
nothing to share), and the registry thresholds say so. At 4096 tokens the new kernel runs at about 29 TFLOPS; the
dequantisation (bf16 tile built by CUDA cores) and the per-pair y buffer round trip are what is left; FP4 block-scaled
MMA (`mma.sync ...e2m1`) was not attempted.

Dense GEMMs, same harness (old `*_mma` -> `*_gemm`, microseconds): NVFP4 M=2048 N=8192 K=2048 2476 -> 1339; M=4096
N=2048 K=4096 1809 -> 1339; M=257 N=512 K=2048 54 -> 31; FP8 W8A8 M=2048 N=8192 2813 -> 1257. Two different
comparisons are behind the accuracy figures, and they must not be mixed up:

* `*_gemm` against the old `*_mma` kernels (dense weights): identical bf16 outputs, rel_l2 0 in all 30 shape/format
  combinations, measured in the standalone C++ harness (not in a test of this repo yet).
* grouped MoE `moe_experts_mma_bf16` against the old `moe_experts_bf16`: not identical, up to 3.5e-4 rel_l2, because
  the summation order differs.

The tests `matmul_nvfp4_gemm_is_bitwise_the_mma_kernel` and `matmul_fp8_gemm_is_bitwise_the_mma_kernel`
(`crates/intarsia-backend-cuda/tests/quantmoe.rs`) assert the first claim (equal output bytes, M in {33, 40, 100, 256})
and `moe_experts_mma_matches_old_kernel` bounds the second at rel_l2 1e-3. They need a GPU and only run in the spark1
acceptance run; nothing has run them yet.

Not done: combining the shared expert and the gate into the experts kernel (needs a new op signature; touches the
shared `ops.rs`/CPU reference), `moe_topk` is already one fused launch (softmax + top-k + renormalise).

## Tests

* Mac: `cargo test -p intarsia-core quant`, `cargo test -p intarsia-backend-cpu --test quant_hf`. The latter checks
  dequantisation, FP8 (both modes), NVFP4 matmul, top-k and a whole 8-expert MoE block against values computed by
  `tools/gen_quantmoe_fixture.py` (torch float8 + e2m1 table + the HF module) from real tensor slices
  (`tests/fixtures/qwen36_quant_slices.safetensors`, 0.5 MB, cut by `tools/extract_quant_slices.py`).
* spark1: `intarsia kernels test --backend cuda --op matmul_fp8` (and `matmul_nvfp4`, `moe_topk`, `moe_experts`,
  `moe_combine`) runs the admission sweep incl. the real 35B-A3B shapes (FP8 8192x2048 ... NVFP4 248320-row head,
  64-expert H=2048 I=512 top-8); `cargo test -p intarsia-backend-cuda --test quantmoe -- --include-ignored --nocapture`
  adds skewed/invalid routing and a microbenchmark.
