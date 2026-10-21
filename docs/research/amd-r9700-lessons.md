# R9700 (gfx1201) lessons carried into IntarsiaML

> **Update (2026-10-21, docs-accuracy pass, see `docs/decisions/LOG.md`):** C12: "No license file found - do not vendor" (libr4d) is superseded: radiance is Apache-2.0 (ADR-0002, `docs/research/radiance-port-plan.md`), and `kernels/hip/allreduce.hip` is derived from it with `kernels/hip/NOTICE-radiance`.

Source: survey of `/opt/code/amd_r9700_pro_tuning` and `/opt/code/vllm-mxfp4`, 2026-10-12. Figures are as reported in the cited files; verify against them before relying on a single number.

## The bar to beat (Qwen3.8-27B dense, MXFP4 W4A8, fp8 KV, TP=2, DFlash2 7 draft tokens)

| Engine | Single-stream decode | 8/12/16-stream total | Prefill 4k/16k/32k |
|---|---|---|---|
| Native radiance 1.2.4 "C2" (C++/HIP, upstream StillDeadcode) | 207 median, 216 mean tok/s, 20.25 ms/step, MAL 3.76 | 611 / 661 / 708 | 5291 / 5256 / 4845 |
| Native radiance 1.4.0 (production since 2026-10-12) | 226 tok/s health check | - | - |
| vLLM-radiance | 150.3 tok/s, 24.0 ms/step, MAL 3.61 | 540 / 537 / 588 | 5335 / 5238 / 5003 |

MoE (Qwen3.5-35B-A3B MXFP4, one card, vLLM stack): 141 tok/s decode; prefill 14.9k/12.6k/10.6k at 4k/16k/34k. llama.cpp Vulkan Q4_K_XL: 112 tok/s decode (unanimis ed2cb0d0).

Decode floor (TP=2 27B): ~12.1 ms from weight bytes vs 24.0 ms measured on vLLM (`r9700-engine/profile/results.md`). Remaining headroom per `r9700-engine/DESIGN.md`: small-M drafter attention (1.5-2 ms), fusion (1-1.5 ms), GEMV tune (1 ms), all-reduce fusion (1 ms) → ~17-18 ms/step.

## Lessons

1. `GPU_MAX_HW_QUEUES=1`: a stray copy stream on the default 4 HW queues gives a permanent ~20% decode slowdown for the life of the process (`moe-slowstart-2026-10-02/results.tsv`).
2. Floor-first profiling: compute the byte floor from safetensors headers, then attribute the gap. Launch overhead was only ~1%; idle gaps (2.74 ms, ~1090 kernels/step), drafter (~3 ms), GEMV at 81% BW, and all-reduce dominated.
3. Decode all-reduce: one-shot P2P kernel, exact bf16, ~8.6 µs × 128 calls/step ≈ 1.1 ms. Fuse RMSNorm + fp8 quant into it (`_nq`). Lossy 6-bit rotated transport (`wht6`) only helps large prefill messages (+11-12% prefill) and changes greedy text: must be opt-in and accuracy-gated. Check message-size thresholds (a hardcoded 48 MB cap silently fell back to RCCL).
4. Speculative decoding acceptance dominates single-stream speed. Drafter must stay FP8 (MXFP4 RTN dropped MAL 2.5→2.21). MAL varies by workload: tools 6.8-7.1, code 5.8, prose 2.7-3.6. Chat-template preambles can change acceptance by 25%.
5. MoE: never dequantize all experts per pass (3.35 ms/layer). Tile tables must be tuned per arch (aiter default 12.6 ms vs 2.4 ms tuned).
6. Verify-attention for spec decode needs a split-KV kernel for q_len>1 (8930 → 475 µs at 18k ctx).
7. Shape traps: per-rank K divisible by 128 for MXFP4 GEMM; padded drafters must keep GQA ratio; KV bytes/token must match target and drafter. Every kernel needs explicit shape preconditions checked at load time.
8. Derive KV memory from measured free VRAM under load; a wrong pin OOMs at startup.
9. Native C++/HIP host: 11 s cold start vs 160 s for vLLM, equal-or-better step time.
10. One measured change at a time, each with a gate.

## Reusable

- libr4d HIP kernels (pinned `R4D_PIN=b9e42ab`): one-shot 2/3-rank all-reduce, MXFP4×FP8 GEMM, GDN recurrent update, paged attention, int2 lm_head + bf16 rerank. **No license file found** — do not vendor without checking with upstream.
- gfx12 WMMA GEMM prototype (`/opt/code/TensorFold-pr621`, branch `gfx12-wmma`): 60 TFLOP/s vs 170 peak.
- Deploy/A-B harness: `deploy/tatooine/native-staged/`, `native-radiance-2026-10-07/`.
