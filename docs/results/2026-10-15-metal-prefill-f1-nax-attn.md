# Metal prefill F1: NAX (Metal 4) flash prefill attention (2026-10-15)

**Status: in progress.** Kernel done and unit-tested; engine gates and end-to-end speed not run yet (filled in below as they finish).

## Outcome so far

`attn_prefill_bf16_flash_nax` (`kernels/metal/attention_flash_nax.metal`, launcher `src/flash_nax.rs`) runs the prefill attention of Qwen3-8B on the M5 neural accelerators (`mpp::tensor_ops::matmul2d` on cooperative tensors, Metal 4). Kernel microbenchmark (`examples/flash_bench`, GPU time from the command-buffer trace, 36 layers, 32/8 heads, head_dim 128, FLOPs `4*D*H*(T*pos0 + T*(T+1)/2)`):

| prompt | chunk | P2 flash (`attn_prefill_bf16_flash`) | NAX | factor |
|---|---:|---:|---:|---:|
| 4k (4096 tokens, summed over chunks) | 256 | 444 ms, 11.2 TFLOPS | 137 ms, 36.0 TFLOPS | 3.2x |
| 4k | 1024 | 395 ms, 12.5 TFLOPS | 135 ms, 36.7 TFLOPS | 2.9x |
| 16k (16384 tokens) | 256 | 7035 ms, 11.3 TFLOPS | 2325 ms, 34.1 TFLOPS | 3.0x |
| 16k | 1024 | 6436 ms, 12.3 TFLOPS | 2437 ms, 32.5 TFLOPS | 2.6x |

A single 256-row chunk at 3840 cached positions runs 41 TFLOPS (403 us per layer against 1353 us for P2).

## How it got there (three attempts at the K / V feed)

The ablations below are the evidence; the experimental kernel is kept in `docs/results/raw/2026-10-15-metal-prefill-f1/experiments/`.

1. K / V fragments by 16-byte lane loads straight from device memory (MLX's structure, `steel_attention_nax.h`): 12.6 TFLOPS, same as P2. Ablation at 3840 / chunk 256: softmax and mask cost ~3%; without any K / V memory traffic the same kernel runs 48.8 TFLOPS (matmul alone 52), with loads from a 16 KB hot window 31.9. The loads were the whole problem.
2. Tiles staged through threadgroup memory (P2's scheme, coalesced 16-byte global loads, register prefetch, fragment reads from padded threadgroup memory): 14.7 TFLOPS with 8 simdgroups. The fragment reads from threadgroup memory cost as much as the device loads.
3. The primitive's own `cooperative_tensor::load` from a device tensor slice (the path the NAX GEMM takes to 55-65 TFLOPS): 41 TFLOPS. Full key tiles use it; the last, partial tile (and so any rows past the visible keys, which must read as zero) uses bounds-checked lane loads, because the tensor slice load is not bounds-checked.

(Remaining sections: design, gates, speed, merge-ready verdict: written when the gates finish.)
