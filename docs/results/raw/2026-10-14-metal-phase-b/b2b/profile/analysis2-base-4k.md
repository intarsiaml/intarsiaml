### base @ 4k: per-kernel trace, 128 decode steps, 616 launches per step

| kernel | n x k | calls/token | us/call | ms/token | share | GB/s |
|---|---|---:|---:|---:|---:|---:|
| matmul_bf16_gemv | 12288x4096 | 72 | 174.7 | 12.582 | 38.2% | 576 |
| matmul_bf16_gemv | 4096x12288 | 36 | 177.2 | 6.380 | 19.4% | 568 |
| matmul_bf16_gemv | 4096x4096 | 72 | 61.2 | 4.404 | 13.4% | 549 |
| attn_decode_bf16 |  | 36 | 110.7 | 3.985 | 12.1% |  |
| logits_bf16_gemv | 151936x4096 | 1 | 2099.8 | 2.100 | 6.4% | 593 |
| matmul_bf16_gemv | 1024x4096 | 72 | 17.6 | 1.266 | 3.8% | 477 |
| rmsnorm_add_bf16 |  | 72 | 14.7 | 1.060 | 3.2% |  |
| rope_bf16 |  | 72 | 6.1 | 0.440 | 1.3% |  |
| qk_norm_bf16 |  | 72 | 3.5 | 0.251 | 0.8% |  |
| kv_append_bf16 |  | 72 | 3.0 | 0.219 | 0.7% |  |
| argmax_f32 |  | 1 | 133.8 | 0.134 | 0.4% |  |
| silu_glu_bf16 |  | 36 | 2.6 | 0.094 | 0.3% |  |
| rmsnorm_bf16 |  | 1 | 13.0 | 0.013 | 0.0% |  |
| embedding_bf16 |  | 1 | 3.1 | 0.003 | 0.0% |  |

Sum of kernel time 32.93 ms/token (isolated command buffers).

| class | launches/token | ms/token | share of kernel time |
|---|---:|---:|---:|
| GEMV | 253 | 26.73 | 81.2% |
| attention (decode + kv_append) | 108 | 4.20 | 12.8% |
| elementwise/row | 255 | 2.00 | 6.1% |

### base @ 4k: command-buffer timeline (stream as normal): last 128 command buffers = 128 steps (78848 dispatches)

- step period (first start to last end / steps): 32.83 ms = 30.5 tok/s
- GPU busy 32.30 ms/step (98.4%), idle 0.528 ms/step
- gaps between consecutive command buffers: mean 532.0 us, max 2742 us, n=127
