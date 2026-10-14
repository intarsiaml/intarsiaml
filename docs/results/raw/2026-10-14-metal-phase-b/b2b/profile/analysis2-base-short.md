### base @ short: per-kernel trace, 128 decode steps, 616 launches per step

| kernel | n x k | calls/token | us/call | ms/token | share | GB/s |
|---|---|---:|---:|---:|---:|---:|
| matmul_bf16_gemv | 12288x4096 | 72 | 178.6 | 12.860 | 42.3% | 564 |
| matmul_bf16_gemv | 4096x12288 | 36 | 179.6 | 6.465 | 21.3% | 561 |
| matmul_bf16_gemv | 4096x4096 | 72 | 62.4 | 4.491 | 14.8% | 538 |
| logits_bf16_gemv | 151936x4096 | 1 | 2100.1 | 2.100 | 6.9% | 593 |
| matmul_bf16_gemv | 1024x4096 | 72 | 18.1 | 1.305 | 4.3% | 463 |
| rmsnorm_add_bf16 |  | 72 | 16.1 | 1.160 | 3.8% |  |
| attn_decode_bf16 |  | 36 | 19.9 | 0.718 | 2.4% |  |
| rope_bf16 |  | 72 | 6.7 | 0.485 | 1.6% |  |
| qk_norm_bf16 |  | 72 | 3.9 | 0.279 | 0.9% |  |
| kv_append_bf16 |  | 72 | 3.4 | 0.244 | 0.8% |  |
| argmax_f32 |  | 1 | 142.7 | 0.143 | 0.5% |  |
| silu_glu_bf16 |  | 36 | 3.0 | 0.110 | 0.4% |  |
| rmsnorm_bf16 |  | 1 | 13.6 | 0.014 | 0.0% |  |
| embedding_bf16 |  | 1 | 3.1 | 0.003 | 0.0% |  |

Sum of kernel time 30.38 ms/token (isolated command buffers).

| class | launches/token | ms/token | share of kernel time |
|---|---:|---:|---:|
| GEMV | 253 | 27.22 | 89.6% |
| elementwise/row | 255 | 2.19 | 7.2% |
| attention (decode + kv_append) | 108 | 0.96 | 3.2% |

### base @ short: command-buffer timeline (stream as normal): last 128 command buffers = 128 steps (78848 dispatches)

- step period (first start to last end / steps): 30.02 ms = 33.3 tok/s
- GPU busy 29.50 ms/step (98.3%), idle 0.517 ms/step
- gaps between consecutive command buffers: mean 520.6 us, max 652 us, n=127
