### new @ 4k: per-kernel trace, 128 decode steps, 544 launches per step

| kernel | n x k | calls/token | us/call | ms/token | share | GB/s |
|---|---|---:|---:|---:|---:|---:|
| matmul_pair_bf16_gemv | 24576x4096 | 36 | 344.3 | 12.395 | 38.7% | 585 |
| matmul_bf16_gemv | 4096x12288 | 36 | 177.0 | 6.373 | 19.9% | 569 |
| matmul_bf16_gemv | 4096x4096 | 72 | 60.8 | 4.378 | 13.7% | 552 |
| attn_decode_bf16 |  | 36 | 110.3 | 3.972 | 12.4% |  |
| logits_bf16_gemv | 151936x4096 | 1 | 2096.7 | 2.097 | 6.5% | 594 |
| matmul_pair_bf16_gemv | 2048x4096 | 36 | 32.3 | 1.161 | 3.6% | 520 |
| rmsnorm_add_bf16 |  | 72 | 9.5 | 0.686 | 2.1% |  |
| rope_bf16 |  | 72 | 6.1 | 0.438 | 1.4% |  |
| qk_norm_bf16 |  | 72 | 3.2 | 0.233 | 0.7% |  |
| kv_append_bf16 |  | 72 | 2.9 | 0.210 | 0.7% |  |
| silu_glu_bf16 |  | 36 | 2.6 | 0.095 | 0.3% |  |
| argmax_f32 |  | 1 | 14.8 | 0.015 | 0.0% |  |
| rmsnorm_bf16 |  | 1 | 5.2 | 0.005 | 0.0% |  |
| embedding_bf16 |  | 1 | 3.1 | 0.003 | 0.0% |  |

Sum of kernel time 32.06 ms/token (isolated command buffers).

| class | launches/token | ms/token | share of kernel time |
|---|---:|---:|---:|
| GEMV | 181 | 26.40 | 82.4% |
| attention (decode + kv_append) | 108 | 4.18 | 13.0% |
| elementwise/row | 255 | 1.48 | 4.6% |

### new @ 4k: command-buffer timeline (stream as normal): last 640 command buffers = 128 steps (69632 dispatches)

- step period (first start to last end / steps): 32.01 ms = 31.2 tok/s
- GPU busy 31.69 ms/step (99.0%), idle 0.313 ms/step
- gaps between consecutive command buffers: mean 62.7 us, max 419 us, n=639
