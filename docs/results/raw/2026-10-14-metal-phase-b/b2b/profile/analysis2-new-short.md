### new @ short: per-kernel trace, 128 decode steps, 544 launches per step

| kernel | n x k | calls/token | us/call | ms/token | share | GB/s |
|---|---|---:|---:|---:|---:|---:|
| matmul_pair_bf16_gemv | 24576x4096 | 36 | 344.0 | 12.383 | 43.2% | 585 |
| matmul_bf16_gemv | 4096x12288 | 36 | 176.7 | 6.360 | 22.2% | 570 |
| matmul_bf16_gemv | 4096x4096 | 72 | 60.8 | 4.376 | 15.3% | 552 |
| logits_bf16_gemv | 151936x4096 | 1 | 2102.0 | 2.102 | 7.3% | 592 |
| matmul_pair_bf16_gemv | 2048x4096 | 36 | 32.2 | 1.158 | 4.0% | 522 |
| rmsnorm_add_bf16 |  | 72 | 9.4 | 0.680 | 2.4% |  |
| attn_decode_bf16 |  | 36 | 17.7 | 0.638 | 2.2% |  |
| rope_bf16 |  | 72 | 6.2 | 0.443 | 1.5% |  |
| qk_norm_bf16 |  | 72 | 3.2 | 0.232 | 0.8% |  |
| kv_append_bf16 |  | 72 | 2.9 | 0.205 | 0.7% |  |
| silu_glu_bf16 |  | 36 | 2.6 | 0.095 | 0.3% |  |
| argmax_f32 |  | 1 | 14.2 | 0.014 | 0.0% |  |
| rmsnorm_bf16 |  | 1 | 5.2 | 0.005 | 0.0% |  |
| embedding_bf16 |  | 1 | 3.3 | 0.003 | 0.0% |  |

Sum of kernel time 28.69 ms/token (isolated command buffers).

| class | launches/token | ms/token | share of kernel time |
|---|---:|---:|---:|
| GEMV | 181 | 26.38 | 91.9% |
| elementwise/row | 255 | 1.47 | 5.1% |
| attention (decode + kv_append) | 108 | 0.84 | 2.9% |

### new @ short: command-buffer timeline (stream as normal): last 640 command buffers = 128 steps (69632 dispatches)

- step period (first start to last end / steps): 29.37 ms = 34.1 tok/s
- GPU busy 29.06 ms/step (99.0%), idle 0.307 ms/step
- gaps between consecutive command buffers: mean 61.5 us, max 410 us, n=639
