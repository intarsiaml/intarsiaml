### short: per-kernel trace, 128 decode steps, 616 launches per step

| kernel | n x k | calls/token | us/call | ms/token | share | GB/s |
|---|---|---:|---:|---:|---:|---:|
| matmul_bf16_gemv | 12288x4096 | 72 | 174.2 | 12.544 | 42.5% | 578 |
| matmul_bf16_gemv | 4096x12288 | 36 | 176.6 | 6.357 | 21.5% | 570 |
| matmul_bf16_gemv | 4096x4096 | 72 | 60.7 | 4.373 | 14.8% | 552 |
| logits_bf16_gemv | 151936x4096 | 1 | 2092.9 | 2.093 | 7.1% | 595 |
| matmul_bf16_gemv | 1024x4096 | 72 | 17.4 | 1.252 | 4.2% | 483 |
| rmsnorm_add_bf16 |  | 72 | 14.9 | 1.072 | 3.6% |  |
| attn_decode_bf16 |  | 36 | 18.2 | 0.655 | 2.2% |  |
| rope_bf16 |  | 72 | 6.3 | 0.455 | 1.5% |  |
| qk_norm_bf16 |  | 72 | 3.8 | 0.275 | 0.9% |  |
| kv_append_bf16 |  | 72 | 3.0 | 0.216 | 0.7% |  |
| argmax_f32 |  | 1 | 137.4 | 0.137 | 0.5% |  |
| silu_glu_bf16 |  | 36 | 2.8 | 0.102 | 0.3% |  |
| rmsnorm_bf16 |  | 1 | 15.4 | 0.015 | 0.1% |  |
| embedding_bf16 |  | 1 | 3.0 | 0.003 | 0.0% |  |

Sum of kernel time 29.55 ms/token; first-start to last-end span (includes the host submit latency between serialized launches) 120.14 ms/token.

| class | ms/token | share of kernel time |
|---|---:|---:|
| GEMV | 26.62 | 90.1% |
| elementwise/row | 2.06 | 7.0% |
| attention | 0.87 | 2.9% |

GEMV weight bytes 15.136 GB/token in 26.62 ms/token = 569 GB/s effective.

### short: command-buffer timeline (stream as normal), last 128 records, dispatches per cb 616..616

- GPU busy per step (cb GPUEnd-GPUStart): 29.32 ms (min 28.73, max 31.88)
- GPU idle between consecutive step cbs: 0.515 ms avg (min 0.428, max 0.776)
- step period (start to start): 29.84 ms = 33.5 tok/s; GPU busy fraction 98.3%
