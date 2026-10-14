### 4k: per-kernel trace, 128 decode steps, 616 launches per step

| kernel | n x k | calls/token | us/call | ms/token | share | GB/s |
|---|---|---:|---:|---:|---:|---:|
| matmul_bf16_gemv | 12288x4096 | 72 | 182.6 | 13.145 | 37.5% | 551 |
| matmul_bf16_gemv | 4096x12288 | 36 | 182.7 | 6.577 | 18.7% | 551 |
| attn_decode_bf16 |  | 36 | 130.0 | 4.679 | 13.3% |  |
| matmul_bf16_gemv | 4096x4096 | 72 | 64.0 | 4.605 | 13.1% | 525 |
| logits_bf16_gemv | 151936x4096 | 1 | 2117.5 | 2.117 | 6.0% | 588 |
| matmul_bf16_gemv | 1024x4096 | 72 | 18.8 | 1.355 | 3.9% | 446 |
| rmsnorm_add_bf16 |  | 72 | 17.2 | 1.238 | 3.5% |  |
| rope_bf16 |  | 72 | 7.0 | 0.507 | 1.4% |  |
| qk_norm_bf16 |  | 72 | 4.3 | 0.306 | 0.9% |  |
| kv_append_bf16 |  | 72 | 3.7 | 0.263 | 0.7% |  |
| argmax_f32 |  | 1 | 148.2 | 0.148 | 0.4% |  |
| silu_glu_bf16 |  | 36 | 3.6 | 0.129 | 0.4% |  |
| rmsnorm_bf16 |  | 1 | 15.1 | 0.015 | 0.0% |  |
| embedding_bf16 |  | 1 | 3.4 | 0.003 | 0.0% |  |

Sum of kernel time 35.09 ms/token; first-start to last-end span (includes the host submit latency between serialized launches) 125.99 ms/token.

| class | ms/token | share of kernel time |
|---|---:|---:|
| GEMV | 27.80 | 79.2% |
| attention | 4.94 | 14.1% |
| elementwise/row | 2.35 | 6.7% |

GEMV weight bytes 15.136 GB/token in 27.80 ms/token = 544 GB/s effective.

### 4k: command-buffer timeline (stream as normal), last 128 records, dispatches per cb 616..616

- GPU busy per step (cb GPUEnd-GPUStart): 32.40 ms (min 31.93, max 34.10)
- GPU idle between consecutive step cbs: 0.542 ms avg (min 0.446, max 0.722)
- step period (start to start): 32.94 ms = 30.4 tok/s; GPU busy fraction 98.4%
