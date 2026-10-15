### 4k-new chunk 256: kernel-mode trace, 6946 launches

| kernel | M | n x k | calls | us/call | total ms | share | TFLOPS |
|---|---|---|---:|---:|---:|---:|---:|
| matmul_bf16 | M=256 | 12288x4096 | 1080 | 4354 | 4702.3 | 43.9% | 5.92 |
| matmul_bf16 | M=256 | 4096x12288 | 540 | 4899 | 2645.6 | 24.7% | 5.26 |
| matmul_bf16 | M=256 | 4096x4096 | 1080 | 1566 | 1691.6 | 15.8% | 5.48 |
| matmul_bf16 | M=256 | 1024x4096 | 1080 | 558 | 602.7 | 5.6% | 3.85 |
| attn_prefill_bf16_flash |  |  | 576 | 706 | 406.4 | 3.8% | 11.98 |
| matmul_bf16 | M=222 | 12288x4096 | 72 | 3862 | 278.1 | 2.6% | 5.79 |
| matmul_bf16 | M=222 | 4096x12288 | 36 | 4823 | 173.6 | 1.6% | 4.63 |
| matmul_bf16 | M=222 | 4096x4096 | 72 | 1510 | 108.7 | 1.0% | 4.93 |
| qk_norm_rope_kv_bf16 |  |  | 576 | 74 | 42.5 | 0.4% |  |
| matmul_bf16 | M=222 | 1024x4096 | 72 | 544 | 39.2 | 0.4% | 3.42 |
| rmsnorm_add_bf16 |  |  | 1152 | 14 | 15.8 | 0.1% |  |
| silu_glu_bf16 |  |  | 576 | 23 | 13.4 | 0.1% |  |
| logits_bf16_gemv | M=1 | 151936x4096 | 1 | 2151 | 2.2 | 0.0% | 0.58 |
| embedding_bf16 |  |  | 16 | 12 | 0.2 | 0.0% |  |
| rmsnorm_bf16 |  |  | 16 | 9 | 0.1 | 0.0% |  |
| argmax_f32 |  |  | 1 | 15 | 0.0 | 0.0% |  |

Sum of kernel time 10722.3 ms; first-start to last-end span 12085.6 ms (kernel mode serialises launches, so the span includes host submit latency).

| class | ms | share |
|---|---:|---:|
| matmul | 10244.0 | 95.5% |
| attention | 406.4 | 3.8% |
| elementwise/row | 72.0 | 0.7% |

Matmul: 56.43 TFLOP in 10244.0 ms = 5.51 TFLOPS aggregate.
Attention: 4.87 TFLOP (causal) in 406.4 ms = 11.98 TFLOPS.

attention ms per chunk (36 layers): c0:2, c1:5, c2:8, c3:12, c4:15, c5:18, c6:21, c7:24, c8:27, c9:31, c10:34, c11:37, c12:40, c13:43, c14:47, c15:43
