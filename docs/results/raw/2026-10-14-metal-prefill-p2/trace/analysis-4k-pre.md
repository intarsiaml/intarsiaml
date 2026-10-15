### 4k-pre chunk 256: kernel-mode trace, 6946 launches

| kernel | M | n x k | calls | us/call | total ms | share | TFLOPS |
|---|---|---|---:|---:|---:|---:|---:|
| attn_prefill_bf16 |  |  | 576 | 13374 | 7703.6 | 42.9% | 0.63 |
| matmul_bf16 | M=256 | 12288x4096 | 1080 | 4332 | 4678.9 | 26.1% | 5.95 |
| matmul_bf16 | M=256 | 4096x12288 | 540 | 4858 | 2623.2 | 14.6% | 5.30 |
| matmul_bf16 | M=256 | 4096x4096 | 1080 | 1550 | 1673.7 | 9.3% | 5.54 |
| matmul_bf16 | M=256 | 1024x4096 | 1080 | 556 | 600.6 | 3.3% | 3.86 |
| matmul_bf16 | M=222 | 12288x4096 | 72 | 3821 | 275.1 | 1.5% | 5.85 |
| matmul_bf16 | M=222 | 4096x12288 | 36 | 4804 | 172.9 | 1.0% | 4.65 |
| matmul_bf16 | M=222 | 4096x4096 | 72 | 1566 | 112.7 | 0.6% | 4.76 |
| qk_norm_rope_kv_bf16 |  |  | 576 | 74 | 42.9 | 0.2% |  |
| matmul_bf16 | M=222 | 1024x4096 | 72 | 585 | 42.1 | 0.2% | 3.18 |
| rmsnorm_add_bf16 |  |  | 1152 | 13 | 15.3 | 0.1% |  |
| silu_glu_bf16 |  |  | 576 | 23 | 13.2 | 0.1% |  |
| logits_bf16_gemv | M=1 | 151936x4096 | 1 | 2569 | 2.6 | 0.0% | 0.48 |
| embedding_bf16 |  |  | 16 | 24 | 0.4 | 0.0% |  |
| rmsnorm_bf16 |  |  | 16 | 9 | 0.1 | 0.0% |  |
| argmax_f32 |  |  | 1 | 14 | 0.0 | 0.0% |  |

Sum of kernel time 17957.5 ms; first-start to last-end span 20400.4 ms (kernel mode serialises launches, so the span includes host submit latency).

| class | ms | share |
|---|---:|---:|
| matmul | 10181.9 | 56.7% |
| attention | 7703.6 | 42.9% |
| elementwise/row | 71.9 | 0.4% |

Matmul: 56.43 TFLOP in 10181.9 ms = 5.54 TFLOPS aggregate.
Attention: 4.87 TFLOP (causal) in 7703.6 ms = 0.63 TFLOPS.

attention ms per chunk (36 layers): c0:25, c1:75, c2:125, c3:175, c4:226, c5:277, c6:327, c7:400, c8:514, c9:615, c10:683, c11:746, c12:808, c13:878, c14:944, c15:886
