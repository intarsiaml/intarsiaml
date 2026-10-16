### 16k-new chunk 256: kernel-mode trace, 28212 launches

| kernel | M | n x k | calls | us/call | total ms | share | TFLOPS |
|---|---|---|---:|---:|---:|---:|---:|
| matmul_bf16 | M=256 | 12288x4096 | 4608 | 4454 | 20524.9 | 41.3% | 5.79 |
| matmul_bf16 | M=256 | 4096x12288 | 2304 | 5004 | 11530.3 | 23.2% | 5.15 |
| matmul_bf16 | M=256 | 4096x4096 | 4608 | 1599 | 7368.0 | 14.8% | 5.37 |
| attn_prefill_bf16_flash |  |  | 2340 | 2932 | 6860.6 | 13.8% | 11.81 |
| matmul_bf16 | M=256 | 1024x4096 | 4608 | 570 | 2628.0 | 5.3% | 3.77 |
| matmul_bf16 | M=194 | 12288x4096 | 72 | 3474 | 250.1 | 0.5% | 5.62 |
| qk_norm_rope_kv_bf16 |  |  | 2340 | 77 | 180.2 | 0.4% |  |
| matmul_bf16 | M=194 | 4096x12288 | 36 | 4142 | 149.1 | 0.3% | 4.72 |
| matmul_bf16 | M=194 | 4096x4096 | 72 | 1315 | 94.7 | 0.2% | 4.95 |
| rmsnorm_add_bf16 |  |  | 4680 | 14 | 64.5 | 0.1% |  |
| silu_glu_bf16 |  |  | 2340 | 24 | 56.8 | 0.1% |  |
| matmul_bf16 | M=194 | 1024x4096 | 72 | 416 | 29.9 | 0.1% | 3.92 |
| logits_bf16_gemv | M=1 | 151936x4096 | 1 | 2737 | 2.7 | 0.0% | 0.45 |
| embedding_bf16 |  |  | 65 | 13 | 0.9 | 0.0% |  |
| rmsnorm_bf16 |  |  | 65 | 8 | 0.5 | 0.0% |  |
| argmax_f32 |  |  | 1 | 16 | 0.0 | 0.0% |  |

Sum of kernel time 49741.3 ms; first-start to last-end span 54717.8 ms (kernel mode serialises launches, so the span includes host submit latency).

| class | ms | share |
|---|---:|---:|
| matmul | 42577.8 | 85.6% |
| attention | 6860.6 | 13.8% |
| elementwise/row | 302.9 | 0.6% |

Matmul: 230.30 TFLOP in 42577.8 ms = 5.41 TFLOPS aggregate.
Attention: 81.06 TFLOP (causal) in 6860.6 ms = 11.81 TFLOPS.

attention ms per chunk (36 layers): c0:2, c1:5, c2:8, c3:12, c4:15, c5:18, c6:21, c7:24, c8:28, c9:31, c10:35, c11:41, c12:43, c13:46, c14:46, c15:49, c16:52, c17:55, c18:58, c19:61, c20:65, c21:71, c22:78, c23:80, c24:80, c25:80, c26:83, c27:88, c28:91, c29:94, c30:97, c31:102, c32:112, c33:114, c34:113, c35:111, c36:115, c37:118, c38:122, c39:126, c40:130, c41:140, c42:146, c43:144, c44:142, c45:142, c46:146, c47:149, c48:153, c49:156, c50:161, c51:175, c52:178, c53:179, c54:171, c55:174, c56:177, c57:182, c58:186, c59:200, c60:218, c61:217, c62:218, c63:207, c64:181
