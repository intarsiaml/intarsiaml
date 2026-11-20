| ms per verify pass | sampled 4k | greedy 4k | sampled 64k | greedy 64k |
|---|---|---|---|---|
| passes in the 256-token window | 132.5 | 122.4 | 133.7 | 116.9 |
| tokens per pass in that window | 1.93 | 2.09 | 1.91 | 2.19 |
| wall (first to last decode kernel) | 21.5 | 18.3 | 39.9 | 35.3 |
| GPU busy (union of kernels) | 18.2 | 17.5 | 36.2 | 34.2 |
| GPU idle (host gaps) | 3.2 | 0.8 | 3.7 | 1.0 |
| attention kernels (verify rows + draft head) | 1.3 | 1.3 | 18.6 | 18.2 |
| dense matmul / GEMV (fp8, nvfp4, bf16, lm head) | 9.6 | 13.2 | 10.1 | 13.0 |
| MoE experts + routing | 5.9 | 6.2 | 6.0 | 6.1 |
| GatedDeltaNet | 0.7 | 0.7 | 0.8 | 0.7 |
| norm / rope / elementwise / copies | 0.6 | 0.6 | 0.6 | 0.6 |
| sampling kernels (argmax, top-k) | 0.0 | 0.0 | 0.0 | 0.0 |
| other | 0.1 | 0.1 | 0.1 | 0.1 |
| D2H copies per pass | 4.03 | 1.26 | 6.56 | 4.82 |
| D2H MB per pass | 4.0 | 1.3 | 6.9 | 6.0 |
