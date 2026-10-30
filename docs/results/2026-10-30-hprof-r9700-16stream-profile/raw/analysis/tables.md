
TP=1 (card 0)

| group | B=1 ms (share) | B=8 ms (share) | B=16 ms (share) |
|---|---:|---:|---:|
| MoE routed experts | - | 6.57 (31 %) | 11.57 (29 %) |
| MoE fused block (B=1: router+top-k+routed+shared) | 1.92 (30 %) | - | - |
| dense fp8 projections | 2.47 (38 %) | 5.48 (25 %) | 11.18 (28 %) |
| GatedDeltaNet (conv, delta rule, gated norm) | 0.24 (4 %) | 1.97 (9 %) | 3.63 (9 %) |
| shared expert | - | 1.09 (5 %) | 2.05 (5 %) |
| lm_head (NVFP4 GEMV) | 0.46 (7 %) | 1.27 (6 %) | 2.53 (6 %) |
| attention (excl. projections) | 0.24 (4 %) | 0.95 (4 %) | 1.79 (4 %) |
| small bf16 GEMVs (router, GDN b/a, shared-gate logit) | 0.00 (0 %) | 0.76 (4 %) | 3.30 (8 %) |
| MoE routing glue (top-k, group, reduce) | - | 0.38 (2 %) | 0.41 (1 %) |
| norms / residual add | 0.29 (4 %) | 0.22 (1 %) | 0.22 (1 %) |
| sampling (argmax) | 0.05 (1 %) | 0.04 (0 %) | 0.04 (0 %) |
| other (embedding, copies) | 0.01 (0 %) | 0.01 (0 %) | 0.01 (0 %) |
| idle between launches inside the step | 0.85 (13 %) | 2.73 (13 %) | 3.18 (8 %) |
| **step wall (profiled)** | **6.52** | **21.49** | **39.91** |
| kernel launches / step | 268 | 869 | 1001 |

TP=2 (rank 0 of 2)

| group | B=1 ms (share) | B=8 ms (share) | B=16 ms (share) |
|---|---:|---:|---:|
| MoE routed experts | - | 4.59 (28 %) | 7.75 (26 %) |
| MoE fused block (B=1: router+top-k+routed+shared) | 1.50 (25 %) | - | - |
| dense fp8 projections | 1.34 (22 %) | 3.12 (19 %) | 6.13 (21 %) |
| GatedDeltaNet (conv, delta rule, gated norm) | 0.21 (4 %) | 1.09 (7 %) | 2.02 (7 %) |
| shared expert | - | 1.00 (6 %) | 1.85 (6 %) |
| lm_head (NVFP4 GEMV) | 0.23 (4 %) | 0.64 (4 %) | 1.29 (4 %) |
| attention (excl. projections) | 0.22 (4 %) | 0.61 (4 %) | 0.97 (3 %) |
| small bf16 GEMVs (router, GDN b/a, shared-gate logit) | 0.00 (0 %) | 0.75 (5 %) | 3.17 (11 %) |
| all-reduce (TP=2) | 0.84 (14 %) | 1.07 (6 %) | 2.17 (7 %) |
| MoE routing glue (top-k, group, reduce) | - | 0.37 (2 %) | 0.38 (1 %) |
| norms / residual add | 0.28 (5 %) | 0.21 (1 %) | 0.21 (1 %) |
| sampling (argmax) | 0.05 (1 %) | 0.05 (0 %) | 0.04 (0 %) |
| other (embedding, copies) | 0.04 (1 %) | 0.21 (1 %) | 0.40 (1 %) |
| idle between launches inside the step | 1.36 (22 %) | 2.90 (17 %) | 3.35 (11 %) |
| **step wall (profiled)** | **6.10** | **16.59** | **29.70** |
| kernel launches / step | 389 | 926 | 1075 |

8->16 TP1

| group | B=8 ms | B=16 ms | delta ms | share of the +18.4 ms | ratio |
|---|---:|---:|---:|---:|---:|
| dense fp8 projections | 5.48 | 11.18 | +5.70 | 31 % | 2.04x |
| MoE routed experts | 6.57 | 11.57 | +5.00 | 27 % | 1.76x |
| small bf16 GEMVs (router, GDN b/a, shared-gate logit) | 0.76 | 3.30 | +2.53 | 14 % | 4.31x |
| GatedDeltaNet (conv, delta rule, gated norm) | 1.97 | 3.63 | +1.66 | 9 % | 1.84x |
| lm_head (NVFP4 GEMV) | 1.27 | 2.53 | +1.26 | 7 % | 1.99x |
| shared expert | 1.09 | 2.05 | +0.96 | 5 % | 1.89x |
| attention (excl. projections) | 0.95 | 1.79 | +0.84 | 5 % | 1.89x |
| idle between launches | 2.73 | 3.18 | +0.45 | 2 % | 1.17x |
| MoE routing glue (top-k, group, reduce) | 0.38 | 0.41 | +0.03 | 0 % | 1.07x |
| norms / residual add | 0.22 | 0.22 | +0.01 | 0 % | 1.04x |
| other (embedding, copies) | 0.01 | 0.01 | +0.00 | 0 % | 1.02x |
| sampling (argmax) | 0.04 | 0.04 | -0.00 | -0 % | 0.99x |
| **step wall** | 21.49 | 39.91 | +18.41 | 100 % | 1.86x |

sweep

| B | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 13 | 14 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| profiled step wall ms | 6.52 | 10.04 | 13.97 | 17.46 | 21.49 | 27.28 | 29.76 | 32.85 | 34.75 | 39.91 |
| launches | 268 | 588 | 828 | 829 | 869 | 870 | 870 | 1001 | 1001 | 1001 |
| ms per extra row (vs previous col) | - | 3.52 | 1.96 | 1.75 | 2.02 | 2.89 | 1.24 | 3.09 | 1.90 | 2.58 |
| dense fp8 projections ms | 2.47 | 2.71 | 3.23 | 4.02 | 5.48 | 6.97 | 7.87 | 8.93 | 9.66 | 11.18 |
| MoE routed experts ms | 0.00 | 0.00 | 3.80 | 5.42 | 6.57 | 7.83 | 9.01 | 9.69 | 10.11 | 11.57 |
| small bf16 GEMVs ms | 0.00 | 0.39 | 0.51 | 0.64 | 0.76 | 2.80 | 1.91 | 2.24 | 2.53 | 3.30 |
| lm_head ms | 0.46 | 0.47 | 0.64 | 1.08 | 1.27 | 1.70 | 1.91 | 2.24 | 2.35 | 2.53 |
| GatedDeltaNet ms | 0.24 | 0.59 | 1.07 | 1.30 | 1.97 | 2.19 | 2.60 | 2.78 | 2.98 | 3.63 |
| shared expert ms | 0.00 | 0.00 | 0.83 | 0.96 | 1.09 | 1.29 | 1.74 | 1.83 | 1.90 | 2.05 |

roofline

| kernel (calls/step) | us/call | ms/step | bytes/call | GB/s | % of 640 | TFLOPS |
|---|---:|---:|---:|---:|---:|---:|
| routed gate_up (70 experts) (40) | 173.6 | 6.94 | 82.58 MB | 476 | 74 % | 3.09 |
| fp8 GEMV N=8192 K=2048 M=8 (80) | 63.5 | 5.08 | 16.78 MB | 264 | 41 % | 4.23 |
| routed down (70 experts) (40) | 115.6 | 4.63 | 41.29 MB | 357 | 56 % | 2.32 |
| GDN delta rule (state r+w, fp32) (30) | 102.3 | 3.07 | 67.11 MB | 656 | 102 % | 0.33 |
| lm_head NVFP4 N=248320 M=4 (4) | 632.2 | 2.53 | 286.06 MB | 452 | 71 % | 6.44 |
| fp8 GEMV N=2048 K=4096 M=5 (80) | 28.5 | 2.28 | 8.39 MB | 294 | 46 % | 2.94 |
| fp8 GEMV N=4096 K=2048 M=8 (60) | 36.3 | 2.18 | 8.39 MB | 231 | 36 % | 3.70 |
| GDN b/a bf16 GEMV (60) | 23.0 | 1.38 | 0.26 MB | 11 | 2 % | 0.18 |
| fp8 GEMV N=2048 K=4096 M=6 (40) | 32.5 | 1.30 | 8.39 MB | 258 | 40 % | 3.09 |
| attn splitkv (ctx 1200) (10) | 126.1 | 1.26 | 39.32 MB | 312 | 49 % | 2.50 |
| shared gate+up (40) | 28.9 | 1.16 | 1.18 MB | 41 | 6 % | 2.32 |
| router bf16 GEMV (40) | 24.2 | 0.97 | 1.05 MB | 43 | 7 % | 0.69 |
| shared-gate bf16 GEMV (40) | 23.7 | 0.95 | 0.00 MB | 0 | 0 % | 0.00 |
| shared down (40) | 22.3 | 0.89 | 0.59 MB | 26 | 4 % | 1.51 |
| fp8 GEMV N=512 K=2048 M=8 (40) | 8.6 | 0.34 | 1.05 MB | 122 | 19 % | 1.95 |
