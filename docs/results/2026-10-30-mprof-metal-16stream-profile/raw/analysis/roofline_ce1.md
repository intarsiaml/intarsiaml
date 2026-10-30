
### B=1

| kernel (ce1, in-stream) | ms / step | launches | us / launch | stream MB / step | unique MB / step | GB/s (stream) | GB/s (unique) | % of 580 (unique) | TFLOPS |
|---|---|---|---|---|---|---|---|---|---|
| fp8 8192x2048 (GDN qkv x30, attn q+gate x10) | 1.986 | 40 | 49.6 | 671 | 671 | 338 | 338 | 58% | 0.68 |
| fp8 2048x4096 (GDN out x30, attn o x10) | 1.194 | 40 | 29.9 | 336 | 336 | 281 | 281 | 48% | 0.56 |
| experts gate-up (fused silu*up) | 0.844 | 40 | 21.1 | 378 | 378 | 447 | 447 | 77% | 1.59 |
| experts down | 0.706 | 40 | 17.6 | 189 | 189 | 268 | 268 | 46% | 0.95 |
| lm_head nvfp4 248320x2048 -> f32 | 0.511 | 1 | 511.1 | 286 | 286 | 560 | 560 | 96% | 1.99 |
| rmsnorm_add (2 per layer) | 0.421 | 80 | 5.3 | - | - | - | - | - | - |
| fp8 4096x2048 (GDN z x30) | 0.359 | 30 | 12.0 | 252 | 252 | 702 | 702 | 121% | 1.40 |
| top-k | 0.330 | 40 | 8.3 | - | - | - | - | - | - |
| delta rule (recurrent state) | 0.317 | 30 | 10.6 | 126 | 126 | 397 | 397 | 68% | 0.40 |
| attention decode split-KV | 0.257 | 10 | 25.7 | 23 | 23 | 88 | 88 | 15% | 0.71 |
| nvfp4 gate / up 512x2048 | 0.240 | 80 | 3.0 | 47 | 47 | 196 | 196 | 34% | 0.70 |
| router + shared-gate GEMV (257x2048) | 0.204 | 40 | 5.1 | 42 | 42 | 206 | 206 | 36% | 0.21 |
| nvfp4 down 2048x512 | 0.196 | 40 | 4.9 | 24 | 24 | 121 | 121 | 21% | 0.43 |
| conv1d + state | 0.142 | 30 | 4.7 | - | - | - | - | - | - |
| in_a + in_b bf16 pair GEMV (64x2048) | 0.123 | 30 | 4.1 | 8 | 8 | 64 | 64 | 11% | 0.06 |
| combine (weighted sum + shared) | 0.122 | 40 | 3.0 | - | - | - | - | - | - |
| gated norm: rmsnorm | 0.096 | 30 | 3.2 | - | - | - | - | - | - |
| gated norm: silu_glu | 0.078 | 30 | 2.6 | - | - | - | - | - | - |
| qk_norm_rope_kv | 0.074 | 10 | 7.4 | - | - | - | - | - | - |
| fp8 512x2048 (attn k, v x10 each) | 0.052 | 20 | 2.6 | 21 | 21 | 403 | 403 | 69% | 0.81 |

total 8.33 ms / step

### B=4

| kernel (ce1, in-stream) | ms / step | launches | us / launch | stream MB / step | unique MB / step | GB/s (stream) | GB/s (unique) | % of 580 (unique) | TFLOPS |
|---|---|---|---|---|---|---|---|---|---|
| experts gate-up (fused silu*up) | 2.615 | 40 | 65.4 | 1278 | 1278 | 489 | 489 | 84% | 2.05 |
| fp8 8192x2048 (GDN qkv x30, attn q+gate x10) | 2.102 | 40 | 52.5 | 671 | 671 | 319 | 319 | 55% | 2.55 |
| fp8 2048x4096 (GDN out x30, attn o x10) | 1.641 | 40 | 41.0 | 336 | 336 | 205 | 205 | 35% | 1.64 |
| experts down | 1.580 | 40 | 39.5 | 639 | 639 | 405 | 405 | 70% | 1.70 |
| lm_head nvfp4 248320x2048 -> f32 | 1.125 | 1 | 1125.0 | 286 | 286 | 254 | 254 | 44% | 3.62 |
| fp8 4096x2048 (GDN z x30) | 1.016 | 30 | 33.9 | 252 | 252 | 248 | 248 | 43% | 1.98 |
| delta rule (recurrent state) | 0.631 | 30 | 21.0 | 503 | 503 | 797 | 797 | 137% | 0.80 |
| attention decode split-KV | 0.603 | 10 | 60.3 | 91 | 91 | 150 | 150 | 26% | 1.20 |
| nvfp4 gate / up 512x2048 | 0.463 | 80 | 5.8 | 47 | 47 | 102 | 102 | 18% | 1.45 |
| rmsnorm_add (2 per layer) | 0.447 | 80 | 5.6 | - | - | - | - | - | - |
| top-k | 0.329 | 40 | 8.2 | - | - | - | - | - | - |
| nvfp4 down 2048x512 | 0.312 | 40 | 7.8 | 24 | 24 | 76 | 76 | 13% | 1.08 |
| fp8 512x2048 (attn k, v x10 each) | 0.307 | 20 | 15.3 | 21 | 21 | 68 | 68 | 12% | 0.55 |
| router + shared-gate GEMV (257x2048) | 0.233 | 40 | 5.8 | 42 | 42 | 181 | 181 | 31% | 0.72 |
| expert gather / grouping | 0.209 | 40 | 5.2 | - | - | - | - | - | - |
| in_a + in_b bf16 pair GEMV (64x2048) | 0.141 | 30 | 4.7 | 8 | 8 | 56 | 56 | 10% | 0.22 |
| combine (weighted sum + shared) | 0.122 | 40 | 3.0 | - | - | - | - | - | - |
| conv1d + state | 0.111 | 30 | 3.7 | - | - | - | - | - | - |
| gated norm: rmsnorm | 0.089 | 30 | 3.0 | - | - | - | - | - | - |
| gated norm: silu_glu | 0.087 | 30 | 2.9 | - | - | - | - | - | - |
| qk_norm_rope_kv | 0.077 | 10 | 7.7 | - | - | - | - | - | - |

total 14.30 ms / step

### B=8

| kernel (ce1, in-stream) | ms / step | launches | us / launch | stream MB / step | unique MB / step | GB/s (stream) | GB/s (unique) | % of 580 (unique) | TFLOPS |
|---|---|---|---|---|---|---|---|---|---|
| experts gate-up (fused silu*up) | 5.336 | 40 | 133.4 | 2365 | 2365 | 443 | 443 | 76% | 2.01 |
| experts down | 2.840 | 40 | 71.0 | 1183 | 1183 | 416 | 416 | 72% | 1.89 |
| fp8 8192x2048 (GDN qkv x30, attn q+gate x10) | 2.322 | 40 | 58.0 | 671 | 671 | 289 | 289 | 50% | 4.62 |
| lm_head nvfp4 248320x2048 -> f32 | 2.063 | 1 | 2063.1 | 286 | 286 | 139 | 139 | 24% | 3.94 |
| fp8 2048x4096 (GDN out x30, attn o x10) | 1.870 | 40 | 46.7 | 336 | 336 | 179 | 179 | 31% | 2.87 |
| delta rule (recurrent state) | 1.118 | 30 | 37.3 | 1007 | 1007 | 901 | 901 | 155% | 0.90 |
| fp8 4096x2048 (GDN z x30) | 1.034 | 30 | 34.5 | 252 | 252 | 243 | 243 | 42% | 3.89 |
| attention decode split-KV | 0.929 | 10 | 92.9 | 182 | 182 | 196 | 196 | 34% | 1.57 |
| rmsnorm_add (2 per layer) | 0.470 | 80 | 5.9 | - | - | - | - | - | - |
| nvfp4 gate / up 512x2048 | 0.453 | 80 | 5.7 | 47 | 47 | 104 | 104 | 18% | 2.96 |
| nvfp4 down 2048x512 | 0.404 | 40 | 10.1 | 24 | 24 | 58 | 58 | 10% | 1.66 |
| top-k | 0.339 | 40 | 8.5 | - | - | - | - | - | - |
| router + shared-gate GEMV (257x2048) | 0.297 | 40 | 7.4 | 42 | 42 | 142 | 142 | 24% | 1.13 |
| fp8 512x2048 (attn k, v x10 each) | 0.246 | 20 | 12.3 | 21 | 21 | 85 | 85 | 15% | 1.37 |
| expert gather / grouping | 0.213 | 40 | 5.3 | - | - | - | - | - | - |
| in_a + in_b bf16 pair GEMV (64x2048) | 0.178 | 30 | 5.9 | 8 | 8 | 44 | 44 | 8% | 0.35 |
| conv1d + state | 0.134 | 30 | 4.5 | - | - | - | - | - | - |
| combine (weighted sum + shared) | 0.125 | 40 | 3.1 | - | - | - | - | - | - |
| gated norm: rmsnorm | 0.092 | 30 | 3.1 | - | - | - | - | - | - |
| gated norm: silu_glu | 0.078 | 30 | 2.6 | - | - | - | - | - | - |
| qk_norm_rope_kv | 0.076 | 10 | 7.6 | - | - | - | - | - | - |

total 20.69 ms / step

### B=16

| kernel (ce1, in-stream) | ms / step | launches | us / launch | stream MB / step | unique MB / step | GB/s (stream) | GB/s (unique) | % of 580 (unique) | TFLOPS |
|---|---|---|---|---|---|---|---|---|---|
| experts gate-up (fused silu*up) | 9.074 | 40 | 226.9 | 3738 | 3738 | 412 | 412 | 71% | 2.36 |
| experts down | 5.226 | 40 | 130.6 | 1869 | 1869 | 358 | 358 | 62% | 2.06 |
| fp8 8192x2048 (GDN qkv x30, attn q+gate x10) | 4.489 | 80 | 56.1 | 1342 | 671 | 299 | 150 | 26% | 4.78 |
| lm_head nvfp4 248320x2048 -> f32 | 4.139 | 2 | 2069.3 | 572 | 286 | 138 | 69 | 12% | 3.93 |
| fp8 2048x4096 (GDN out x30, attn o x10) | 3.223 | 80 | 40.3 | 671 | 336 | 208 | 104 | 18% | 3.33 |
| delta rule (recurrent state) | 2.266 | 30 | 75.5 | 2013 | 2013 | 889 | 889 | 153% | 0.89 |
| fp8 4096x2048 (GDN z x30) | 1.954 | 60 | 32.6 | 503 | 252 | 258 | 129 | 22% | 4.12 |
| attention decode split-KV | 1.756 | 10 | 175.6 | 364 | 364 | 207 | 207 | 36% | 1.66 |
| nvfp4 gate / up 512x2048 | 0.931 | 160 | 5.8 | 94 | 47 | 101 | 51 | 9% | 2.88 |
| nvfp4 down 2048x512 | 0.731 | 80 | 9.1 | 47 | 24 | 65 | 32 | 6% | 1.84 |
| router + shared-gate GEMV (257x2048) | 0.600 | 80 | 7.5 | 84 | 42 | 140 | 70 | 12% | 1.12 |
| fp8 512x2048 (attn k, v x10 each) | 0.484 | 40 | 12.1 | 42 | 21 | 87 | 43 | 7% | 1.39 |
| rmsnorm_add (2 per layer) | 0.475 | 80 | 5.9 | - | - | - | - | - | - |
| top-k | 0.344 | 40 | 8.6 | - | - | - | - | - | - |
| in_a + in_b bf16 pair GEMV (64x2048) | 0.338 | 60 | 5.6 | 16 | 8 | 47 | 23 | 4% | 0.37 |
| expert gather / grouping | 0.225 | 40 | 5.6 | - | - | - | - | - | - |
| conv1d + state | 0.186 | 30 | 6.2 | - | - | - | - | - | - |
| combine (weighted sum + shared) | 0.126 | 40 | 3.1 | - | - | - | - | - | - |
| silu_glu | 0.104 | 40 | 2.6 | - | - | - | - | - | - |
| gated norm: rmsnorm | 0.104 | 30 | 3.5 | - | - | - | - | - | - |
| gated norm: silu_glu | 0.088 | 30 | 2.9 | - | - | - | - | - | - |
| qk_norm_rope_kv | 0.080 | 10 | 8.0 | - | - | - | - | - | - |

total 37.03 ms / step
