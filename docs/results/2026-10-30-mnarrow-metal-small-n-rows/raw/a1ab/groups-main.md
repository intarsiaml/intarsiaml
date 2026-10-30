# a1ab/ce1-main-r1.jsonl.gz + a1ab/ce1-main-r2.jsonl.gz + a1ab/ce1-main-r3.jsonl.gz + a1ab/ce1-main-r4.jsonl.gz + a1ab/ce1-main-r5.jsonl.gz mode=ce1: ms per decode step by op group (mean over steps after 4 warm-up)
| group / op | B=8 | B=9 | B=12 | B=16 | share B=16 | launches B=1 | launches B=16 | 8->16 delta ms | x |
|---|---|---|---|---|---|---|---|---|---|
| **MoE routed experts** | 8.27 | 9.35 | 11.65 | 14.46 | 39.6% | | | +6.20 | 1.75 |
| &nbsp;&nbsp;experts gate-up (fused silu*up) | 5.139 | 5.937 | 7.337 | 8.965 | 24.5% | 0 | 40 | +3.827 | 1.74 |
| &nbsp;&nbsp;experts down | 2.802 | 3.077 | 3.973 | 5.157 | 14.1% | 0 | 40 | +2.356 | 1.84 |
| &nbsp;&nbsp;expert gather / grouping | 0.209 | 0.212 | 0.215 | 0.220 | 0.6% | 0 | 40 | +0.012 | 1.06 |
| &nbsp;&nbsp;combine (weighted sum + shared) | 0.121 | 0.120 | 0.120 | 0.122 | 0.3% | 0 | 40 | +0.001 | 1.01 |
| **dense fp8 projections** | 5.43 | 9.93 | 10.01 | 10.04 | 27.5% | | | +4.62 | 1.85 |
| &nbsp;&nbsp;fp8 8192x2048 (GDN qkv x30, attn q+gate x10) | 2.290 | 4.223 | 4.446 | 4.453 | 12.2% | 0 | 80 | +2.163 | 1.94 |
| &nbsp;&nbsp;fp8 2048x4096 (GDN out x30, attn o x10) | 1.871 | 3.252 | 3.165 | 3.182 | 8.7% | 0 | 80 | +1.312 | 1.70 |
| &nbsp;&nbsp;fp8 4096x2048 (GDN z x30) | 1.019 | 1.902 | 1.918 | 1.924 | 5.3% | 0 | 60 | +0.905 | 1.89 |
| &nbsp;&nbsp;fp8 512x2048 (attn k, v x10 each) | 0.248 | 0.557 | 0.481 | 0.485 | 1.3% | 0 | 40 | +0.236 | 1.95 |
| **lm_head + argmax** | 2.07 | 2.51 | 3.21 | 4.13 | 11.3% | | | +2.06 | 2.00 |
| &nbsp;&nbsp;lm_head nvfp4 248320x2048 -> f32 | 2.050 | 2.482 | 3.180 | 4.106 | 11.2% | 0 | 2 | +2.056 | 2.00 |
| &nbsp;&nbsp;argmax | 0.022 | 0.024 | 0.026 | 0.029 | 0.1% | 0 | 1 | +0.007 | 1.31 |
| **GDN layers** | 1.58 | 1.82 | 2.25 | 2.92 | 8.0% | | | +1.34 | 1.85 |
| &nbsp;&nbsp;delta rule (recurrent state) | 1.101 | 1.230 | 1.623 | 2.215 | 6.1% | 0 | 30 | +1.114 | 2.01 |
| &nbsp;&nbsp;in_a + in_b bf16 pair GEMV (64x2048) | 0.178 | 0.274 | 0.295 | 0.336 | 0.9% | 0 | 60 | +0.158 | 1.89 |
| &nbsp;&nbsp;conv1d + state | 0.133 | 0.142 | 0.159 | 0.184 | 0.5% | 0 | 30 | +0.050 | 1.38 |
| &nbsp;&nbsp;gated norm: rmsnorm | 0.089 | 0.090 | 0.094 | 0.100 | 0.3% | 0 | 30 | +0.010 | 1.12 |
| &nbsp;&nbsp;gated norm: silu_glu | 0.077 | 0.079 | 0.081 | 0.087 | 0.2% | 0 | 30 | +0.010 | 1.13 |
| **full attention (core)** | 1.03 | 1.12 | 1.41 | 1.85 | 5.1% | | | +0.82 | 1.80 |
| &nbsp;&nbsp;attention decode split-KV | 0.916 | 1.008 | 1.301 | 1.732 | 4.7% | 0 | 10 | +0.816 | 1.89 |
| &nbsp;&nbsp;qk_norm_rope_kv | 0.075 | 0.074 | 0.076 | 0.079 | 0.2% | 0 | 10 | +0.003 | 1.04 |
| &nbsp;&nbsp;sigmoid_mul | 0.023 | 0.023 | 0.024 | 0.026 | 0.1% | 0 | 10 | +0.003 | 1.14 |
| &nbsp;&nbsp;gate_split | 0.010 | 0.009 | 0.009 | 0.010 | 0.0% | 0 | 10 | -0.001 | 0.91 |
| **shared expert** | 0.80 | 1.33 | 1.50 | 1.74 | 4.8% | | | +0.94 | 2.17 |
| &nbsp;&nbsp;nvfp4 gate / up 512x2048 | 0.423 | 0.663 | 0.788 | 0.922 | 2.5% | 0 | 160 | +0.499 | 2.18 |
| &nbsp;&nbsp;nvfp4 down 2048x512 | 0.378 | 0.565 | 0.612 | 0.715 | 2.0% | 0 | 80 | +0.337 | 1.89 |
| &nbsp;&nbsp;silu_glu | 0.000 | 0.100 | 0.102 | 0.104 | 0.3% | 0 | 40 | +0.103 | 236.83 |
| **router + top-k** | 0.63 | 0.76 | 0.82 | 0.94 | 2.6% | | | +0.30 | 1.48 |
| &nbsp;&nbsp;router + shared-gate GEMV (257x2048) | 0.296 | 0.434 | 0.492 | 0.595 | 1.6% | 0 | 80 | +0.299 | 2.01 |
| &nbsp;&nbsp;top-k | 0.336 | 0.329 | 0.333 | 0.340 | 0.9% | 0 | 40 | +0.005 | 1.01 |
| **norms / embedding** | 0.47 | 0.47 | 0.47 | 0.48 | 1.3% | | | +0.01 | 1.01 |
| &nbsp;&nbsp;rmsnorm_add (2 per layer) | 0.465 | 0.463 | 0.463 | 0.472 | 1.3% | 0 | 80 | +0.007 | 1.02 |
| &nbsp;&nbsp;final rmsnorm | 0.005 | 0.005 | 0.005 | 0.005 | 0.0% | 0 | 1 | -0.000 | 0.97 |
| &nbsp;&nbsp;embedding | 0.003 | 0.003 | 0.003 | 0.003 | 0.0% | 0 | 1 | +0.000 | 1.02 |
| **total (sum of records)** | 20.28 | 27.28 | 31.32 | 36.57 | 100% | | | +16.29 | 1.80 |
