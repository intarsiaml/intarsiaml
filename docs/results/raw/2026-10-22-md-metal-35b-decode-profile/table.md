launch correction c = 0.82 us (ce1 attributed sum 8462 - cb1 GPU busy 7837) / 764; roofline 560 GB/s
| phase | launches / token | us / token | % of GPU busy | weight+KV MB / token | achieved GB/s | at-roofline us | excess us |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp8 GEMV gdn qkv 8192x2048 + z 4096x2048 | 60 | 1743 | 22.2 % | 755.0 | 433 | 1348 | 395 |
| MoE experts gate/up (8 of 256, nvfp4) | 40 | 926 | 11.8 % | 377.5 | 408 | 674 | 252 |
| fp8 GEMV gdn out 2048x4096 | 30 | 855 | 10.9 % | 251.7 | 294 | 449 | 405 |
| MoE experts down (8 of 256, nvfp4) | 40 | 655 | 8.4 % | 188.7 | 288 | 337 | 318 |
| shared expert (gate, up, silu_glu, down; nvfp4) | 160 | 621 | 7.9 % | 70.8 | 114 | 126 | 495 |
| lm_head nvfp4 248320x2048 | 1 | 487 | 6.2 % | 286.1 | 587 | 511 | -23 |
| fp8 GEMV attn q+gate 8192x2048 + k, v 512x2048 | 30 | 450 | 5.7 % | 188.7 | 420 | 337 | 113 |
| rmsnorm / rmsnorm_add (all layers) | 81 | 352 | 4.5 % | 0.0 | - | 0 | 352 |
| MoE topk | 40 | 281 | 3.6 % | 0.0 | - | 0 | 281 |
| gdn delta_rule (f32 state in + out) | 30 | 276 | 3.5 % | 125.8 | 456 | 225 | 51 |
| fp8 GEMV attn o 2048x4096 | 10 | 271 | 3.5 % | 83.9 | 310 | 150 | 121 |
| attn_decode split-KV (1k ctx) | 10 | 259 | 3.3 % | 20.5 | 79 | 37 | 223 |
| MoE router+shared-gate pair (bf16 257x2048) | 40 | 162 | 2.1 % | 42.1 | 260 | 75 | 87 |
| gdn gated rmsnorm + silu_glu | 60 | 111 | 1.4 % | 0.0 | - | 0 | 111 |
| gdn conv | 30 | 103 | 1.3 % | 0.0 | - | 0 | 103 |
| attn qk_norm_rope_kv + gate_split + sigmoid_mul | 30 | 98 | 1.3 % | 0.0 | - | 0 | 98 |
| gdn a/b pair bf16 | 30 | 86 | 1.1 % | 7.9 | 92 | 14 | 72 |
| MoE combine | 40 | 80 | 1.0 % | 0.0 | - | 0 | 80 |
| embedding + argmax | 2 | 20 | 0.3 % | 0.0 | - | 0 | 20 |
| total | 764 | 7837 | 100 % | 2399 | 306 | 4283 | 3554 |
