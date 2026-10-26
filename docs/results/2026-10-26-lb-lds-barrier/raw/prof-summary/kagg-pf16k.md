### pf16k: kernel time base 1708.63 ms, dev 1694.31 ms (-0.84 %)

| kernel | launches | base total ms | dev total ms | base mean us | dev mean us | delta |
|---|---:|---:|---:|---:|---:|---:|
| `void moe_wmma_kernel_r7<true, 64, 45>(ik_bf16 const*, unsigned char const*, unsigned char ` | 160 / 160 | 349.371 | 341.242 | 2183.57 | 2132.76 | -2.3 % |
| `void (anonymous namespace)::gdn_delta_rows_v3_kernel<128>(ik_bf16 const*, ik_bf16 const*, ` | 600 / 600 | 327.116 | 327.829 | 545.19 | 546.38 | +0.2 % |
| `void (anonymous namespace)::flash_hpb_kernel<256, 32, 4, 5>(ik_bf16 const*, ik_bf16 const*` | 40 / 40 | 274.408 | 274.129 | 6860.19 | 6853.23 | -0.1 % |
| `void moe_wmma_kernel_r7<false, 64, 25>(ik_bf16 const*, unsigned char const*, unsigned char` | 320 / 320 | 201.278 | 199.826 | 628.99 | 624.46 | -0.7 % |
| `void f8_gemm_kernel<4, 4, 4, 2, ik_bf16>(unsigned char const*, unsigned char const*, float` | 160 / 160 | 120.226 | 119.112 | 751.42 | 744.45 | -0.9 % |
| `void f8_gemm_kernel<4, 4, 2, 4, ik_bf16>(unsigned char const*, unsigned char const*, float` | 360 / 360 | 105.131 | 103.828 | 292.03 | 288.41 | -1.2 % |
| `void dq_gemm_kernel<(anonymous namespace)::Nvfp4Policy, 4, 2, 2, 4, ik_bf16>(DqArgs, ik_bf` | 480 / 480 | 82.822 | 82.157 | 172.55 | 171.16 | -0.8 % |
| `(anonymous namespace)::moe_reduce_v8_kernel(ik_bf16 const*, int const*, float const*, ik_b` | 160 / 160 | 40.420 | 40.444 | 252.63 | 252.77 | +0.1 % |
| `void (anonymous namespace)::gdn_conv_kernel<false, false>(ik_bf16 const*, ik_bf16 const*, ` | 120 / 120 | 35.742 | 35.907 | 297.85 | 299.23 | +0.5 % |
| `void (anonymous namespace)::gemm_wmma_kernel<4, 2, 2, 4, ik_bf16>(ik_bf16 const*, ik_bf16 ` | 560 / 560 | 28.594 | 26.286 | 51.06 | 46.94 | -8.1 % |
| `f8_actq_kernel(ik_bf16 const*, unsigned char*, float const*, long)` | 520 / 520 | 26.484 | 26.505 | 50.93 | 50.97 | +0.1 % |
| `void (anonymous namespace)::rmsnorm_kernel<true, true>(ik_bf16 const*, ik_bf16 const*, ik_` | 321 / 321 | 26.227 | 25.888 | 81.71 | 80.65 | -1.3 % |
| `void (anonymous namespace)::gated_rmsnorm_kernel<true>(ik_bf16 const*, ik_bf16 const*, ik_` | 120 / 120 | 22.785 | 22.738 | 189.87 | 189.48 | -0.2 % |
| `void (anonymous namespace)::moe_combine_kernel<ik_bf16>(ik_bf16 const*, ik_bf16 const*, ik` | 160 / 160 | 20.433 | 20.498 | 127.70 | 128.11 | +0.3 % |
| `void (anonymous namespace)::gate_qk_norm_rope_kv_kernel<ik_bf16>(ik_bf16 const*, ik_bf16 c` | 50 / 50 | 12.801 | 12.721 | 256.02 | 254.41 | -0.6 % |
| `__amd_rocclr_copyBuffer` | 658 / 658 | 8.521 | 8.147 | 12.95 | 12.38 | -4.4 % |
| `(anonymous namespace)::sigmoid_mul_kernel(ik_bf16 const*, ik_bf16 const*, ik_bf16*, long)` | 40 / 40 | 6.647 | 7.006 | 166.18 | 175.14 | +5.4 % |
| `(anonymous namespace)::moe_topk_v2_kernel(ik_bf16 const*, int, int, int, int, float*, int*` | 160 / 160 | 3.388 | 3.393 | 21.18 | 21.21 | +0.2 % |
| `(anonymous namespace)::moe_group_kernel(int const*, int, int, int*, int*, int*)` | 160 / 160 | 3.263 | 3.272 | 20.39 | 20.45 | +0.3 % |
| `__amd_rocclr_fillBufferAligned` | 61 / 61 | 2.145 | 2.370 | 35.16 | 38.85 | +10.5 % |
| `void (anonymous namespace)::moe_block_kernel<4, 8>((anonymous namespace)::BlockArgs)` | 40 / 40 | 2.026 | 2.040 | 50.66 | 50.99 | +0.7 % |
| `void (anonymous namespace)::silu_glu_kernel<true>(ik_bf16 const*, ik_bf16 const*, ik_bf16*` | 160 / 160 | 1.730 | 1.738 | 10.81 | 10.86 | +0.4 % |
| `void dq_gemm_kernel<(anonymous namespace)::Fp8PolicyW16, 4, 2, 2, 4, ik_bf16>(DqArgs, ik_b` | 520 / 520 | 1.620 | 1.634 | 3.12 | 3.14 | +0.9 % |
| `void (anonymous namespace)::fp8_gemv_pair_kernel<1, ik_bf16, 2, 8>(ik_bf16 const*, unsigne` | 30 / 30 | 1.321 | 1.316 | 44.03 | 43.87 | -0.4 % |
| `void (anonymous namespace)::splitkv_kernel<8, 32, 2, false>(ik_bf16 const*, ik_bf16 const*` | 10 / 10 | 0.842 | 0.819 | 84.21 | 81.86 | -2.8 % |
| `void (anonymous namespace)::fp8_gemv_gated_kernel<2, ik_bf16>(ik_bf16 const*, ik_bf16 cons` | 30 / 30 | 0.592 | 0.596 | 19.75 | 19.86 | +0.6 % |
| `void (anonymous namespace)::nvfp4_gemv_v2_kernel<1, float, false>(ik_bf16 const*, unsigned` | 1 / 1 | 0.457 | 0.457 | 456.67 | 457.15 | +0.1 % |
| `void (anonymous namespace)::rmsnorm_kernel<false, true>(ik_bf16 const*, ik_bf16 const*, ik` | 5 / 5 | 0.411 | 0.432 | 82.14 | 86.41 | +5.2 % |
| `void (anonymous namespace)::fp8_gemv_kernel<1, 2, ik_bf16, 2>(ik_bf16 const*, unsigned cha` | 10 / 10 | 0.318 | 0.317 | 31.75 | 31.71 | -0.1 % |
| `void (anonymous namespace)::rmsnorm_add_pair_kernel<1, 8>(ik_bf16 const*, ik_bf16 const*, ` | 69 / 69 | 0.308 | 0.301 | 4.46 | 4.36 | -2.2 % |
| `void (anonymous namespace)::embedding_kernel<true>(int const*, ik_bf16 const*, ik_bf16*, l` | 5 / 5 | 0.286 | 0.451 | 57.26 | 90.12 | +57.4 % |
| `(anonymous namespace)::gdn_conv_state_kernel(ik_bf16 const*, float*, long, int, int)` | 120 / 120 | 0.281 | 0.280 | 2.34 | 2.33 | -0.5 % |
| `void (anonymous namespace)::gdn_conv_delta_kernel<128, 2, 1, 4>(ik_bf16 const*, ik_bf16 co` | 30 / 30 | 0.262 | 0.262 | 8.73 | 8.74 | +0.1 % |
| `void (anonymous namespace)::fp8_gemv_sigmoid_kernel<2>(ik_bf16 const*, ik_bf16 const*, uns` | 10 / 10 | 0.171 | 0.173 | 17.11 | 17.26 | +0.8 % |
| `void (anonymous namespace)::combine_kernel<256, 16, false>(float const*, HIP_vector_type<f` | 10 / 10 | 0.079 | 0.079 | 7.89 | 7.92 | +0.5 % |
| `void (anonymous namespace)::fp8_gemv_norm_pair_kernel<1>(ik_bf16 const*, ik_bf16 const*, i` | 10 / 10 | 0.074 | 0.073 | 7.36 | 7.31 | -0.7 % |
| `void (anonymous namespace)::argmax_kernel<float>(float const*, int*, long)` | 1 / 1 | 0.049 | 0.048 | 48.58 | 48.08 | -1.0 % |
| `void (anonymous namespace)::gemv_pair_kernel<1, 8, ik_bf16>(ik_bf16 const*, ik_bf16 const*` | 1 / 1 | 0.002 | 0.002 | 2.48 | 2.34 | -5.7 % |
