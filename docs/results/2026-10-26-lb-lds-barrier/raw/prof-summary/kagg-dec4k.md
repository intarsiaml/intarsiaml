### dec4k: kernel time base 1202.61 ms, dev 1195.38 ms (-0.60 %)

| kernel | launches | base total ms | dev total ms | base mean us | dev mean us | delta |
|---|---:|---:|---:|---:|---:|---:|
| `void (anonymous namespace)::moe_block_kernel<4, 8>((anonymous namespace)::BlockArgs)` | 5120 / 5120 | 244.437 | 244.519 | 47.74 | 47.76 | +0.0 % |
| `void (anonymous namespace)::fp8_gemv_pair_kernel<1, ik_bf16, 2, 8>(ik_bf16 const*, unsigne` | 3840 / 3840 | 167.809 | 167.410 | 43.70 | 43.60 | -0.2 % |
| `void moe_wmma_kernel_r7<true, 64, 45>(ik_bf16 const*, unsigned char const*, unsigned char ` | 40 / 40 | 90.011 | 87.164 | 2250.28 | 2179.11 | -3.2 % |
| `void (anonymous namespace)::gdn_delta_rows_v3_kernel<128>(ik_bf16 const*, ik_bf16 const*, ` | 180 / 180 | 87.398 | 87.850 | 485.54 | 488.06 | +0.5 % |
| `void (anonymous namespace)::fp8_gemv_gated_kernel<2, ik_bf16>(ik_bf16 const*, ik_bf16 cons` | 3840 / 3840 | 75.067 | 75.122 | 19.55 | 19.56 | +0.1 % |
| `void (anonymous namespace)::nvfp4_gemv_v2_kernel<1, float, false>(ik_bf16 const*, unsigned` | 128 / 128 | 58.592 | 58.565 | 457.75 | 457.54 | -0.0 % |
| `void moe_wmma_kernel_r7<false, 64, 25>(ik_bf16 const*, unsigned char const*, unsigned char` | 80 / 80 | 49.042 | 48.744 | 613.03 | 609.30 | -0.6 % |
| `void (anonymous namespace)::fp8_gemv_kernel<1, 2, ik_bf16, 2>(ik_bf16 const*, unsigned cha` | 1280 / 1280 | 40.382 | 40.277 | 31.55 | 31.47 | -0.3 % |
| `void (anonymous namespace)::splitkv_kernel<8, 32, 2, false>(ik_bf16 const*, ik_bf16 const*` | 1280 / 1280 | 38.941 | 37.983 | 30.42 | 29.67 | -2.5 % |
| `void (anonymous namespace)::rmsnorm_add_pair_kernel<1, 8>(ik_bf16 const*, ik_bf16 const*, ` | 8832 / 8832 | 36.028 | 35.350 | 4.08 | 4.00 | -1.9 % |
| `void (anonymous namespace)::gdn_conv_delta_kernel<128, 2, 1, 4>(ik_bf16 const*, ik_bf16 co` | 3840 / 3840 | 30.888 | 30.762 | 8.04 | 8.01 | -0.4 % |
| `void moe_wmma_kernel<true, 32>(ik_bf16 const*, unsigned char const*, unsigned char const*,` | 40 / 40 | 30.494 | 30.668 | 762.36 | 766.71 | +0.6 % |
| `void f8_gemm_kernel<4, 4, 4, 2, ik_bf16>(unsigned char const*, unsigned char const*, float` | 40 / 40 | 30.114 | 29.743 | 752.85 | 743.58 | -1.2 % |
| `void f8_gemm_kernel<4, 4, 2, 4, ik_bf16>(unsigned char const*, unsigned char const*, float` | 90 / 90 | 26.512 | 26.352 | 294.58 | 292.80 | -0.6 % |
| `void dq_gemm_kernel<(anonymous namespace)::Nvfp4Policy, 4, 2, 2, 4, ik_bf16>(DqArgs, ik_bf` | 240 / 240 | 26.320 | 26.221 | 109.67 | 109.25 | -0.4 % |
| `void (anonymous namespace)::flash_hpb_kernel<256, 32, 4, 5>(ik_bf16 const*, ik_bf16 const*` | 20 / 20 | 22.034 | 22.079 | 1101.72 | 1103.93 | +0.2 % |
| `void (anonymous namespace)::fp8_gemv_sigmoid_kernel<2>(ik_bf16 const*, ik_bf16 const*, uns` | 1280 / 1280 | 21.385 | 21.383 | 16.71 | 16.71 | -0.0 % |
| `void (anonymous namespace)::gemm_wmma_kernel<4, 2, 2, 4, ik_bf16>(ik_bf16 const*, ik_bf16 ` | 280 / 280 | 11.682 | 10.972 | 41.72 | 39.19 | -6.1 % |
| `(anonymous namespace)::moe_reduce_v8_kernel(ik_bf16 const*, int const*, float const*, ik_b` | 80 / 80 | 10.579 | 10.590 | 132.24 | 132.37 | +0.1 % |
| `void (anonymous namespace)::gate_qk_norm_rope_kv_kernel<ik_bf16>(ik_bf16 const*, ik_bf16 c` | 1300 / 1300 | 10.561 | 10.463 | 8.12 | 8.05 | -0.9 % |
| `void moe_wmma_kernel_r7<false, 32, 41>(ik_bf16 const*, unsigned char const*, unsigned char` | 40 / 40 | 10.314 | 9.837 | 257.84 | 245.94 | -4.6 % |
| `void (anonymous namespace)::gdn_conv_kernel<false, false>(ik_bf16 const*, ik_bf16 const*, ` | 60 / 60 | 9.674 | 9.688 | 161.23 | 161.46 | +0.1 % |
| `__amd_rocclr_copyBuffer` | 977 / 977 | 8.823 | 8.772 | 9.03 | 8.98 | -0.6 % |
| `void (anonymous namespace)::fp8_gemv_norm_pair_kernel<1>(ik_bf16 const*, ik_bf16 const*, i` | 1280 / 1280 | 8.819 | 8.713 | 6.89 | 6.81 | -1.2 % |
| `void (anonymous namespace)::combine_kernel<256, 16, false>(float const*, HIP_vector_type<f` | 1280 / 1280 | 8.084 | 8.043 | 6.32 | 6.28 | -0.5 % |
| `void f8_gemm_kernel<4, 2, 2, 4, ik_bf16>(unsigned char const*, unsigned char const*, float` | 130 / 130 | 7.512 | 7.381 | 57.78 | 56.78 | -1.7 % |
| `f8_actq_kernel(ik_bf16 const*, unsigned char*, float const*, long)` | 260 / 260 | 7.380 | 7.385 | 28.38 | 28.41 | +0.1 % |
| `void (anonymous namespace)::rmsnorm_kernel<true, true>(ik_bf16 const*, ik_bf16 const*, ik_` | 288 / 288 | 7.366 | 7.351 | 25.58 | 25.52 | -0.2 % |
| `void (anonymous namespace)::gated_rmsnorm_kernel<true>(ik_bf16 const*, ik_bf16 const*, ik_` | 60 / 60 | 6.212 | 6.205 | 103.54 | 103.41 | -0.1 % |
| `void (anonymous namespace)::argmax_kernel<float>(float const*, int*, long)` | 128 / 128 | 5.789 | 5.800 | 45.22 | 45.31 | +0.2 % |
| `void (anonymous namespace)::moe_combine_kernel<ik_bf16>(ik_bf16 const*, ik_bf16 const*, ik` | 80 / 80 | 5.545 | 5.663 | 69.31 | 70.79 | +2.1 % |
| `__amd_rocclr_fillBufferAligned` | 61 / 61 | 2.203 | 2.277 | 36.12 | 37.33 | +3.4 % |
| `(anonymous namespace)::sigmoid_mul_kernel(ik_bf16 const*, ik_bf16 const*, ik_bf16*, long)` | 20 / 20 | 1.853 | 1.788 | 92.67 | 89.41 | -3.5 % |
| `(anonymous namespace)::moe_topk_v2_kernel(ik_bf16 const*, int, int, int, int, float*, int*` | 80 / 80 | 1.067 | 1.065 | 13.33 | 13.31 | -0.2 % |
| `(anonymous namespace)::moe_group_kernel(int const*, int, int, int*, int*, int*)` | 80 / 80 | 0.942 | 0.949 | 11.77 | 11.87 | +0.8 % |
| `void (anonymous namespace)::embedding_kernel<true>(int const*, ik_bf16 const*, ik_bf16*, l` | 130 / 130 | 0.649 | 0.298 | 4.99 | 2.30 | -54.0 % |
| `void dq_gemm_kernel<(anonymous namespace)::Fp8PolicyW16, 4, 2, 2, 4, ik_bf16>(DqArgs, ik_b` | 260 / 260 | 0.607 | 0.617 | 2.33 | 2.37 | +1.7 % |
| `void (anonymous namespace)::silu_glu_kernel<true>(ik_bf16 const*, ik_bf16 const*, ik_bf16*` | 80 / 80 | 0.573 | 0.571 | 7.16 | 7.14 | -0.3 % |
| `void (anonymous namespace)::rmsnorm_kernel<false, true>(ik_bf16 const*, ik_bf16 const*, ik` | 130 / 130 | 0.534 | 0.380 | 4.11 | 2.92 | -28.8 % |
| `void (anonymous namespace)::gemv_pair_kernel<1, 8, ik_bf16>(ik_bf16 const*, ik_bf16 const*` | 128 / 128 | 0.253 | 0.251 | 1.98 | 1.96 | -1.1 % |
| `(anonymous namespace)::gdn_conv_state_kernel(ik_bf16 const*, float*, long, int, int)` | 60 / 60 | 0.130 | 0.130 | 2.16 | 2.17 | +0.5 % |
