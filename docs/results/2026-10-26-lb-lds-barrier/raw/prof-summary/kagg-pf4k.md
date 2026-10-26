### pf4k: kernel time base 388.54 ms, dev 384.88 ms (-0.94 %)

| kernel | launches | base total ms | dev total ms | base mean us | dev mean us | delta |
|---|---:|---:|---:|---:|---:|---:|
| `void moe_wmma_kernel_r7<true, 64, 45>(ik_bf16 const*, unsigned char const*, unsigned char ` | 40 / 40 | 89.465 | 87.447 | 2236.62 | 2186.17 | -2.3 % |
| `void (anonymous namespace)::gdn_delta_rows_v3_kernel<128>(ik_bf16 const*, ik_bf16 const*, ` | 150 / 150 | 81.146 | 81.675 | 540.97 | 544.50 | +0.7 % |
| `void moe_wmma_kernel_r7<false, 64, 25>(ik_bf16 const*, unsigned char const*, unsigned char` | 80 / 80 | 48.911 | 48.544 | 611.39 | 606.79 | -0.8 % |
| `void f8_gemm_kernel<4, 4, 4, 2, ik_bf16>(unsigned char const*, unsigned char const*, float` | 40 / 40 | 30.057 | 29.678 | 751.43 | 741.96 | -1.3 % |
| `void f8_gemm_kernel<4, 4, 2, 4, ik_bf16>(unsigned char const*, unsigned char const*, float` | 90 / 90 | 26.607 | 26.177 | 295.63 | 290.85 | -1.6 % |
| `void dq_gemm_kernel<(anonymous namespace)::Nvfp4Policy, 4, 2, 2, 4, ik_bf16>(DqArgs, ik_bf` | 120 / 120 | 20.563 | 20.475 | 171.36 | 170.62 | -0.4 % |
| `void (anonymous namespace)::flash_hpb_kernel<256, 32, 4, 5>(ik_bf16 const*, ik_bf16 const*` | 10 / 10 | 17.741 | 17.697 | 1774.12 | 1769.67 | -0.3 % |
| `(anonymous namespace)::moe_reduce_v8_kernel(ik_bf16 const*, int const*, float const*, ik_b` | 40 / 40 | 10.049 | 10.050 | 251.23 | 251.24 | +0.0 % |
| `void (anonymous namespace)::gdn_conv_kernel<false, false>(ik_bf16 const*, ik_bf16 const*, ` | 30 / 30 | 8.917 | 8.947 | 297.24 | 298.23 | +0.3 % |
| `__amd_rocclr_copyBuffer` | 565 / 565 | 7.834 | 7.320 | 13.87 | 12.96 | -6.6 % |
| `void (anonymous namespace)::gemm_wmma_kernel<4, 2, 2, 4, ik_bf16>(ik_bf16 const*, ik_bf16 ` | 140 / 140 | 7.108 | 6.516 | 50.77 | 46.55 | -8.3 % |
| `f8_actq_kernel(ik_bf16 const*, unsigned char*, float const*, long)` | 130 / 130 | 6.611 | 6.638 | 50.85 | 51.06 | +0.4 % |
| `void (anonymous namespace)::rmsnorm_kernel<true, true>(ik_bf16 const*, ik_bf16 const*, ik_` | 81 / 81 | 6.488 | 6.465 | 80.10 | 79.82 | -0.4 % |
| `void (anonymous namespace)::gated_rmsnorm_kernel<true>(ik_bf16 const*, ik_bf16 const*, ik_` | 30 / 30 | 5.698 | 5.741 | 189.94 | 191.37 | +0.8 % |
| `void (anonymous namespace)::moe_combine_kernel<ik_bf16>(ik_bf16 const*, ik_bf16 const*, ik` | 40 / 40 | 5.096 | 5.093 | 127.39 | 127.34 | -0.0 % |
| `void (anonymous namespace)::gate_qk_norm_rope_kv_kernel<ik_bf16>(ik_bf16 const*, ik_bf16 c` | 20 / 20 | 3.218 | 3.213 | 160.90 | 160.65 | -0.2 % |
| `__amd_rocclr_fillBufferAligned` | 61 / 61 | 2.196 | 2.563 | 35.99 | 42.01 | +16.7 % |
| `void (anonymous namespace)::moe_block_kernel<4, 8>((anonymous namespace)::BlockArgs)` | 40 / 40 | 2.062 | 2.065 | 51.55 | 51.63 | +0.2 % |
| `(anonymous namespace)::sigmoid_mul_kernel(ik_bf16 const*, ik_bf16 const*, ik_bf16*, long)` | 10 / 10 | 1.697 | 1.749 | 169.70 | 174.89 | +3.1 % |
| `void (anonymous namespace)::fp8_gemv_pair_kernel<1, ik_bf16, 2, 8>(ik_bf16 const*, unsigne` | 30 / 30 | 1.324 | 1.322 | 44.12 | 44.08 | -0.1 % |
| `(anonymous namespace)::moe_topk_v2_kernel(ik_bf16 const*, int, int, int, int, float*, int*` | 40 / 40 | 0.856 | 0.856 | 21.40 | 21.39 | -0.0 % |
| `(anonymous namespace)::moe_group_kernel(int const*, int, int, int*, int*, int*)` | 40 / 40 | 0.792 | 0.791 | 19.80 | 19.78 | -0.1 % |
| `void (anonymous namespace)::fp8_gemv_gated_kernel<2, ik_bf16>(ik_bf16 const*, ik_bf16 cons` | 30 / 30 | 0.601 | 0.603 | 20.04 | 20.09 | +0.2 % |
| `void (anonymous namespace)::nvfp4_gemv_v2_kernel<1, float, false>(ik_bf16 const*, unsigned` | 1 / 1 | 0.461 | 0.461 | 460.73 | 460.61 | -0.0 % |
| `void (anonymous namespace)::silu_glu_kernel<true>(ik_bf16 const*, ik_bf16 const*, ik_bf16*` | 40 / 40 | 0.433 | 0.433 | 10.83 | 10.83 | +0.1 % |
| `void (anonymous namespace)::embedding_kernel<true>(int const*, ik_bf16 const*, ik_bf16*, l` | 2 / 2 | 0.413 | 0.078 | 206.72 | 38.99 | -81.1 % |
| `void dq_gemm_kernel<(anonymous namespace)::Fp8PolicyW16, 4, 2, 2, 4, ik_bf16>(DqArgs, ik_b` | 130 / 130 | 0.402 | 0.412 | 3.09 | 3.17 | +2.5 % |
| `void (anonymous namespace)::splitkv_kernel<8, 32, 2, false>(ik_bf16 const*, ik_bf16 const*` | 10 / 10 | 0.340 | 0.323 | 34.04 | 32.30 | -5.1 % |
| `void (anonymous namespace)::fp8_gemv_kernel<1, 2, ik_bf16, 2>(ik_bf16 const*, unsigned cha` | 10 / 10 | 0.320 | 0.318 | 31.95 | 31.82 | -0.4 % |
| `void (anonymous namespace)::rmsnorm_add_pair_kernel<1, 8>(ik_bf16 const*, ik_bf16 const*, ` | 69 / 69 | 0.303 | 0.302 | 4.39 | 4.37 | -0.4 % |
| `void (anonymous namespace)::gdn_conv_delta_kernel<128, 2, 1, 4>(ik_bf16 const*, ik_bf16 co` | 30 / 30 | 0.272 | 0.271 | 9.06 | 9.04 | -0.3 % |
| `void (anonymous namespace)::fp8_gemv_sigmoid_kernel<2>(ik_bf16 const*, ik_bf16 const*, uns` | 10 / 10 | 0.171 | 0.171 | 17.10 | 17.11 | +0.1 % |
| `void (anonymous namespace)::rmsnorm_kernel<false, true>(ik_bf16 const*, ik_bf16 const*, ik` | 2 / 2 | 0.110 | 0.216 | 54.83 | 107.77 | +96.6 % |
| `void (anonymous namespace)::combine_kernel<256, 16, false>(float const*, HIP_vector_type<f` | 10 / 10 | 0.079 | 0.080 | 7.92 | 8.00 | +1.0 % |
| `void (anonymous namespace)::fp8_gemv_norm_pair_kernel<1>(ik_bf16 const*, ik_bf16 const*, i` | 10 / 10 | 0.073 | 0.073 | 7.34 | 7.30 | -0.6 % |
| `(anonymous namespace)::gdn_conv_state_kernel(ik_bf16 const*, float*, long, int, int)` | 30 / 30 | 0.071 | 0.070 | 2.36 | 2.34 | -0.8 % |
| `void (anonymous namespace)::argmax_kernel<float>(float const*, int*, long)` | 1 / 1 | 0.047 | 0.048 | 47.42 | 48.34 | +1.9 % |
| `void (anonymous namespace)::gemv_pair_kernel<1, 8, ik_bf16>(ik_bf16 const*, ik_bf16 const*` | 1 / 1 | 0.002 | 0.002 | 2.34 | 2.46 | +5.1 % |
