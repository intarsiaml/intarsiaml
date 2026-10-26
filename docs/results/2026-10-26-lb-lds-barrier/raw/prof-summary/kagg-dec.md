### dec: kernel time base 1119.60 ms, dev 1118.79 ms (-0.07 %)

| kernel | launches | base total ms | dev total ms | base mean us | dev mean us | delta |
|---|---:|---:|---:|---:|---:|---:|
| `void (anonymous namespace)::moe_block_kernel<4, 8>((anonymous namespace)::BlockArgs)` | 7680 / 7680 | 369.772 | 370.722 | 48.15 | 48.27 | +0.3 % |
| `void (anonymous namespace)::fp8_gemv_pair_kernel<1, ik_bf16, 2, 8>(ik_bf16 const*, unsigne` | 5760 / 5760 | 254.747 | 254.060 | 44.23 | 44.11 | -0.3 % |
| `void (anonymous namespace)::fp8_gemv_gated_kernel<2, ik_bf16>(ik_bf16 const*, ik_bf16 cons` | 5760 / 5760 | 112.353 | 112.919 | 19.51 | 19.60 | +0.5 % |
| `void (anonymous namespace)::nvfp4_gemv_v2_kernel<1, float, false>(ik_bf16 const*, unsigned` | 192 / 192 | 88.177 | 88.053 | 459.26 | 458.61 | -0.1 % |
| `void (anonymous namespace)::fp8_gemv_kernel<1, 2, ik_bf16, 2>(ik_bf16 const*, unsigned cha` | 1920 / 1920 | 60.621 | 60.571 | 31.57 | 31.55 | -0.1 % |
| `void (anonymous namespace)::rmsnorm_add_pair_kernel<1, 8>(ik_bf16 const*, ik_bf16 const*, ` | 13248 / 13248 | 53.587 | 53.287 | 4.04 | 4.02 | -0.6 % |
| `void (anonymous namespace)::gdn_conv_delta_kernel<128, 2, 1, 4>(ik_bf16 const*, ik_bf16 co` | 5790 / 5790 | 46.596 | 46.766 | 8.05 | 8.08 | +0.4 % |
| `void (anonymous namespace)::fp8_gemv_sigmoid_kernel<2>(ik_bf16 const*, ik_bf16 const*, uns` | 1920 / 1920 | 32.828 | 32.817 | 17.10 | 17.09 | -0.0 % |
| `void (anonymous namespace)::splitkv_kernel<8, 32, 2, false>(ik_bf16 const*, ik_bf16 const*` | 1930 / 1930 | 23.768 | 23.014 | 12.32 | 11.92 | -3.2 % |
| `void (anonymous namespace)::fp8_gemv_norm_pair_kernel<1>(ik_bf16 const*, ik_bf16 const*, i` | 1920 / 1920 | 13.190 | 13.138 | 6.87 | 6.84 | -0.4 % |
| `void (anonymous namespace)::gate_qk_norm_rope_kv_kernel<ik_bf16>(ik_bf16 const*, ik_bf16 c` | 1930 / 1930 | 10.710 | 10.747 | 5.55 | 5.57 | +0.4 % |
| `__amd_rocclr_copyBuffer` | 1138 / 1138 | 8.896 | 8.791 | 7.82 | 7.73 | -1.2 % |
| `void (anonymous namespace)::argmax_kernel<float>(float const*, int*, long)` | 192 / 192 | 8.701 | 8.743 | 45.32 | 45.54 | +0.5 % |
| `void dq_gemm_kernel<(anonymous namespace)::Nvfp4Policy, 4, 1, 2, 1, ik_bf16>(DqArgs, ik_bf` | 120 / 120 | 7.153 | 6.987 | 59.60 | 58.22 | -2.3 % |
| `(anonymous namespace)::moe_gate_up_kernel(ik_bf16 const*, unsigned char const*, unsigned c` | 40 / 40 | 6.431 | 6.237 | 160.78 | 155.91 | -3.0 % |
| `void f8_gemm_kernel<4, 1, 2, 1, ik_bf16>(unsigned char const*, unsigned char const*, float` | 130 / 130 | 4.990 | 4.598 | 38.38 | 35.37 | -7.8 % |
| `(anonymous namespace)::moe_down_kernel(ik_bf16 const*, unsigned char const*, unsigned char` | 40 / 40 | 4.598 | 4.487 | 114.96 | 112.17 | -2.4 % |
| `void (anonymous namespace)::combine_kernel<256, 16, false>(float const*, HIP_vector_type<f` | 1930 / 1930 | 4.239 | 4.320 | 2.20 | 2.24 | +1.9 % |
| `void (anonymous namespace)::gemm_wmma_kernel<4, 2, 2, 2, ik_bf16>(ik_bf16 const*, ik_bf16 ` | 140 / 140 | 3.010 | 3.039 | 21.50 | 21.71 | +0.9 % |
| `__amd_rocclr_fillBufferAligned` | 61 / 61 | 2.044 | 2.288 | 33.52 | 37.51 | +11.9 % |
| `void (anonymous namespace)::rmsnorm_kernel<true, true>(ik_bf16 const*, ik_bf16 const*, ik_` | 272 / 272 | 0.667 | 0.666 | 2.45 | 2.45 | -0.1 % |
| `f8_actq_kernel(ik_bf16 const*, unsigned char*, float const*, long)` | 130 / 130 | 0.426 | 0.425 | 3.27 | 3.27 | -0.1 % |
| `void (anonymous namespace)::rmsnorm_kernel<false, true>(ik_bf16 const*, ik_bf16 const*, ik` | 193 / 193 | 0.408 | 0.409 | 2.11 | 2.12 | +0.2 % |
| `void (anonymous namespace)::gemv_pair_kernel<1, 8, ik_bf16>(ik_bf16 const*, ik_bf16 const*` | 192 / 192 | 0.377 | 0.380 | 1.97 | 1.98 | +0.8 % |
| `void (anonymous namespace)::embedding_kernel<true>(int const*, ik_bf16 const*, ik_bf16*, l` | 193 / 193 | 0.337 | 0.339 | 1.75 | 1.76 | +0.6 % |
| `(anonymous namespace)::moe_topk_v2_kernel(ik_bf16 const*, int, int, int, int, float*, int*` | 40 / 40 | 0.211 | 0.210 | 5.26 | 5.25 | -0.3 % |
| `void dq_gemm_kernel<(anonymous namespace)::Fp8PolicyW16, 4, 1, 2, 1, ik_bf16>(DqArgs, ik_b` | 130 / 130 | 0.173 | 0.176 | 1.33 | 1.35 | +1.4 % |
| `(anonymous namespace)::moe_reduce_v8_kernel(ik_bf16 const*, int const*, float const*, ik_b` | 40 / 40 | 0.165 | 0.172 | 4.13 | 4.30 | +4.0 % |
| `(anonymous namespace)::moe_group_kernel(int const*, int, int, int*, int*, int*)` | 40 / 40 | 0.120 | 0.121 | 3.01 | 3.02 | +0.4 % |
| `void (anonymous namespace)::gated_rmsnorm_kernel<true>(ik_bf16 const*, ik_bf16 const*, ik_` | 30 / 30 | 0.099 | 0.098 | 3.28 | 3.25 | -1.0 % |
| `void (anonymous namespace)::silu_glu_kernel<true>(ik_bf16 const*, ik_bf16 const*, ik_bf16*` | 40 / 40 | 0.093 | 0.093 | 2.32 | 2.32 | -0.3 % |
| `void (anonymous namespace)::moe_combine_kernel<ik_bf16>(ik_bf16 const*, ik_bf16 const*, ik` | 40 / 40 | 0.086 | 0.087 | 2.15 | 2.17 | +1.2 % |
| `(anonymous namespace)::sigmoid_mul_kernel(ik_bf16 const*, ik_bf16 const*, ik_bf16*, long)` | 10 / 10 | 0.029 | 0.028 | 2.86 | 2.82 | -1.7 % |
