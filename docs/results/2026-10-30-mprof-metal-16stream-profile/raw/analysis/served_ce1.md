
#### 8-stream burst (ce1 trace)

* wall 13.93 s for 4096 tokens = 294.1 tok/s aggregate (ce1 mode)
* prefill steps: 8 x 238 ms = 1.90 s (13.7 % of wall)
* ramp decode steps (rows 1..7, between the prefills): 15 steps, 0.23 s (1.6 %)
* ramp idle between steps: 13 ms
* steady D8: 504 steps, 11.30 s (81.1 %), 22.41 ms / step (GPU span first->last record), wall per step 23.15 ms = 345.6 tok/s decode-only
* tail (draining streams): 7 steps, 0.11 s (0.8 %)
* ce1 ms / steady step by group: MoE routed experts 8.45, dense fp8 projections 6.04, lm_head + argmax 2.47, GDN layers 1.79, full attention (core) 1.25, shared expert 0.94, router + top-k 0.71, norms / embedding 0.52; total 22.18
* launches per steady step: matmul_fp8_gemv_rows 130.0, matmul_nvfp4_gemv_rows 121.0, rmsnorm_add_bf16 80.0, matmul_pair_bf16_gemv_rows 70.0, silu_glu_bf16 70.0, moe_topk_bf16 40.0, moe_experts_bf16_rows 40.0, moe_experts_bf16_rows#1 40.0, moe_experts_bf16_rows#2 40.0, moe_combine_bf16 40.0, rmsnorm_bf16 31.0, gdn_conv_bf16_rows 30.0, gdn_delta_rule_bf16_rows 30.0, gate_split_bf16 10.0, qk_norm_rope_kv_bf16_rows 10.0, attn_decode_bf16_splitkv_rows 10.0, sigmoid_mul_bf16 10.0, embedding_bf16 1.0, argmax_f32 1.0

#### 16-stream burst (ce1 trace)

* wall 26.08 s for 8192 tokens = 314.1 tok/s aggregate (ce1 mode)
* prefill steps: 16 x 245 ms = 3.92 s (15.0 % of wall)
* ramp decode steps (rows 1..15, between the prefills): 31 steps, 0.65 s (2.5 %)
* ramp idle between steps: 29 ms
* steady D16: 496 steps, 20.58 s (78.9 %), 41.49 ms / step (GPU span first->last record), wall per step 42.47 ms = 376.7 tok/s decode-only
* tail (draining streams): 15 steps, 0.39 s (1.5 %)
* ce1 ms / steady step by group: MoE routed experts 15.68, dense fp8 projections 11.47, lm_head + argmax 4.91, GDN layers 3.21, full attention (core) 2.28, shared expert 2.00, router + top-k 1.05, norms / embedding 0.53; total 41.14
* launches per steady step: matmul_fp8_gemv_rows 260.0, matmul_nvfp4_gemv_rows 242.0, matmul_pair_bf16_gemv_rows 140.0, rmsnorm_add_bf16 80.0, silu_glu_bf16 70.0, moe_topk_bf16 40.0, moe_experts_bf16_rows 40.0, moe_experts_bf16_rows#1 40.0, moe_experts_bf16_rows#2 40.0, moe_combine_bf16 40.0, rmsnorm_bf16 31.0, gdn_conv_bf16_rows 30.0, gdn_delta_rule_bf16_rows 30.0, gate_split_bf16 10.0, qk_norm_rope_kv_bf16_rows 10.0, attn_decode_bf16_splitkv_rows 10.0, sigmoid_mul_bf16 10.0, embedding_bf16 1.0, argmax_f32 1.0
