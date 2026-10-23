# Qualification quick report: Qwen3.6-35B-A3B-NVFP4 on cuda

_Quick section, generated unix 1792779861, intarsia 0.0.1 build `pf-dev-055775da`._

## Headline

**Verdict: FAIL** (9 stages: 3 pass, 2 fail, 0 error, 4 skipped)

| | |
|---|---|
| Model | Qwen3.6-35B-A3B-NVFP4 (`/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5`) |
| Backend / device | cuda / ? (?) |
| Recipe | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| Decode | not measured |
| Prefill | not measured |
| Accuracy (short golden) | **FAIL** (prefill top-1 98.27% KL 6.83e-3; decode top-1 99.38% KL 6.99e-3) |
| Accuracy (format reference, same weights) | skipped: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| Quick gate | **PASS**: perplexity 1.21, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 3m27s |

### Failures

- `acc-short` **FAIL**: baseline gate failed
- `acc-oracle` **FAIL**: the logits moved against the recorded dump /home/nvidia/intarsia-decfuse/pf-golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.cuda.selfgolden.safetensors: mean KL(21-way)                                 8.6497e-3  (limit 1.0000e-4)  FAIL; max position KL(21-way)                         4.1821e-1  (li

### Skipped

- `doctor`: skipped by --skip
- `kernels`: skipped by --skip
- `acc-format`: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE)
- `bench`: skipped by --skip

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | SKIP | 0s | skipped by --skip |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 38s | 36 ops resolved, exactness fast, loaded in 37.7s |
| 4 | `kernels` kernels test on the model's shapes | quick | SKIP | 0s | skipped by --skip |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 39s | perplexity 1.21, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| 7 | `acc-short` short accuracy against a golden | quick | FAIL | 40s | baseline gate failed |
| 8 | `acc-oracle` short prefill accuracy against the f64 CPU oracle (same weights, deterministic reference) | quick | FAIL | 90s | the logits moved against the recorded dump /home/nvidia/intarsia-decfuse/pf-golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.cuda.selfgolden.safetensors: mean KL(21-way)                                 8.6497e-3  (limit 1.0000e-4)  FAIL; max position KL(21-way)                         4.1821e-1  (li |
| 9 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | SKIP | 0s | skipped by --skip |

### Notes

- `acc-short`: gated against the recorded baseline /home/nvidia/intarsia-decfuse/pf-golden/baselines/qwen36-35b-a3b-nvfp4-vllm-top20.cuda.baseline.json (commit unstamped) and its noise envelope

## Speed

No speed numbers.

## Accuracy

**acc-short** against `/home/nvidia/intarsia-decfuse/pf-golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1441 | 98.27 | 91.56 | 6.83e-3 | 6.65e-3 | 1.04e0 | 1.62e-3 | 0 | 4.8 | pass |
| decode | 160 | 99.38 | 93.50 | 6.99e-3 | 7.34e-3 | 1.10e-1 | 2.89e-3 | 0 | 90.0 | pass |

**Quick gate**: perplexity 1.21, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) (PASS)
- plan: 36 ops resolved, exactness fast, loaded in 37.7s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_bf16 | exact | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 1 |
| rmsnorm_add | rmsnorm_add_bf16 | deterministic | yes | 11 |
| qk_norm | qk_norm_bf16 | deterministic | yes | 20 |
| rope | rope_bf16 | deterministic | yes | 20 |
| matmul | matmul_bf16 | fast | yes | 0 |
| matmul | matmul_bf16_gemv | deterministic | yes | 0 |
| kv_append | kv_append_bf16 | exact | yes | 20 |
| attn_prefill | attn_prefill_bf16_flash | deterministic | yes | 0 |
| attn_prefill | attn_prefill_bf16_splitkv | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 10 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 0 |
| argmax | argmax_f32 | exact | yes | 1 |
| matmul_fp8 | matmul_fp8_gemm_lt | deterministic | yes | 0 |
| matmul_fp8 | matmul_fp8_gemv | deterministic | yes | 20 |
| matmul_nvfp4 | matmul_nvfp4_gemm | deterministic | yes | 0 |
| matmul_nvfp4 | matmul_nvfp4_gemm_reg | deterministic | yes | 0 |
| matmul_nvfp4 | matmul_nvfp4_gemv | deterministic | yes | 41 |
| matmul_nvfp4 | matmul_nvfp4_mma | deterministic | yes | 0 |
| moe_topk | moe_topk_bf16 | deterministic | yes | 40 |
| moe_experts | moe_experts_bf16 | deterministic | yes | 0 |
| moe_experts | moe_experts_decode_v2_bf16 | deterministic | yes | 40 |
| moe_experts | moe_experts_mma_bf16 | deterministic | yes | 0 |
| moe_experts | moe_experts_w4a16_bf16 | deterministic | yes | 0 |
| moe_combine | moe_combine_bf16 | deterministic | yes | 40 |
| gate_split | gate_split_bf16 | deterministic | yes | 10 |
| sigmoid_mul | sigmoid_mul_bf16 | deterministic | yes | 10 |
| gdn_conv | gdn_conv_bf16 | deterministic | yes | 30 |
| gdn_delta_rule | gdn_delta_rule_bf16_split | deterministic | yes | 30 |
| matmul_pair | matmul_pair_bf16_gemv | deterministic | yes | 1 |
| matmul_fp8_pair | matmul_fp8_pair_gemv | deterministic | yes | 40 |
| matmul_nvfp4_glu | matmul_nvfp4_glu_gemm | deterministic | yes | 0 |
| matmul_nvfp4_glu | matmul_nvfp4_glu_gemv | deterministic | yes | 40 |
| gated_rmsnorm | gated_rmsnorm_bf16 | deterministic | yes | 0 |
| rmsnorm_add_matmul_pair | rmsnorm_add_matmul_pair_bf16_gemv | deterministic | yes | 69 |
| gated_rmsnorm_matmul_fp8 | gated_rmsnorm_matmul_fp8_gemv | deterministic | yes | 30 |

- kernels test: skipped by --skip (SKIP)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | pf-dev-055775da | spark-fa00 | 0 | `-` |
| `recipe` | pf-dev-055775da | spark-fa00 | 1792779653 | `recipe::analyse(/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5)` |
| `plan` | pf-dev-055775da | spark-fa00 | 1792779654 | `/home/nvidia/intarsia-decfuse/pf-dev/target/acc/release/intarsia plan show --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --admit --json` |
| `kernels` | pf-dev-055775da | spark-fa00 | 0 | `/home/nvidia/intarsia-decfuse/pf-dev/target/acc/release/intarsia plan show --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --admit --json` |
| `gate` | pf-dev-055775da | spark-fa00 | 1792779691 | `/home/nvidia/intarsia-decfuse/pf-dev/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --quick --max-seq 2048` |
| `acc-format` | pf-dev-055775da | spark-fa00 | 1792779731 | `-` |
| `acc-short` | pf-dev-055775da | spark-fa00 | 1792779731 | `/home/nvidia/intarsia-decfuse/pf-dev/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsia-decfuse/pf-golden/qw` |
| `acc-oracle` | pf-dev-055775da | spark-fa00 | 1792779771 | `/home/nvidia/intarsia-decfuse/pf-dev/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --self-golden /home/nvidia/intarsia-decfuse/pf-gold` |
| `bench` | pf-dev-055775da | spark-fa00 | 0 | `/home/nvidia/intarsia-decfuse/pf-dev/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --self-golden /home/nvidia/intarsia-decfuse/pf-gold` |

Model fingerprint: `fnv1a64:25b7234fc8ebc6ee`.


## Artifacts

- `plan.json` (plan)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `logs/acc-oracle-drift.out` (acc-oracle)
- `state.json` (stage records; `--resume` reads it)
