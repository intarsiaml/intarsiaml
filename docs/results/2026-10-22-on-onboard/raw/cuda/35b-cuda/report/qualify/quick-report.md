# Qualification quick report: Qwen3.6-35B-A3B-NVFP4 on cuda

_Quick section, generated unix 1792654244, intarsia 0.0.1 build `head=716b4e4 code-tree=79c4400f0040 dirty=no`._

## Headline

**Verdict: PASS** (8 stages: 7 pass, 0 fail, 0 error, 1 skipped)

| | |
|---|---|
| Model | Qwen3.6-35B-A3B-NVFP4 (`/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5`) |
| Backend / device | cuda / NVIDIA GB10 (sm_121) |
| Recipe | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| Decode | 128 tok: **84.1** tok/s (86% of floor); 1024 tok: **83.6** tok/s (85% of floor); 4096 tok: **81.1** tok/s (83% of floor) |
| Prefill | 128 tok: **1282** tok/s; 1024 tok: **4640** tok/s; 4096 tok: **5383** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 96.83% KL 7.65e-3; decode top-1 98.44% KL 1.19e-2) |
| Accuracy (format reference, same weights) | skipped: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| Quick gate | **PASS**: perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 5m49s |
| Reference | http://100.120.88.104:8000/v1 |

### Failures

None.

### Skipped

- `acc-format`: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE)

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 1s | NVIDIA GB10 (sm_121), 54 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 70s | 36 ops resolved, exactness fast, loaded in 69.9s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 44s | 54 kernels, 1056 cases in 43.5s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 74s | perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| 7 | `acc-short` short accuracy against a golden | quick | PASS | 69s | prefill top-1 96.83% KL 7.65e-3; decode top-1 98.44% KL 1.19e-2 |
| 8 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 1m31s | decode 84.1 tok/s at 128 tokens (86% of the byte floor) |

### Notes

- `doctor`: warning disk_free:.: 271.5 GB free of 4032 GB (7% free)
- `doctor`: warning disk_free:/home/nvidia: 271.5 GB free of 4032 GB (7% free)

## Speed

**quick bench** (floor: 2.25 GB/token at 221 GB/s; gb/token: estimated from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 221 GB/s, the GB10 measured figure)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 131 | 0.101 | 1282 | 84.10 | 4.21 | 98.0 | 85.8 |
| 1015 | 0.219 | 4640 | 83.63 | 0.17 | 98.0 | 85.3 |
| 4185 | 0.782 | 5383 | 81.14 | 0.21 | 98.0 | 82.8 |

## Accuracy

**acc-short** against `/home/nvidia/onboard-runs/35b-cuda/golden-fresh/Qwen3.6-35B-A3B-NVFP4-vllm-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 441 | 96.83 | 95.19 | 7.65e-3 | 7.65e-3 | 2.53e-1 | 1.84e-3 | 0 | 32.7 | pass |
| decode | 128 | 98.44 | 95.16 | 1.19e-2 | 1.28e-2 | 4.13e-1 | 4.33e-3 | 0 | 78.1 | pass |

**Quick gate**: perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) (PASS)
- plan: 36 ops resolved, exactness fast, loaded in 69.9s (PASS)

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
| matmul_fp8 | matmul_fp8_gemm | deterministic | yes | 0 |
| matmul_fp8 | matmul_fp8_gemm_reg | deterministic | yes | 0 |
| matmul_fp8 | matmul_fp8_gemv | deterministic | yes | 20 |
| matmul_fp8 | matmul_fp8_mma | deterministic | yes | 0 |
| matmul_nvfp4 | matmul_nvfp4_gemm | deterministic | yes | 0 |
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
| matmul_nvfp4_glu | matmul_nvfp4_glu_gemv | deterministic | yes | 40 |
| gated_rmsnorm | gated_rmsnorm_bf16 | deterministic | yes | 0 |
| rmsnorm_add_matmul_pair | rmsnorm_add_matmul_pair_bf16_gemv | deterministic | yes | 69 |
| gated_rmsnorm_matmul_fp8 | gated_rmsnorm_matmul_fp8_gemv | deterministic | yes | 30 |

- kernels test: 54 kernels, 1056 cases in 43.5s (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | head=716b4e4 code-tree=79c4400f0040 dirty=no | spark-4e77 | 1792653894 | `/home/nvidia/intarsia-decfuse/on/target/acc/release/intarsia doctor --backend cuda --device 0 --out /home/nvidia/onboard-runs/35b-cuda/report/qualify/doctor.json` |
| `recipe` | head=716b4e4 code-tree=79c4400f0040 dirty=no | spark-4e77 | 1792653894 | `recipe::analyse(/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5)` |
| `plan` | head=716b4e4 code-tree=79c4400f0040 dirty=no | spark-4e77 | 1792653895 | `/home/nvidia/intarsia-decfuse/on/target/acc/release/intarsia plan show --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 38 --admit --json` |
| `kernels` | head=716b4e4 code-tree=79c4400f0040 dirty=no | spark-4e77 | 1792653965 | `/home/nvidia/intarsia-decfuse/on/target/acc/release/intarsia kernels test --backend cuda --device 0 --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5` |
| `gate` | head=716b4e4 code-tree=79c4400f0040 dirty=no | spark-4e77 | 1792654009 | `/home/nvidia/intarsia-decfuse/on/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 38 --quick --max-seq 2048` |
| `acc-format` | head=716b4e4 code-tree=79c4400f0040 dirty=no | spark-4e77 | 1792654244 | `-` |
| `acc-short` | head=716b4e4 code-tree=79c4400f0040 dirty=no | spark-4e77 | 1792654083 | `/home/nvidia/intarsia-decfuse/on/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 38 --golden /home/nvidia/onboard-runs/35b-cuda/golden-fres` |
| `bench` | head=716b4e4 code-tree=79c4400f0040 dirty=no | spark-4e77 | 1792654152 | `/home/nvidia/intarsia-decfuse/on/target/acc/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3.6-35B-A3B-NVFP4 --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-Qwen3.6-35B-A3B-NVFP4-cuda-bench --out-dir /home/nvidia/onboard-runs/35b-cuda/` |

Model fingerprint: `fnv1a64:25b7234fc8ebc6ee`.


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/kernels.out` (kernels)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `bench/qualify-Qwen3.6-35B-A3B-NVFP4-cuda-bench.json` (bench)
- `logs/bench-bench.out` (bench)
- `logs/bench-server.err` (bench)
- `state.json` (stage records; `--resume` reads it)
