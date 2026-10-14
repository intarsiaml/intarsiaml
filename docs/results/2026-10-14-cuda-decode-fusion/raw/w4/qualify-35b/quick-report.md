# Qualification quick report: Qwen3.6-35B-A3B-NVFP4 on cuda

_Quick section, generated unix 1791938314, intarsia 0.0.1 build `head=8a13590 code-tree=194ebbf659dc dirty=no`._

## Headline

**Verdict: PASS** (7 stages: 7 pass, 0 fail, 0 error, 0 skipped)

| | |
|---|---|
| Model | Qwen3.6-35B-A3B-NVFP4 (`/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5`) |
| Backend / device | cuda / NVIDIA GB10 (sm_121) |
| Recipe | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| Decode | 128 tok: **80.7** tok/s (82% of floor); 1024 tok: **80.0** tok/s (82% of floor); 4096 tok: **77.5** tok/s (79% of floor) |
| Prefill | 128 tok: **1306** tok/s; 1024 tok: **3938** tok/s; 4096 tok: **4719** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 97.92% KL 5.87e-3; decode top-1 98.12% KL 1.54e-2) |
| Quick gate | **PASS**: perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 4m12s |

### Failures

None.

### Skipped

None.

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 1s | NVIDIA GB10 (sm_121), 41 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 52s | 32 ops resolved, exactness fast, loaded in 51.9s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 31s | 41 kernels, 569 cases in 30.6s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 43s | perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-short` short accuracy against a golden | quick | PASS | 52s | prefill top-1 97.92% KL 5.87e-3; decode top-1 98.12% KL 1.54e-2 |
| 7 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 73s | decode 80.7 tok/s at 128 tokens (82% of the byte floor) |

### Notes

- `doctor`: warning disk_free:.: 195.0 GB free of 4032 GB (5% free)
- `doctor`: warning disk_free:/home/nvidia: 195.0 GB free of 4032 GB (5% free)

## Speed

**quick bench** (floor: 2.25 GB/token at 221 GB/s; gb/token: estimated from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 221 GB/s, the GB10 measured figure)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 131 | 0.100 | 1306 | 80.70 | 0.03 | 98.0 | 82.3 |
| 999 | 0.255 | 3938 | 80.01 | 0.11 | 98.0 | 81.6 |
| 4087 | 0.866 | 4719 | 77.47 | 0.15 | 98.0 | 79.0 |

## Accuracy

**acc-short** against `/home/nvidia/intarsiaml-golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1441 | 97.92 | 91.81 | 5.87e-3 | 5.43e-3 | 6.48e-1 | 1.41e-3 | 0 | 5.6 | pass |
| decode | 160 | 98.12 | 92.75 | 1.54e-2 | 1.59e-2 | 1.33e0 | 3.36e-3 | 0 | 88.8 | pass |

**Quick gate**: perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) (PASS)
- plan: 32 ops resolved, exactness fast, loaded in 51.9s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_bf16 | exact | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 1 |
| rmsnorm_add | rmsnorm_add_bf16 | deterministic | yes | 80 |
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
| matmul_fp8 | matmul_fp8_gemv | deterministic | yes | 50 |
| matmul_fp8 | matmul_fp8_mma | deterministic | yes | 0 |
| matmul_nvfp4 | matmul_nvfp4_gemm | deterministic | yes | 0 |
| matmul_nvfp4 | matmul_nvfp4_gemv | deterministic | yes | 41 |
| matmul_nvfp4 | matmul_nvfp4_mma | deterministic | yes | 0 |
| moe_topk | moe_topk_bf16 | deterministic | yes | 40 |
| moe_experts | moe_experts_bf16 | deterministic | yes | 0 |
| moe_experts | moe_experts_decode_v2_bf16 | deterministic | yes | 40 |
| moe_experts | moe_experts_mma_bf16 | deterministic | yes | 0 |
| moe_combine | moe_combine_bf16 | deterministic | yes | 40 |
| gate_split | gate_split_bf16 | deterministic | yes | 10 |
| sigmoid_mul | sigmoid_mul_bf16 | deterministic | yes | 10 |
| gdn_conv | gdn_conv_bf16 | deterministic | yes | 30 |
| gdn_delta_rule | gdn_delta_rule_bf16 | deterministic | yes | 30 |
| matmul_pair | matmul_pair_bf16_gemv | deterministic | yes | 70 |
| matmul_fp8_pair | matmul_fp8_pair_gemv | deterministic | yes | 40 |
| matmul_nvfp4_glu | matmul_nvfp4_glu_gemv | deterministic | yes | 40 |
| gated_rmsnorm | gated_rmsnorm_bf16 | deterministic | yes | 30 |

- kernels test: 41 kernels, 569 cases in 30.6s (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | head=8a13590 code-tree=194ebbf659dc dirty=no | spark-4e77 | 1791938062 | `/home/nvidia/intarsia-decfuse/dev/target/acc/release/intarsia doctor --backend cuda --device 0 --out /home/nvidia/intarsia-decfuse/out/w4/qualify-35b/doctor.json` |
| `recipe` | head=8a13590 code-tree=194ebbf659dc dirty=no | spark-4e77 | 1791938063 | `recipe::analyse(/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5)` |
| `plan` | head=8a13590 code-tree=194ebbf659dc dirty=no | spark-4e77 | 1791938063 | `/home/nvidia/intarsia-decfuse/dev/target/acc/release/intarsia plan show --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --admit --json` |
| `kernels` | head=8a13590 code-tree=194ebbf659dc dirty=no | spark-4e77 | 1791938115 | `/home/nvidia/intarsia-decfuse/dev/target/acc/release/intarsia kernels test --backend cuda --device 0 --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5` |
| `gate` | head=8a13590 code-tree=194ebbf659dc dirty=no | spark-4e77 | 1791938146 | `/home/nvidia/intarsia-decfuse/dev/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --quick --max-seq 2048` |
| `acc-short` | head=8a13590 code-tree=194ebbf659dc dirty=no | spark-4e77 | 1791938189 | `/home/nvidia/intarsia-decfuse/dev/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsiaml-golden/qwen36-35b-a3b` |
| `bench` | head=8a13590 code-tree=194ebbf659dc dirty=no | spark-4e77 | 1791938241 | `/home/nvidia/intarsia-decfuse/dev/target/acc/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3.6-35B-A3B-NVFP4 --sizes 128,1024,4096 --gen 128 --reps 5 --warmup 1 --thinking off --label qualify-Qwen3.6-35B-A3B-NVFP4-cuda-bench --out-dir /home/nvidia/intarsia-decfuse/out/` |

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
