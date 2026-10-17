# Qualification quick report: Qwen3.6-35B-A3B-NVFP4 on cuda

_Quick section, generated unix 1792261235, intarsia 0.0.1 build `head=ff42756 code-tree=dec5cb3168d0 dirty=no`._

## Headline

**Verdict: PASS** (7 stages: 7 pass, 0 fail, 0 error, 0 skipped)

| | |
|---|---|
| Model | Qwen3.6-35B-A3B-NVFP4 (`/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5`) |
| Backend / device | cuda / NVIDIA GB10 (sm_121) |
| Recipe | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| Decode | 128 tok: **81.1** tok/s (83% of floor); 1024 tok: **80.5** tok/s (82% of floor); 4096 tok: **77.8** tok/s (79% of floor) |
| Prefill | 128 tok: **1285** tok/s; 1024 tok: **3966** tok/s; 4096 tok: **4453** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 97.92% KL 5.87e-3; decode top-1 98.12% KL 1.54e-2) |
| Quick gate | **PASS**: perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 3m50s |

### Failures

None.

### Skipped

None.

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 1s | NVIDIA GB10 (sm_121), 43 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 43s | 34 ops resolved, exactness fast, loaded in 43.4s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 32s | 43 kernels, 595 cases in 31.6s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 40s | perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-short` short accuracy against a golden | quick | PASS | 40s | prefill top-1 97.92% KL 5.87e-3; decode top-1 98.12% KL 1.54e-2 |
| 7 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 73s | decode 81.1 tok/s at 128 tokens (83% of the byte floor) |

### Notes

- `doctor`: warning disk_free:.: 93.4 GB free of 4032 GB (2% free)
- `doctor`: warning disk_free:/home/nvidia: 93.4 GB free of 4032 GB (2% free)

## Speed

**quick bench** (floor: 2.25 GB/token at 221 GB/s; gb/token: estimated from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 221 GB/s, the GB10 measured figure)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 129 | 0.100 | 1285 | 81.12 | 0.08 | 98.0 | 82.8 |
| 1019 | 0.258 | 3966 | 80.49 | 0.11 | 98.0 | 82.1 |
| 4227 | 0.950 | 4453 | 77.82 | 0.13 | 98.0 | 79.4 |

## Accuracy

**acc-short** against `/home/nvidia/intarsiaml-golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1441 | 97.92 | 91.81 | 5.87e-3 | 5.43e-3 | 6.48e-1 | 1.41e-3 | 0 | 5.6 | pass |
| decode | 160 | 98.12 | 92.75 | 1.54e-2 | 1.59e-2 | 1.33e0 | 3.36e-3 | 0 | 88.8 | pass |

**Quick gate**: perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) (PASS)
- plan: 34 ops resolved, exactness fast, loaded in 43.4s (PASS)

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
| matmul_fp8 | matmul_fp8_gemv | deterministic | yes | 20 |
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
| matmul_pair | matmul_pair_bf16_gemv | deterministic | yes | 1 |
| matmul_fp8_pair | matmul_fp8_pair_gemv | deterministic | yes | 40 |
| matmul_nvfp4_glu | matmul_nvfp4_glu_gemv | deterministic | yes | 40 |
| gated_rmsnorm | gated_rmsnorm_bf16 | deterministic | yes | 0 |
| rmsnorm_add_matmul_pair | rmsnorm_add_matmul_pair_bf16_gemv | deterministic | yes | 69 |
| gated_rmsnorm_matmul_fp8 | gated_rmsnorm_matmul_fp8_gemv | deterministic | yes | 30 |

- kernels test: 43 kernels, 595 cases in 31.6s (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | head=ff42756 code-tree=dec5cb3168d0 dirty=no | spark-4e77 | 1792261005 | `/home/nvidia/intarsia-decfuse/cudanorm/target/acc/release/intarsia doctor --backend cuda --device 0 --out /home/nvidia/intarsia-decfuse/out/b2/qualify-35b/doctor.json` |
| `recipe` | head=ff42756 code-tree=dec5cb3168d0 dirty=no | spark-4e77 | 1792261006 | `recipe::analyse(/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5)` |
| `plan` | head=ff42756 code-tree=dec5cb3168d0 dirty=no | spark-4e77 | 1792261006 | `/home/nvidia/intarsia-decfuse/cudanorm/target/acc/release/intarsia plan show --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --admit --json` |
| `kernels` | head=ff42756 code-tree=dec5cb3168d0 dirty=no | spark-4e77 | 1792261050 | `/home/nvidia/intarsia-decfuse/cudanorm/target/acc/release/intarsia kernels test --backend cuda --device 0 --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5` |
| `gate` | head=ff42756 code-tree=dec5cb3168d0 dirty=no | spark-4e77 | 1792261082 | `/home/nvidia/intarsia-decfuse/cudanorm/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --quick --max-seq 2048` |
| `acc-short` | head=ff42756 code-tree=dec5cb3168d0 dirty=no | spark-4e77 | 1792261122 | `/home/nvidia/intarsia-decfuse/cudanorm/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsiaml-golden/qwen36-35` |
| `bench` | head=ff42756 code-tree=dec5cb3168d0 dirty=no | spark-4e77 | 1792261162 | `/home/nvidia/intarsia-decfuse/cudanorm/target/acc/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3.6-35B-A3B-NVFP4 --sizes 128,1024,4096 --gen 128 --reps 5 --warmup 1 --thinking off --label qualify-Qwen3.6-35B-A3B-NVFP4-cuda-bench --out-dir /home/nvidia/intarsia-decfuse` |

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
