# Qualification report: Qwen3.6-35B-A3B-NVFP4 on cuda

_Full (quick + in-depth) section, generated unix 1792658271, intarsia 0.0.1 build `head=296c8c3 code-tree=57dfbb7bb429 dirty=no`._

## Headline

**Verdict: FAIL** (15 stages: 3 pass, 1 fail, 0 error, 11 skipped)

| | |
|---|---|
| Model | Qwen3.6-35B-A3B-NVFP4 (`/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5`) |
| Backend / device | cuda / NVIDIA GB10 (sm_121) |
| Recipe | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| Decode | not measured |
| Prefill | not measured |
| Accuracy (short golden) | skipped: skipped by --skip |
| Accuracy (format reference, same weights) | skipped: skipped by --skip |
| Accuracy (long golden) | **FAIL** (prefill top-1 94.17% KL 4.27e-2; decode top-1 97.19% KL 1.05e-2) |
| Accuracy (4k golden) | skipped: skipped by --skip |
| Quick gate | **SKIP**: skipped by --skip |
| Wall time | quick 45s, in-depth 4m48s, total 5m32s |

### Failures

- `acc-long` **FAIL**: prefix reuse at chunk 1024: 5 judged, 0 skipped: prefix reuse differs from a cold run

### Skipped

- `kernels`: skipped by --skip
- `gate`: skipped by --skip
- `acc-format`: skipped by --skip
- `acc-short`: skipped by --skip
- `acc-oracle`: no CPU-oracle golden for 'Qwen3.6-35B-A3B-NVFP4' (--oracle-golden, or oracle/<model>-cpu-oracle*.safetensors in a golden directory)
- `bench`: skipped by --skip
- `acc-4k`: skipped by --skip
- `envelope`: skipped by --skip
- `bench-long`: skipped by --skip
- `ab`: skipped by --skip
- `tp`: skipped by --skip

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 1s | NVIDIA GB10 (sm_121), 54 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 44s | 36 ops resolved, exactness fast, loaded in 43.6s |
| 4 | `kernels` kernels test on the model's shapes | quick | SKIP | 0s | skipped by --skip |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | SKIP | 0s | skipped by --skip |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | skipped by --skip |
| 7 | `acc-short` short accuracy against a golden | quick | SKIP | 0s | skipped by --skip |
| 8 | `acc-oracle` short prefill accuracy against the f64 CPU oracle (same weights, deterministic reference) | quick | SKIP | 0s | no CPU-oracle golden for 'Qwen3.6-35B-A3B-NVFP4' (--oracle-golden, or oracle/<model>-cpu-oracle*.safetensors in a golden directory) |
| 9 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | SKIP | 0s | skipped by --skip |
| 10 | `acc-long` accuracy --mode all on the long golden (prefill, decode, prefix reuse) | full | FAIL | 4m48s | prefix reuse at chunk 1024: 5 judged, 0 skipped: prefix reuse differs from a cold run |
| 11 | `acc-4k` accuracy --mode all on the 4k golden (prefill, decode, prefix reuse past 2 chunks of 2048) | full | SKIP | 0s | skipped by --skip |
| 12 | `envelope` noise envelope and baseline for regression gating | full | SKIP | 0s | skipped by --skip |
| 13 | `bench-long` long-context bench up to 16k | full | SKIP | 0s | skipped by --skip |
| 14 | `ab` A/B against the reference server via localeval | full | SKIP | 0s | skipped by --skip |
| 15 | `tp` tensor-parallel variant | full | SKIP | 0s | skipped by --skip |

### Notes

- `doctor`: warning disk_free:.: 272.2 GB free of 4032 GB (7% free)
- `doctor`: warning disk_free:/home/nvidia: 272.2 GB free of 4032 GB (7% free)

## Speed

No speed numbers.

## Accuracy

**acc-long** against `/home/nvidia/intarsiaml-golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors` (given with --golden; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 36887 | 94.17 | 90.36 | 4.27e-2 | 4.58e-2 | 2.51e1 | 7.73e-3 | 0 | 0.2 | pass |
| decode | 320 | 97.19 | 89.75 | 1.05e-2 | 9.58e-3 | 2.49e-1 | 3.24e-3 | 0 | 65.0 | pass |

**Quick gate**: skipped by --skip

## Plan and recipe

- recipe: builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) (PASS)
- plan: 36 ops resolved, exactness fast, loaded in 43.6s (PASS)

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

- kernels test: skipped by --skip (SKIP)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | head=296c8c3 code-tree=57dfbb7bb429 dirty=no | spark-4e77 | 1792657939 | `/home/nvidia/intarsia-decfuse/fx5/target/acc/release/intarsia doctor --backend cuda --device 0 --out /home/nvidia/intarsia-decfuse/out/fx5q/qualify/doctor.json` |
| `recipe` | head=296c8c3 code-tree=57dfbb7bb429 dirty=no | spark-4e77 | 1792657939 | `recipe::analyse(/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5)` |
| `plan` | head=296c8c3 code-tree=57dfbb7bb429 dirty=no | spark-4e77 | 1792657940 | `/home/nvidia/intarsia-decfuse/fx5/target/acc/release/intarsia plan show --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 40 --admit --json` |
| `kernels` | head=296c8c3 code-tree=57dfbb7bb429 dirty=no | spark-4e77 | 0 | `/home/nvidia/intarsia-decfuse/fx5/target/acc/release/intarsia plan show --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 40 --admit --json` |
| `gate` | head=296c8c3 code-tree=57dfbb7bb429 dirty=no | spark-4e77 | 0 | `/home/nvidia/intarsia-decfuse/fx5/target/acc/release/intarsia plan show --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 40 --admit --json` |
| `acc-format` | head=296c8c3 code-tree=57dfbb7bb429 dirty=no | spark-4e77 | 0 | `/home/nvidia/intarsia-decfuse/fx5/target/acc/release/intarsia plan show --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 40 --admit --json` |
| `acc-short` | head=296c8c3 code-tree=57dfbb7bb429 dirty=no | spark-4e77 | 0 | `/home/nvidia/intarsia-decfuse/fx5/target/acc/release/intarsia plan show --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 40 --admit --json` |
| `acc-oracle` | head=296c8c3 code-tree=57dfbb7bb429 dirty=no | spark-4e77 | 1792657983 | `-` |
| `bench` | head=296c8c3 code-tree=57dfbb7bb429 dirty=no | spark-4e77 | 0 | `-` |
| `acc-long` | head=296c8c3 code-tree=57dfbb7bb429 dirty=no | spark-4e77 | 1792657983 | `/home/nvidia/intarsia-decfuse/fx5/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 40 --golden /home/nvidia/intarsiaml-golden/qwen36-35b-a3b` |
| `acc-4k` | head=296c8c3 code-tree=57dfbb7bb429 dirty=no | spark-4e77 | 0 | `/home/nvidia/intarsia-decfuse/fx5/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 40 --golden /home/nvidia/intarsiaml-golden/qwen36-35b-a3b` |
| `envelope` | head=296c8c3 code-tree=57dfbb7bb429 dirty=no | spark-4e77 | 0 | `/home/nvidia/intarsia-decfuse/fx5/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 40 --golden /home/nvidia/intarsiaml-golden/qwen36-35b-a3b` |
| `bench-long` | head=296c8c3 code-tree=57dfbb7bb429 dirty=no | spark-4e77 | 0 | `/home/nvidia/intarsia-decfuse/fx5/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 40 --golden /home/nvidia/intarsiaml-golden/qwen36-35b-a3b` |
| `ab` | head=296c8c3 code-tree=57dfbb7bb429 dirty=no | spark-4e77 | 0 | `/home/nvidia/intarsia-decfuse/fx5/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 40 --golden /home/nvidia/intarsiaml-golden/qwen36-35b-a3b` |
| `tp` | head=296c8c3 code-tree=57dfbb7bb429 dirty=no | spark-4e77 | 0 | `/home/nvidia/intarsia-decfuse/fx5/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 40 --golden /home/nvidia/intarsiaml-golden/qwen36-35b-a3b` |

Model fingerprint: `fnv1a64:25b7234fc8ebc6ee`.


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/acc-long.out` (acc-long)
- `baselines/acc-long.baseline.json` (acc-long)
- `logs/acc-long-reuse.out` (acc-long)
- `state.json` (stage records; `--resume` reads it)
