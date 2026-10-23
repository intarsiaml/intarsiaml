# Qualification report: Qwen3.6-35B-A3B-NVFP4 on cuda

_Full (quick + in-depth) section, generated unix 1792777209, intarsia 0.0.1 build `main-055775da`._

## Headline

**Verdict: ERROR** (15 stages: 6 pass, 1 fail, 1 error, 7 skipped)

| | |
|---|---|
| Model | Qwen3.6-35B-A3B-NVFP4 (`/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5`) |
| Backend / device | cuda / ? (?) |
| Recipe | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| Decode | not measured |
| Prefill | not measured |
| Accuracy (short golden) | **PASS** (prefill top-1 97.92% KL 5.87e-3; decode top-1 98.12% KL 1.54e-2) |
| Accuracy (format reference, same weights) | skipped: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| Accuracy (long golden) | **PASS** (prefill top-1 94.17% KL 4.27e-2; decode top-1 97.19% KL 1.05e-2) |
| Accuracy (4k golden) | **FAIL** (prefill top-1 92.81% KL 8.14e-2; decode top-1 98.96% KL 7.96e-3) |
| Quick gate | **PASS**: perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 3m40s, in-depth 5m18s, total 8m58s |

### Failures

- `acc-4k` **FAIL**: prefix reuse at chunk 1024: 3 judged, 0 skipped: prefix reuse differs from a cold run
- `envelope` **ERROR**: measuring the noise envelope with 'chunk' exit 1: Error: the reference baseline or this build carries no code stamp, so they cannot be shown to be one build (build through scripts/phase1b-acceptance.sh, or --allow-unstamped for development)

### Skipped

- `doctor`: skipped by --skip
- `kernels`: skipped by --skip
- `acc-format`: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE)
- `bench`: skipped by --skip
- `bench-long`: skipped by --skip
- `ab`: skipped by --skip
- `tp`: skipped by --skip

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | SKIP | 0s | skipped by --skip |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 37s | 39 ops resolved, exactness fast, loaded in 36.8s |
| 4 | `kernels` kernels test on the model's shapes | quick | SKIP | 0s | skipped by --skip |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 38s | perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| 7 | `acc-short` short accuracy against a golden | quick | PASS | 39s | prefill top-1 97.92% KL 5.87e-3; decode top-1 98.12% KL 1.54e-2 |
| 8 | `acc-oracle` short prefill accuracy against the f64 CPU oracle (same weights, deterministic reference) | quick | PASS | 1m46s | prefill top-1 95.82% KL 1.01e-2 |
| 9 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | SKIP | 0s | skipped by --skip |
| 10 | `acc-long` accuracy --mode all on the long golden (prefill, decode, prefix reuse) | full | PASS | 1m45s | prefill top-1 94.17% KL 4.27e-2; decode top-1 97.19% KL 1.05e-2 |
| 11 | `acc-4k` accuracy --mode all on the 4k golden (prefill, decode, prefix reuse past 2 chunks of 2048) | full | FAIL | 2m55s | prefix reuse at chunk 1024: 3 judged, 0 skipped: prefix reuse differs from a cold run |
| 12 | `envelope` noise envelope and baseline for regression gating | full | ERROR | 37s | measuring the noise envelope with 'chunk' exit 1: Error: the reference baseline or this build carries no code stamp, so they cannot be shown to be one build (build through scripts/phase1b-acceptance.sh, or --allow-unstamped for development) |
| 13 | `bench-long` long-context bench up to 16k | full | SKIP | 0s | skipped by --skip |
| 14 | `ab` A/B against the reference server via localeval | full | SKIP | 0s | skipped by --skip |
| 15 | `tp` tensor-parallel variant | full | SKIP | 0s | skipped by --skip |

### Notes

- `acc-short`: gated against the recorded baseline /home/nvidia/intarsia-decfuse/pf-golden/baselines/qwen36-35b-a3b-nvfp4-vllm-top20.cuda.baseline.json (commit unstamped) and its noise envelope
- `acc-oracle`: floor: mean KL(oracle \|\| engine) over the oracle's top-64 + tail, scored prefill rows, <= 2.0e-2 (measured, docs/results/2026-10-22-fx1-oracle-gate.md); top-1 not gated. The floor guards absolute accuracy; it does not see a 1 % scale bug, the drift check does
- `acc-oracle`: drift check: logits unchanged against the recorded dump /home/nvidia/intarsia-decfuse/pf-golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.cuda.selfgolden.safetensors
- `acc-long`: prefix reuse is gated in acc-4k, on the 4k golden
- `acc-4k`: no baseline for backend cuda next to the golden or in golden/baselines: KL, tail, prefix-reuse gates only, top-1 not gated

## Speed

No speed numbers.

## Accuracy

**acc-short** against `/home/nvidia/intarsia-decfuse/pf-golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1441 | 97.92 | 91.81 | 5.87e-3 | 5.43e-3 | 6.48e-1 | 1.41e-3 | 0 | 5.6 | pass |
| decode | 160 | 98.12 | 92.75 | 1.54e-2 | 1.59e-2 | 1.33e0 | 3.36e-3 | 0 | 88.8 | pass |

**acc-long** against `/home/nvidia/intarsia-decfuse/pf-golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 36887 | 94.17 | 90.36 | 4.27e-2 | 4.58e-2 | 2.51e1 | 7.73e-3 | 0 | 0.2 | pass |
| decode | 320 | 97.19 | 89.75 | 1.05e-2 | 9.58e-3 | 2.49e-1 | 3.24e-3 | 0 | 65.0 | pass |

**acc-4k** against `/home/nvidia/intarsia-decfuse/pf-golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 12759 | 92.81 | 89.92 | 8.14e-2 | 9.26e-2 | 2.14e1 | 1.22e-2 | 0 | 0.2 | pass |
| decode | 192 | 98.96 | 90.62 | 7.96e-3 | 7.43e-3 | 1.21e-1 | 1.67e-3 | 0 | 87.5 | pass |

**Regression gating artifacts**: measuring the noise envelope with 'chunk' exit 1: Error: the reference baseline or this build carries no code stamp, so they cannot be shown to be one build (build through scripts/phase1b-acceptance.sh, or --allow-unstamped for development)

``

**Quick gate**: perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) (PASS)
- plan: 39 ops resolved, exactness fast, loaded in 36.8s (PASS)

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
| matmul_fp8_pair | matmul_fp8_pair_gemm_reg | deterministic | yes | 0 |
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
| `doctor` | main-055775da | spark-fa00 | 0 | `-` |
| `recipe` | main-055775da | spark-fa00 | 1792776671 | `recipe::analyse(/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5)` |
| `plan` | main-055775da | spark-fa00 | 1792776671 | `/home/nvidia/intarsia-decfuse/pf-base/target/acc/release/intarsia plan show --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --admit --json` |
| `kernels` | main-055775da | spark-fa00 | 0 | `/home/nvidia/intarsia-decfuse/pf-base/target/acc/release/intarsia plan show --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --admit --json` |
| `gate` | main-055775da | spark-fa00 | 1792776708 | `/home/nvidia/intarsia-decfuse/pf-base/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --quick --max-seq 2048` |
| `acc-format` | main-055775da | spark-fa00 | 1792776746 | `-` |
| `acc-short` | main-055775da | spark-fa00 | 1792776746 | `/home/nvidia/intarsia-decfuse/pf-base/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsia-decfuse/pf-golden/q` |
| `acc-oracle` | main-055775da | spark-fa00 | 1792776785 | `/home/nvidia/intarsia-decfuse/pf-base/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --self-golden /home/nvidia/intarsia-decfuse/pf-gol` |
| `bench` | main-055775da | spark-fa00 | 0 | `/home/nvidia/intarsia-decfuse/pf-base/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --self-golden /home/nvidia/intarsia-decfuse/pf-gol` |
| `acc-long` | main-055775da | spark-fa00 | 1792776891 | `/home/nvidia/intarsia-decfuse/pf-base/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsia-decfuse/pf-golden/q` |
| `acc-4k` | main-055775da | spark-fa00 | 1792776997 | `/home/nvidia/intarsia-decfuse/pf-base/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsia-decfuse/pf-golden/q` |
| `envelope` | main-055775da | spark-fa00 | 1792777173 | `/home/nvidia/intarsia-decfuse/pf-base/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsia-decfuse/pf-golden/q` |
| `bench-long` | main-055775da | spark-fa00 | 0 | `/home/nvidia/intarsia-decfuse/pf-base/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsia-decfuse/pf-golden/q` |
| `ab` | main-055775da | spark-fa00 | 0 | `/home/nvidia/intarsia-decfuse/pf-base/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsia-decfuse/pf-golden/q` |
| `tp` | main-055775da | spark-fa00 | 0 | `/home/nvidia/intarsia-decfuse/pf-base/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsia-decfuse/pf-golden/q` |

Model fingerprint: `fnv1a64:25b7234fc8ebc6ee`.


## Artifacts

- `plan.json` (plan)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `logs/acc-oracle.out` (acc-oracle)
- `baselines/acc-oracle.baseline.json` (acc-oracle)
- `baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.cuda.selfgolden.safetensors` (acc-oracle)
- `logs/acc-long.out` (acc-long)
- `baselines/acc-long.baseline.json` (acc-long)
- `logs/acc-4k.out` (acc-4k)
- `baselines/acc-4k.baseline.json` (acc-4k)
- `logs/acc-4k-reuse.out` (acc-4k)
- `state.json` (stage records; `--resume` reads it)
