# Qualification report: Qwen3.6-35B-A3B-NVFP4 on cuda

_Full (quick + in-depth) section, generated unix 1792779653, intarsia 0.0.1 build `pf-dev-055775da`._

## Headline

**Verdict: ERROR** (15 stages: 5 pass, 2 fail, 1 error, 7 skipped)

| | |
|---|---|
| Model | Qwen3.6-35B-A3B-NVFP4 (`/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5`) |
| Backend / device | cuda / ? (?) |
| Recipe | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| Decode | not measured |
| Prefill | not measured |
| Accuracy (short golden) | **PASS** (prefill top-1 97.92% KL 5.87e-3; decode top-1 98.12% KL 1.54e-2) |
| Accuracy (format reference, same weights) | skipped: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| Accuracy (long golden) | **PASS** (prefill top-1 94.09% KL 4.56e-2; decode top-1 96.56% KL 1.04e-2) |
| Accuracy (4k golden) | **FAIL** (prefill top-1 92.60% KL 9.10e-2; decode top-1 97.40% KL 9.95e-3) |
| Quick gate | **PASS**: perplexity 1.21, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 3m35s, in-depth 5m11s, total 8m46s |

### Failures

- `acc-oracle` **FAIL**: the logits moved against the recorded dump /home/nvidia/intarsia-decfuse/pf-golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.cuda.selfgolden.safetensors: mean KL(21-way)                                 8.0064e-3  (limit 1.0000e-4)  FAIL; max position KL(21-way)                         9.1842e-1  (li
- `acc-4k` **FAIL**: prefix reuse differs from a cold run; prefix reuse at chunk 1024: 3 judged, 0 skipped: prefix reuse differs from a cold run
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
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 39s | 39 ops resolved, exactness fast, loaded in 38.9s |
| 4 | `kernels` kernels test on the model's shapes | quick | SKIP | 0s | skipped by --skip |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 37s | perplexity 1.21, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| 7 | `acc-short` short accuracy against a golden | quick | PASS | 38s | prefill top-1 97.92% KL 5.87e-3; decode top-1 98.12% KL 1.54e-2 |
| 8 | `acc-oracle` short prefill accuracy against the f64 CPU oracle (same weights, deterministic reference) | quick | FAIL | 1m40s | the logits moved against the recorded dump /home/nvidia/intarsia-decfuse/pf-golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.cuda.selfgolden.safetensors: mean KL(21-way)                                 8.0064e-3  (limit 1.0000e-4)  FAIL; max position KL(21-way)                         9.1842e-1  (li |
| 9 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | SKIP | 0s | skipped by --skip |
| 10 | `acc-long` accuracy --mode all on the long golden (prefill, decode, prefix reuse) | full | PASS | 1m53s | prefill top-1 94.09% KL 4.56e-2; decode top-1 96.56% KL 1.04e-2 |
| 11 | `acc-4k` accuracy --mode all on the 4k golden (prefill, decode, prefix reuse past 2 chunks of 2048) | full | FAIL | 2m43s | prefix reuse differs from a cold run; prefix reuse at chunk 1024: 3 judged, 0 skipped: prefix reuse differs from a cold run |
| 12 | `envelope` noise envelope and baseline for regression gating | full | ERROR | 34s | measuring the noise envelope with 'chunk' exit 1: Error: the reference baseline or this build carries no code stamp, so they cannot be shown to be one build (build through scripts/phase1b-acceptance.sh, or --allow-unstamped for development) |
| 13 | `bench-long` long-context bench up to 16k | full | SKIP | 0s | skipped by --skip |
| 14 | `ab` A/B against the reference server via localeval | full | SKIP | 0s | skipped by --skip |
| 15 | `tp` tensor-parallel variant | full | SKIP | 0s | skipped by --skip |

### Notes

- `acc-short`: gated against the recorded baseline /home/nvidia/intarsia-decfuse/pf-golden/baselines/qwen36-35b-a3b-nvfp4-vllm-top20.cuda.baseline.json (commit unstamped) and its noise envelope
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
| prefill | 36887 | 94.09 | 90.30 | 4.56e-2 | 4.93e-2 | 2.54e1 | 7.93e-3 | 0 | 0.3 | pass |
| decode | 320 | 96.56 | 90.12 | 1.04e-2 | 9.65e-3 | 3.46e-1 | 3.00e-3 | 0 | 37.5 | pass |

**acc-4k** against `/home/nvidia/intarsia-decfuse/pf-golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 12759 | 92.60 | 89.61 | 9.10e-2 | 1.06e-1 | 2.46e1 | 1.34e-2 | 0 | 0.6 | pass |
| decode | 192 | 97.40 | 91.46 | 9.95e-3 | 9.30e-3 | 2.30e-1 | 1.48e-3 | 0 | 57.3 | pass |

**Regression gating artifacts**: measuring the noise envelope with 'chunk' exit 1: Error: the reference baseline or this build carries no code stamp, so they cannot be shown to be one build (build through scripts/phase1b-acceptance.sh, or --allow-unstamped for development)

``

**Quick gate**: perplexity 1.21, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) (PASS)
- plan: 39 ops resolved, exactness fast, loaded in 38.9s (PASS)

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
| `doctor` | pf-dev-055775da | spark-fa00 | 0 | `-` |
| `recipe` | pf-dev-055775da | spark-fa00 | 1792779126 | `recipe::analyse(/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5)` |
| `plan` | pf-dev-055775da | spark-fa00 | 1792779127 | `/home/nvidia/intarsia-decfuse/pf-dev/target/acc/release/intarsia plan show --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --admit --json` |
| `kernels` | pf-dev-055775da | spark-fa00 | 0 | `/home/nvidia/intarsia-decfuse/pf-dev/target/acc/release/intarsia plan show --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --admit --json` |
| `gate` | pf-dev-055775da | spark-fa00 | 1792779166 | `/home/nvidia/intarsia-decfuse/pf-dev/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --quick --max-seq 2048` |
| `acc-format` | pf-dev-055775da | spark-fa00 | 1792779203 | `-` |
| `acc-short` | pf-dev-055775da | spark-fa00 | 1792779203 | `/home/nvidia/intarsia-decfuse/pf-dev/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsia-decfuse/pf-golden/qw` |
| `acc-oracle` | pf-dev-055775da | spark-fa00 | 1792779241 | `/home/nvidia/intarsia-decfuse/pf-dev/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --self-golden /home/nvidia/intarsia-decfuse/pf-gold` |
| `bench` | pf-dev-055775da | spark-fa00 | 0 | `/home/nvidia/intarsia-decfuse/pf-dev/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --self-golden /home/nvidia/intarsia-decfuse/pf-gold` |
| `acc-long` | pf-dev-055775da | spark-fa00 | 1792779341 | `/home/nvidia/intarsia-decfuse/pf-dev/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsia-decfuse/pf-golden/qw` |
| `acc-4k` | pf-dev-055775da | spark-fa00 | 1792779455 | `/home/nvidia/intarsia-decfuse/pf-dev/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsia-decfuse/pf-golden/qw` |
| `envelope` | pf-dev-055775da | spark-fa00 | 1792779619 | `/home/nvidia/intarsia-decfuse/pf-dev/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsia-decfuse/pf-golden/qw` |
| `bench-long` | pf-dev-055775da | spark-fa00 | 0 | `/home/nvidia/intarsia-decfuse/pf-dev/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsia-decfuse/pf-golden/qw` |
| `ab` | pf-dev-055775da | spark-fa00 | 0 | `/home/nvidia/intarsia-decfuse/pf-dev/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsia-decfuse/pf-golden/qw` |
| `tp` | pf-dev-055775da | spark-fa00 | 0 | `/home/nvidia/intarsia-decfuse/pf-dev/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsia-decfuse/pf-golden/qw` |

Model fingerprint: `fnv1a64:25b7234fc8ebc6ee`.


## Artifacts

- `plan.json` (plan)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `logs/acc-oracle-drift.out` (acc-oracle)
- `logs/acc-long.out` (acc-long)
- `baselines/acc-long.baseline.json` (acc-long)
- `logs/acc-4k.out` (acc-4k)
- `baselines/acc-4k.baseline.json` (acc-4k)
- `logs/acc-4k-reuse.out` (acc-4k)
- `state.json` (stage records; `--resume` reads it)
