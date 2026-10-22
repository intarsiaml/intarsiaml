# Qualification report: Qwen3.6-35B-A3B-NVFP4 on cuda

> **Onboard verdict: PASS.** recipe: PASS; smoke: PASS; quick: PASS; in-depth: PASS;

_Full (quick + in-depth) section, generated unix 1792654724, intarsia 0.0.1 build `head=716b4e4 code-tree=79c4400f0040 dirty=no`._

## Headline

**Verdict: PASS** (14 stages: 11 pass, 0 fail, 0 error, 3 skipped)

| | |
|---|---|
| Model | Qwen3.6-35B-A3B-NVFP4 (`/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5`) |
| Backend / device | cuda / NVIDIA GB10 (sm_121) |
| Recipe | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| Decode | 128 tok: **84.1** tok/s (86% of floor); 1024 tok: **83.6** tok/s (85% of floor); 4096 tok: **81.1** tok/s (83% of floor) |
| Prefill | 128 tok: **1282** tok/s; 1024 tok: **4640** tok/s; 4096 tok: **5383** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 96.83% KL 7.65e-3; decode top-1 98.44% KL 1.19e-2) |
| Accuracy (format reference, same weights) | skipped: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| Accuracy (long golden) | **PASS** (prefill top-1 97.85% KL 5.56e-3; decode top-1 97.40% KL 1.01e-2) |
| Accuracy (4k golden) | **PASS** (prefill top-1 92.40% KL 9.08e-2; decode top-1 97.92% KL 9.53e-3) |
| Quick gate | **PASS**: perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 5m49s, in-depth 7m59s, total 13m49s |
| Reference | http://100.120.88.104:8000/v1 |

### Failures

None.

### Skipped

- `acc-format`: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE)
- `ab`: localeval is not at /opt/code/deepeval (--localeval-dir); the A/B must run through localeval, never an ad-hoc script
- `tp`: --tp not given (or 1)

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
| 9 | `acc-long` accuracy --mode all on the long golden (prefill, decode, prefix reuse) | full | PASS | 79s | prefill top-1 97.85% KL 5.56e-3; decode top-1 97.40% KL 1.01e-2 |
| 10 | `acc-4k` accuracy --mode all on the 4k golden (prefill, decode, prefix reuse past 2 chunks of 2048) | full | PASS | 1m59s | prefill top-1 92.40% KL 9.08e-2; decode top-1 97.92% KL 9.53e-3 |
| 11 | `envelope` noise envelope and baseline for regression gating | full | PASS | 3m19s | baseline and envelope (chunk + attn-simt) written; they gate this build: PASS |
| 12 | `bench-long` long-context bench up to 16k | full | PASS | 82s | decode 82.9 tok/s at 1024 tokens (84% of the byte floor) |
| 13 | `ab` A/B against the reference server via localeval | full | SKIP | 0s | localeval is not at /opt/code/deepeval (--localeval-dir); the A/B must run through localeval, never an ad-hoc script |
| 14 | `tp` tensor-parallel variant | full | SKIP | 0s | --tp not given (or 1) |

### Notes

- `doctor`: warning disk_free:.: 271.5 GB free of 4032 GB (7% free)
- `doctor`: warning disk_free:/home/nvidia: 271.5 GB free of 4032 GB (7% free)
- `acc-long`: prefix reuse is gated in acc-4k, on the 4k golden
- `acc-4k`: no baseline for backend cuda next to the 4k golden or in golden/baselines: KL, tail, prefix-reuse gates only, top-1 not gated

## Speed

**quick bench** (floor: 2.25 GB/token at 221 GB/s; gb/token: estimated from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 221 GB/s, the GB10 measured figure)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 131 | 0.101 | 1282 | 84.10 | 4.21 | 98.0 | 85.8 |
| 1015 | 0.219 | 4640 | 83.63 | 0.17 | 98.0 | 85.3 |
| 4185 | 0.782 | 5383 | 81.14 | 0.21 | 98.0 | 82.8 |

**long-context bench** (floor: 2.25 GB/token at 221 GB/s; gb/token: estimated from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 221 GB/s, the GB10 measured figure)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 1020 | 0.219 | 4653 | 82.86 | 0.01 | 98.0 | 84.5 |
| 4078 | 0.701 | 5815 | 80.56 | 0.08 | 98.0 | 82.2 |
| 16721 | 3.113 | 5371 | 72.86 | 0.07 | 98.0 | 74.3 |

## Accuracy

**acc-short** against `/home/nvidia/onboard-runs/35b-cuda/golden-fresh/Qwen3.6-35B-A3B-NVFP4-vllm-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 441 | 96.83 | 95.19 | 7.65e-3 | 7.65e-3 | 2.53e-1 | 1.84e-3 | 0 | 32.7 | pass |
| decode | 128 | 98.44 | 95.16 | 1.19e-2 | 1.28e-2 | 4.13e-1 | 4.33e-3 | 0 | 78.1 | pass |

**acc-long** against `/home/nvidia/onboard-runs/35b-cuda/golden-fresh/Qwen3.6-35B-A3B-NVFP4-vllm-long-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1441 | 97.85 | 91.94 | 5.56e-3 | 5.13e-3 | 3.86e-1 | 1.48e-3 | 0 | 9.4 | pass |
| decode | 500 | 97.40 | 93.36 | 1.01e-2 | 1.02e-2 | 5.04e-1 | 1.70e-3 | 0 | 65.4 | pass |

**acc-4k** against `/home/nvidia/onboard-runs/35b-cuda/golden-fresh/Qwen3.6-35B-A3B-NVFP4-vllm-4k-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 12759 | 92.40 | 89.90 | 9.08e-2 | 1.06e-1 | 2.45e1 | 1.27e-2 | 0 | 0.2 | pass |
| decode | 192 | 97.92 | 91.25 | 9.53e-3 | 8.49e-3 | 1.88e-1 | 3.02e-3 | 0 | 85.9 | pass |

**Regression gating artifacts**: baseline and envelope (chunk + attn-simt) written; they gate this build: PASS

`intarsia accuracy --model <dir> --backend cuda --golden /home/nvidia/onboard-runs/35b-cuda/golden-fresh/Qwen3.6-35B-A3B-NVFP4-vllm-top20.safetensors --baseline baselines/acc-short.baseline.json --noise-envelope baselines/envelope.json  (copy both files next to the golden, e.g. golden/baselines/)`

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
| `acc-long` | head=716b4e4 code-tree=79c4400f0040 dirty=no | spark-4e77 | 1792654244 | `/home/nvidia/intarsia-decfuse/on/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 38 --golden /home/nvidia/onboard-runs/35b-cuda/golden-fres` |
| `acc-4k` | head=716b4e4 code-tree=79c4400f0040 dirty=no | spark-4e77 | 1792654322 | `/home/nvidia/intarsia-decfuse/on/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 38 --golden /home/nvidia/onboard-runs/35b-cuda/golden-fres` |
| `envelope` | head=716b4e4 code-tree=79c4400f0040 dirty=no | spark-4e77 | 1792654442 | `/home/nvidia/intarsia-decfuse/on/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 38 --golden /home/nvidia/onboard-runs/35b-cuda/golden-fres` |
| `bench-long` | head=716b4e4 code-tree=79c4400f0040 dirty=no | spark-4e77 | 1792654642 | `/home/nvidia/intarsia-decfuse/on/target/acc/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3.6-35B-A3B-NVFP4 --sizes 1024,4096,16384 --gen 64 --reps 2 --warmup 1 --thinking off --label qualify-Qwen3.6-35B-A3B-NVFP4-cuda-bench-long --out-dir /home/nvidia/onboard-runs/35b` |
| `ab` | head=716b4e4 code-tree=79c4400f0040 dirty=no | spark-4e77 | 1792654724 | `-` |
| `tp` | head=716b4e4 code-tree=79c4400f0040 dirty=no | spark-4e77 | 1792654724 | `-` |

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
- `logs/acc-long.out` (acc-long)
- `baselines/acc-long.baseline.json` (acc-long)
- `logs/acc-4k.out` (acc-4k)
- `baselines/acc-4k.baseline.json` (acc-4k)
- `baselines/acc-short.baseline.json` (envelope)
- `baselines/envelope.json` (envelope)
- `logs/baseline-selfcheck.out` (envelope)
- `bench/qualify-Qwen3.6-35B-A3B-NVFP4-cuda-bench-long.json` (bench-long)
- `logs/bench-long-bench.out` (bench-long)
- `logs/bench-long-server.err` (bench-long)
- `state.json` (stage records; `--resume` reads it)

## Stage timeline

All times UTC. The qualify stages ran inside the `quick` / `in-depth` stages. A stage that was killed and resumed shows its last attempt; the qualify rows show every sub-stage with the time it actually ran.

| Stage | Tier | Verdict | Start | End | Duration | Note |
|---|---|---|---|---|---|---|
| recipe | prepare | PASS | 2026-10-22T07:23:20Z | 2026-10-22T07:23:20Z | 0s |  |
| smoke | prepare | PASS | 2026-10-22T07:23:20Z | 2026-10-22T07:24:53Z | 1m33s |  |
| quick | quick | PASS | 2026-10-22T07:24:53Z | 2026-10-22T07:30:44Z | 5m50s |  |
| publish-quick | quick | PASS | 2026-10-22T07:30:44Z | 2026-10-22T07:30:44Z | 0s |  |
| in-depth | in-depth | PASS | 2026-10-22T07:30:44Z | 2026-10-22T07:38:44Z | 8m00s |  |
| doctor | qualify quick | PASS | 2026-10-22T07:24:54Z | 2026-10-22T07:24:55Z | 1s |  |
| recipe | qualify quick | PASS | 2026-10-22T07:24:54Z | 2026-10-22T07:24:54Z | 0s |  |
| plan | qualify quick | PASS | 2026-10-22T07:24:55Z | 2026-10-22T07:26:05Z | 70s |  |
| kernels | qualify quick | PASS | 2026-10-22T07:26:05Z | 2026-10-22T07:26:49Z | 44s |  |
| gate | qualify quick | PASS | 2026-10-22T07:26:49Z | 2026-10-22T07:28:03Z | 74s |  |
| acc-format | qualify quick | SKIP | 2026-10-22T07:30:44Z | 2026-10-22T07:30:44Z | 0s |  |
| acc-short | qualify quick | PASS | 2026-10-22T07:28:03Z | 2026-10-22T07:29:12Z | 69s |  |
| bench | qualify quick | PASS | 2026-10-22T07:29:12Z | 2026-10-22T07:30:43Z | 1m31s |  |
| acc-long | qualify full | PASS | 2026-10-22T07:30:44Z | 2026-10-22T07:32:03Z | 79s |  |
| acc-4k | qualify full | PASS | 2026-10-22T07:32:02Z | 2026-10-22T07:34:02Z | 1m59s |  |
| envelope | qualify full | PASS | 2026-10-22T07:34:02Z | 2026-10-22T07:37:22Z | 3m19s |  |
| bench-long | qualify full | PASS | 2026-10-22T07:37:22Z | 2026-10-22T07:38:44Z | 82s |  |
| ab | qualify full | SKIP | 2026-10-22T07:38:44Z | 2026-10-22T07:38:44Z | 0s |  |
| tp | qualify full | SKIP | 2026-10-22T07:38:44Z | 2026-10-22T07:38:44Z | 0s |  |

_Published 2026-10-22T07:38:44Z by `intarsia onboard` (report directory `/home/nvidia/onboard-runs/35b-cuda/report`)._
