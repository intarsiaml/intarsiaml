# Qualification report: Qwen3.6-35B-A3B-NVFP4 on hip

_Full (quick + in-depth) section, generated unix 1792654978, intarsia 0.0.1 build `head=4d3e38e code-tree=9cc7bec56a44 dirty=no`._

## Headline

**Verdict: FAIL** (14 stages: 10 pass, 1 fail, 0 error, 3 skipped)

| | |
|---|---|
| Model | Qwen3.6-35B-A3B-NVFP4 (`/models/Qwen3.6-35B-A3B-NVFP4`) |
| Backend / device | hip / AMD Radeon Graphics (gfx1201) |
| Recipe | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| Decode | 128 tok: **159.8** tok/s (58% of floor); 1024 tok: **159.3** tok/s (57% of floor); 4096 tok: **155.1** tok/s (56% of floor) |
| Prefill | 128 tok: **1959** tok/s; 1024 tok: **8085** tok/s; 4096 tok: **8725** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 97.51% KL 1.09e-2; decode top-1 97.66% KL 8.79e-3) |
| Accuracy (format reference, same weights) | skipped: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| Accuracy (long golden) | **PASS** (prefill top-1 98.54% KL 7.12e-3; decode top-1 97.00% KL 1.08e-2) |
| Accuracy (4k golden) | **FAIL** (prefill top-1 92.73% KL 8.86e-2; decode top-1 96.88% KL 9.76e-3) |
| Quick gate | **PASS**: perplexity 1.21, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 76s, in-depth 84s, total 2m39s |

### Failures

- `acc-4k` **FAIL**: prefix reuse differs from a cold run

### Skipped

- `acc-format`: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE)
- `ab`: no --reference server given
- `tp`: --tp not given (or 1)

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | AMD Radeon Graphics (gfx1201), 54 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 9s | 30 ops resolved, exactness deterministic, loaded in 9.0s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 39s | 54 kernels, 995 cases in 38.7s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 6s | perplexity 1.21, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| 7 | `acc-short` short accuracy against a golden | quick | PASS | 5s | prefill top-1 97.51% KL 1.09e-2; decode top-1 97.66% KL 8.79e-3 |
| 8 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 17s | decode 159.8 tok/s at 128 tokens (58% of the byte floor) |
| 9 | `acc-long` accuracy --mode all on the long golden (prefill, decode, prefix reuse) | full | PASS | 11s | prefill top-1 98.54% KL 7.12e-3; decode top-1 97.00% KL 1.08e-2 |
| 10 | `acc-4k` accuracy --mode all on the 4k golden (prefill, decode, prefix reuse past 2 chunks of 2048) | full | FAIL | 40s | prefix reuse differs from a cold run |
| 11 | `envelope` noise envelope and baseline for regression gating | full | PASS | 16s | baseline and envelope (chunk + attn-simt) written; they gate this build: PASS |
| 12 | `bench-long` long-context bench up to 16k | full | PASS | 18s | decode 159.1 tok/s at 1024 tokens (57% of the byte floor) |
| 13 | `ab` A/B against the reference server via localeval | full | SKIP | 0s | no --reference server given |
| 14 | `tp` tensor-parallel variant | full | SKIP | 0s | --tp not given (or 1) |

### Notes

- `acc-long`: prefix reuse is gated in acc-4k, on the 4k golden
- `acc-4k`: no baseline for backend hip next to the 4k golden or in golden/baselines: KL, tail, prefix-reuse gates only, top-1 not gated

## Speed

**quick bench** (floor: 2.25 GB/token at 627 GB/s; gb/token: estimated from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 627 GB/s, the R9700 figure of docs/results/2026-10-13-phase2a-hip.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 132 | 0.067 | 1959 | 159.79 | 0.20 | 278.1 | 57.5 |
| 1028 | 0.128 | 8085 | 159.34 | 0.05 | 278.1 | 57.3 |
| 4185 | 0.480 | 8725 | 155.06 | 0.05 | 278.1 | 55.8 |

**long-context bench** (floor: 2.25 GB/token at 627 GB/s; gb/token: estimated from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 627 GB/s, the R9700 figure of docs/results/2026-10-13-phase2a-hip.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 992 | 0.139 | 7178 | 159.08 | 0.21 | 278.1 | 57.2 |
| 4073 | 0.423 | 9620 | 154.91 | 0.31 | 278.1 | 55.7 |
| 16719 | 2.125 | 7867 | 141.40 | 0.01 | 278.1 | 50.8 |

## Accuracy

**acc-short** against `/work/onboard-runs/golden-fresh/Qwen3.6-35B-A3B-NVFP4-vllm-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 441 | 97.51 | 95.33 | 1.09e-2 | 1.16e-2 | 5.81e-1 | 2.87e-3 | 0 | 42.9 | pass |
| decode | 128 | 97.66 | 94.38 | 8.79e-3 | 8.99e-3 | 1.57e-1 | 2.97e-3 | 0 | 78.1 | pass |

**acc-long** against `/work/onboard-runs/golden-fresh/Qwen3.6-35B-A3B-NVFP4-vllm-long-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1441 | 98.54 | 91.37 | 7.12e-3 | 6.87e-3 | 1.06e0 | 1.67e-3 | 0 | 13.2 | pass |
| decode | 500 | 97.00 | 93.52 | 1.08e-2 | 1.08e-2 | 6.73e-1 | 1.59e-3 | 0 | 53.2 | pass |

**acc-4k** against `/work/onboard-runs/golden-fresh/Qwen3.6-35B-A3B-NVFP4-vllm-4k-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 12759 | 92.73 | 90.09 | 8.86e-2 | 1.01e-1 | 2.16e1 | 1.21e-2 | 0 | 0.6 | pass |
| decode | 192 | 96.88 | 91.56 | 9.76e-3 | 8.71e-3 | 2.28e-1 | 2.98e-3 | 0 | 72.9 | pass |

**Regression gating artifacts**: baseline and envelope (chunk + attn-simt) written; they gate this build: PASS

`intarsia accuracy --model <dir> --backend hip --golden /work/onboard-runs/golden-fresh/Qwen3.6-35B-A3B-NVFP4-vllm-top20.safetensors --baseline baselines/acc-short.baseline.json --noise-envelope baselines/envelope.json  (copy both files next to the golden, e.g. golden/baselines/)`

**Quick gate**: perplexity 1.21, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) (PASS)
- plan: 30 ops resolved, exactness deterministic, loaded in 9.0s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_bf16 | exact | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 1 |
| rmsnorm_add | rmsnorm_add_bf16 | deterministic | yes | 1 |
| matmul | matmul_bf16 | deterministic | yes | 0 |
| matmul | matmul_bf16_gemv | deterministic | yes | 0 |
| attn_prefill | attn_prefill_bf16_flash_v2 | deterministic | yes | 0 |
| attn_prefill | attn_prefill_bf16_splitkv | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 10 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 0 |
| argmax | argmax_f32 | exact | yes | 1 |
| matmul_fp8 | matmul_fp8_gemv_v2 | deterministic | yes | 10 |
| matmul_fp8 | matmul_fp8_mma_f8 | deterministic | yes | 0 |
| matmul_nvfp4 | matmul_nvfp4_gemv_v2 | deterministic | yes | 1 |
| matmul_nvfp4 | matmul_nvfp4_mma | deterministic | yes | 0 |
| moe_topk | moe_topk_bf16_v2 | deterministic | yes | 0 |
| moe_experts | moe_experts_bf16_v2 | deterministic | yes | 0 |
| moe_combine | moe_combine_bf16 | deterministic | yes | 0 |
| sigmoid_mul | sigmoid_mul_bf16 | deterministic | yes | 0 |
| gdn_conv | gdn_conv_bf16 | deterministic | yes | 0 |
| gdn_delta_rule | gdn_delta_rule_bf16_v2 | deterministic | yes | 0 |
| matmul_pair | matmul_pair_bf16_gemv | deterministic | yes | 1 |
| matmul_fp8_pair | matmul_fp8_pair_gemv | deterministic | yes | 30 |
| gated_rmsnorm | gated_rmsnorm_bf16 | deterministic | yes | 0 |
| rmsnorm_add_matmul_pair | rmsnorm_add_matmul_pair_bf16_gemv | deterministic | yes | 69 |
| gated_rmsnorm_matmul_fp8 | gated_rmsnorm_matmul_fp8_gemv | deterministic | yes | 30 |
| rmsnorm_add_matmul_fp8_pair | rmsnorm_add_matmul_fp8_pair_gemv | deterministic | yes | 10 |
| sigmoid_mul_matmul_fp8 | sigmoid_mul_matmul_fp8_gemv | deterministic | yes | 10 |
| gate_split_qk_norm_rope_kv | gate_split_qk_norm_rope_kv_bf16 | deterministic | yes | 10 |
| gdn_conv_delta | gdn_conv_delta_bf16 | deterministic | yes | 30 |
| moe_block | moe_block_bf16 | deterministic | yes | 40 |

- kernels test: 54 kernels, 995 cases in 38.7s (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | head=4d3e38e code-tree=9cc7bec56a44 dirty=no | tatooine | 1792654818 | `/work/on-intarsia doctor --backend hip --device 0 --out /work/onboard-runs/35b-hip-fresh/report/qualify/doctor.json` |
| `recipe` | head=4d3e38e code-tree=9cc7bec56a44 dirty=no | tatooine | 1792654818 | `recipe::analyse(/models/Qwen3.6-35B-A3B-NVFP4)` |
| `plan` | head=4d3e38e code-tree=9cc7bec56a44 dirty=no | tatooine | 1792654818 | `/work/on-intarsia plan show --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --admit --json` |
| `kernels` | head=4d3e38e code-tree=9cc7bec56a44 dirty=no | tatooine | 1792654827 | `/work/on-intarsia kernels test --backend hip --device 0 --model /models/Qwen3.6-35B-A3B-NVFP4` |
| `gate` | head=4d3e38e code-tree=9cc7bec56a44 dirty=no | tatooine | 1792654866 | `/work/on-intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --quick --max-seq 2048` |
| `acc-format` | head=4d3e38e code-tree=9cc7bec56a44 dirty=no | tatooine | 1792654894 | `-` |
| `acc-short` | head=4d3e38e code-tree=9cc7bec56a44 dirty=no | tatooine | 1792654872 | `/work/on-intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --golden /work/onboard-runs/golden-fresh/Qwen3.6-35B-A3B-NVFP4-vllm-top20.safetensors --mode both --decode-positions 16 --max-seq 4096 --write-baseline /work/onboard-runs/35b-hip-fresh/report/qu` |
| `bench` | head=4d3e38e code-tree=9cc7bec56a44 dirty=no | tatooine | 1792654877 | `/work/on-intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3.6-35B-A3B-NVFP4 --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-Qwen3.6-35B-A3B-NVFP4-hip-bench --out-dir /work/onboard-runs/35b-hip-fresh/report/qualify/bench --tp 1 --fingerprint mode` |
| `acc-long` | head=4d3e38e code-tree=9cc7bec56a44 dirty=no | tatooine | 1792654894 | `/work/on-intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --golden /work/onboard-runs/golden-fresh/Qwen3.6-35B-A3B-NVFP4-vllm-long-top20.safetensors --mode both --decode-positions 64 --max-seq 8192 --write-baseline /work/onboard-runs/35b-hip-fresh/repo` |
| `acc-4k` | head=4d3e38e code-tree=9cc7bec56a44 dirty=no | tatooine | 1792654905 | `/work/on-intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --golden /work/onboard-runs/golden-fresh/Qwen3.6-35B-A3B-NVFP4-vllm-4k-top20.safetensors --mode all --decode-positions 64 --max-seq 8192 --write-baseline /work/onboard-runs/35b-hip-fresh/report/` |
| `envelope` | head=4d3e38e code-tree=9cc7bec56a44 dirty=no | tatooine | 1792654944 | `/work/on-intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --golden /work/onboard-runs/golden-fresh/Qwen3.6-35B-A3B-NVFP4-vllm-top20.safetensors --mode both --decode-positions 16 --max-seq 4096 --baseline /work/onboard-runs/35b-hip-fresh/report/qualify/` |
| `bench-long` | head=4d3e38e code-tree=9cc7bec56a44 dirty=no | tatooine | 1792654960 | `/work/on-intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3.6-35B-A3B-NVFP4 --sizes 1024,4096,16384 --gen 64 --reps 2 --warmup 1 --thinking off --label qualify-Qwen3.6-35B-A3B-NVFP4-hip-bench-long --out-dir /work/onboard-runs/35b-hip-fresh/report/qualify/bench --tp 1 --fingerprin` |
| `ab` | head=4d3e38e code-tree=9cc7bec56a44 dirty=no | tatooine | 1792654978 | `-` |
| `tp` | head=4d3e38e code-tree=9cc7bec56a44 dirty=no | tatooine | 1792654978 | `-` |

Model fingerprint: `fnv1a64:9ca75aef5131bb81`.


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/kernels.out` (kernels)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `bench/qualify-Qwen3.6-35B-A3B-NVFP4-hip-bench.json` (bench)
- `logs/bench-bench.out` (bench)
- `logs/bench-server.err` (bench)
- `logs/acc-long.out` (acc-long)
- `baselines/acc-long.baseline.json` (acc-long)
- `logs/acc-4k.out` (acc-4k)
- `baselines/acc-4k.baseline.json` (acc-4k)
- `baselines/acc-short.baseline.json` (envelope)
- `baselines/envelope.json` (envelope)
- `logs/baseline-selfcheck.out` (envelope)
- `bench/qualify-Qwen3.6-35B-A3B-NVFP4-hip-bench-long.json` (bench-long)
- `logs/bench-long-bench.out` (bench-long)
- `logs/bench-long-server.err` (bench-long)
- `state.json` (stage records; `--resume` reads it)
