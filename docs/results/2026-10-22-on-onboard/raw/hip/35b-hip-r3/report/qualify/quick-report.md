# Qualification quick report: Qwen3.6-35B-A3B-NVFP4 on hip

_Quick section, generated unix 1792640924, intarsia 0.0.1 build `head=3661de9 code-tree=ad6b9ff260d1 dirty=no`._

## Headline

**Verdict: PASS** (8 stages: 7 pass, 0 fail, 0 error, 1 skipped)

| | |
|---|---|
| Model | Qwen3.6-35B-A3B-NVFP4 (`/models/Qwen3.6-35B-A3B-NVFP4`) |
| Backend / device | hip / AMD Radeon Graphics (gfx1201) |
| Recipe | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| Decode | 128 tok: **160.3** tok/s (58% of floor); 1024 tok: **160.0** tok/s (58% of floor); 4096 tok: **155.4** tok/s (56% of floor) |
| Prefill | 128 tok: **1968** tok/s; 1024 tok: **6535** tok/s; 4096 tok: **8684** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 98.61% KL 7.63e-3; decode top-1 98.75% KL 1.18e-2) |
| Accuracy (format reference, same weights) | skipped: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| Quick gate | **PASS**: perplexity 1.21, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 80s |

### Failures

None.

### Skipped

- `acc-format`: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE)

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | AMD Radeon Graphics (gfx1201), 54 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 9s | 30 ops resolved, exactness deterministic, loaded in 9.3s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 39s | 54 kernels, 995 cases in 38.7s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 6s | perplexity 1.21, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| 7 | `acc-short` short accuracy against a golden | quick | PASS | 8s | prefill top-1 98.61% KL 7.63e-3; decode top-1 98.75% KL 1.18e-2 |
| 8 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 17s | decode 160.3 tok/s at 128 tokens (58% of the byte floor) |

### Notes

None.

## Speed

**quick bench** (floor: 2.25 GB/token at 627 GB/s; gb/token: estimated from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 627 GB/s, the R9700 figure of docs/results/2026-10-13-phase2a-hip.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 127 | 0.066 | 1968 | 160.32 | 0.36 | 278.1 | 57.6 |
| 987 | 0.149 | 6535 | 160.01 | 0.13 | 278.1 | 57.5 |
| 4141 | 0.477 | 8684 | 155.38 | 0.14 | 278.1 | 55.9 |

## Accuracy

**acc-short** against `golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1441 | 98.61 | 91.30 | 7.63e-3 | 7.42e-3 | 7.11e-1 | 1.80e-3 | 0 | 31.0 | pass |
| decode | 160 | 98.75 | 92.88 | 1.18e-2 | 1.17e-2 | 7.31e-1 | 2.27e-3 | 0 | 88.8 | pass |

**Quick gate**: perplexity 1.21, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) (PASS)
- plan: 30 ops resolved, exactness deterministic, loaded in 9.3s (PASS)

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
| `doctor` | head=3661de9 code-tree=ad6b9ff260d1 dirty=no | tatooine | 1792640844 | `/work/on-intarsia doctor --backend hip --device 0 --out /work/onboard-runs/35b-hip-r3/report/qualify/doctor.json` |
| `recipe` | head=3661de9 code-tree=ad6b9ff260d1 dirty=no | tatooine | 1792640845 | `recipe::analyse(/models/Qwen3.6-35B-A3B-NVFP4)` |
| `plan` | head=3661de9 code-tree=ad6b9ff260d1 dirty=no | tatooine | 1792640845 | `/work/on-intarsia plan show --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --admit --json` |
| `kernels` | head=3661de9 code-tree=ad6b9ff260d1 dirty=no | tatooine | 1792640854 | `/work/on-intarsia kernels test --backend hip --device 0 --model /models/Qwen3.6-35B-A3B-NVFP4` |
| `gate` | head=3661de9 code-tree=ad6b9ff260d1 dirty=no | tatooine | 1792640893 | `/work/on-intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --quick --max-seq 2048` |
| `acc-format` | head=3661de9 code-tree=ad6b9ff260d1 dirty=no | tatooine | 1792640924 | `-` |
| `acc-short` | head=3661de9 code-tree=ad6b9ff260d1 dirty=no | tatooine | 1792640899 | `/work/on-intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --golden golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --decode-positions 16 --max-seq 4096 --write-baseline /work/onboard-runs/35b-hip-r3/report/qualify/baselines/acc-short.bas` |
| `bench` | head=3661de9 code-tree=ad6b9ff260d1 dirty=no | tatooine | 1792640907 | `/work/on-intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3.6-35B-A3B-NVFP4 --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-Qwen3.6-35B-A3B-NVFP4-hip-bench --out-dir /work/onboard-runs/35b-hip-r3/report/qualify/bench --tp 1 --fingerprint model=Q` |

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
- `state.json` (stage records; `--resume` reads it)
