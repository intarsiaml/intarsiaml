# Qualification quick report: Qwen3.6-35B-A3B-NVFP4 on hip

_Quick section, generated unix 1792730293, intarsia 0.0.1 build `unstamped`._

## Headline

**Verdict: PASS** (9 stages: 8 pass, 0 fail, 0 error, 1 skipped)

| | |
|---|---|
| Model | Qwen3.6-35B-A3B-NVFP4 (`/models/Qwen3.6-35B-A3B-NVFP4`) |
| Backend / device | hip / AMD Radeon Graphics (gfx1201) |
| Recipe | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| Decode | 128 tok: **160.5** tok/s (58% of floor); 1024 tok: **160.0** tok/s (58% of floor); 4096 tok: **155.0** tok/s (56% of floor) |
| Prefill | 128 tok: **1939** tok/s; 1024 tok: **7942** tok/s; 4096 tok: **8889** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 98.61% KL 7.63e-3; decode top-1 98.75% KL 1.18e-2) |
| Accuracy (format reference, same weights) | skipped: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| Quick gate | **PASS**: perplexity 1.21, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 2m03s |

### Failures

None.

### Skipped

- `acc-format`: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE)

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | AMD Radeon Graphics (gfx1201), 55 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 9s | 30 ops resolved, exactness deterministic, loaded in 9.2s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 41s | 55 kernels, 1079 cases in 40.8s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 6s | perplexity 1.21, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| 7 | `acc-short` short accuracy against a golden | quick | PASS | 8s | prefill top-1 98.61% KL 7.63e-3; decode top-1 98.75% KL 1.18e-2 |
| 8 | `acc-oracle` short prefill accuracy against the f64 CPU oracle (same weights, deterministic reference) | quick | PASS | 36s | prefill top-1 95.86% KL 1.07e-2 |
| 9 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 23s | decode 160.5 tok/s at 128 tokens (58% of the byte floor) |

### Notes

- `acc-short`: gated against the recorded baseline /work/golden/baselines/qwen36-35b-a3b-nvfp4-vllm-top20.hip.baseline.json (commit unstamped) and its noise envelope
- `acc-oracle`: floor: mean KL(oracle \|\| engine) over the oracle's top-64 + tail, scored prefill rows, <= 2.0e-2 (measured, docs/results/2026-10-22-fx1-oracle-gate.md); top-1 not gated. The floor guards absolute accuracy; it does not see a 1 % scale bug, the drift check does
- `acc-oracle`: drift check: logits unchanged against the recorded dump golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.hip.selfgolden.safetensors

## Speed

**quick bench** (floor: 2.25 GB/token at 627 GB/s; gb/token: estimated from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 627 GB/s, the R9700 figure of docs/results/2026-10-13-phase2a-hip.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 127 | 0.065 | 1939 | 160.54 | 0.41 | 278.1 | 57.7 |
| 1019 | 0.130 | 7942 | 159.99 | 0.05 | 278.1 | 57.5 |
| 4207 | 0.472 | 8889 | 155.04 | 0.15 | 278.1 | 55.7 |

## Accuracy

**acc-short** against `/models/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1441 | 98.61 | 91.30 | 7.63e-3 | 7.42e-3 | 7.11e-1 | 1.80e-3 | 0 | 31.0 | pass |
| decode | 160 | 98.75 | 92.88 | 1.18e-2 | 1.17e-2 | 7.31e-1 | 2.27e-3 | 0 | 88.8 | pass |

**Quick gate**: perplexity 1.21, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) (PASS)
- plan: 30 ops resolved, exactness deterministic, loaded in 9.2s (PASS)

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

- kernels test: 55 kernels, 1079 cases in 40.8s (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | unstamped | tatooine | 1792730170 | `/target/release/intarsia doctor --backend hip --device 0 --out /work/bench-out/qn/qual-35b/doctor.json` |
| `recipe` | unstamped | tatooine | 1792730170 | `recipe::analyse(/models/Qwen3.6-35B-A3B-NVFP4)` |
| `plan` | unstamped | tatooine | 1792730171 | `/target/release/intarsia plan show --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --admit --json` |
| `kernels` | unstamped | tatooine | 1792730180 | `/target/release/intarsia kernels test --backend hip --device 0 --model /models/Qwen3.6-35B-A3B-NVFP4` |
| `gate` | unstamped | tatooine | 1792730221 | `/target/release/intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --quick --max-seq 2048` |
| `acc-format` | unstamped | tatooine | 1792730227 | `-` |
| `acc-short` | unstamped | tatooine | 1792730227 | `/target/release/intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --golden /models/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --decode-positions 16 --max-seq 4096 --write-baseline /work/bench-out/qn/qual-35b/baselines/acc-short.basel` |
| `acc-oracle` | unstamped | tatooine | 1792730235 | `/target/release/intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --self-golden golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.hip.selfgolden.safetensors --mode prefill --decode-positions 16 --max-seq 4096 --model-id Qwen3.6-35B-A3B-NVFP4 --allow-same` |
| `bench` | unstamped | tatooine | 1792730271 | `/target/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3.6-35B-A3B-NVFP4 --sizes 128,1024,4096 --gen 128 --reps 5 --warmup 1 --thinking off --label qualify-Qwen3.6-35B-A3B-NVFP4-hip-bench --out-dir /work/bench-out/qn/qual-35b/bench --tp 1 --fingerprint model=Qwen3.6-35B` |

Model fingerprint: `fnv1a64:9ca75aef5131bb81`.

Build stamp `unstamped` means a plain `cargo build`: baselines written here gate this binary, but `--pre-change` provenance checks refuse them as a "before".


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/kernels.out` (kernels)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `logs/acc-oracle.out` (acc-oracle)
- `baselines/acc-oracle.baseline.json` (acc-oracle)
- `baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.hip.selfgolden.safetensors` (acc-oracle)
- `bench/qualify-Qwen3.6-35B-A3B-NVFP4-hip-bench.json` (bench)
- `logs/bench-bench.out` (bench)
- `logs/bench-server.err` (bench)
- `state.json` (stage records; `--resume` reads it)
