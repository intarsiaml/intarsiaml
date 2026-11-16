# Qualification quick report: Qwen3.6-35B-A3B-NVFP4 on hip

_Quick section, generated unix 1794761567, intarsia 0.0.1 build `head=4379dd0 code-tree=a5d1066ac581 dirty=no`._

## Headline

**Verdict: MISSING-BASELINE** (9 stages: 4 pass, 0 fail, 2 missing baseline, 0 error, 3 skipped)

| | |
|---|---|
| Model | Qwen3.6-35B-A3B-NVFP4 (`/models/Qwen3.6-35B-A3B-NVFP4`) |
| Backend / device | hip / AMD Radeon Graphics (gfx1201) |
| Recipe | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| Decode | not measured |
| Prefill | not measured |
| Accuracy (short golden) | **MISSING-BASELINE** (prefill top-1 98.47% KL 6.00e-3; decode top-1 98.75% KL 1.11e-2) |
| Accuracy (format reference, same weights) | skipped: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| Quick gate | **PASS**: perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 43s |

### Failures

- `acc-short` **MISSING-BASELINE**: MISSING-BASELINE: no baseline for backend hip next to the golden or in golden/baselines: KL, tail, prefix-reuse gates only, top-1 not gated. Qwen3.6-35B-A3B-NVFP4 has no full-precision golden (MoE / hybrid / quantized, golden from a reference server), so acc-short's floor alone does not gate it (it 
- `acc-oracle` **MISSING-BASELINE**: MISSING-BASELINE: no recorded dump qwen36-35b-a3b-nvfp4-cpu-oracle.hip.selfgolden.safetensors in golden/baselines; the oracle floor passed (prefill top-1 95.77% KL 1.04e-2), but on this model only a recorded same-backend dump can see a 1 % scale bug. Record it from clean main: copy baselines/qwen36-

### Skipped

- `kernels`: skipped by --skip
- `acc-format`: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE)
- `bench`: skipped by --skip

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | AMD Radeon Graphics (gfx1201), 95 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 9s | 31 ops resolved, exactness deterministic, loaded in 9.2s |
| 4 | `kernels` kernels test on the model's shapes | quick | SKIP | 0s | skipped by --skip |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 6s | perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| 7 | `acc-short` short accuracy against a golden | quick | MISSING-BASELINE | 8s | MISSING-BASELINE: no baseline for backend hip next to the golden or in golden/baselines: KL, tail, prefix-reuse gates only, top-1 not gated. Qwen3.6-35B-A3B-NVFP4 has no full-precision golden (MoE / hybrid / quantized, golden from a reference server), so acc-short's floor alone does not gate it (it  |
| 8 | `acc-oracle` short prefill accuracy against the f64 CPU oracle (same weights, deterministic reference) | quick | MISSING-BASELINE | 20s | MISSING-BASELINE: no recorded dump qwen36-35b-a3b-nvfp4-cpu-oracle.hip.selfgolden.safetensors in golden/baselines; the oracle floor passed (prefill top-1 95.77% KL 1.04e-2), but on this model only a recorded same-backend dump can see a 1 % scale bug. Record it from clean main: copy baselines/qwen36- |
| 9 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | SKIP | 0s | skipped by --skip |

### Notes

None.

## Speed

No speed numbers.

## Accuracy

**acc-short** against `/models/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors` (given with --golden; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1441 | 98.47 | 91.70 | 6.00e-3 | 5.69e-3 | 6.19e-1 | 1.35e-3 | 0 | 6.7 | pass |
| decode | 160 | 98.75 | 93.38 | 1.11e-2 | 1.11e-2 | 7.31e-1 | 2.26e-3 | 0 | 88.8 | pass |

**Quick gate**: perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) (PASS)
- plan: 31 ops resolved, exactness deterministic, loaded in 9.2s (PASS)

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
| gdn_delta_rule | gdn_delta_rule_bf16_chunk | deterministic | yes | 0 |
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

- kernels test: skipped by --skip (SKIP)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | head=4379dd0 code-tree=a5d1066ac581 dirty=no | tatooine | 1794761524 | `/target/release/intarsia doctor --backend hip --device 0 --out /work/.gdn-default-logs/1115-1139/q-rec/doctor.json` |
| `recipe` | head=4379dd0 code-tree=a5d1066ac581 dirty=no | tatooine | 1794761524 | `recipe::analyse(/models/Qwen3.6-35B-A3B-NVFP4)` |
| `plan` | head=4379dd0 code-tree=a5d1066ac581 dirty=no | tatooine | 1794761525 | `/target/release/intarsia plan show --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --admit --json` |
| `kernels` | head=4379dd0 code-tree=a5d1066ac581 dirty=no | tatooine | 0 | `/target/release/intarsia plan show --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --admit --json` |
| `gate` | head=4379dd0 code-tree=a5d1066ac581 dirty=no | tatooine | 1794761534 | `/target/release/intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --quick --max-seq 2048` |
| `acc-format` | head=4379dd0 code-tree=a5d1066ac581 dirty=no | tatooine | 1794761540 | `-` |
| `acc-short` | head=4379dd0 code-tree=a5d1066ac581 dirty=no | tatooine | 1794761540 | `/target/release/intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --golden /models/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --decode-positions 16 --max-seq 4096 --write-baseline /work/.gdn-default-logs/1115-1139/q-rec/baselines/acc` |
| `acc-oracle` | head=4379dd0 code-tree=a5d1066ac581 dirty=no | tatooine | 1794761547 | `/target/release/intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --golden /work/golden/oracle/qwen36-35b-a3b-nvfp4-cpu-oracle.safetensors --mode prefill --decode-positions 16 --max-seq 4352 --write-baseline /work/.gdn-default-logs/1115-1139/q-rec/basel` |
| `bench` | head=4379dd0 code-tree=a5d1066ac581 dirty=no | tatooine | 0 | `/target/release/intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --golden /work/golden/oracle/qwen36-35b-a3b-nvfp4-cpu-oracle.safetensors --mode prefill --decode-positions 16 --max-seq 4352 --write-baseline /work/.gdn-default-logs/1115-1139/q-rec/basel` |

Model fingerprint: `fnv1a64:9ca75aef5131bb81`.


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `logs/acc-oracle.out` (acc-oracle)
- `baselines/acc-oracle.baseline.json` (acc-oracle)
- `baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.hip.selfgolden.safetensors` (acc-oracle)
- `state.json` (stage records; `--resume` reads it)
