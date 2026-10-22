# Qualification quick report: Qwen3.6-35B-A3B-NVFP4 on metal

_Quick section, generated unix 1792679551, intarsia 0.0.1 build `head=ec603e1 code-tree=6eac9c190c06 dirty=no`._

## Headline

**Verdict: PASS** (9 stages: 3 pass, 0 fail, 0 error, 6 skipped)

| | |
|---|---|
| Model | Qwen3.6-35B-A3B-NVFP4 (`/Users/josef/Models/Qwen3.6-35B-A3B-NVFP4`) |
| Backend / device | metal / Apple M5 Max (applegpu_g17s) |
| Recipe | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| Decode | not measured |
| Prefill | not measured |
| Accuracy (short golden) | skipped: skipped by --skip |
| Accuracy (format reference, same weights) | skipped: skipped by --skip |
| Quick gate | **SKIP**: skipped by --skip |
| Wall time | quick 5s |

### Failures

None.

### Skipped

- `kernels`: skipped by --skip
- `gate`: skipped by --skip
- `acc-format`: skipped by --skip
- `acc-short`: skipped by --skip
- `acc-oracle`: skipped by --skip
- `bench`: skipped by --skip

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | Apple M5 Max (applegpu_g17s), 39 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 5s | 21 ops resolved, exactness deterministic, loaded in 4.7s |
| 4 | `kernels` kernels test on the model's shapes | quick | SKIP | 0s | skipped by --skip |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | SKIP | 0s | skipped by --skip |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | skipped by --skip |
| 7 | `acc-short` short accuracy against a golden | quick | SKIP | 0s | skipped by --skip |
| 8 | `acc-oracle` short prefill accuracy against the f64 CPU oracle (same weights, deterministic reference) | quick | SKIP | 0s | skipped by --skip |
| 9 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | SKIP | 0s | skipped by --skip |

### Notes

None.

## Speed

No speed numbers.

## Accuracy

**Quick gate**: skipped by --skip

No golden comparison ran (see Skipped).

## Plan and recipe

- recipe: builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) (PASS)
- plan: 21 ops resolved, exactness deterministic, loaded in 4.7s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_bf16 | exact | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 31 |
| rmsnorm_add | rmsnorm_add_bf16 | deterministic | yes | 80 |
| attn_prefill | attn_prefill_bf16_flash | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 10 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 70 |
| argmax | argmax_f32 | exact | yes | 1 |
| matmul_fp8 | matmul_fp8_gemv | deterministic | yes | 130 |
| matmul_fp8 | matmul_fp8_mma | deterministic | yes | 0 |
| matmul_nvfp4 | matmul_nvfp4_gemv | deterministic | yes | 121 |
| matmul_nvfp4 | matmul_nvfp4_mma | deterministic | yes | 0 |
| moe_topk | moe_topk_bf16 | deterministic | yes | 40 |
| moe_experts | moe_experts_bf16 | deterministic | yes | 40 |
| moe_combine | moe_combine_bf16 | deterministic | yes | 40 |
| gate_split | gate_split_bf16 | deterministic | yes | 10 |
| sigmoid_mul | sigmoid_mul_bf16 | deterministic | yes | 10 |
| gdn_conv | gdn_conv_bf16 | deterministic | yes | 30 |
| gdn_delta_rule | gdn_delta_rule_bf16 | deterministic | yes | 30 |
| matmul_pair | matmul_pair_bf16_gemv | deterministic | yes | 70 |
| matmul_pair | matmul_pair_bf16_nax | deterministic | yes | 0 |
| qk_norm_rope_kv | qk_norm_rope_kv_bf16 | deterministic | yes | 10 |

- kernels test: skipped by --skip (SKIP)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | head=ec603e1 code-tree=6eac9c190c06 dirty=no | endor.kuepker.haus | 1792679546 | `/opt/code/wt-gc-target/release/intarsia doctor --backend metal --device 0 --out /opt/code/wt-gc/docs/results/2026-10-22-gc-gate-coverage/raw/metal-acc4k/q-1-clean/doctor.json` |
| `recipe` | head=ec603e1 code-tree=6eac9c190c06 dirty=no | endor.kuepker.haus | 1792679546 | `recipe::analyse(/Users/josef/Models/Qwen3.6-35B-A3B-NVFP4)` |
| `plan` | head=ec603e1 code-tree=6eac9c190c06 dirty=no | endor.kuepker.haus | 1792679547 | `/opt/code/wt-gc-target/release/intarsia plan show --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 45 --admit --json` |
| `kernels` | head=ec603e1 code-tree=6eac9c190c06 dirty=no | endor.kuepker.haus | 0 | `/opt/code/wt-gc-target/release/intarsia plan show --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 45 --admit --json` |
| `gate` | head=ec603e1 code-tree=6eac9c190c06 dirty=no | endor.kuepker.haus | 0 | `/opt/code/wt-gc-target/release/intarsia plan show --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 45 --admit --json` |
| `acc-format` | head=ec603e1 code-tree=6eac9c190c06 dirty=no | endor.kuepker.haus | 0 | `/opt/code/wt-gc-target/release/intarsia plan show --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 45 --admit --json` |
| `acc-short` | head=ec603e1 code-tree=6eac9c190c06 dirty=no | endor.kuepker.haus | 0 | `/opt/code/wt-gc-target/release/intarsia plan show --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 45 --admit --json` |
| `acc-oracle` | head=ec603e1 code-tree=6eac9c190c06 dirty=no | endor.kuepker.haus | 0 | `/opt/code/wt-gc-target/release/intarsia plan show --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 45 --admit --json` |
| `bench` | head=ec603e1 code-tree=6eac9c190c06 dirty=no | endor.kuepker.haus | 0 | `/opt/code/wt-gc-target/release/intarsia plan show --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 45 --admit --json` |

Model fingerprint: `fnv1a64:9ca75aef5131bb81`.


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `state.json` (stage records; `--resume` reads it)
