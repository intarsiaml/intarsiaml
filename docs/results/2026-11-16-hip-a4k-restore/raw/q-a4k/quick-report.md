# Qualification quick report: Qwen3.6-35B-A3B-NVFP4 on hip

_Quick section, generated unix 1794790693, intarsia 0.0.1 build `head=f2b1056 code-tree=a04365425877 dirty=no`._

## Headline

**Verdict: PASS** (9 stages: 3 pass, 0 fail, 0 error, 6 skipped)

| | |
|---|---|
| Model | Qwen3.6-35B-A3B-NVFP4 (`/models/Qwen3.6-35B-A3B-NVFP4`) |
| Backend / device | hip / AMD Radeon Graphics (gfx1201) |
| Recipe | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| Decode | not measured |
| Prefill | not measured |
| Accuracy (short golden) | skipped: skipped by --skip |
| Accuracy (format reference, same weights) | skipped: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| Quick gate | **SKIP**: skipped by --skip |
| Wall time | quick 10s |

### Failures

None.

### Skipped

- `kernels`: skipped by --skip
- `gate`: skipped by --skip
- `acc-format`: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE)
- `acc-short`: skipped by --skip
- `acc-oracle`: skipped by --skip
- `bench`: skipped by --skip

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | AMD Radeon Graphics (gfx1201), 95 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 9s | 31 ops resolved, exactness deterministic, loaded in 9.5s |
| 4 | `kernels` kernels test on the model's shapes | quick | SKIP | 0s | skipped by --skip |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | SKIP | 0s | skipped by --skip |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
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
- plan: 31 ops resolved, exactness deterministic, loaded in 9.5s (PASS)

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
| `doctor` | head=f2b1056 code-tree=a04365425877 dirty=no | tatooine | 1794790682 | `/target/release/intarsia doctor --backend hip --device 0 --out /work/.a4k-restore-logs/w1/q-a4k/doctor.json` |
| `recipe` | head=f2b1056 code-tree=a04365425877 dirty=no | tatooine | 1794790683 | `recipe::analyse(/models/Qwen3.6-35B-A3B-NVFP4)` |
| `plan` | head=f2b1056 code-tree=a04365425877 dirty=no | tatooine | 1794790683 | `/target/release/intarsia plan show --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --admit --json` |
| `kernels` | head=f2b1056 code-tree=a04365425877 dirty=no | tatooine | 0 | `/target/release/intarsia plan show --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --admit --json` |
| `gate` | head=f2b1056 code-tree=a04365425877 dirty=no | tatooine | 0 | `/target/release/intarsia plan show --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --admit --json` |
| `acc-format` | head=f2b1056 code-tree=a04365425877 dirty=no | tatooine | 1794790693 | `-` |
| `acc-short` | head=f2b1056 code-tree=a04365425877 dirty=no | tatooine | 0 | `-` |
| `acc-oracle` | head=f2b1056 code-tree=a04365425877 dirty=no | tatooine | 0 | `-` |
| `bench` | head=f2b1056 code-tree=a04365425877 dirty=no | tatooine | 0 | `-` |

Model fingerprint: `fnv1a64:9ca75aef5131bb81`.


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `state.json` (stage records; `--resume` reads it)
