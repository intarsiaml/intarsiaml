# Qualification quick report: Qwen3.8-27B-Quark-AWQ-MXFP4 on hip

_Quick section, generated unix 1794796057, intarsia 0.0.1 build `head=f219eec code-tree=aa4946c84c9a dirty=no`._

## Headline

**Verdict: FAIL** (9 stages: 2 pass, 1 fail, 0 error, 6 skipped)

| | |
|---|---|
| Model | Qwen3.8-27B-Quark-AWQ-MXFP4 (`/models/Qwen3.8-27B-Quark-AWQ-MXFP4`) |
| Backend / device | hip / AMD Radeon Graphics (gfx1201) |
| Recipe | builtin:qwen3_5_vl covers this model (45/45 tensor templates resolve, quant mxfp4) |
| Decode | not measured |
| Prefill | not measured |
| Accuracy (short golden) | skipped: blocked by 'plan' (FAIL): 1 op(s) selected a kernel that did not pass admission: attn_decode:attn_decode_bf16_splitkv_rows |
| Accuracy (format reference, same weights) | skipped: skipped by --skip |
| Quick gate | **SKIP**: blocked by 'plan' (FAIL): 1 op(s) selected a kernel that did not pass admission: attn_decode:attn_decode_bf16_splitkv_rows |
| Wall time | quick 2m04s |

### Failures

- `plan` **FAIL**: 1 op(s) selected a kernel that did not pass admission: attn_decode:attn_decode_bf16_splitkv_rows

### Skipped

- `kernels`: skipped by --skip
- `gate`: blocked by 'plan' (FAIL): 1 op(s) selected a kernel that did not pass admission: attn_decode:attn_decode_bf16_splitkv_rows
- `acc-format`: skipped by --skip
- `acc-short`: blocked by 'plan' (FAIL): 1 op(s) selected a kernel that did not pass admission: attn_decode:attn_decode_bf16_splitkv_rows
- `acc-oracle`: skipped by --skip
- `bench`: skipped by --skip

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | AMD Radeon Graphics (gfx1201), 95 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_vl covers this model (45/45 tensor templates resolve, quant mxfp4) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | FAIL | 2m04s | 1 op(s) selected a kernel that did not pass admission: attn_decode:attn_decode_bf16_splitkv_rows |
| 4 | `kernels` kernels test on the model's shapes | quick | SKIP | 0s | skipped by --skip |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | SKIP | 0s | blocked by 'plan' (FAIL): 1 op(s) selected a kernel that did not pass admission: attn_decode:attn_decode_bf16_splitkv_rows |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | skipped by --skip |
| 7 | `acc-short` short accuracy against a golden | quick | SKIP | 0s | blocked by 'plan' (FAIL): 1 op(s) selected a kernel that did not pass admission: attn_decode:attn_decode_bf16_splitkv_rows |
| 8 | `acc-oracle` short prefill accuracy against the f64 CPU oracle (same weights, deterministic reference) | quick | SKIP | 0s | skipped by --skip |
| 9 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | SKIP | 0s | skipped by --skip |

### Notes

None.

## Speed

No speed numbers.

## Accuracy

**Quick gate**: blocked by 'plan' (FAIL): 1 op(s) selected a kernel that did not pass admission: attn_decode:attn_decode_bf16_splitkv_rows

No golden comparison ran (see Skipped).

## Plan and recipe

- recipe: builtin:qwen3_5_vl covers this model (45/45 tensor templates resolve, quant mxfp4) (PASS)
- plan: 1 op(s) selected a kernel that did not pass admission: attn_decode:attn_decode_bf16_splitkv_rows (FAIL)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_bf16 | exact | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 1 |
| rmsnorm_add | rmsnorm_add_bf16 | deterministic | yes | 128 |
| add | add_bf16 | deterministic | yes | 0 |
| matmul | matmul_bf16 | deterministic | yes | 0 |
| matmul | matmul_bf16_gemv | deterministic | yes | 0 |
| attn_prefill | attn_prefill_bf16_flash_v2 | deterministic | yes | 0 |
| attn_prefill | attn_prefill_bf16_splitkv | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 16 |
| attn_decode | attn_decode_bf16_splitkv_rows | deterministic | no | 0 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 64 |
| argmax | argmax_f32 | exact | yes | 1 |
| matmul_fp8 | matmul_fp8_gemv_rows | deterministic | yes | 0 |
| matmul_fp8 | matmul_fp8_gemv_v2 | deterministic | yes | 1 |
| matmul_fp8 | matmul_fp8_mma_f8 | deterministic | yes | 0 |
| sigmoid_mul | sigmoid_mul_bf16 | deterministic | yes | 16 |
| gdn_conv | gdn_conv_bf16 | deterministic | yes | 0 |
| gdn_delta_rule | gdn_delta_rule_bf16_chunk | deterministic | yes | 0 |
| gdn_delta_rule | gdn_delta_rule_bf16_v2 | deterministic | yes | 0 |
| matmul_pair | matmul_pair_bf16_gemv | deterministic | yes | 0 |
| gated_rmsnorm | gated_rmsnorm_bf16 | deterministic | yes | 48 |
| gate_split_qk_norm_rope_kv | gate_split_qk_norm_rope_kv_bf16 | deterministic | yes | 16 |
| gdn_conv_delta | gdn_conv_delta_bf16 | deterministic | yes | 48 |
| matmul_mxfp4 | matmul_mxfp4_a8_gemv | deterministic | yes | 144 |
| matmul_mxfp4 | matmul_mxfp4_a8_gemv_rows | deterministic | yes | 0 |
| matmul_mxfp4 | matmul_mxfp4_a8_mma | deterministic | yes | 0 |
| matmul_mxfp4_pair | matmul_mxfp4_pair_a8_gemv | deterministic | yes | 176 |
| matmul_mxfp4_pair | matmul_mxfp4_pair_a8_gemv_rows | deterministic | yes | 0 |
| matmul_mxfp4_pair | matmul_mxfp4_pair_a8_mma | deterministic | yes | 0 |

- kernels test: skipped by --skip (SKIP)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | head=f219eec code-tree=aa4946c84c9a dirty=no | tatooine | 1794795932 | `/target/release/intarsia doctor --backend hip --device 0 --out /work/.b27-logs/main/q-rec-w4a4/doctor.json` |
| `recipe` | head=f219eec code-tree=aa4946c84c9a dirty=no | tatooine | 1794795933 | `recipe::analyse(/models/Qwen3.8-27B-Quark-AWQ-MXFP4)` |
| `plan` | head=f219eec code-tree=aa4946c84c9a dirty=no | tatooine | 1794795933 | `/target/release/intarsia plan show --model /models/Qwen3.8-27B-Quark-AWQ-MXFP4 --backend hip --device 0 --mem-budget 29 --admit --json` |
| `kernels` | head=f219eec code-tree=aa4946c84c9a dirty=no | tatooine | 0 | `/target/release/intarsia plan show --model /models/Qwen3.8-27B-Quark-AWQ-MXFP4 --backend hip --device 0 --mem-budget 29 --admit --json` |
| `gate` | head=f219eec code-tree=aa4946c84c9a dirty=no | tatooine | 0 | `/target/release/intarsia plan show --model /models/Qwen3.8-27B-Quark-AWQ-MXFP4 --backend hip --device 0 --mem-budget 29 --admit --json` |
| `acc-format` | head=f219eec code-tree=aa4946c84c9a dirty=no | tatooine | 0 | `/target/release/intarsia plan show --model /models/Qwen3.8-27B-Quark-AWQ-MXFP4 --backend hip --device 0 --mem-budget 29 --admit --json` |
| `acc-short` | head=f219eec code-tree=aa4946c84c9a dirty=no | tatooine | 0 | `/target/release/intarsia plan show --model /models/Qwen3.8-27B-Quark-AWQ-MXFP4 --backend hip --device 0 --mem-budget 29 --admit --json` |
| `acc-oracle` | head=f219eec code-tree=aa4946c84c9a dirty=no | tatooine | 0 | `/target/release/intarsia plan show --model /models/Qwen3.8-27B-Quark-AWQ-MXFP4 --backend hip --device 0 --mem-budget 29 --admit --json` |
| `bench` | head=f219eec code-tree=aa4946c84c9a dirty=no | tatooine | 0 | `/target/release/intarsia plan show --model /models/Qwen3.8-27B-Quark-AWQ-MXFP4 --backend hip --device 0 --mem-budget 29 --admit --json` |

Model fingerprint: `fnv1a64:c493d7657945124b`.


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `state.json` (stage records; `--resume` reads it)
