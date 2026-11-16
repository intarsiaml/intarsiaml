# Qualification report: Qwen3.8-27B-Quark-AWQ-MXFP4 on hip

_Full (quick + in-depth) section, generated unix 1794797761, intarsia 0.0.1 build `head=fd8a3af code-tree=9ccac5d2913f dirty=yes`._

## Headline

**Verdict: PASS** (15 stages: 4 pass, 0 fail, 0 error, 11 skipped)

| | |
|---|---|
| Model | Qwen3.8-27B-Quark-AWQ-MXFP4 (`/models/Qwen3.8-27B-Quark-AWQ-MXFP4`) |
| Backend / device | hip / AMD Radeon Graphics (gfx1201) |
| Recipe | builtin:qwen3_5_vl covers this model (45/45 tensor templates resolve, quant mxfp4) |
| Decode | not measured |
| Prefill | not measured |
| Accuracy (short golden) | skipped: skipped by --skip |
| Accuracy (format reference, same weights) | skipped: skipped by --skip |
| Accuracy (long golden) | skipped: skipped by --skip |
| Accuracy (4k golden) | skipped: skipped by --skip |
| Quick gate | **SKIP**: skipped by --skip |
| Wall time | quick 5m26s, in-depth 0s, total 5m26s |

### Failures

None.

### Skipped

- `gate`: skipped by --skip
- `acc-format`: skipped by --skip
- `acc-short`: skipped by --skip
- `acc-oracle`: skipped by --skip
- `bench`: skipped by --skip
- `acc-long`: skipped by --skip
- `acc-4k`: skipped by --skip
- `envelope`: skipped by --skip
- `bench-long`: skipped by --skip
- `ab`: skipped by --skip
- `tp`: skipped by --skip

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | AMD Radeon Graphics (gfx1201), 95 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_vl covers this model (45/45 tensor templates resolve, quant mxfp4) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 1m38s | 29 ops resolved, exactness deterministic, loaded in 98.5s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 3m48s | 91 kernels, 2219 cases in 227.9s; incomplete: 10 kernel(s) had no cases in this sweep (untested, not failed; plan-time admission tests them) |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | SKIP | 0s | skipped by --skip |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | skipped by --skip |
| 7 | `acc-short` short accuracy against a golden | quick | SKIP | 0s | skipped by --skip |
| 8 | `acc-oracle` short prefill accuracy against the f64 CPU oracle (same weights, deterministic reference) | quick | SKIP | 0s | skipped by --skip |
| 9 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | SKIP | 0s | skipped by --skip |
| 10 | `acc-long` accuracy --mode all on the long golden (prefill, decode, prefix reuse) | full | SKIP | 0s | skipped by --skip |
| 11 | `acc-4k` accuracy --mode all on the 4k golden (prefill, decode, prefix reuse past 2 chunks of 2048) | full | SKIP | 0s | skipped by --skip |
| 12 | `envelope` noise envelope and baseline for regression gating | full | SKIP | 0s | skipped by --skip |
| 13 | `bench-long` long-context bench up to 16k | full | SKIP | 0s | skipped by --skip |
| 14 | `ab` A/B against the reference server via localeval | full | SKIP | 0s | skipped by --skip |
| 15 | `tp` tensor-parallel variant | full | SKIP | 0s | skipped by --skip |

### Notes

None.

## Speed

No speed numbers.

## Accuracy

**Quick gate**: skipped by --skip

No golden comparison ran (see Skipped).

## Plan and recipe

- recipe: builtin:qwen3_5_vl covers this model (45/45 tensor templates resolve, quant mxfp4) (PASS)
- plan: 29 ops resolved, exactness deterministic, loaded in 98.5s (PASS)

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
| attn_decode | attn_decode_bf16_splitkv_rows | deterministic | yes | 0 |
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

- kernels test: 91 kernels, 2219 cases in 227.9s; incomplete: 10 kernel(s) had no cases in this sweep (untested, not failed; plan-time admission tests them) (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | head=fd8a3af code-tree=9ccac5d2913f dirty=yes | tatooine | 1794797434 | `/target/release/intarsia doctor --backend hip --device 0 --out /work/.q27-logs/w1/qual/doctor.json` |
| `recipe` | head=fd8a3af code-tree=9ccac5d2913f dirty=yes | tatooine | 1794797434 | `recipe::analyse(/models/Qwen3.8-27B-Quark-AWQ-MXFP4)` |
| `plan` | head=fd8a3af code-tree=9ccac5d2913f dirty=yes | tatooine | 1794797434 | `/target/release/intarsia plan show --model /models/Qwen3.8-27B-Quark-AWQ-MXFP4 --backend hip --device 0 --mem-budget 29 --admit --json` |
| `kernels` | head=fd8a3af code-tree=9ccac5d2913f dirty=yes | tatooine | 1794797533 | `/target/release/intarsia kernels test --backend hip --device 0 --model /models/Qwen3.8-27B-Quark-AWQ-MXFP4` |
| `gate` | head=fd8a3af code-tree=9ccac5d2913f dirty=yes | tatooine | 0 | `/target/release/intarsia kernels test --backend hip --device 0 --model /models/Qwen3.8-27B-Quark-AWQ-MXFP4` |
| `acc-format` | head=fd8a3af code-tree=9ccac5d2913f dirty=yes | tatooine | 0 | `/target/release/intarsia kernels test --backend hip --device 0 --model /models/Qwen3.8-27B-Quark-AWQ-MXFP4` |
| `acc-short` | head=fd8a3af code-tree=9ccac5d2913f dirty=yes | tatooine | 0 | `/target/release/intarsia kernels test --backend hip --device 0 --model /models/Qwen3.8-27B-Quark-AWQ-MXFP4` |
| `acc-oracle` | head=fd8a3af code-tree=9ccac5d2913f dirty=yes | tatooine | 0 | `/target/release/intarsia kernels test --backend hip --device 0 --model /models/Qwen3.8-27B-Quark-AWQ-MXFP4` |
| `bench` | head=fd8a3af code-tree=9ccac5d2913f dirty=yes | tatooine | 0 | `/target/release/intarsia kernels test --backend hip --device 0 --model /models/Qwen3.8-27B-Quark-AWQ-MXFP4` |
| `acc-long` | head=fd8a3af code-tree=9ccac5d2913f dirty=yes | tatooine | 0 | `/target/release/intarsia kernels test --backend hip --device 0 --model /models/Qwen3.8-27B-Quark-AWQ-MXFP4` |
| `acc-4k` | head=fd8a3af code-tree=9ccac5d2913f dirty=yes | tatooine | 0 | `/target/release/intarsia kernels test --backend hip --device 0 --model /models/Qwen3.8-27B-Quark-AWQ-MXFP4` |
| `envelope` | head=fd8a3af code-tree=9ccac5d2913f dirty=yes | tatooine | 0 | `/target/release/intarsia kernels test --backend hip --device 0 --model /models/Qwen3.8-27B-Quark-AWQ-MXFP4` |
| `bench-long` | head=fd8a3af code-tree=9ccac5d2913f dirty=yes | tatooine | 0 | `/target/release/intarsia kernels test --backend hip --device 0 --model /models/Qwen3.8-27B-Quark-AWQ-MXFP4` |
| `ab` | head=fd8a3af code-tree=9ccac5d2913f dirty=yes | tatooine | 0 | `/target/release/intarsia kernels test --backend hip --device 0 --model /models/Qwen3.8-27B-Quark-AWQ-MXFP4` |
| `tp` | head=fd8a3af code-tree=9ccac5d2913f dirty=yes | tatooine | 0 | `/target/release/intarsia kernels test --backend hip --device 0 --model /models/Qwen3.8-27B-Quark-AWQ-MXFP4` |

Model fingerprint: `fnv1a64:c493d7657945124b`.


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/kernels.out` (kernels)
- `state.json` (stage records; `--resume` reads it)
