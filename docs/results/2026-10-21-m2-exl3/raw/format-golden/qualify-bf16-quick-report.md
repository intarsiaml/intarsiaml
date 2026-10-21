# Qualification quick report: Qwen3-8B on cuda

_Quick section, generated unix 1792606130, intarsia 0.0.1 build `m2-fg-d65dfe10`._

## Headline

**Verdict: PASS** (8 stages: 3 pass, 0 fail, 0 error, 5 skipped)

| | |
|---|---|
| Model | Qwen3-8B (`/mnt/models/Qwen3-8B`) |
| Backend / device | cuda / NVIDIA GB10 (sm_121) |
| Recipe | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| Decode | not measured |
| Prefill | not measured |
| Accuracy (short golden) | skipped: skipped by --skip |
| Accuracy (format reference, same weights) | skipped: the weights are not quantized: no format reference applies |
| Quick gate | **SKIP**: skipped by --skip |
| Wall time | quick 1m59s |

### Failures

None.

### Skipped

- `kernels`: skipped by --skip
- `gate`: skipped by --skip
- `acc-short`: skipped by --skip
- `acc-format`: the weights are not quantized: no format reference applies
- `bench`: skipped by --skip

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 3s | NVIDIA GB10 (sm_121), 54 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 1m57s | 16 ops resolved, exactness fast, loaded in 117.3s |
| 4 | `kernels` kernels test on the model's shapes | quick | SKIP | 0s | skipped by --skip |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | SKIP | 0s | skipped by --skip |
| 6 | `acc-short` short accuracy against a golden | quick | SKIP | 0s | skipped by --skip |
| 7 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | the weights are not quantized: no format reference applies |
| 8 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | SKIP | 0s | skipped by --skip |

### Notes

None.

## Speed

No speed numbers.

## Accuracy

**Quick gate**: skipped by --skip

No golden comparison ran (see Skipped).

## Plan and recipe

- recipe: builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) (PASS)
- plan: 16 ops resolved, exactness fast, loaded in 117.3s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_bf16 | exact | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 1 |
| rmsnorm_add | rmsnorm_add_bf16 | deterministic | yes | 72 |
| qk_norm | qk_norm_bf16 | deterministic | yes | 72 |
| rope | rope_bf16 | deterministic | yes | 72 |
| matmul | matmul_bf16 | fast | yes | 0 |
| matmul | matmul_bf16_gemv | deterministic | yes | 108 |
| logits | logits_bf16 | fast | yes | 0 |
| logits | logits_bf16_gemv | deterministic | yes | 1 |
| kv_append | kv_append_bf16 | exact | yes | 72 |
| attn_prefill | attn_prefill_bf16_flash | deterministic | yes | 0 |
| attn_prefill | attn_prefill_bf16_splitkv | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 36 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 36 |
| argmax | argmax_f32 | exact | yes | 1 |
| matmul_pair | matmul_pair_bf16_gemv | deterministic | yes | 72 |

- kernels test: skipped by --skip (SKIP)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | m2-fg-d65dfe10 | spark-4e77 | 1792606010 | `/home/nvidia/intarsia-decfuse/m2-fg/target/acc/release/intarsia doctor --backend cuda --device 0 --out /home/nvidia/m2-fg-qualify/bf16/doctor.json` |
| `recipe` | m2-fg-d65dfe10 | spark-4e77 | 1792606013 | `recipe::analyse(/mnt/models/Qwen3-8B)` |
| `plan` | m2-fg-d65dfe10 | spark-4e77 | 1792606013 | `/home/nvidia/intarsia-decfuse/m2-fg/target/acc/release/intarsia plan show --model /mnt/models/Qwen3-8B --backend cuda --device 0 --admit --json` |
| `kernels` | m2-fg-d65dfe10 | spark-4e77 | 0 | `/home/nvidia/intarsia-decfuse/m2-fg/target/acc/release/intarsia plan show --model /mnt/models/Qwen3-8B --backend cuda --device 0 --admit --json` |
| `gate` | m2-fg-d65dfe10 | spark-4e77 | 0 | `/home/nvidia/intarsia-decfuse/m2-fg/target/acc/release/intarsia plan show --model /mnt/models/Qwen3-8B --backend cuda --device 0 --admit --json` |
| `acc-short` | m2-fg-d65dfe10 | spark-4e77 | 0 | `/home/nvidia/intarsia-decfuse/m2-fg/target/acc/release/intarsia plan show --model /mnt/models/Qwen3-8B --backend cuda --device 0 --admit --json` |
| `acc-format` | m2-fg-d65dfe10 | spark-4e77 | 1792606130 | `-` |
| `bench` | m2-fg-d65dfe10 | spark-4e77 | 0 | `-` |

Model fingerprint: `fnv1a64:e6a70935b68a5561`.


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `state.json` (stage records; `--resume` reads it)
