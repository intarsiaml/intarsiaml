# Qualification quick report: Qwen3-8B on metal

_Quick section, generated unix 1792071527, intarsia 0.0.1 build `head=b71ee56 code-tree=20fc48756134 dirty=no`._

## Headline

**Verdict: PASS** (7 stages: 3 pass, 0 fail, 0 error, 4 skipped)

| | |
|---|---|
| Model | Qwen3-8B (`/Users/josef/Models/Qwen3-8B`) |
| Backend / device | metal / Apple M5 Max (applegpu_g17s) |
| Recipe | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| Decode | not measured |
| Prefill | not measured |
| Accuracy (short golden) | skipped: skipped by --skip |
| Quick gate | **SKIP**: skipped by --skip |
| Wall time | quick 47s |

### Failures

None.

### Skipped

- `kernels`: skipped by --skip
- `gate`: skipped by --skip
- `acc-short`: skipped by --skip
- `bench`: skipped by --skip

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | Apple M5 Max (applegpu_g17s), 23 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 46s | 14 ops resolved, exactness deterministic, loaded in 46.3s |
| 4 | `kernels` kernels test on the model's shapes | quick | SKIP | 0s | skipped by --skip |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | SKIP | 0s | skipped by --skip |
| 6 | `acc-short` short accuracy against a golden | quick | SKIP | 0s | skipped by --skip |
| 7 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | SKIP | 0s | skipped by --skip |

### Notes

None.

## Speed

No speed numbers.

## Accuracy

**Quick gate**: skipped by --skip

No golden comparison ran (see Skipped).

## Plan and recipe

- recipe: builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) (PASS)
- plan: 14 ops resolved, exactness deterministic, loaded in 46.3s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_bf16 | exact | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 1 |
| rmsnorm_add | rmsnorm_add_bf16 | deterministic | yes | 72 |
| matmul | matmul_bf16_gemv | deterministic | yes | 108 |
| matmul | matmul_bf16_nax | deterministic | yes | 0 |
| logits | logits_bf16 | deterministic | yes | 0 |
| logits | logits_bf16_gemv | deterministic | yes | 1 |
| attn_prefill | attn_prefill_bf16_flash_nax | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 36 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 36 |
| argmax | argmax_f32 | exact | yes | 1 |
| matmul_pair | matmul_pair_bf16_gemv | deterministic | yes | 72 |
| matmul_pair | matmul_pair_bf16_nax | deterministic | yes | 0 |
| qk_norm_rope_kv | qk_norm_rope_kv_bf16 | deterministic | yes | 36 |

- kernels test: skipped by --skip (SKIP)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | head=b71ee56 code-tree=20fc48756134 dirty=no | endor.kuepker.haus | 1792071480 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-b doctor --backend metal --device 0 --out /opt/code/intarsiaml-golden-4k/docs/results/raw/2026-10-15-metal-golden-4k/round3/safe/doctor.json` |
| `recipe` | head=b71ee56 code-tree=20fc48756134 dirty=no | endor.kuepker.haus | 1792071481 | `recipe::analyse(/Users/josef/Models/Qwen3-8B)` |
| `plan` | head=b71ee56 code-tree=20fc48756134 dirty=no | endor.kuepker.haus | 1792071481 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-b plan show --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --admit --json` |
| `kernels` | head=b71ee56 code-tree=20fc48756134 dirty=no | endor.kuepker.haus | 0 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-b plan show --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --admit --json` |
| `gate` | head=b71ee56 code-tree=20fc48756134 dirty=no | endor.kuepker.haus | 0 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-b plan show --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --admit --json` |
| `acc-short` | head=b71ee56 code-tree=20fc48756134 dirty=no | endor.kuepker.haus | 0 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-b plan show --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --admit --json` |
| `bench` | head=b71ee56 code-tree=20fc48756134 dirty=no | endor.kuepker.haus | 0 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-b plan show --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --admit --json` |

Model fingerprint: `fnv1a64:e6a70935b68a5561`.


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `state.json` (stage records; `--resume` reads it)
