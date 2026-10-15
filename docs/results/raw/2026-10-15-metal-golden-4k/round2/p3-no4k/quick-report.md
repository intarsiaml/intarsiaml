# Qualification quick report: Qwen3-8B on metal

_Quick section, generated unix 1792069949, intarsia 0.0.1 build `head=03ee339 code-tree=336fc7542763 dirty=no`._

## Headline

**Verdict: PASS** (7 stages: 4 pass, 0 fail, 0 error, 3 skipped)

| | |
|---|---|
| Model | Qwen3-8B (`/Users/josef/Models/Qwen3-8B`) |
| Backend / device | metal / Apple M5 Max (applegpu_g17s) |
| Recipe | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| Decode | not measured |
| Prefill | not measured |
| Accuracy (short golden) | **PASS** (prefill top-1 100.00% KL 2.83e-4; decode top-1 100.00% KL 3.31e-4) |
| Quick gate | **SKIP**: skipped by --skip |
| Wall time | quick 51s |

### Failures

None.

### Skipped

- `kernels`: skipped by --skip
- `gate`: skipped by --skip
- `bench`: skipped by --skip

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | Apple M5 Max (applegpu_g17s), 23 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 44s | 14 ops resolved, exactness deterministic, loaded in 43.5s |
| 4 | `kernels` kernels test on the model's shapes | quick | SKIP | 0s | skipped by --skip |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | SKIP | 0s | skipped by --skip |
| 6 | `acc-short` short accuracy against a golden | quick | PASS | 7s | prefill top-1 100.00% KL 2.83e-4; decode top-1 100.00% KL 3.31e-4 |
| 7 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | SKIP | 0s | skipped by --skip |

### Notes

None.

## Speed

No speed numbers.

## Accuracy

**acc-short** against `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/gold-no4k/qwen3-8b-fp32.safetensors` (found by model name; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 434 | 100.00 | 98.89 | 2.83e-4 | 2.20e-4 | 1.51e-2 | 4.46e-4 | 0 | 100.0 | pass |
| decode | 128 | 100.00 | 98.59 | 3.31e-4 | 2.98e-4 | 7.94e-3 | 3.07e-4 | 0 | 100.0 | pass |

**Quick gate**: skipped by --skip

## Plan and recipe

- recipe: builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) (PASS)
- plan: 14 ops resolved, exactness deterministic, loaded in 43.5s (PASS)

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
| `doctor` | head=03ee339 code-tree=336fc7542763 dirty=no | endor.kuepker.haus | 1792069898 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-p32 doctor --backend metal --device 0 --out /opt/code/intarsiaml-golden-4k/docs/results/raw/2026-10-15-metal-golden-4k/round2/p3-no4k/doctor.json` |
| `recipe` | head=03ee339 code-tree=336fc7542763 dirty=no | endor.kuepker.haus | 1792069899 | `recipe::analyse(/Users/josef/Models/Qwen3-8B)` |
| `plan` | head=03ee339 code-tree=336fc7542763 dirty=no | endor.kuepker.haus | 1792069899 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-p32 plan show --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --admit --json` |
| `kernels` | head=03ee339 code-tree=336fc7542763 dirty=no | endor.kuepker.haus | 0 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-p32 plan show --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --admit --json` |
| `gate` | head=03ee339 code-tree=336fc7542763 dirty=no | endor.kuepker.haus | 0 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-p32 plan show --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --admit --json` |
| `acc-short` | head=03ee339 code-tree=336fc7542763 dirty=no | endor.kuepker.haus | 1792069942 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-p32 accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --golden /private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/gold-no4k/q` |
| `bench` | head=03ee339 code-tree=336fc7542763 dirty=no | endor.kuepker.haus | 0 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-p32 accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --golden /private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/gold-no4k/q` |

Model fingerprint: `fnv1a64:e6a70935b68a5561`.


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `state.json` (stage records; `--resume` reads it)
