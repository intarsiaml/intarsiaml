# Qualification report: Qwen3-8B on metal

_Full (quick + in-depth) section, generated unix 1792071776, intarsia 0.0.1 build `head=59af184 code-tree=1e0dfabdf84a dirty=no`._

## Headline

**Verdict: PASS** (13 stages: 5 pass, 0 fail, 0 error, 8 skipped)

| | |
|---|---|
| Model | Qwen3-8B (`/Users/josef/Models/Qwen3-8B`) |
| Backend / device | metal / Apple M5 Max (applegpu_g17s) |
| Recipe | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| Decode | not measured |
| Prefill | not measured |
| Accuracy (short golden) | skipped: skipped by --skip |
| Accuracy (long golden) | **PASS** (prefill top-1 99.72% KL 3.70e-4; decode top-1 99.80% KL 3.54e-4) |
| Accuracy (4k golden) | **PASS** (prefill top-1 99.28% KL 2.44e-4; decode top-1 98.96% KL 2.57e-4) |
| Quick gate | **SKIP**: skipped by --skip |
| Wall time | quick 43s, in-depth 81s, total 2m04s |

### Failures

None.

### Skipped

- `kernels`: skipped by --skip
- `gate`: skipped by --skip
- `acc-short`: skipped by --skip
- `bench`: skipped by --skip
- `envelope`: skipped by --skip
- `bench-long`: skipped by --skip
- `ab`: skipped by --skip
- `tp`: skipped by --skip

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | Apple M5 Max (applegpu_g17s), 23 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 43s | 14 ops resolved, exactness deterministic, loaded in 43.0s |
| 4 | `kernels` kernels test on the model's shapes | quick | SKIP | 0s | skipped by --skip |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | SKIP | 0s | skipped by --skip |
| 6 | `acc-short` short accuracy against a golden | quick | SKIP | 0s | skipped by --skip |
| 7 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | SKIP | 0s | skipped by --skip |
| 8 | `acc-long` accuracy --mode all on the long golden (prefill, decode, prefix reuse) | full | PASS | 23s | prefill top-1 99.72% KL 3.70e-4; decode top-1 99.80% KL 3.54e-4 |
| 9 | `acc-4k` accuracy --mode all on the 4k golden (prefill, decode, prefix reuse past 2 chunks of 2048) | full | PASS | 58s | prefill top-1 99.28% KL 2.44e-4; decode top-1 98.96% KL 2.57e-4 |
| 10 | `envelope` noise envelope and baseline for regression gating | full | SKIP | 0s | skipped by --skip |
| 11 | `bench-long` long-context bench up to 16k | full | SKIP | 0s | skipped by --skip |
| 12 | `ab` A/B against the reference server via localeval | full | SKIP | 0s | skipped by --skip |
| 13 | `tp` tensor-parallel variant | full | SKIP | 0s | skipped by --skip |

### Notes

- `acc-long`: prefix reuse is gated in acc-4k, on the 4k golden
- `acc-4k`: baseline /opt/code/intarsiaml-golden-4k/golden/baselines/qwen3-8b-4k-fp32-top20.metal.baseline.json not used (recorded at prefill chunk 256, this run uses 2048): KL, tail, prefix-reuse gates only, top-1 not gated

## Speed

No speed numbers.

## Accuracy

**acc-long** against `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/gold/qwen3-8b-long-fp32.safetensors` (found by model name; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1434 | 99.72 | 98.68 | 3.70e-4 | 3.38e-4 | 2.16e-2 | 3.34e-4 | 0 | 56.3 | pass |
| decode | 500 | 99.80 | 98.84 | 3.54e-4 | 3.33e-4 | 4.49e-2 | 2.27e-4 | 0 | 90.0 | pass |

**acc-4k** against `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/gold/qwen3-8b-4k-fp32-top20.safetensors` (found by model name; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 12685 | 99.28 | 98.94 | 2.44e-4 | 2.27e-4 | 2.78e-1 | 6.81e-4 | 0 | 2.1 | pass |
| decode | 192 | 98.96 | 98.96 | 2.57e-4 | 2.55e-4 | 8.05e-3 | 3.22e-4 | 0 | 71.9 | pass |

**Quick gate**: skipped by --skip

## Plan and recipe

- recipe: builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) (PASS)
- plan: 14 ops resolved, exactness deterministic, loaded in 43.0s (PASS)

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
| `doctor` | head=59af184 code-tree=1e0dfabdf84a dirty=no | endor.kuepker.haus | 1792071651 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-p3b doctor --backend metal --device 0 --out /opt/code/intarsiaml-golden-4k/docs/results/raw/2026-10-15-metal-golden-4k/round3/p3-chunk-mismatch/doctor.json` |
| `recipe` | head=59af184 code-tree=1e0dfabdf84a dirty=no | endor.kuepker.haus | 1792071652 | `recipe::analyse(/Users/josef/Models/Qwen3-8B)` |
| `plan` | head=59af184 code-tree=1e0dfabdf84a dirty=no | endor.kuepker.haus | 1792071652 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-p3b plan show --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --admit --json` |
| `kernels` | head=59af184 code-tree=1e0dfabdf84a dirty=no | endor.kuepker.haus | 0 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-p3b plan show --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --admit --json` |
| `gate` | head=59af184 code-tree=1e0dfabdf84a dirty=no | endor.kuepker.haus | 0 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-p3b plan show --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --admit --json` |
| `acc-short` | head=59af184 code-tree=1e0dfabdf84a dirty=no | endor.kuepker.haus | 0 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-p3b plan show --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --admit --json` |
| `bench` | head=59af184 code-tree=1e0dfabdf84a dirty=no | endor.kuepker.haus | 0 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-p3b plan show --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --admit --json` |
| `acc-long` | head=59af184 code-tree=1e0dfabdf84a dirty=no | endor.kuepker.haus | 1792071695 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-p3b accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --golden /private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/gold/qwen3-` |
| `acc-4k` | head=59af184 code-tree=1e0dfabdf84a dirty=no | endor.kuepker.haus | 1792071718 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-p3b accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --golden /private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/gold/qwen3-` |
| `envelope` | head=59af184 code-tree=1e0dfabdf84a dirty=no | endor.kuepker.haus | 0 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-p3b accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --golden /private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/gold/qwen3-` |
| `bench-long` | head=59af184 code-tree=1e0dfabdf84a dirty=no | endor.kuepker.haus | 0 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-p3b accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --golden /private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/gold/qwen3-` |
| `ab` | head=59af184 code-tree=1e0dfabdf84a dirty=no | endor.kuepker.haus | 0 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-p3b accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --golden /private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/gold/qwen3-` |
| `tp` | head=59af184 code-tree=1e0dfabdf84a dirty=no | endor.kuepker.haus | 0 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-p3b accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --golden /private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/gold/qwen3-` |

Model fingerprint: `fnv1a64:e6a70935b68a5561`.


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/acc-long.out` (acc-long)
- `baselines/acc-long.baseline.json` (acc-long)
- `logs/acc-4k.out` (acc-4k)
- `baselines/acc-4k.baseline.json` (acc-4k)
- `state.json` (stage records; `--resume` reads it)
