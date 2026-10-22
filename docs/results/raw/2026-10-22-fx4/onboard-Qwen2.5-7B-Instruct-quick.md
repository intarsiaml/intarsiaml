# Qualification quick report: Qwen2.5-7B-Instruct on metal

> **Onboard verdict: FAIL.** recipe: PASS; smoke: PASS; quick: FAIL (qualify exit 1 (a gate failed): gate FAIL: sanity FAIL: 48 tokens, max 4-gram repeat 4: non-empty; acc-short FAIL: prefill: top-1 (all rows) 99.30% (need >= 99.5); prefill: full-vocab KL (scored rows, n=1272) 1.405e-3 (need <= 1.0e-3); decode: top-1 (all rows) 98.12% (need >= 99.5));

_Quick section, generated unix 1792644809, intarsia 0.0.1 build `unstamped`._

## Headline

**Verdict: FAIL** (8 stages: 5 pass, 2 fail, 0 error, 1 skipped)

| | |
|---|---|
| Model | Qwen2.5-7B-Instruct (`/Users/josef/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28`) |
| Backend / device | metal / Apple M5 Max (applegpu_g17s) |
| Recipe | builtin:qwen2 covers this model (21/21 tensor templates resolve, quant bf16) |
| Decode | 128 tok: **31.7** tok/s (73% of floor); 1024 tok: **32.2** tok/s (74% of floor); 4096 tok: **27.6** tok/s (64% of floor) |
| Prefill | 128 tok: **1494** tok/s; 1024 tok: **1014** tok/s; 4096 tok: **553** tok/s |
| Accuracy (short golden) | **FAIL** (prefill top-1 99.30% KL 3.16e-3; decode top-1 98.12% KL 5.18e-3) |
| Accuracy (format reference, same weights) | skipped: the weights are not quantized: no format reference applies |
| Quick gate | **FAIL**: sanity FAIL: 48 tokens, max 4-gram repeat 4: non-empty |
| Wall time | quick 5m12s |
| Reference | /opt/code/wt-fx4/golden/fx4/qwen2.5-7b-instruct-fp32.safetensors |

### Failures

- `gate` **FAIL**: sanity FAIL: 48 tokens, max 4-gram repeat 4: non-empty
- `acc-short` **FAIL**: prefill: top-1 (all rows) 99.30% (need >= 99.5); prefill: full-vocab KL (scored rows, n=1272) 1.405e-3 (need <= 1.0e-3); decode: top-1 (all rows) 98.12% (need >= 99.5)

### Skipped

- `acc-format`: the weights are not quantized: no format reference applies

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 1s | Apple M5 Max (applegpu_g17s), 30 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen2 covers this model (21/21 tensor templates resolve, quant bf16) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 79s | 16 ops resolved, exactness deterministic, loaded in 78.5s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 69s | 30 kernels, 408 cases in 69.0s; incomplete: 1 kernel(s) had no cases in this sweep (untested, not failed; plan-time admission tests them) |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | FAIL | 15s | sanity FAIL: 48 tokens, max 4-gram repeat 4: non-empty |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | the weights are not quantized: no format reference applies |
| 7 | `acc-short` short accuracy against a golden | quick | FAIL | 20s | prefill: top-1 (all rows) 99.30% (need >= 99.5); prefill: full-vocab KL (scored rows, n=1272) 1.405e-3 (need <= 1.0e-3); decode: top-1 (all rows) 98.12% (need >= 99.5) |
| 8 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 2m08s | decode 31.7 tok/s at 128 tokens (73% of the byte floor) |

### Notes

- `bench`: bench warning: size 128: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 1024: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 4096: 0 graph replays during the measured reps (decode ran eagerly)

## Speed

**quick bench** (floor: 14.14 GB/token at 614 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 614 GB/s, the Apple M5 Max (40-core GPU) spec figure of docs/decisions/0009-native-metal-backend.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 124 | 0.084 | 1494 | 31.73 | 1.09 | 43.4 | 73.1 |
| 1011 | 0.996 | 1014 | 32.17 | 0.33 | 43.4 | 74.1 |
| 4156 | 7.516 | 553 | 27.61 | 0.28 | 43.4 | 63.6 |

## Accuracy

**acc-short** against `/opt/code/wt-fx4/golden/fx4/qwen2.5-7b-instruct-fp32.safetensors` (given with --golden; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1434 | 99.30 | 97.99 | 3.16e-3 | 2.13e-3 | 7.16e-1 | 1.45e-3 | 0 | 29.8 | FAIL |
| decode | 160 | 98.12 | 98.12 | 5.18e-3 | 2.81e-3 | 3.78e-1 | 2.10e-3 | 0 | 78.1 | FAIL |

**Quick gate**: sanity FAIL: 48 tokens, max 4-gram repeat 4: non-empty

## Plan and recipe

- recipe: builtin:qwen2 covers this model (21/21 tensor templates resolve, quant bf16) (PASS)
- plan: 16 ops resolved, exactness deterministic, loaded in 78.5s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_bf16 | exact | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 1 |
| rmsnorm_add | rmsnorm_add_bf16 | deterministic | yes | 56 |
| rope | rope_bf16 | deterministic | yes | 56 |
| matmul | matmul_bf16_gemv | deterministic | yes | 84 |
| matmul | matmul_bf16_nax | deterministic | yes | 0 |
| logits | logits_bf16 | deterministic | yes | 0 |
| logits | logits_bf16_gemv | deterministic | yes | 1 |
| kv_append | kv_append_bf16 | exact | yes | 56 |
| attn_prefill | attn_prefill_bf16 | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 28 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 28 |
| argmax | argmax_f32 | exact | yes | 1 |
| matmul_pair | matmul_pair_bf16_gemv | deterministic | yes | 56 |
| matmul_pair | matmul_pair_bf16_nax | deterministic | yes | 0 |
| add_bias | add_bias_bf16 | deterministic | yes | 84 |

- kernels test: 30 kernels, 408 cases in 69.0s; incomplete: 1 kernel(s) had no cases in this sweep (untested, not failed; plan-time admission tests them) (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | unstamped | endor.kuepker.haus | 1792644496 | `/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/fx4/bin/intarsia-onb doctor --backend metal --device 0 --out /private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/fx4/onboard/Qwen2.5-7B-Instruct/qualify/doctor.json` |
| `recipe` | unstamped | endor.kuepker.haus | 1792644497 | `recipe::analyse(/Users/josef/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28)` |
| `plan` | unstamped | endor.kuepker.haus | 1792644497 | `/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/fx4/bin/intarsia-onb plan show --model /Users/josef/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 --backend metal --device 0 --admit --json` |
| `kernels` | unstamped | endor.kuepker.haus | 1792644576 | `/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/fx4/bin/intarsia-onb kernels test --backend metal --device 0 --model /Users/josef/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28` |
| `gate` | unstamped | endor.kuepker.haus | 1792644645 | `/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/fx4/bin/intarsia-onb accuracy --model /Users/josef/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 --backend metal --device 0 --quick --max-seq 204` |
| `acc-format` | unstamped | endor.kuepker.haus | 1792644660 | `-` |
| `acc-short` | unstamped | endor.kuepker.haus | 1792644660 | `/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/fx4/bin/intarsia-onb accuracy --model /Users/josef/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 --backend metal --device 0 --golden /opt/code/wt` |
| `bench` | unstamped | endor.kuepker.haus | 1792644680 | `/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/fx4/bin/intarsia-onb bench speed --url http://127.0.0.1:8090 --model qualify-Qwen2.5-7B-Instruct --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-Qwen2.5-7B-Instruct-metal-b` |

Model fingerprint: `fnv1a64:04b1e29159d2d6f7`.

Build stamp `unstamped` means a plain `cargo build`: baselines written here gate this binary, but `--pre-change` provenance checks refuse them as a "before".


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/kernels.out` (kernels)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `bench/qualify-Qwen2.5-7B-Instruct-metal-bench.json` (bench)
- `logs/bench-bench.out` (bench)
- `logs/bench-server.err` (bench)
- `state.json` (stage records; `--resume` reads it)

## Stage timeline

All times UTC. The qualify stages ran inside the `quick` / `in-depth` stages. A stage that was killed and resumed shows its last attempt; the qualify rows show every sub-stage with the time it actually ran.

| Stage | Tier | Verdict | Start | End | Duration | Note |
|---|---|---|---|---|---|---|
| recipe | prepare | PASS | 2026-10-22T04:48:11Z | 2026-10-22T04:48:11Z | 0s |  |
| smoke | prepare | PASS | 2026-10-22T04:48:11Z | 2026-10-22T04:48:16Z | 5s |  |
| quick | quick | FAIL | 2026-10-22T04:48:16Z | 2026-10-22T04:53:29Z | 5m12s |  |
| doctor | qualify quick | PASS | 2026-10-22T04:48:16Z | 2026-10-22T04:48:17Z | 1s |  |
| recipe | qualify quick | PASS | 2026-10-22T04:48:17Z | 2026-10-22T04:48:17Z | 0s |  |
| plan | qualify quick | PASS | 2026-10-22T04:48:17Z | 2026-10-22T04:49:36Z | 79s |  |
| kernels | qualify quick | PASS | 2026-10-22T04:49:36Z | 2026-10-22T04:50:45Z | 69s |  |
| gate | qualify quick | FAIL | 2026-10-22T04:50:45Z | 2026-10-22T04:51:00Z | 15s |  |
| acc-format | qualify quick | SKIP | 2026-10-22T04:51:00Z | 2026-10-22T04:51:00Z | 0s |  |
| acc-short | qualify quick | FAIL | 2026-10-22T04:51:00Z | 2026-10-22T04:51:20Z | 20s |  |
| bench | qualify quick | PASS | 2026-10-22T04:51:20Z | 2026-10-22T04:53:29Z | 2m08s |  |

_Published 2026-10-22T04:53:29Z by `intarsia onboard` (report directory `/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/fx4/onboard/Qwen2.5-7B-Instruct`)._
