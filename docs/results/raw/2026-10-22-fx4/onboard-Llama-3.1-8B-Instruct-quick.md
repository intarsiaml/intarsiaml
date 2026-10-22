# Qualification quick report: Llama-3.1-8B-Instruct on metal

> **Onboard verdict: FAIL.** recipe: PASS; smoke: PASS; quick: FAIL (qualify exit 1 (a gate failed): acc-short FAIL: prefill: top-1 (all rows) 99.09% (need >= 99.5); decode: top-1 (all rows) 99.38% (need >= 99.5));

_Quick section, generated unix 1792644491, intarsia 0.0.1 build `unstamped`._

## Headline

**Verdict: FAIL** (8 stages: 6 pass, 1 fail, 0 error, 1 skipped)

| | |
|---|---|
| Model | Llama-3.1-8B-Instruct (`/Users/josef/.cache/huggingface/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/4699cc75b550f9c6f3173fb80f4703b62d946aa5`) |
| Backend / device | metal / Apple M5 Max (applegpu_g17s) |
| Recipe | builtin:llama covers this model (21/21 tensor templates resolve, quant bf16) |
| Decode | 128 tok: **23.2** tok/s (57% of floor); 1024 tok: **23.9** tok/s (58% of floor); 4096 tok: **22.4** tok/s (55% of floor) |
| Prefill | 128 tok: **1081** tok/s; 1024 tok: **1306** tok/s; 4096 tok: **1533** tok/s |
| Accuracy (short golden) | **FAIL** (prefill top-1 99.09% KL 5.33e-4; decode top-1 99.38% KL 7.77e-4) |
| Accuracy (format reference, same weights) | skipped: the weights are not quantized: no format reference applies |
| Quick gate | **PASS**: perplexity 1.28, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 5m13s |
| Reference | /opt/code/wt-fx4/golden/fx4/llama-3.1-8b-instruct-fp32.safetensors |

### Failures

- `acc-short` **FAIL**: prefill: top-1 (all rows) 99.09% (need >= 99.5); decode: top-1 (all rows) 99.38% (need >= 99.5)

### Skipped

- `acc-format`: the weights are not quantized: no format reference applies

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 1s | Apple M5 Max (applegpu_g17s), 30 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:llama covers this model (21/21 tensor templates resolve, quant bf16) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 59s | 15 ops resolved, exactness deterministic, loaded in 58.9s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 80s | 30 kernels, 422 cases in 79.8s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 16s | perplexity 1.28, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | the weights are not quantized: no format reference applies |
| 7 | `acc-short` short accuracy against a golden | quick | FAIL | 19s | prefill: top-1 (all rows) 99.09% (need >= 99.5); decode: top-1 (all rows) 99.38% (need >= 99.5) |
| 8 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 2m19s | decode 23.2 tok/s at 128 tokens (57% of the byte floor) |

### Notes

- `bench`: bench warning: size 128: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 1024: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 4096: 0 graph replays during the measured reps (decode ran eagerly)

## Speed

**quick bench** (floor: 15.01 GB/token at 614 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 614 GB/s, the Apple M5 Max (40-core GPU) spec figure of docs/decisions/0009-native-metal-backend.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 125 | 0.114 | 1081 | 23.24 | 0.53 | 40.9 | 56.8 |
| 1023 | 0.783 | 1306 | 23.93 | 0.27 | 40.9 | 58.5 |
| 4060 | 2.634 | 1533 | 22.42 | 0.85 | 40.9 | 54.8 |

## Accuracy

**acc-short** against `/opt/code/wt-fx4/golden/fx4/llama-3.1-8b-instruct-fp32.safetensors` (given with --golden; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1533 | 99.09 | 98.59 | 5.33e-4 | 4.96e-4 | 7.26e-2 | 4.68e-4 | 0 | 83.6 | FAIL |
| decode | 160 | 99.38 | 99.12 | 7.77e-4 | 7.17e-4 | 5.96e-2 | 6.54e-4 | 0 | 93.8 | FAIL |

**Quick gate**: perplexity 1.28, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:llama covers this model (21/21 tensor templates resolve, quant bf16) (PASS)
- plan: 15 ops resolved, exactness deterministic, loaded in 58.9s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_bf16 | exact | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 1 |
| rmsnorm_add | rmsnorm_add_bf16 | deterministic | yes | 64 |
| matmul | matmul_bf16_gemv | deterministic | yes | 96 |
| matmul | matmul_bf16_nax | deterministic | yes | 0 |
| logits | logits_bf16 | deterministic | yes | 0 |
| logits | logits_bf16_gemv | deterministic | yes | 1 |
| kv_append | kv_append_bf16 | exact | yes | 64 |
| attn_prefill | attn_prefill_bf16_flash_nax | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 32 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 32 |
| argmax | argmax_f32 | exact | yes | 1 |
| matmul_pair | matmul_pair_bf16_gemv | deterministic | yes | 64 |
| matmul_pair | matmul_pair_bf16_nax | deterministic | yes | 0 |
| rope_scaled | rope_scaled_bf16 | deterministic | yes | 64 |

- kernels test: 30 kernels, 422 cases in 79.8s (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | unstamped | endor.kuepker.haus | 1792644177 | `/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/fx4/bin/intarsia-onb doctor --backend metal --device 0 --out /private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/fx4/onboard/Llama-3.1-8B-Instruct/qualify/doctor.json` |
| `recipe` | unstamped | endor.kuepker.haus | 1792644178 | `recipe::analyse(/Users/josef/.cache/huggingface/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/4699cc75b550f9c6f3173fb80f4703b62d946aa5)` |
| `plan` | unstamped | endor.kuepker.haus | 1792644178 | `/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/fx4/bin/intarsia-onb plan show --model /Users/josef/.cache/huggingface/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/4699cc75b550f9c6f3173fb80f4703b62d946aa5 --backend metal --device 0 --admit --json` |
| `kernels` | unstamped | endor.kuepker.haus | 1792644236 | `/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/fx4/bin/intarsia-onb kernels test --backend metal --device 0 --model /Users/josef/.cache/huggingface/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/4699cc75b550f9c6f3173fb80f4703b62d946aa5` |
| `gate` | unstamped | endor.kuepker.haus | 1792644316 | `/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/fx4/bin/intarsia-onb accuracy --model /Users/josef/.cache/huggingface/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/4699cc75b550f9c6f3173fb80f4703b62d946aa5 --backend metal --device 0 --quick --max-se` |
| `acc-format` | unstamped | endor.kuepker.haus | 1792644332 | `-` |
| `acc-short` | unstamped | endor.kuepker.haus | 1792644332 | `/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/fx4/bin/intarsia-onb accuracy --model /Users/josef/.cache/huggingface/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/4699cc75b550f9c6f3173fb80f4703b62d946aa5 --backend metal --device 0 --golden /opt/co` |
| `bench` | unstamped | endor.kuepker.haus | 1792644351 | `/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/fx4/bin/intarsia-onb bench speed --url http://127.0.0.1:8090 --model qualify-Llama-3.1-8B-Instruct --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-Llama-3.1-8B-Instruct-met` |

Model fingerprint: `fnv1a64:5331eca5701f4126`.

Build stamp `unstamped` means a plain `cargo build`: baselines written here gate this binary, but `--pre-change` provenance checks refuse them as a "before".


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/kernels.out` (kernels)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `bench/qualify-Llama-3.1-8B-Instruct-metal-bench.json` (bench)
- `logs/bench-bench.out` (bench)
- `logs/bench-server.err` (bench)
- `state.json` (stage records; `--resume` reads it)

## Stage timeline

All times UTC. The qualify stages ran inside the `quick` / `in-depth` stages. A stage that was killed and resumed shows its last attempt; the qualify rows show every sub-stage with the time it actually ran.

| Stage | Tier | Verdict | Start | End | Duration | Note |
|---|---|---|---|---|---|---|
| recipe | prepare | PASS | 2026-10-22T04:42:54Z | 2026-10-22T04:42:54Z | 0s |  |
| smoke | prepare | PASS | 2026-10-22T04:42:54Z | 2026-10-22T04:42:57Z | 3s |  |
| quick | quick | FAIL | 2026-10-22T04:42:57Z | 2026-10-22T04:48:11Z | 5m13s |  |
| doctor | qualify quick | PASS | 2026-10-22T04:42:57Z | 2026-10-22T04:42:58Z | 1s |  |
| recipe | qualify quick | PASS | 2026-10-22T04:42:58Z | 2026-10-22T04:42:58Z | 0s |  |
| plan | qualify quick | PASS | 2026-10-22T04:42:58Z | 2026-10-22T04:43:57Z | 59s |  |
| kernels | qualify quick | PASS | 2026-10-22T04:43:56Z | 2026-10-22T04:45:16Z | 80s |  |
| gate | qualify quick | PASS | 2026-10-22T04:45:16Z | 2026-10-22T04:45:32Z | 16s |  |
| acc-format | qualify quick | SKIP | 2026-10-22T04:45:32Z | 2026-10-22T04:45:32Z | 0s |  |
| acc-short | qualify quick | FAIL | 2026-10-22T04:45:32Z | 2026-10-22T04:45:51Z | 19s |  |
| bench | qualify quick | PASS | 2026-10-22T04:45:51Z | 2026-10-22T04:48:11Z | 2m19s |  |

_Published 2026-10-22T04:48:11Z by `intarsia onboard` (report directory `/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/fx4/onboard/Llama-3.1-8B-Instruct`)._
