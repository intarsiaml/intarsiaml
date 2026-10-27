# Qualification quick report: lfm2.5-1.2b-instruct on metal

> **Onboard verdict: PASS.** recipe: PASS; smoke: PASS; quick: PASS;

_Quick section, generated unix 1793108520, intarsia 0.0.1 build `unstamped`._

## Headline

**Verdict: PASS** (9 stages: 7 pass, 0 fail, 0 error, 2 skipped)

| | |
|---|---|
| Model | lfm2.5-1.2b-instruct (`/Users/josef/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/0f604ada3f766f9f257460c4c9f0b5d6f69d431b`) |
| Backend / device | metal / Apple M5 Max (applegpu_g17s) |
| Recipe | builtin:lfm2 covers this model (33/33 tensor templates resolve, quant bf16) |
| Decode | 128 tok: **217.1** tok/s (83% of floor); 1024 tok: **194.2** tok/s (74% of floor); 4096 tok: **182.4** tok/s (70% of floor) |
| Prefill | 128 tok: **8626** tok/s; 1024 tok: **17141** tok/s; 4096 tok: **16741** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 99.55% KL 5.88e-4; decode top-1 99.22% KL 6.97e-4) |
| Accuracy (format reference, same weights) | skipped: the weights are not quantized: no format reference applies |
| Quick gate | **PASS**: perplexity 6.91, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 1m46s |

### Failures

None.

### Skipped

- `acc-format`: the weights are not quantized: no format reference applies
- `acc-oracle`: no CPU-oracle golden for 'lfm2.5-1.2b-instruct' (--oracle-golden, or oracle/<model>-cpu-oracle*.safetensors in a golden directory)

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | Apple M5 Max (applegpu_g17s), 52 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:lfm2 covers this model (33/33 tensor templates resolve, quant bf16) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 19s | 15 ops resolved, exactness deterministic, loaded in 18.8s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 38s | 47 kernels, 865 cases in 37.5s; incomplete: 1 kernel(s) had no cases in this sweep (untested, not failed; plan-time admission tests them) |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 2s | perplexity 6.91, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | the weights are not quantized: no format reference applies |
| 7 | `acc-short` short accuracy against a golden | quick | PASS | 22s | prefill top-1 99.55% KL 5.88e-4; decode top-1 99.22% KL 6.97e-4 |
| 8 | `acc-oracle` short prefill accuracy against the f64 CPU oracle (same weights, deterministic reference) | quick | SKIP | 0s | no CPU-oracle golden for 'lfm2.5-1.2b-instruct' (--oracle-golden, or oracle/<model>-cpu-oracle*.safetensors in a golden directory) |
| 9 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 26s | decode 217.1 tok/s at 128 tokens (83% of the byte floor) |

### Notes

- `acc-short`: no baseline for backend metal next to the golden or in golden/baselines: KL, tail, prefix-reuse gates only, top-1 not gated
- `bench`: bench warning: size 128: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 1024: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 4096: 0 graph replays during the measured reps (decode ran eagerly)

## Speed

**quick bench** (floor: 2.34 GB/token at 614 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 614 GB/s, the Apple M5 Max (40-core GPU) spec figure of docs/decisions/0009-native-metal-backend.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 123 | 0.015 | 8626 | 217.14 | 1.29 | 262.3 | 82.8 |
| 1023 | 0.060 | 17141 | 194.21 | 19.60 | 262.3 | 74.0 |
| 4081 | 0.244 | 16741 | 182.41 | 10.39 | 262.3 | 69.5 |

## Accuracy

**acc-short** against `raw/nab2/onboard-1/qualify/goldens/lfm2.5-1.2b-instruct-short.safetensors` (made with tools/golden.py from /Users/josef/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/0f604ada3f766f9f257460c4c9f0b5d6f69d431b (HF transformers fp32 on cpu, bf16 spread recorded) in 20s; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 440 | 99.55 | 99.36 | 5.88e-4 | 5.24e-4 | 2.86e-2 | 1.70e-3 | 0 | 90.0 | pass |
| decode | 128 | 99.22 | 98.91 | 6.97e-4 | 5.97e-4 | 1.58e-2 | 2.70e-3 | 0 | 93.8 | pass |

**Quick gate**: perplexity 6.91, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:lfm2 covers this model (33/33 tensor templates resolve, quant bf16) (PASS)
  - TODO config.json: model error: config.json missing 'rms_norm_eps' (no default is assumed); the engine reads the dimensions with the Hugging Face key names, map different names in a [config] table (docs/adding-a-model.md)
  - TODO layer: no attention or linear-attention block recognised (module groups in the layer: ["conv", "feed_forward"]); a new block or a different tensor layout is needed
  - TODO layer.ffn: no dense FFN (gate_proj / up_proj / down_proj) or MoE (experts.{e}.*) block recognised in the layer
  - TODO layer: no attention or linear-attention block recognised (module groups in the layer: ["feed_forward", "self_attn"]); a new block or a different tensor layout is needed
  - TODO layer.ffn: no dense FFN (gate_proj / up_proj / down_proj) or MoE (experts.{e}.*) block recognised in the layer
  - TODO top.final_norm: no final norm tensor (model.norm.weight or similar) in the checkpoint; name it
  - TODO top.lm_head: no lm_head tensor in the checkpoint and tie_word_embeddings is false or absent in the config; the loader will refuse it. Name the head tensor or set tie_word_embeddings
  - TODO top: 1 checkpoint tensor(s) outside the layer stack are claimed by no block, e.g. ["model.embedding_norm.weight"]; decide whether the model needs them
  - TODO layer: 14 per-layer tensor(s) are claimed by no block, e.g. ["conv.conv.weight", "conv.in_proj.weight", "conv.out_proj.weight", "feed_forward.w1.weight", "feed_forward.w2.weight", "feed_forward.w3.weight"] ... (relative to model.layers.<i>.); extra norms, biases or unknown modules need a block or an explicit decision to ignore them
- plan: 15 ops resolved, exactness deterministic, loaded in 18.8s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_bf16 | exact | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 1 |
| rmsnorm_add | rmsnorm_add_bf16 | deterministic | yes | 32 |
| matmul | matmul_bf16_gemv | deterministic | yes | 48 |
| matmul | matmul_bf16_nax | deterministic | yes | 0 |
| logits | logits_bf16 | deterministic | yes | 0 |
| logits | logits_bf16_gemv | deterministic | yes | 1 |
| attn_prefill | attn_prefill_bf16_flash | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 6 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 16 |
| argmax | argmax_f32 | exact | yes | 1 |
| matmul_pair | matmul_pair_bf16_gemv | deterministic | yes | 22 |
| matmul_pair | matmul_pair_bf16_nax | deterministic | yes | 0 |
| qk_norm_rope_kv | qk_norm_rope_kv_bf16 | deterministic | yes | 6 |
| short_conv | short_conv_bf16 | deterministic | yes | 10 |

- kernels test: 47 kernels, 865 cases in 37.5s; incomplete: 1 kernel(s) had no cases in this sweep (untested, not failed; plan-time admission tests them) (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | unstamped | endor.kuepker.haus | 1793108413 | `/opt/code/intarsiaml/.worktrees/nab2/target/release/intarsia doctor --backend metal --device 0 --out raw/nab2/onboard-1/qualify/doctor.json` |
| `recipe` | unstamped | endor.kuepker.haus | 1793108414 | `recipe::analyse(/Users/josef/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/0f604ada3f766f9f257460c4c9f0b5d6f69d431b)` |
| `plan` | unstamped | endor.kuepker.haus | 1793108414 | `/opt/code/intarsiaml/.worktrees/nab2/target/release/intarsia plan show --model /Users/josef/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/0f604ada3f766f9f257460c4c9f0b5d6f69d431b --backend metal --device 0 --mem-budget 14 --admit --json` |
| `kernels` | unstamped | endor.kuepker.haus | 1793108432 | `/opt/code/intarsiaml/.worktrees/nab2/target/release/intarsia kernels test --backend metal --device 0 --model /Users/josef/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/0f604ada3f766f9f257460c4c9f0b5d6f69d431b` |
| `gate` | unstamped | endor.kuepker.haus | 1793108470 | `/opt/code/intarsiaml/.worktrees/nab2/target/release/intarsia accuracy --model /Users/josef/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/0f604ada3f766f9f257460c4c9f0b5d6f69d431b --backend metal --device 0 --mem-budget 14 --quick --max-seq 2048` |
| `acc-format` | unstamped | endor.kuepker.haus | 1793108472 | `-` |
| `acc-short` | unstamped | endor.kuepker.haus | 1793108472 | `/opt/code/intarsiaml/.worktrees/nab2/target/release/intarsia accuracy --model /Users/josef/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/0f604ada3f766f9f257460c4c9f0b5d6f69d431b --backend metal --device 0 --mem-budget 14 --golden raw/nab2/onboard-1/qualify/goldens/lfm2.5-1.` |
| `acc-oracle` | unstamped | endor.kuepker.haus | 1793108494 | `-` |
| `bench` | unstamped | endor.kuepker.haus | 1793108494 | `/opt/code/intarsiaml/.worktrees/nab2/target/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-lfm2.5-1.2b-instruct --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-lfm2.5-1.2b-instruct-metal-bench --out-dir raw/nab2/onboard-1/qualify/bench --t` |

Model fingerprint: `fnv1a64:3efd17d91ee2e94c`.

Build stamp `unstamped` means a plain `cargo build`: baselines written here gate this binary, but `--pre-change` provenance checks refuse them as a "before".


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/kernels.out` (kernels)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `bench/qualify-lfm2.5-1.2b-instruct-metal-bench.json` (bench)
- `logs/bench-bench.out` (bench)
- `logs/bench-server.err` (bench)
- `state.json` (stage records; `--resume` reads it)

## Stage timeline

All times UTC. The qualify stages ran inside the `quick` / `in-depth` stages. A stage that was killed and resumed shows its last attempt; the qualify rows show every sub-stage with the time it actually ran.

| Stage | Tier | Verdict | Start | End | Duration | Note |
|---|---|---|---|---|---|---|
| recipe | prepare | PASS | 2026-10-27T13:40:13Z | 2026-10-27T13:40:13Z | 0s |  |
| smoke | prepare | PASS | 2026-10-27T13:40:13Z | 2026-10-27T13:40:13Z | 0s |  |
| quick | quick | PASS | 2026-10-27T13:40:13Z | 2026-10-27T13:42:00Z | 1m47s |  |
| doctor | qualify quick | PASS | 2026-10-27T13:40:13Z | 2026-10-27T13:40:13Z | 0s |  |
| recipe | qualify quick | PASS | 2026-10-27T13:40:14Z | 2026-10-27T13:40:14Z | 0s |  |
| plan | qualify quick | PASS | 2026-10-27T13:40:14Z | 2026-10-27T13:40:33Z | 19s |  |
| kernels | qualify quick | PASS | 2026-10-27T13:40:32Z | 2026-10-27T13:41:10Z | 38s |  |
| gate | qualify quick | PASS | 2026-10-27T13:41:10Z | 2026-10-27T13:41:12Z | 2s |  |
| acc-format | qualify quick | SKIP | 2026-10-27T13:41:12Z | 2026-10-27T13:41:12Z | 0s |  |
| acc-short | qualify quick | PASS | 2026-10-27T13:41:12Z | 2026-10-27T13:41:34Z | 22s |  |
| acc-oracle | qualify quick | SKIP | 2026-10-27T13:41:34Z | 2026-10-27T13:41:34Z | 0s |  |
| bench | qualify quick | PASS | 2026-10-27T13:41:34Z | 2026-10-27T13:42:00Z | 26s |  |

_Published 2026-10-27T13:42:00Z by `intarsia onboard` (report directory `raw/nab2/onboard-1`)._
