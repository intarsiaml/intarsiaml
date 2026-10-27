# Qualification report: lfm2.5-1.2b-instruct on metal

> **Onboard verdict: PASS.** recipe: PASS; smoke: PASS; quick: PASS; in-depth: PASS;

_Full (quick + in-depth) section, generated unix 1793109194, intarsia 0.0.1 build `unstamped`._

## Headline

**Verdict: PASS** (15 stages: 11 pass, 0 fail, 0 error, 4 skipped)

| | |
|---|---|
| Model | lfm2.5-1.2b-instruct (`/Users/josef/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/0f604ada3f766f9f257460c4c9f0b5d6f69d431b`) |
| Backend / device | metal / Apple M5 Max (applegpu_g17s) |
| Recipe | builtin:lfm2 covers this model (33/33 tensor templates resolve, quant bf16) |
| Decode | 128 tok: **217.1** tok/s (83% of floor); 1024 tok: **194.2** tok/s (74% of floor); 4096 tok: **182.4** tok/s (70% of floor) |
| Prefill | 128 tok: **8626** tok/s; 1024 tok: **17141** tok/s; 4096 tok: **16741** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 99.55% KL 5.88e-4; decode top-1 99.22% KL 6.97e-4) |
| Accuracy (format reference, same weights) | skipped: the weights are not quantized: no format reference applies |
| Accuracy (long golden) | **PASS** (prefill top-1 99.31% KL 6.22e-4; decode top-1 98.41% KL 4.20e-4) |
| Accuracy (4k golden) | **PASS** (prefill top-1 98.80% KL 4.15e-4; decode top-1 99.48% KL 2.13e-4) |
| Quick gate | **PASS**: perplexity 6.91, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 1m46s, in-depth 85s, total 3m11s |

### Failures

None.

### Skipped

- `acc-format`: the weights are not quantized: no format reference applies
- `acc-oracle`: no CPU-oracle golden for 'lfm2.5-1.2b-instruct' (--oracle-golden, or oracle/<model>-cpu-oracle*.safetensors in a golden directory)
- `ab`: no --reference server given
- `tp`: --tp not given (or 1)

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
| 10 | `acc-long` accuracy --mode all on the long golden (prefill, decode, prefix reuse) | full | PASS | 7s | prefill top-1 99.31% KL 6.22e-4; decode top-1 98.41% KL 4.20e-4 |
| 11 | `acc-4k` accuracy --mode all on the 4k golden (prefill, decode, prefix reuse past 2 chunks of 2048) | full | PASS | 32s | prefill top-1 98.80% KL 4.15e-4; decode top-1 99.48% KL 2.13e-4 |
| 12 | `envelope` noise envelope and baseline for regression gating | full | PASS | 6s | baseline and envelope (chunk) written; they gate this build: PASS |
| 13 | `bench-long` long-context bench up to 16k | full | PASS | 40s | decode 157.8 tok/s at 1024 tokens (60% of the byte floor) |
| 14 | `ab` A/B against the reference server via localeval | full | SKIP | 0s | no --reference server given |
| 15 | `tp` tensor-parallel variant | full | SKIP | 0s | --tp not given (or 1) |

### Notes

- `acc-short`: no baseline for backend metal next to the golden or in golden/baselines: KL, tail, prefix-reuse gates only, top-1 not gated
- `bench`: bench warning: size 128: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 1024: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 4096: 0 graph replays during the measured reps (decode ran eagerly)
- `acc-long`: prefix reuse is gated in acc-4k, on the 4k golden
- `acc-4k`: prefix reuse at chunk 256: 3 judged, 0 skipped
- `acc-4k`: no baseline for backend metal next to the golden or in golden/baselines: KL, tail, prefix-reuse gates only, top-1 not gated
- `bench-long`: bench warning: size 1024: 0 graph replays during the measured reps (decode ran eagerly)
- `bench-long`: bench warning: size 4096: 0 graph replays during the measured reps (decode ran eagerly)
- `bench-long`: bench warning: size 16384: 0 graph replays during the measured reps (decode ran eagerly)

## Speed

**quick bench** (floor: 2.34 GB/token at 614 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 614 GB/s, the Apple M5 Max (40-core GPU) spec figure of docs/decisions/0009-native-metal-backend.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 123 | 0.015 | 8626 | 217.14 | 1.29 | 262.3 | 82.8 |
| 1023 | 0.060 | 17141 | 194.21 | 19.60 | 262.3 | 74.0 |
| 4081 | 0.244 | 16741 | 182.41 | 10.39 | 262.3 | 69.5 |

**long-context bench** (floor: 2.34 GB/token at 614 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 614 GB/s, the Apple M5 Max (40-core GPU) spec figure of docs/decisions/0009-native-metal-backend.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 1017 | 0.072 | 14090 | 157.77 | 19.16 | 262.3 | 60.1 |
| 4076 | 0.432 | 9449 | 150.81 | 8.94 | 262.3 | 57.5 |
| 16281 | 1.822 | 8936 | 182.58 | 19.60 | 262.3 | 69.6 |

## Accuracy

**acc-short** against `raw/nab2/onboard-1/qualify/goldens/lfm2.5-1.2b-instruct-short.safetensors` (made with tools/golden.py from /Users/josef/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/0f604ada3f766f9f257460c4c9f0b5d6f69d431b (HF transformers fp32 on cpu, bf16 spread recorded) in 20s; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 440 | 99.55 | 99.36 | 5.88e-4 | 5.24e-4 | 2.86e-2 | 1.70e-3 | 0 | 90.0 | pass |
| decode | 128 | 99.22 | 98.91 | 6.97e-4 | 5.97e-4 | 1.58e-2 | 2.70e-3 | 0 | 93.8 | pass |

**acc-long** against `raw/nab2/goldens/lfm2.5-1.2b-instruct-long-fp32.safetensors` (found by model name; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1440 | 99.31 | 98.76 | 6.22e-4 | 5.33e-4 | 2.86e-2 | 1.54e-3 | 0 | 32.4 | pass |
| decode | 502 | 98.41 | 99.04 | 4.20e-4 | 3.59e-4 | 1.64e-2 | 1.28e-3 | 0 | 74.1 | pass |

**acc-4k** against `raw/nab2/goldens/lfm2.5-1.2b-instruct-4k-fp32-top20.safetensors` (found by model name; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 12996 | 98.80 | 98.55 | 4.15e-4 | 3.73e-4 | 4.99e-2 | 1.71e-3 | 0 | 0.4 | pass |
| decode | 192 | 99.48 | 99.27 | 2.13e-4 | 1.91e-4 | 2.92e-3 | 7.44e-4 | 0 | 93.8 | pass |

**Regression gating artifacts**: baseline and envelope (chunk) written; they gate this build: PASS

`intarsia accuracy --model <dir> --backend metal --golden raw/nab2/onboard-1/qualify/goldens/lfm2.5-1.2b-instruct-short.safetensors --baseline baselines/acc-short.baseline.json --noise-envelope baselines/envelope.json  (copy both files next to the golden, e.g. golden/baselines/)`

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
| `acc-format` | unstamped | endor.kuepker.haus | 1793109155 | `-` |
| `acc-short` | unstamped | endor.kuepker.haus | 1793108472 | `/opt/code/intarsiaml/.worktrees/nab2/target/release/intarsia accuracy --model /Users/josef/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/0f604ada3f766f9f257460c4c9f0b5d6f69d431b --backend metal --device 0 --mem-budget 14 --golden raw/nab2/onboard-1/qualify/goldens/lfm2.5-1.` |
| `acc-oracle` | unstamped | endor.kuepker.haus | 1793109155 | `-` |
| `bench` | unstamped | endor.kuepker.haus | 1793108494 | `/opt/code/intarsiaml/.worktrees/nab2/target/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-lfm2.5-1.2b-instruct --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-lfm2.5-1.2b-instruct-metal-bench --out-dir raw/nab2/onboard-1/qualify/bench --t` |
| `acc-long` | unstamped | endor.kuepker.haus | 1793109155 | `/opt/code/intarsiaml/.worktrees/nab2/target/release/intarsia accuracy --model /Users/josef/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/0f604ada3f766f9f257460c4c9f0b5d6f69d431b --backend metal --device 0 --mem-budget 14 --golden raw/nab2/goldens/lfm2.5-1.2b-instruct-long-f` |
| `acc-4k` | unstamped | endor.kuepker.haus | 1793109162 | `/opt/code/intarsiaml/.worktrees/nab2/target/release/intarsia accuracy --model /Users/josef/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/0f604ada3f766f9f257460c4c9f0b5d6f69d431b --backend metal --device 0 --mem-budget 14 --golden raw/nab2/goldens/lfm2.5-1.2b-instruct-4k-fp3` |
| `envelope` | unstamped | endor.kuepker.haus | 1793108520 | `/opt/code/intarsiaml/.worktrees/nab2/target/release/intarsia accuracy --model /Users/josef/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/0f604ada3f766f9f257460c4c9f0b5d6f69d431b --backend metal --device 0 --mem-budget 14 --golden raw/nab2/onboard-1/qualify/goldens/lfm2.5-1.` |
| `bench-long` | unstamped | endor.kuepker.haus | 1793108526 | `/opt/code/intarsiaml/.worktrees/nab2/target/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-lfm2.5-1.2b-instruct --sizes 1024,4096,16384 --gen 64 --reps 2 --warmup 1 --thinking off --label qualify-lfm2.5-1.2b-instruct-metal-bench-long --out-dir raw/nab2/onboard-1/qualify/ben` |
| `ab` | unstamped | endor.kuepker.haus | 1793109194 | `-` |
| `tp` | unstamped | endor.kuepker.haus | 1793109194 | `-` |

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
- `logs/acc-long.out` (acc-long)
- `baselines/acc-long.baseline.json` (acc-long)
- `logs/acc-4k.out` (acc-4k)
- `baselines/acc-4k.baseline.json` (acc-4k)
- `logs/acc-4k-reuse.out` (acc-4k)
- `baselines/acc-short.baseline.json` (envelope)
- `baselines/envelope.json` (envelope)
- `logs/baseline-selfcheck.out` (envelope)
- `bench/qualify-lfm2.5-1.2b-instruct-metal-bench-long.json` (bench-long)
- `logs/bench-long-bench.out` (bench-long)
- `logs/bench-long-server.err` (bench-long)
- `state.json` (stage records; `--resume` reads it)

## Stage timeline

All times UTC. The qualify stages ran inside the `quick` / `in-depth` stages. A stage that was killed and resumed shows its last attempt; the qualify rows show every sub-stage with the time it actually ran.

| Stage | Tier | Verdict | Start | End | Duration | Note |
|---|---|---|---|---|---|---|
| recipe | prepare | PASS | 2026-10-27T13:40:13Z | 2026-10-27T13:40:13Z | 0s |  |
| smoke | prepare | PASS | 2026-10-27T13:40:13Z | 2026-10-27T13:40:13Z | 0s |  |
| quick | quick | PASS | 2026-10-27T13:40:13Z | 2026-10-27T13:42:00Z | 1m47s |  |
| publish-quick | quick | PASS | 2026-10-27T13:42:00Z | 2026-10-27T13:42:00Z | 0s |  |
| in-depth | in-depth | PASS | 2026-10-27T13:52:35Z | 2026-10-27T13:53:14Z | 39s |  |
| publish-in-depth | in-depth | PASS | 2026-10-27T13:42:46Z | 2026-10-27T13:42:46Z | 0s |  |
| doctor | qualify quick | PASS | 2026-10-27T13:40:13Z | 2026-10-27T13:40:13Z | 0s |  |
| recipe | qualify quick | PASS | 2026-10-27T13:40:14Z | 2026-10-27T13:40:14Z | 0s |  |
| plan | qualify quick | PASS | 2026-10-27T13:40:14Z | 2026-10-27T13:40:33Z | 19s |  |
| kernels | qualify quick | PASS | 2026-10-27T13:40:32Z | 2026-10-27T13:41:10Z | 38s |  |
| gate | qualify quick | PASS | 2026-10-27T13:41:10Z | 2026-10-27T13:41:12Z | 2s |  |
| acc-format | qualify quick | SKIP | 2026-10-27T13:52:35Z | 2026-10-27T13:52:35Z | 0s |  |
| acc-short | qualify quick | PASS | 2026-10-27T13:41:12Z | 2026-10-27T13:41:34Z | 22s |  |
| acc-oracle | qualify quick | SKIP | 2026-10-27T13:52:35Z | 2026-10-27T13:52:35Z | 0s |  |
| bench | qualify quick | PASS | 2026-10-27T13:41:34Z | 2026-10-27T13:42:00Z | 26s |  |
| acc-long | qualify full | PASS | 2026-10-27T13:52:35Z | 2026-10-27T13:52:42Z | 7s |  |
| acc-4k | qualify full | PASS | 2026-10-27T13:52:42Z | 2026-10-27T13:53:14Z | 32s |  |
| envelope | qualify full | PASS | 2026-10-27T13:42:00Z | 2026-10-27T13:42:06Z | 6s |  |
| bench-long | qualify full | PASS | 2026-10-27T13:42:06Z | 2026-10-27T13:42:46Z | 40s |  |
| ab | qualify full | SKIP | 2026-10-27T13:53:14Z | 2026-10-27T13:53:14Z | 0s |  |
| tp | qualify full | SKIP | 2026-10-27T13:53:14Z | 2026-10-27T13:53:14Z | 0s |  |

_Published 2026-10-27T13:53:14Z by `intarsia onboard` (report directory `raw/nab2/onboard-1`)._
