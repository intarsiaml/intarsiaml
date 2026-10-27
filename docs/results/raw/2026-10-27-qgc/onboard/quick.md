# Qualification quick report: qwen3.5-4b on metal

> **Onboard verdict: PASS.** recipe: PASS; smoke: PASS; quick: PASS;

_Quick section, generated unix 1793098996, intarsia 0.0.1 build `unstamped`._

## Headline

**Verdict: PASS** (9 stages: 7 pass, 0 fail, 0 error, 2 skipped)

| | |
|---|---|
| Model | qwen3.5-4b (`/Users/josef/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`) |
| Backend / device | metal / Apple M5 Max (applegpu_g17s) |
| Recipe | builtin:qwen3_5_vl covers this model (45/45 tensor templates resolve, quant bf16) |
| Decode | 128 tok: **53.1** tok/s (73% of floor); 1024 tok: **61.2** tok/s (84% of floor); 4096 tok: **61.3** tok/s (84% of floor) |
| Prefill | 128 tok: **2373** tok/s; 1024 tok: **4356** tok/s; 4096 tok: **4501** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 99.77% KL 1.23e-4; decode top-1 99.22% KL 2.03e-4) |
| Accuracy (format reference, same weights) | skipped: the weights are not quantized: no format reference applies |
| Quick gate | **PASS**: perplexity 2.02, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 5m36s |

### Failures

None.

### Skipped

- `acc-format`: the weights are not quantized: no format reference applies
- `acc-oracle`: no CPU-oracle golden for 'qwen3.5-4b' (--oracle-golden, or oracle/<model>-cpu-oracle*.safetensors in a golden directory)

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | Apple M5 Max (applegpu_g17s), 51 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_vl covers this model (45/45 tensor templates resolve, quant bf16) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 30s | 19 ops resolved, exactness deterministic, loaded in 29.8s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 44s | 46 kernels, 845 cases in 44.2s; incomplete: 1 kernel(s) had no cases in this sweep (untested, not failed; plan-time admission tests them) |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 7s | perplexity 2.02, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | the weights are not quantized: no format reference applies |
| 7 | `acc-short` short accuracy against a golden | quick | PASS | 3m20s | prefill top-1 99.77% KL 1.23e-4; decode top-1 99.22% KL 2.03e-4 |
| 8 | `acc-oracle` short prefill accuracy against the f64 CPU oracle (same weights, deterministic reference) | quick | SKIP | 0s | no CPU-oracle golden for 'qwen3.5-4b' (--oracle-golden, or oracle/<model>-cpu-oracle*.safetensors in a golden directory) |
| 9 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 54s | decode 53.1 tok/s at 128 tokens (73% of the byte floor) |

### Notes

- `acc-short`: no baseline for backend metal next to the golden or in golden/baselines: KL, tail, prefix-reuse gates only, top-1 not gated
- `bench`: bench warning: size 128: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 1024: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 4096: 0 graph replays during the measured reps (decode ran eagerly)

## Speed

**quick bench** (floor: 8.41 GB/token at 614 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 614 GB/s, the Apple M5 Max (40-core GPU) spec figure of docs/decisions/0009-native-metal-backend.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 134 | 0.056 | 2373 | 53.12 | 2.82 | 73.0 | 72.8 |
| 1034 | 0.238 | 4356 | 61.20 | 0.35 | 73.0 | 83.8 |
| 4195 | 0.927 | 4501 | 61.31 | 0.56 | 73.0 | 84.0 |

## Accuracy

**acc-short** against `/opt/code/intarsiaml/.worktrees/qgc/runs/onboard-2/qualify/goldens/qwen3.5-4b-short.safetensors` (made with tools/golden.py from /Users/josef/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a (HF transformers fp32 on cpu, bf16 spread recorded) in 194s; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 441 | 99.77 | 99.41 | 1.23e-4 | 1.12e-4 | 8.41e-3 | 3.13e-4 | 0 | 98.0 | pass |
| decode | 128 | 99.22 | 98.44 | 2.03e-4 | 1.91e-4 | 4.95e-3 | 3.78e-4 | 0 | 93.0 | pass |

**Quick gate**: perplexity 2.02, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3_5_vl covers this model (45/45 tensor templates resolve, quant bf16) (PASS)
- plan: 19 ops resolved, exactness deterministic, loaded in 29.8s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_bf16 | exact | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 25 |
| rmsnorm_add | rmsnorm_add_bf16 | deterministic | yes | 64 |
| matmul | matmul_bf16_gemv | deterministic | yes | 120 |
| matmul | matmul_bf16_nax | deterministic | yes | 0 |
| logits | logits_bf16 | deterministic | yes | 0 |
| logits | logits_bf16_gemv | deterministic | yes | 1 |
| attn_prefill | attn_prefill_bf16_flash | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 8 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 56 |
| argmax | argmax_f32 | exact | yes | 1 |
| gate_split | gate_split_bf16 | deterministic | yes | 8 |
| sigmoid_mul | sigmoid_mul_bf16 | deterministic | yes | 8 |
| gdn_conv | gdn_conv_bf16 | deterministic | yes | 24 |
| gdn_delta_rule | gdn_delta_rule_bf16 | deterministic | yes | 24 |
| gdn_delta_rule | gdn_delta_rule_bf16_pre | deterministic | yes | 0 |
| matmul_pair | matmul_pair_bf16_gemv | deterministic | yes | 64 |
| matmul_pair | matmul_pair_bf16_nax | deterministic | yes | 0 |
| qk_norm_rope_kv | qk_norm_rope_kv_bf16 | deterministic | yes | 8 |

- kernels test: 46 kernels, 845 cases in 44.2s; incomplete: 1 kernel(s) had no cases in this sweep (untested, not failed; plan-time admission tests them) (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | unstamped | endor.kuepker.haus | 1793098660 | `/opt/code/intarsiaml/.worktrees/qgc/target/release/intarsia doctor --backend metal --device 0 --out /opt/code/intarsiaml/.worktrees/qgc/runs/onboard-2/qualify/doctor.json` |
| `recipe` | unstamped | endor.kuepker.haus | 1793098660 | `recipe::analyse(/Users/josef/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)` |
| `plan` | unstamped | endor.kuepker.haus | 1793098660 | `/opt/code/intarsiaml/.worktrees/qgc/target/release/intarsia plan show --model /Users/josef/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --backend metal --device 0 --admit --json` |
| `kernels` | unstamped | endor.kuepker.haus | 1793098690 | `/opt/code/intarsiaml/.worktrees/qgc/target/release/intarsia kernels test --backend metal --device 0 --model /Users/josef/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| `gate` | unstamped | endor.kuepker.haus | 1793098734 | `/opt/code/intarsiaml/.worktrees/qgc/target/release/intarsia accuracy --model /Users/josef/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --backend metal --device 0 --quick --max-seq 2048` |
| `acc-format` | unstamped | endor.kuepker.haus | 1793098741 | `-` |
| `acc-short` | unstamped | endor.kuepker.haus | 1793098741 | `/opt/code/intarsiaml/.worktrees/qgc/target/release/intarsia accuracy --model /Users/josef/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --backend metal --device 0 --golden /opt/code/intarsiaml/.worktrees/qgc/runs/onboard-2/qualify/goldens/qwen3.5-` |
| `acc-oracle` | unstamped | endor.kuepker.haus | 1793098942 | `-` |
| `bench` | unstamped | endor.kuepker.haus | 1793098942 | `/opt/code/intarsiaml/.worktrees/qgc/target/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-qwen3.5-4b --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-qwen3.5-4b-metal-bench --out-dir /opt/code/intarsiaml/.worktrees/qgc/runs/onboard-2/qualif` |

Model fingerprint: `fnv1a64:25692ed81280f6ab`.

Build stamp `unstamped` means a plain `cargo build`: baselines written here gate this binary, but `--pre-change` provenance checks refuse them as a "before".


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/kernels.out` (kernels)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `bench/qualify-qwen3.5-4b-metal-bench.json` (bench)
- `logs/bench-bench.out` (bench)
- `logs/bench-server.err` (bench)
- `state.json` (stage records; `--resume` reads it)

## Stage timeline

All times UTC. The qualify stages ran inside the `quick` / `in-depth` stages. A stage that was killed and resumed shows its last attempt; the qualify rows show every sub-stage with the time it actually ran.

| Stage | Tier | Verdict | Start | End | Duration | Note |
|---|---|---|---|---|---|---|
| recipe | prepare | PASS | 2026-10-27T10:57:38Z | 2026-10-27T10:57:38Z | 0s |  |
| smoke | prepare | PASS | 2026-10-27T10:57:38Z | 2026-10-27T10:57:40Z | 1s |  |
| quick | quick | PASS | 2026-10-27T10:57:40Z | 2026-10-27T11:03:16Z | 5m36s |  |
| doctor | qualify quick | PASS | 2026-10-27T10:57:40Z | 2026-10-27T10:57:40Z | 0s |  |
| recipe | qualify quick | PASS | 2026-10-27T10:57:40Z | 2026-10-27T10:57:40Z | 0s |  |
| plan | qualify quick | PASS | 2026-10-27T10:57:40Z | 2026-10-27T10:58:10Z | 30s |  |
| kernels | qualify quick | PASS | 2026-10-27T10:58:10Z | 2026-10-27T10:58:54Z | 44s |  |
| gate | qualify quick | PASS | 2026-10-27T10:58:54Z | 2026-10-27T10:59:01Z | 7s |  |
| acc-format | qualify quick | SKIP | 2026-10-27T10:59:01Z | 2026-10-27T10:59:01Z | 0s |  |
| acc-short | qualify quick | PASS | 2026-10-27T10:59:01Z | 2026-10-27T11:02:22Z | 3m20s |  |
| acc-oracle | qualify quick | SKIP | 2026-10-27T11:02:22Z | 2026-10-27T11:02:22Z | 0s |  |
| bench | qualify quick | PASS | 2026-10-27T11:02:22Z | 2026-10-27T11:03:16Z | 54s |  |

_Published 2026-10-27T11:03:16Z by `intarsia onboard` (report directory `/opt/code/intarsiaml/.worktrees/qgc/runs/onboard-2`)._
