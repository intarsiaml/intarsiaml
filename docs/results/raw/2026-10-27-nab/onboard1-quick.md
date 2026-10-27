# Qualification quick report: Qwen3.5-2B-MLX-4bit on metal

> **Onboard verdict: PASS.** recipe: PASS; smoke: PASS; quick: PASS;

_Quick section, generated unix 1793102445, intarsia 0.0.1 build `unstamped`._

## Headline

**Verdict: PASS** (9 stages: 7 pass, 0 fail, 0 error, 2 skipped)

| | |
|---|---|
| Model | Qwen3.5-2B-MLX-4bit (`/Users/josef/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-MLX-4bit/snapshots/93760be4f1f69842a46bc13dbdc0f19e291392a3`) |
| Backend / device | metal / Apple M5 Max (applegpu_g17s) |
| Recipe | builtin:qwen3_5_vl covers this model (45/45 tensor templates resolve, quant bf16 (MLX affine 4-bit dequantised at load)) |
| Decode | 128 tok: **107.2** tok/s (66% of floor); 1024 tok: **105.0** tok/s (64% of floor); 4096 tok: **105.7** tok/s (65% of floor) |
| Prefill | 128 tok: **4593** tok/s; 1024 tok: **10958** tok/s; 4096 tok: **8874** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 99.55% KL 1.39e-4; decode top-1 100.00% KL 9.74e-5) |
| Accuracy (format reference, same weights) | skipped: no format-reference golden for 'Qwen3.5-2B-MLX-4bit' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| Quick gate | **PASS**: perplexity 4.58, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 2m29s |
| Reference | /opt/code/intarsiaml/.worktrees/nab/.work/qwen3.5-2b-mlx4-dequant |

### Failures

None.

### Skipped

- `acc-format`: no format-reference golden for 'Qwen3.5-2B-MLX-4bit' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE)
- `acc-oracle`: no CPU-oracle golden for 'Qwen3.5-2B-MLX-4bit' (--oracle-golden, or oracle/<model>-cpu-oracle*.safetensors in a golden directory)

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | Apple M5 Max (applegpu_g17s), 51 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_vl covers this model (45/45 tensor templates resolve, quant bf16 (MLX affine 4-bit dequantised at load)) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 19s | 19 ops resolved, exactness deterministic, loaded in 18.8s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 32s | 46 kernels, 845 cases in 31.4s; incomplete: 1 kernel(s) had no cases in this sweep (untested, not failed; plan-time admission tests them) |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 5s | perplexity 4.58, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | no format-reference golden for 'Qwen3.5-2B-MLX-4bit' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| 7 | `acc-short` short accuracy against a golden | quick | PASS | 62s | prefill top-1 99.55% KL 1.39e-4; decode top-1 100.00% KL 9.74e-5 |
| 8 | `acc-oracle` short prefill accuracy against the f64 CPU oracle (same weights, deterministic reference) | quick | SKIP | 0s | no CPU-oracle golden for 'Qwen3.5-2B-MLX-4bit' (--oracle-golden, or oracle/<model>-cpu-oracle*.safetensors in a golden directory) |
| 9 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 32s | decode 107.2 tok/s at 128 tokens (66% of the byte floor) |

### Notes

- `acc-short`: no baseline for backend metal next to the golden or in golden/baselines: KL, tail, prefix-reuse gates only, top-1 not gated
- `bench`: bench warning: size 128: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 1024: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 4096: 0 graph replays during the measured reps (decode ran eagerly)

## Speed

**quick bench** (floor: 3.76 GB/token at 614 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 614 GB/s, the Apple M5 Max (40-core GPU) spec figure of docs/decisions/0009-native-metal-backend.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 130 | 0.028 | 4593 | 107.25 | 4.18 | 163.1 | 65.7 |
| 1021 | 0.093 | 10958 | 105.04 | 3.12 | 163.1 | 64.4 |
| 4207 | 0.481 | 8874 | 105.69 | 4.71 | 163.1 | 64.8 |

## Accuracy

**acc-short** against `/opt/code/intarsiaml/.worktrees/nab/.work/onboard1/qualify/goldens/Qwen3.5-2B-MLX-4bit-short.safetensors` (made with tools/golden.py from /opt/code/intarsiaml/.worktrees/nab/.work/qwen3.5-2b-mlx4-dequant (HF transformers fp32 on cpu, bf16 spread recorded) in 57s; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 441 | 99.55 | 99.27 | 1.39e-4 | 1.19e-4 | 1.11e-2 | 5.05e-4 | 0 | 80.3 | pass |
| decode | 128 | 100.00 | 98.91 | 9.74e-5 | 8.01e-5 | 1.22e-3 | 4.68e-4 | 0 | 100.0 | pass |

**Quick gate**: perplexity 4.58, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3_5_vl covers this model (45/45 tensor templates resolve, quant bf16 (MLX affine 4-bit dequantised at load)) (PASS)
- plan: 19 ops resolved, exactness deterministic, loaded in 18.8s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_bf16 | exact | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 19 |
| rmsnorm_add | rmsnorm_add_bf16 | deterministic | yes | 48 |
| matmul | matmul_bf16_gemv | deterministic | yes | 90 |
| matmul | matmul_bf16_nax | deterministic | yes | 0 |
| logits | logits_bf16 | deterministic | yes | 0 |
| logits | logits_bf16_gemv | deterministic | yes | 1 |
| attn_prefill | attn_prefill_bf16_flash | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 6 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 42 |
| argmax | argmax_f32 | exact | yes | 1 |
| gate_split | gate_split_bf16 | deterministic | yes | 6 |
| sigmoid_mul | sigmoid_mul_bf16 | deterministic | yes | 6 |
| gdn_conv | gdn_conv_bf16 | deterministic | yes | 18 |
| gdn_delta_rule | gdn_delta_rule_bf16 | deterministic | yes | 18 |
| gdn_delta_rule | gdn_delta_rule_bf16_pre | deterministic | yes | 0 |
| matmul_pair | matmul_pair_bf16_gemv | deterministic | yes | 48 |
| matmul_pair | matmul_pair_bf16_nax | deterministic | yes | 0 |
| qk_norm_rope_kv | qk_norm_rope_kv_bf16 | deterministic | yes | 6 |

- kernels test: 46 kernels, 845 cases in 31.4s; incomplete: 1 kernel(s) had no cases in this sweep (untested, not failed; plan-time admission tests them) (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | unstamped | endor.kuepker.haus | 1793102295 | `/opt/code/intarsiaml/.worktrees/nab/target/release/intarsia doctor --backend metal --device 0 --out /opt/code/intarsiaml/.worktrees/nab/.work/onboard1/qualify/doctor.json` |
| `recipe` | unstamped | endor.kuepker.haus | 1793102296 | `recipe::analyse(/Users/josef/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-MLX-4bit/snapshots/93760be4f1f69842a46bc13dbdc0f19e291392a3)` |
| `plan` | unstamped | endor.kuepker.haus | 1793102296 | `/opt/code/intarsiaml/.worktrees/nab/target/release/intarsia plan show --model /Users/josef/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-MLX-4bit/snapshots/93760be4f1f69842a46bc13dbdc0f19e291392a3 --backend metal --device 0 --admit --json` |
| `kernels` | unstamped | endor.kuepker.haus | 1793102314 | `/opt/code/intarsiaml/.worktrees/nab/target/release/intarsia kernels test --backend metal --device 0 --model /Users/josef/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-MLX-4bit/snapshots/93760be4f1f69842a46bc13dbdc0f19e291392a3` |
| `gate` | unstamped | endor.kuepker.haus | 1793102346 | `/opt/code/intarsiaml/.worktrees/nab/target/release/intarsia accuracy --model /Users/josef/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-MLX-4bit/snapshots/93760be4f1f69842a46bc13dbdc0f19e291392a3 --backend metal --device 0 --quick --max-seq 2048` |
| `acc-format` | unstamped | endor.kuepker.haus | 1793102351 | `-` |
| `acc-short` | unstamped | endor.kuepker.haus | 1793102351 | `/opt/code/intarsiaml/.worktrees/nab/target/release/intarsia accuracy --model /Users/josef/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-MLX-4bit/snapshots/93760be4f1f69842a46bc13dbdc0f19e291392a3 --backend metal --device 0 --golden /opt/code/intarsiaml/.worktrees/nab/.work/onboard1/qualif` |
| `acc-oracle` | unstamped | endor.kuepker.haus | 1793102413 | `-` |
| `bench` | unstamped | endor.kuepker.haus | 1793102413 | `/opt/code/intarsiaml/.worktrees/nab/target/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3.5-2B-MLX-4bit --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-Qwen3.5-2B-MLX-4bit-metal-bench --out-dir /opt/code/intarsiaml/.worktrees/nab/.wo` |

Model fingerprint: `fnv1a64:dc7e74d5648eaad5`.

Build stamp `unstamped` means a plain `cargo build`: baselines written here gate this binary, but `--pre-change` provenance checks refuse them as a "before".


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/kernels.out` (kernels)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `bench/qualify-Qwen3.5-2B-MLX-4bit-metal-bench.json` (bench)
- `logs/bench-bench.out` (bench)
- `logs/bench-server.err` (bench)
- `state.json` (stage records; `--resume` reads it)

## Stage timeline

All times UTC. The qualify stages ran inside the `quick` / `in-depth` stages. A stage that was killed and resumed shows its last attempt; the qualify rows show every sub-stage with the time it actually ran.

| Stage | Tier | Verdict | Start | End | Duration | Note |
|---|---|---|---|---|---|---|
| recipe | prepare | PASS | 2026-10-27T11:58:14Z | 2026-10-27T11:58:14Z | 0s |  |
| smoke | prepare | PASS | 2026-10-27T11:58:14Z | 2026-10-27T11:58:15Z | 1s |  |
| quick | quick | PASS | 2026-10-27T11:58:15Z | 2026-10-27T12:00:45Z | 2m30s |  |
| doctor | qualify quick | PASS | 2026-10-27T11:58:15Z | 2026-10-27T11:58:15Z | 0s |  |
| recipe | qualify quick | PASS | 2026-10-27T11:58:16Z | 2026-10-27T11:58:16Z | 0s |  |
| plan | qualify quick | PASS | 2026-10-27T11:58:16Z | 2026-10-27T11:58:35Z | 19s |  |
| kernels | qualify quick | PASS | 2026-10-27T11:58:34Z | 2026-10-27T11:59:06Z | 32s |  |
| gate | qualify quick | PASS | 2026-10-27T11:59:06Z | 2026-10-27T11:59:11Z | 5s |  |
| acc-format | qualify quick | SKIP | 2026-10-27T11:59:11Z | 2026-10-27T11:59:11Z | 0s |  |
| acc-short | qualify quick | PASS | 2026-10-27T11:59:11Z | 2026-10-27T12:00:13Z | 62s |  |
| acc-oracle | qualify quick | SKIP | 2026-10-27T12:00:13Z | 2026-10-27T12:00:13Z | 0s |  |
| bench | qualify quick | PASS | 2026-10-27T12:00:13Z | 2026-10-27T12:00:45Z | 32s |  |

_Published 2026-10-27T12:00:45Z by `intarsia onboard` (report directory `/opt/code/intarsiaml/.worktrees/nab/.work/onboard1`)._
