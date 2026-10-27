# Qualification report: Qwen3.5-2B-MLX-4bit on metal

> **Onboard verdict: PASS.** recipe: PASS; smoke: PASS; quick: PASS; in-depth: PASS;

_Full (quick + in-depth) section, generated unix 1793104282, intarsia 0.0.1 build `unstamped`._

## Headline

**Verdict: PASS** (15 stages: 11 pass, 0 fail, 0 error, 4 skipped)

| | |
|---|---|
| Model | Qwen3.5-2B-MLX-4bit (`/Users/josef/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-MLX-4bit/snapshots/93760be4f1f69842a46bc13dbdc0f19e291392a3`) |
| Backend / device | metal / Apple M5 Max (applegpu_g17s) |
| Recipe | builtin:qwen3_5_vl covers this model (45/45 tensor templates resolve, quant bf16 (MLX affine 4-bit dequantised at load)) |
| Decode | 128 tok: **129.6** tok/s (79% of floor); 1024 tok: **132.2** tok/s (81% of floor); 4096 tok: **131.2** tok/s (80% of floor) |
| Prefill | 128 tok: **5532** tok/s; 1024 tok: **10157** tok/s; 4096 tok: **10188** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 99.55% KL 1.39e-4; decode top-1 100.00% KL 9.74e-5) |
| Accuracy (format reference, same weights) | skipped: no format-reference golden for 'Qwen3.5-2B-MLX-4bit' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| Accuracy (long golden) | **PASS** (prefill top-1 98.75% KL 6.23e-4; decode top-1 99.40% KL 1.69e-4) |
| Accuracy (4k golden) | **PASS** (prefill top-1 99.30% KL 2.89e-4; decode top-1 99.48% KL 1.61e-4) |
| Quick gate | **PASS**: perplexity 4.58, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 2m16s, in-depth 1m57s, total 4m14s |
| Reference | /opt/code/intarsiaml/.worktrees/nab/.work/qwen3.5-2b-mlx4-dequant |

### Failures

None.

### Skipped

- `acc-format`: no format-reference golden for 'Qwen3.5-2B-MLX-4bit' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE)
- `acc-oracle`: no CPU-oracle golden for 'Qwen3.5-2B-MLX-4bit' (--oracle-golden, or oracle/<model>-cpu-oracle*.safetensors in a golden directory)
- `ab`: --reference is a model directory, not a server URL: there is nothing to A/B against
- `tp`: --tp not given (or 1)

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | Apple M5 Max (applegpu_g17s), 51 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_vl covers this model (45/45 tensor templates resolve, quant bf16 (MLX affine 4-bit dequantised at load)) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 16s | 19 ops resolved, exactness deterministic, loaded in 16.3s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 31s | 46 kernels, 845 cases in 30.9s; incomplete: 1 kernel(s) had no cases in this sweep (untested, not failed; plan-time admission tests them) |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 5s | perplexity 4.58, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | no format-reference golden for 'Qwen3.5-2B-MLX-4bit' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| 7 | `acc-short` short accuracy against a golden | quick | PASS | 54s | prefill top-1 99.55% KL 1.39e-4; decode top-1 100.00% KL 9.74e-5 |
| 8 | `acc-oracle` short prefill accuracy against the f64 CPU oracle (same weights, deterministic reference) | quick | SKIP | 0s | no CPU-oracle golden for 'Qwen3.5-2B-MLX-4bit' (--oracle-golden, or oracle/<model>-cpu-oracle*.safetensors in a golden directory) |
| 9 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 30s | decode 129.6 tok/s at 128 tokens (79% of the byte floor) |
| 10 | `acc-long` accuracy --mode all on the long golden (prefill, decode, prefix reuse) | full | PASS | 16s | prefill top-1 98.75% KL 6.23e-4; decode top-1 99.40% KL 1.69e-4 |
| 11 | `acc-4k` accuracy --mode all on the 4k golden (prefill, decode, prefix reuse past 2 chunks of 2048) | full | PASS | 60s | prefill top-1 99.30% KL 2.89e-4; decode top-1 99.48% KL 1.61e-4 |
| 12 | `envelope` noise envelope and baseline for regression gating | full | PASS | 11s | baseline and envelope (chunk) written; they gate this build: PASS |
| 13 | `bench-long` long-context bench up to 16k | full | PASS | 31s | decode 129.3 tok/s at 1024 tokens (79% of the byte floor) |
| 14 | `ab` A/B against the reference server via localeval | full | SKIP | 0s | --reference is a model directory, not a server URL: there is nothing to A/B against |
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

**quick bench** (floor: 3.76 GB/token at 614 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 614 GB/s, the Apple M5 Max (40-core GPU) spec figure of docs/decisions/0009-native-metal-backend.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 130 | 0.024 | 5532 | 129.55 | 1.05 | 163.1 | 79.4 |
| 1030 | 0.103 | 10157 | 132.15 | 0.32 | 163.1 | 81.0 |
| 4212 | 0.412 | 10188 | 131.18 | 2.39 | 163.1 | 80.4 |

**long-context bench** (floor: 3.76 GB/token at 614 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 614 GB/s, the Apple M5 Max (40-core GPU) spec figure of docs/decisions/0009-native-metal-backend.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 1003 | 0.090 | 11144 | 129.27 | 2.03 | 163.1 | 79.2 |
| 4108 | 0.411 | 10003 | 131.85 | 0.13 | 163.1 | 80.8 |
| 16768 | 2.568 | 6530 | 127.50 | 0.22 | 163.1 | 78.2 |

## Accuracy

**acc-short** against `/opt/code/intarsiaml/.worktrees/nab/.work/onboard2/qualify/goldens/Qwen3.5-2B-MLX-4bit-short.safetensors` (made with tools/golden.py from /opt/code/intarsiaml/.worktrees/nab/.work/qwen3.5-2b-mlx4-dequant (HF transformers fp32 on cpu, bf16 spread recorded) in 49s; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 441 | 99.55 | 99.27 | 1.39e-4 | 1.19e-4 | 1.11e-2 | 5.05e-4 | 0 | 80.3 | pass |
| decode | 128 | 100.00 | 98.91 | 9.74e-5 | 8.01e-5 | 1.22e-3 | 4.68e-4 | 0 | 100.0 | pass |

**acc-long** against `/opt/code/intarsiaml/.worktrees/nab/.work/goldens/Qwen3.5-2B-MLX-4bit-long-fp32.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1441 | 98.75 | 98.54 | 6.23e-4 | 5.30e-4 | 1.11e-2 | 1.13e-3 | 0 | 29.8 | pass |
| decode | 500 | 99.40 | 99.16 | 1.69e-4 | 1.43e-4 | 5.85e-3 | 5.50e-4 | 0 | 91.6 | pass |

**acc-4k** against `/opt/code/intarsiaml/.worktrees/nab/.work/goldens/Qwen3.5-2B-MLX-4bit-4k-fp32-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 12759 | 99.30 | 98.98 | 2.89e-4 | 2.70e-4 | 7.67e-1 | 9.54e-4 | 0 | 2.6 | pass |
| decode | 192 | 99.48 | 98.85 | 1.61e-4 | 1.54e-4 | 1.09e-3 | 5.29e-4 | 0 | 72.9 | pass |

**Regression gating artifacts**: baseline and envelope (chunk) written; they gate this build: PASS

`intarsia accuracy --model <dir> --backend metal --golden /opt/code/intarsiaml/.worktrees/nab/.work/onboard2/qualify/goldens/Qwen3.5-2B-MLX-4bit-short.safetensors --baseline baselines/acc-short.baseline.json --noise-envelope baselines/envelope.json  (copy both files next to the golden, e.g. golden/baselines/)`

**Quick gate**: perplexity 4.58, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3_5_vl covers this model (45/45 tensor templates resolve, quant bf16 (MLX affine 4-bit dequantised at load)) (PASS)
- plan: 19 ops resolved, exactness deterministic, loaded in 16.3s (PASS)

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

- kernels test: 46 kernels, 845 cases in 30.9s; incomplete: 1 kernel(s) had no cases in this sweep (untested, not failed; plan-time admission tests them) (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | unstamped | endor.kuepker.haus | 1793104027 | `/opt/code/intarsiaml/.worktrees/nab/target/release/intarsia doctor --backend metal --device 0 --out /opt/code/intarsiaml/.worktrees/nab/.work/onboard2/qualify/doctor.json` |
| `recipe` | unstamped | endor.kuepker.haus | 1793104028 | `recipe::analyse(/Users/josef/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-MLX-4bit/snapshots/93760be4f1f69842a46bc13dbdc0f19e291392a3)` |
| `plan` | unstamped | endor.kuepker.haus | 1793104028 | `/opt/code/intarsiaml/.worktrees/nab/target/release/intarsia plan show --model /Users/josef/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-MLX-4bit/snapshots/93760be4f1f69842a46bc13dbdc0f19e291392a3 --backend metal --device 0 --admit --json` |
| `kernels` | unstamped | endor.kuepker.haus | 1793104044 | `/opt/code/intarsiaml/.worktrees/nab/target/release/intarsia kernels test --backend metal --device 0 --model /Users/josef/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-MLX-4bit/snapshots/93760be4f1f69842a46bc13dbdc0f19e291392a3` |
| `gate` | unstamped | endor.kuepker.haus | 1793104075 | `/opt/code/intarsiaml/.worktrees/nab/target/release/intarsia accuracy --model /Users/josef/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-MLX-4bit/snapshots/93760be4f1f69842a46bc13dbdc0f19e291392a3 --backend metal --device 0 --quick --max-seq 2048` |
| `acc-format` | unstamped | endor.kuepker.haus | 1793104164 | `-` |
| `acc-short` | unstamped | endor.kuepker.haus | 1793104080 | `/opt/code/intarsiaml/.worktrees/nab/target/release/intarsia accuracy --model /Users/josef/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-MLX-4bit/snapshots/93760be4f1f69842a46bc13dbdc0f19e291392a3 --backend metal --device 0 --golden /opt/code/intarsiaml/.worktrees/nab/.work/onboard2/qualif` |
| `acc-oracle` | unstamped | endor.kuepker.haus | 1793104164 | `-` |
| `bench` | unstamped | endor.kuepker.haus | 1793104135 | `/opt/code/intarsiaml/.worktrees/nab/target/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3.5-2B-MLX-4bit --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-Qwen3.5-2B-MLX-4bit-metal-bench --out-dir /opt/code/intarsiaml/.worktrees/nab/.wo` |
| `acc-long` | unstamped | endor.kuepker.haus | 1793104164 | `/opt/code/intarsiaml/.worktrees/nab/target/release/intarsia accuracy --model /Users/josef/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-MLX-4bit/snapshots/93760be4f1f69842a46bc13dbdc0f19e291392a3 --backend metal --device 0 --golden /opt/code/intarsiaml/.worktrees/nab/.work/goldens/Qwen3.5` |
| `acc-4k` | unstamped | endor.kuepker.haus | 1793104180 | `/opt/code/intarsiaml/.worktrees/nab/target/release/intarsia accuracy --model /Users/josef/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-MLX-4bit/snapshots/93760be4f1f69842a46bc13dbdc0f19e291392a3 --backend metal --device 0 --golden /opt/code/intarsiaml/.worktrees/nab/.work/goldens/Qwen3.5` |
| `envelope` | unstamped | endor.kuepker.haus | 1793104240 | `/opt/code/intarsiaml/.worktrees/nab/target/release/intarsia accuracy --model /Users/josef/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-MLX-4bit/snapshots/93760be4f1f69842a46bc13dbdc0f19e291392a3 --backend metal --device 0 --golden /opt/code/intarsiaml/.worktrees/nab/.work/onboard2/qualif` |
| `bench-long` | unstamped | endor.kuepker.haus | 1793104251 | `/opt/code/intarsiaml/.worktrees/nab/target/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3.5-2B-MLX-4bit --sizes 1024,4096,16384 --gen 64 --reps 2 --warmup 1 --thinking off --label qualify-Qwen3.5-2B-MLX-4bit-metal-bench-long --out-dir /opt/code/intarsiaml/.worktrees/n` |
| `ab` | unstamped | endor.kuepker.haus | 1793104282 | `-` |
| `tp` | unstamped | endor.kuepker.haus | 1793104282 | `-` |

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
- `logs/acc-long.out` (acc-long)
- `baselines/acc-long.baseline.json` (acc-long)
- `logs/acc-4k.out` (acc-4k)
- `baselines/acc-4k.baseline.json` (acc-4k)
- `logs/acc-4k-reuse.out` (acc-4k)
- `baselines/acc-short.baseline.json` (envelope)
- `baselines/envelope.json` (envelope)
- `logs/baseline-selfcheck.out` (envelope)
- `bench/qualify-Qwen3.5-2B-MLX-4bit-metal-bench-long.json` (bench-long)
- `logs/bench-long-bench.out` (bench-long)
- `logs/bench-long-server.err` (bench-long)
- `state.json` (stage records; `--resume` reads it)

## Stage timeline

All times UTC. The qualify stages ran inside the `quick` / `in-depth` stages. A stage that was killed and resumed shows its last attempt; the qualify rows show every sub-stage with the time it actually ran.

| Stage | Tier | Verdict | Start | End | Duration | Note |
|---|---|---|---|---|---|---|
| recipe | prepare | PASS | 2026-10-27T12:27:06Z | 2026-10-27T12:27:06Z | 0s |  |
| smoke | prepare | PASS | 2026-10-27T12:27:06Z | 2026-10-27T12:27:07Z | 2s |  |
| quick | quick | PASS | 2026-10-27T12:27:07Z | 2026-10-27T12:29:24Z | 2m17s |  |
| publish-quick | quick | PASS | 2026-10-27T12:29:24Z | 2026-10-27T12:29:24Z | 0s |  |
| in-depth | in-depth | PASS | 2026-10-27T12:29:24Z | 2026-10-27T12:31:22Z | 1m57s |  |
| doctor | qualify quick | PASS | 2026-10-27T12:27:07Z | 2026-10-27T12:27:07Z | 0s |  |
| recipe | qualify quick | PASS | 2026-10-27T12:27:08Z | 2026-10-27T12:27:08Z | 0s |  |
| plan | qualify quick | PASS | 2026-10-27T12:27:08Z | 2026-10-27T12:27:24Z | 16s |  |
| kernels | qualify quick | PASS | 2026-10-27T12:27:24Z | 2026-10-27T12:27:55Z | 31s |  |
| gate | qualify quick | PASS | 2026-10-27T12:27:55Z | 2026-10-27T12:28:00Z | 5s |  |
| acc-format | qualify quick | SKIP | 2026-10-27T12:29:24Z | 2026-10-27T12:29:24Z | 0s |  |
| acc-short | qualify quick | PASS | 2026-10-27T12:28:00Z | 2026-10-27T12:28:54Z | 54s |  |
| acc-oracle | qualify quick | SKIP | 2026-10-27T12:29:24Z | 2026-10-27T12:29:24Z | 0s |  |
| bench | qualify quick | PASS | 2026-10-27T12:28:55Z | 2026-10-27T12:29:25Z | 30s |  |
| acc-long | qualify full | PASS | 2026-10-27T12:29:24Z | 2026-10-27T12:29:40Z | 16s |  |
| acc-4k | qualify full | PASS | 2026-10-27T12:29:40Z | 2026-10-27T12:30:40Z | 60s |  |
| envelope | qualify full | PASS | 2026-10-27T12:30:40Z | 2026-10-27T12:30:51Z | 11s |  |
| bench-long | qualify full | PASS | 2026-10-27T12:30:51Z | 2026-10-27T12:31:22Z | 31s |  |
| ab | qualify full | SKIP | 2026-10-27T12:31:22Z | 2026-10-27T12:31:22Z | 0s |  |
| tp | qualify full | SKIP | 2026-10-27T12:31:22Z | 2026-10-27T12:31:22Z | 0s |  |

_Published 2026-10-27T12:31:22Z by `intarsia onboard` (report directory `/opt/code/intarsiaml/.worktrees/nab/.work/onboard2`)._
