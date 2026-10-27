# Qualification report: Qwen3.5-4B on metal

> **Onboard verdict: FAIL.** recipe: PASS; smoke: PASS; quick: FAIL (qualify exit 1 (a gate failed): acc-short FAIL: decode: top-1 (all rows) 98.75% (need >= 99.5)); in-depth: FAIL (qualify exit 1 (a gate failed): acc-short FAIL: decode: top-1 (all rows) 98.75% (need >= 99.5); acc-long FAIL: prefill: top-1 (all rows) 99.08% (need >= 99.5); decode: top-1 (all rows) 99.48% (need >= 99.5));

_Full (quick + in-depth) section, generated unix 1793094302, intarsia 0.0.1 build `unstamped`._

## Headline

**Verdict: FAIL** (15 stages: 8 pass, 2 fail, 0 error, 5 skipped)

| | |
|---|---|
| Model | Qwen3.5-4B (`/Users/josef/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`) |
| Backend / device | metal / Apple M5 Max (applegpu_g17s) |
| Recipe | builtin:qwen3_5_vl covers this model (45/45 tensor templates resolve, quant bf16) |
| Decode | 128 tok: **54.1** tok/s (74% of floor); 1024 tok: **60.5** tok/s (83% of floor); 4096 tok: **60.7** tok/s (83% of floor) |
| Prefill | 128 tok: **2180** tok/s; 1024 tok: **4182** tok/s; 4096 tok: **4068** tok/s |
| Accuracy (short golden) | **FAIL** (prefill top-1 99.72% KL 1.60e-4; decode top-1 98.75% KL 1.85e-4) |
| Accuracy (format reference, same weights) | skipped: the weights are not quantized: no format reference applies |
| Accuracy (long golden) | **FAIL** (prefill top-1 99.08% KL 3.14e-3; decode top-1 99.48% KL 1.45e-4) |
| Accuracy (4k golden) | skipped: the 4k golden is the long golden: covered by acc-long |
| Quick gate | **PASS**: perplexity 2.02, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 2m34s, in-depth 3m19s, total 5m53s |

### Failures

- `acc-short` **FAIL**: decode: top-1 (all rows) 98.75% (need >= 99.5)
- `acc-long` **FAIL**: prefill: top-1 (all rows) 99.08% (need >= 99.5); decode: top-1 (all rows) 99.48% (need >= 99.5)

### Skipped

- `acc-format`: the weights are not quantized: no format reference applies
- `acc-oracle`: no CPU-oracle golden for 'Qwen3.5-4B' (--oracle-golden, or oracle/<model>-cpu-oracle*.safetensors in a golden directory)
- `acc-4k`: the 4k golden is the long golden: covered by acc-long
- `ab`: no --reference server given
- `tp`: --tp not given (or 1)

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | Apple M5 Max (applegpu_g17s), 51 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_vl covers this model (45/45 tensor templates resolve, quant bf16) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 29s | 19 ops resolved, exactness deterministic, loaded in 29.2s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 42s | 46 kernels, 845 cases in 41.8s; incomplete: 1 kernel(s) had no cases in this sweep (untested, not failed; plan-time admission tests them) |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 7s | perplexity 2.02, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | the weights are not quantized: no format reference applies |
| 7 | `acc-short` short accuracy against a golden | quick | FAIL | 13s | decode: top-1 (all rows) 98.75% (need >= 99.5) |
| 8 | `acc-oracle` short prefill accuracy against the f64 CPU oracle (same weights, deterministic reference) | quick | SKIP | 0s | no CPU-oracle golden for 'Qwen3.5-4B' (--oracle-golden, or oracle/<model>-cpu-oracle*.safetensors in a golden directory) |
| 9 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 63s | decode 54.1 tok/s at 128 tokens (74% of the byte floor) |
| 10 | `acc-long` accuracy --mode all on the long golden (prefill, decode, prefix reuse) | full | FAIL | 1m57s | prefill: top-1 (all rows) 99.08% (need >= 99.5); decode: top-1 (all rows) 99.48% (need >= 99.5) |
| 11 | `acc-4k` accuracy --mode all on the 4k golden (prefill, decode, prefix reuse past 2 chunks of 2048) | full | SKIP | 0s | the 4k golden is the long golden: covered by acc-long |
| 12 | `envelope` noise envelope and baseline for regression gating | full | PASS | 24s | baseline and envelope (chunk) written; they gate this build: PASS |
| 13 | `bench-long` long-context bench up to 16k | full | PASS | 58s | decode 61.1 tok/s at 1024 tokens (84% of the byte floor) |
| 14 | `ab` A/B against the reference server via localeval | full | SKIP | 0s | no --reference server given |
| 15 | `tp` tensor-parallel variant | full | SKIP | 0s | --tp not given (or 1) |

### Notes

- `acc-short`: no baseline for backend metal next to the golden or in golden/baselines: KL, tail, prefix-reuse gates only, top-1 not gated
- `bench`: bench warning: size 128: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 1024: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 4096: 0 graph replays during the measured reps (decode ran eagerly)
- `acc-long`: prefix reuse at chunk 256: 3 judged, 0 skipped
- `bench-long`: bench warning: size 1024: 0 graph replays during the measured reps (decode ran eagerly)
- `bench-long`: bench warning: size 4096: 0 graph replays during the measured reps (decode ran eagerly)
- `bench-long`: bench warning: size 16384: 0 graph replays during the measured reps (decode ran eagerly)

## Speed

**quick bench** (floor: 8.41 GB/token at 614 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 614 GB/s, the Apple M5 Max (40-core GPU) spec figure of docs/decisions/0009-native-metal-backend.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 132 | 0.060 | 2180 | 54.13 | 1.79 | 73.0 | 74.2 |
| 1019 | 0.246 | 4182 | 60.54 | 0.53 | 73.0 | 82.9 |
| 4241 | 1.045 | 4068 | 60.70 | 0.13 | 73.0 | 83.2 |

**long-context bench** (floor: 8.41 GB/token at 614 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 614 GB/s, the Apple M5 Max (40-core GPU) spec figure of docs/decisions/0009-native-metal-backend.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 988 | 0.222 | 4464 | 61.06 | 0.58 | 73.0 | 83.7 |
| 4041 | 1.153 | 3504 | 60.22 | 0.42 | 73.0 | 82.5 |
| 16638 | 6.411 | 2598 | 58.71 | 0.36 | 73.0 | 80.4 |

## Accuracy

**acc-short** against `/opt/code/intarsiaml/.worktrees/nmd/.nmd/golden/qwen3.5-4b-fp32.safetensors` (found by model name; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1441 | 99.72 | 98.93 | 1.60e-4 | 1.43e-4 | 1.22e-2 | 3.76e-4 | 0 | 85.4 | pass |
| decode | 160 | 98.75 | 98.75 | 1.85e-4 | 1.70e-4 | 4.95e-3 | 4.12e-4 | 0 | 90.0 | FAIL |

**acc-long** against `/opt/code/intarsiaml/.worktrees/nmd/.nmd/golden/qwen3.5-4b-4k-fp32-top20.safetensors` (found by model name; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 12759 | 99.08 | 98.89 | 3.14e-3 | 3.19e-3 | 1.55e1 | 9.00e-4 | 0 | 4.3 | FAIL |
| decode | 192 | 99.48 | 99.17 | 1.45e-4 | 1.46e-4 | 1.59e-3 | 2.44e-4 | 0 | 93.8 | FAIL |

**Regression gating artifacts**: baseline and envelope (chunk) written; they gate this build: PASS

`intarsia accuracy --model <dir> --backend metal --golden /opt/code/intarsiaml/.worktrees/nmd/.nmd/golden/qwen3.5-4b-fp32.safetensors --baseline baselines/acc-short.baseline.json --noise-envelope baselines/envelope.json  (copy both files next to the golden, e.g. golden/baselines/)`

**Quick gate**: perplexity 2.02, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3_5_vl covers this model (45/45 tensor templates resolve, quant bf16) (PASS)
- plan: 19 ops resolved, exactness deterministic, loaded in 29.2s (PASS)

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

- kernels test: 46 kernels, 845 cases in 41.8s; incomplete: 1 kernel(s) had no cases in this sweep (untested, not failed; plan-time admission tests them) (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | unstamped | endor.kuepker.haus | 1793093948 | `/opt/code/intarsiaml/.worktrees/nmd/target/release/intarsia doctor --backend metal --device 0 --out .nmd/onboard-2/qualify/doctor.json` |
| `recipe` | unstamped | endor.kuepker.haus | 1793093949 | `recipe::analyse(/Users/josef/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)` |
| `plan` | unstamped | endor.kuepker.haus | 1793093949 | `/opt/code/intarsiaml/.worktrees/nmd/target/release/intarsia plan show --model /Users/josef/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --backend metal --device 0 --admit --json` |
| `kernels` | unstamped | endor.kuepker.haus | 1793093978 | `/opt/code/intarsiaml/.worktrees/nmd/target/release/intarsia kernels test --backend metal --device 0 --model /Users/josef/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| `gate` | unstamped | endor.kuepker.haus | 1793094020 | `/opt/code/intarsiaml/.worktrees/nmd/target/release/intarsia accuracy --model /Users/josef/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --backend metal --device 0 --quick --max-seq 2048` |
| `acc-format` | unstamped | endor.kuepker.haus | 1793094103 | `-` |
| `acc-short` | unstamped | endor.kuepker.haus | 1793094027 | `/opt/code/intarsiaml/.worktrees/nmd/target/release/intarsia accuracy --model /Users/josef/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --backend metal --device 0 --golden /opt/code/intarsiaml/.worktrees/nmd/.nmd/golden/qwen3.5-4b-fp32.safetensors` |
| `acc-oracle` | unstamped | endor.kuepker.haus | 1793094103 | `-` |
| `bench` | unstamped | endor.kuepker.haus | 1793094040 | `/opt/code/intarsiaml/.worktrees/nmd/target/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3.5-4B --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-Qwen3.5-4B-metal-bench --out-dir .nmd/onboard-2/qualify/bench --tp 1 --fingerprint model=Q` |
| `acc-long` | unstamped | endor.kuepker.haus | 1793094103 | `/opt/code/intarsiaml/.worktrees/nmd/target/release/intarsia accuracy --model /Users/josef/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --backend metal --device 0 --golden /opt/code/intarsiaml/.worktrees/nmd/.nmd/golden/qwen3.5-4b-4k-fp32-top20.sa` |
| `acc-4k` | unstamped | endor.kuepker.haus | 1793094220 | `-` |
| `envelope` | unstamped | endor.kuepker.haus | 1793094220 | `/opt/code/intarsiaml/.worktrees/nmd/target/release/intarsia accuracy --model /Users/josef/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --backend metal --device 0 --golden /opt/code/intarsiaml/.worktrees/nmd/.nmd/golden/qwen3.5-4b-fp32.safetensors` |
| `bench-long` | unstamped | endor.kuepker.haus | 1793094245 | `/opt/code/intarsiaml/.worktrees/nmd/target/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3.5-4B --sizes 1024,4096,16384 --gen 64 --reps 2 --warmup 1 --thinking off --label qualify-Qwen3.5-4B-metal-bench-long --out-dir .nmd/onboard-2/qualify/bench --tp 1 --fingerprint m` |
| `ab` | unstamped | endor.kuepker.haus | 1793094302 | `-` |
| `tp` | unstamped | endor.kuepker.haus | 1793094302 | `-` |

Model fingerprint: `fnv1a64:25692ed81280f6ab`.

Build stamp `unstamped` means a plain `cargo build`: baselines written here gate this binary, but `--pre-change` provenance checks refuse them as a "before".


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/kernels.out` (kernels)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `bench/qualify-Qwen3.5-4B-metal-bench.json` (bench)
- `logs/bench-bench.out` (bench)
- `logs/bench-server.err` (bench)
- `logs/acc-long.out` (acc-long)
- `baselines/acc-long.baseline.json` (acc-long)
- `logs/acc-long-reuse.out` (acc-long)
- `baselines/acc-short.baseline.json` (envelope)
- `baselines/envelope.json` (envelope)
- `logs/baseline-selfcheck.out` (envelope)
- `bench/qualify-Qwen3.5-4B-metal-bench-long.json` (bench-long)
- `logs/bench-long-bench.out` (bench-long)
- `logs/bench-long-server.err` (bench-long)
- `state.json` (stage records; `--resume` reads it)

## Stage timeline

All times UTC. The qualify stages ran inside the `quick` / `in-depth` stages. A stage that was killed and resumed shows its last attempt; the qualify rows show every sub-stage with the time it actually ran.

| Stage | Tier | Verdict | Start | End | Duration | Note |
|---|---|---|---|---|---|---|
| recipe | prepare | PASS | 2026-10-27T09:33:58Z | 2026-10-27T09:33:58Z | 0s |  |
| smoke | prepare | PASS | 2026-10-27T09:39:07Z | 2026-10-27T09:39:08Z | 1s | resumed; first attempt started 2026-10-27T09:33:58Z |
| publish-quick | quick | PASS | 2026-10-27T09:41:43Z | 2026-10-27T09:41:43Z | 0s |  |
| quick | quick | FAIL | 2026-10-27T09:39:08Z | 2026-10-27T09:41:43Z | 2m34s |  |
| in-depth | in-depth | FAIL | 2026-10-27T09:41:43Z | 2026-10-27T09:45:02Z | 3m19s |  |
| doctor | qualify quick | PASS | 2026-10-27T09:39:08Z | 2026-10-27T09:39:08Z | 0s |  |
| recipe | qualify quick | PASS | 2026-10-27T09:39:09Z | 2026-10-27T09:39:09Z | 0s |  |
| plan | qualify quick | PASS | 2026-10-27T09:39:09Z | 2026-10-27T09:39:38Z | 29s |  |
| kernels | qualify quick | PASS | 2026-10-27T09:39:38Z | 2026-10-27T09:40:20Z | 42s |  |
| gate | qualify quick | PASS | 2026-10-27T09:40:20Z | 2026-10-27T09:40:27Z | 7s |  |
| acc-format | qualify quick | SKIP | 2026-10-27T09:41:43Z | 2026-10-27T09:41:43Z | 0s |  |
| acc-short | qualify quick | FAIL | 2026-10-27T09:40:27Z | 2026-10-27T09:40:40Z | 13s |  |
| acc-oracle | qualify quick | SKIP | 2026-10-27T09:41:43Z | 2026-10-27T09:41:43Z | 0s |  |
| bench | qualify quick | PASS | 2026-10-27T09:40:40Z | 2026-10-27T09:41:43Z | 63s |  |
| acc-long | qualify full | FAIL | 2026-10-27T09:41:43Z | 2026-10-27T09:43:41Z | 1m57s |  |
| acc-4k | qualify full | SKIP | 2026-10-27T09:43:40Z | 2026-10-27T09:43:40Z | 0s |  |
| envelope | qualify full | PASS | 2026-10-27T09:43:40Z | 2026-10-27T09:44:04Z | 24s |  |
| bench-long | qualify full | PASS | 2026-10-27T09:44:05Z | 2026-10-27T09:45:03Z | 58s |  |
| ab | qualify full | SKIP | 2026-10-27T09:45:02Z | 2026-10-27T09:45:02Z | 0s |  |
| tp | qualify full | SKIP | 2026-10-27T09:45:02Z | 2026-10-27T09:45:02Z | 0s |  |

_Published 2026-10-27T09:45:02Z by `intarsia onboard` (report directory `.nmd/onboard-2`)._
