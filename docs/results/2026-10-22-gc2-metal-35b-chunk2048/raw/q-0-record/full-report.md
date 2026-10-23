# Qualification report: Qwen3.6-35B-A3B-NVFP4 on metal

_Full (quick + in-depth) section, generated unix 1792720886, intarsia 0.0.1 build `unstamped`._

## Headline

**Verdict: FAIL** (15 stages: 5 pass, 1 fail, 2 missing baseline, 0 error, 7 skipped)

| | |
|---|---|
| Model | Qwen3.6-35B-A3B-NVFP4 (`/Users/josef/Models/Qwen3.6-35B-A3B-NVFP4`) |
| Backend / device | metal / Apple M5 Max (applegpu_g17s) |
| Recipe | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| Decode | not measured |
| Prefill | not measured |
| Accuracy (short golden) | **MISSING-BASELINE** (prefill top-1 98.33% KL 7.04e-3; decode top-1 98.75% KL 1.11e-2) |
| Accuracy (format reference, same weights) | skipped: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| Accuracy (long golden) | **PASS** (prefill top-1 94.25% KL 4.03e-2; decode top-1 96.25% KL 9.51e-3) |
| Accuracy (4k golden) | **FAIL** (prefill top-1 93.14% KL 8.14e-2; decode top-1 98.96% KL 7.22e-3) |
| Quick gate | **SKIP**: skipped by --skip |
| Wall time | quick 40s, in-depth 5m57s, total 6m36s |

### Failures

- `acc-short` **MISSING-BASELINE**: MISSING-BASELINE: no baseline for backend metal next to the golden or in golden/baselines: KL, tail, prefix-reuse gates only, top-1 not gated. Qwen3.6-35B-A3B-NVFP4 has no full-precision golden (MoE / hybrid / quantized, golden from a reference server), so acc-short's floor alone does not gate it (i
- `acc-oracle` **MISSING-BASELINE**: MISSING-BASELINE: no recorded dump qwen36-35b-a3b-nvfp4-cpu-oracle.metal.selfgolden.safetensors in golden/baselines; the oracle floor passed (prefill top-1 96.26% KL 1.09e-2), but on this model only a recorded same-backend dump can see a 1 % scale bug. Record it from clean main: copy baselines/qwen3
- `acc-4k` **FAIL**: prefix reuse differs from a cold run; prefix reuse at chunk 1024: 3 judged, 0 skipped: prefix reuse differs from a cold run

### Skipped

- `kernels`: skipped by --skip
- `gate`: skipped by --skip
- `acc-format`: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE)
- `bench`: skipped by --skip
- `bench-long`: skipped by --skip
- `ab`: skipped by --skip
- `tp`: skipped by --skip

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | Apple M5 Max (applegpu_g17s), 41 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 7s | 21 ops resolved, exactness deterministic, loaded in 7.2s |
| 4 | `kernels` kernels test on the model's shapes | quick | SKIP | 0s | skipped by --skip |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | SKIP | 0s | skipped by --skip |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| 7 | `acc-short` short accuracy against a golden | quick | MISSING-BASELINE | 12s | MISSING-BASELINE: no baseline for backend metal next to the golden or in golden/baselines: KL, tail, prefix-reuse gates only, top-1 not gated. Qwen3.6-35B-A3B-NVFP4 has no full-precision golden (MoE / hybrid / quantized, golden from a reference server), so acc-short's floor alone does not gate it (i |
| 8 | `acc-oracle` short prefill accuracy against the f64 CPU oracle (same weights, deterministic reference) | quick | MISSING-BASELINE | 20s | MISSING-BASELINE: no recorded dump qwen36-35b-a3b-nvfp4-cpu-oracle.metal.selfgolden.safetensors in golden/baselines; the oracle floor passed (prefill top-1 96.26% KL 1.09e-2), but on this model only a recorded same-backend dump can see a 1 % scale bug. Record it from clean main: copy baselines/qwen3 |
| 9 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | SKIP | 0s | skipped by --skip |
| 10 | `acc-long` accuracy --mode all on the long golden (prefill, decode, prefix reuse) | full | PASS | 2m02s | prefill top-1 94.25% KL 4.03e-2; decode top-1 96.25% KL 9.51e-3 |
| 11 | `acc-4k` accuracy --mode all on the 4k golden (prefill, decode, prefix reuse past 2 chunks of 2048) | full | FAIL | 3m30s | prefix reuse differs from a cold run; prefix reuse at chunk 1024: 3 judged, 0 skipped: prefix reuse differs from a cold run |
| 12 | `envelope` noise envelope and baseline for regression gating | full | PASS | 24s | baseline and envelope (chunk) written; they gate this build: PASS |
| 13 | `bench-long` long-context bench up to 16k | full | SKIP | 0s | skipped by --skip |
| 14 | `ab` A/B against the reference server via localeval | full | SKIP | 0s | skipped by --skip |
| 15 | `tp` tensor-parallel variant | full | SKIP | 0s | skipped by --skip |

### Notes

- `acc-long`: prefix reuse is gated in acc-4k, on the 4k golden
- `acc-4k`: no baseline for backend metal next to the golden or in golden/baselines: KL, tail, prefix-reuse gates only, top-1 not gated

## Speed

No speed numbers.

## Accuracy

**acc-short** against `/opt/code/wt-gc/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors` (given with --golden; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1441 | 98.33 | 91.30 | 7.04e-3 | 6.91e-3 | 7.96e-1 | 1.75e-3 | 0 | 6.6 | pass |
| decode | 160 | 98.75 | 92.25 | 1.11e-2 | 1.16e-2 | 4.65e-1 | 3.16e-3 | 0 | 88.8 | pass |

**acc-long** against `/opt/code/intarsiaml/golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 36887 | 94.25 | 90.37 | 4.03e-2 | 4.26e-2 | 2.23e1 | 7.47e-3 | 0 | 0.3 | pass |
| decode | 320 | 96.25 | 89.75 | 9.51e-3 | 8.83e-3 | 1.44e-1 | 2.82e-3 | 0 | 48.1 | pass |

**acc-4k** against `golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 12759 | 93.14 | 90.18 | 8.14e-2 | 9.09e-2 | 2.21e1 | 1.15e-2 | 0 | 0.8 | pass |
| decode | 192 | 98.96 | 90.62 | 7.22e-3 | 6.90e-3 | 1.12e-1 | 1.79e-3 | 0 | 91.7 | pass |

**Regression gating artifacts**: baseline and envelope (chunk) written; they gate this build: PASS

`intarsia accuracy --model <dir> --backend metal --golden /opt/code/wt-gc/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --baseline baselines/acc-short.baseline.json --noise-envelope baselines/envelope.json  (copy both files next to the golden, e.g. golden/baselines/)`

**Quick gate**: skipped by --skip

## Plan and recipe

- recipe: builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) (PASS)
- plan: 21 ops resolved, exactness deterministic, loaded in 7.2s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_bf16 | exact | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 31 |
| rmsnorm_add | rmsnorm_add_bf16 | deterministic | yes | 80 |
| attn_prefill | attn_prefill_bf16_flash | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 10 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 70 |
| argmax | argmax_f32 | exact | yes | 1 |
| matmul_fp8 | matmul_fp8_gemv | deterministic | yes | 130 |
| matmul_fp8 | matmul_fp8_mma | deterministic | yes | 0 |
| matmul_nvfp4 | matmul_nvfp4_gemv | deterministic | yes | 121 |
| matmul_nvfp4 | matmul_nvfp4_mma | deterministic | yes | 0 |
| moe_topk | moe_topk_bf16 | deterministic | yes | 40 |
| moe_experts | moe_experts_bf16 | deterministic | yes | 40 |
| moe_combine | moe_combine_bf16 | deterministic | yes | 40 |
| gate_split | gate_split_bf16 | deterministic | yes | 10 |
| sigmoid_mul | sigmoid_mul_bf16 | deterministic | yes | 10 |
| gdn_conv | gdn_conv_bf16 | deterministic | yes | 30 |
| gdn_delta_rule | gdn_delta_rule_bf16 | deterministic | yes | 30 |
| matmul_pair | matmul_pair_bf16_gemv | deterministic | yes | 70 |
| matmul_pair | matmul_pair_bf16_nax | deterministic | yes | 0 |
| qk_norm_rope_kv | qk_norm_rope_kv_bf16 | deterministic | yes | 10 |

- kernels test: skipped by --skip (SKIP)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | unstamped | endor.kuepker.haus | 1792720489 | `/opt/code/wt-gc-target/release/intarsia doctor --backend metal --device 0 --out /opt/code/wt-gc/docs/results/2026-10-22-gc2-metal-35b-chunk2048/raw/q-0-record/doctor.json` |
| `recipe` | unstamped | endor.kuepker.haus | 1792720489 | `recipe::analyse(/Users/josef/Models/Qwen3.6-35B-A3B-NVFP4)` |
| `plan` | unstamped | endor.kuepker.haus | 1792720489 | `/opt/code/wt-gc-target/release/intarsia plan show --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 40 --admit --json` |
| `kernels` | unstamped | endor.kuepker.haus | 0 | `/opt/code/wt-gc-target/release/intarsia plan show --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 40 --admit --json` |
| `gate` | unstamped | endor.kuepker.haus | 0 | `/opt/code/wt-gc-target/release/intarsia plan show --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 40 --admit --json` |
| `acc-format` | unstamped | endor.kuepker.haus | 1792720497 | `-` |
| `acc-short` | unstamped | endor.kuepker.haus | 1792720497 | `/opt/code/wt-gc-target/release/intarsia accuracy --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 40 --golden /opt/code/wt-gc/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --decode-positions 16 --max-seq 4096 --write-baseline /opt/code/wt-gc/` |
| `acc-oracle` | unstamped | endor.kuepker.haus | 1792720509 | `/opt/code/wt-gc-target/release/intarsia accuracy --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 40 --golden /opt/code/wt-gc/golden/oracle/qwen36-35b-a3b-nvfp4-cpu-oracle.safetensors --mode prefill --decode-positions 16 --max-seq 4352 --write-baseline /opt/c` |
| `bench` | unstamped | endor.kuepker.haus | 0 | `/opt/code/wt-gc-target/release/intarsia accuracy --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 40 --golden /opt/code/wt-gc/golden/oracle/qwen36-35b-a3b-nvfp4-cpu-oracle.safetensors --mode prefill --decode-positions 16 --max-seq 4352 --write-baseline /opt/c` |
| `acc-long` | unstamped | endor.kuepker.haus | 1792720528 | `/opt/code/wt-gc-target/release/intarsia accuracy --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 40 --golden /opt/code/intarsiaml/golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors --mode both --decode-positions 64 --max-seq 12288 --write-baseline /opt/` |
| `acc-4k` | unstamped | endor.kuepker.haus | 1792720651 | `/opt/code/wt-gc-target/release/intarsia accuracy --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 40 --golden golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors --mode reuse --decode-positions 64 --max-seq 8192 --write-baseline /opt/code/wt-gc/docs/results` |
| `envelope` | unstamped | endor.kuepker.haus | 1792720862 | `/opt/code/wt-gc-target/release/intarsia accuracy --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 40 --golden /opt/code/wt-gc/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --decode-positions 16 --max-seq 4096 --baseline /opt/code/wt-gc/docs/r` |
| `bench-long` | unstamped | endor.kuepker.haus | 0 | `/opt/code/wt-gc-target/release/intarsia accuracy --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 40 --golden /opt/code/wt-gc/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --decode-positions 16 --max-seq 4096 --baseline /opt/code/wt-gc/docs/r` |
| `ab` | unstamped | endor.kuepker.haus | 0 | `/opt/code/wt-gc-target/release/intarsia accuracy --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 40 --golden /opt/code/wt-gc/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --decode-positions 16 --max-seq 4096 --baseline /opt/code/wt-gc/docs/r` |
| `tp` | unstamped | endor.kuepker.haus | 0 | `/opt/code/wt-gc-target/release/intarsia accuracy --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 40 --golden /opt/code/wt-gc/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --decode-positions 16 --max-seq 4096 --baseline /opt/code/wt-gc/docs/r` |

Model fingerprint: `fnv1a64:9ca75aef5131bb81`.

Build stamp `unstamped` means a plain `cargo build`: baselines written here gate this binary, but `--pre-change` provenance checks refuse them as a "before".


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `logs/acc-oracle.out` (acc-oracle)
- `baselines/acc-oracle.baseline.json` (acc-oracle)
- `baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.metal.selfgolden.safetensors` (acc-oracle)
- `logs/acc-long.out` (acc-long)
- `baselines/acc-long.baseline.json` (acc-long)
- `logs/acc-4k.out` (acc-4k)
- `baselines/acc-4k.baseline.json` (acc-4k)
- `logs/acc-4k-reuse.out` (acc-4k)
- `baselines/acc-short.baseline.json` (envelope)
- `baselines/envelope.json` (envelope)
- `logs/baseline-selfcheck.out` (envelope)
- `state.json` (stage records; `--resume` reads it)
