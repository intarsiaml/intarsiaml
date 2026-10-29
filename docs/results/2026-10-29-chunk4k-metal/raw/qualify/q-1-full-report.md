# Qualification report: Qwen3.6-35B-A3B-NVFP4 on metal

_Full (quick + in-depth) section, generated unix 1793298932, intarsia 0.0.1 build `unstamped`._

## Headline

**Verdict: FAIL** (15 stages: 9 pass, 1 fail, 0 error, 5 skipped)

| | |
|---|---|
| Model | Qwen3.6-35B-A3B-NVFP4 (`/Users/josef/Models/Qwen3.6-35B-A3B-NVFP4`) |
| Backend / device | metal / Apple M5 Max (applegpu_g17s) |
| Recipe | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| Decode | 128 tok: **123.7** tok/s (45% of floor); 1024 tok: **117.0** tok/s (43% of floor); 4096 tok: **114.6** tok/s (42% of floor) |
| Prefill | 128 tok: **1761** tok/s; 1024 tok: **4409** tok/s; 4096 tok: **4248** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 97.85% KL 7.93e-3; decode top-1 97.50% KL 1.24e-2) |
| Accuracy (format reference, same weights) | skipped: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| Accuracy (long golden) | **PASS** (prefill top-1 94.20% KL 4.19e-2; decode top-1 98.44% KL 9.54e-3) |
| Accuracy (4k golden) | **FAIL** (prefill top-1 92.66% KL 8.16e-2; decode top-1 98.96% KL 9.69e-3) |
| Quick gate | **PASS**: perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 1m59s, in-depth 2m58s, total 4m57s |

### Failures

- `acc-4k` **FAIL**: prefix reuse at chunk 1024: 3 judged, 0 skipped: prefix reuse differs from a cold run

### Skipped

- `acc-format`: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE)
- `envelope`: skipped by --skip
- `bench-long`: skipped by --skip
- `ab`: skipped by --skip
- `tp`: skipped by --skip

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | Apple M5 Max (applegpu_g17s), 53 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 5s | 24 ops resolved, exactness deterministic, loaded in 4.6s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 31s | 48 kernels, 924 cases in 30.8s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 9s | perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| 7 | `acc-short` short accuracy against a golden | quick | PASS | 14s | prefill top-1 97.85% KL 7.93e-3; decode top-1 97.50% KL 1.24e-2 |
| 8 | `acc-oracle` short prefill accuracy against the f64 CPU oracle (same weights, deterministic reference) | quick | PASS | 38s | prefill top-1 96.04% KL 1.07e-2 |
| 9 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 22s | decode 123.7 tok/s at 128 tokens (45% of the byte floor) |
| 10 | `acc-long` accuracy --mode all on the long golden (prefill, decode, prefix reuse) | full | PASS | 87s | prefill top-1 94.20% KL 4.19e-2; decode top-1 98.44% KL 9.54e-3 |
| 11 | `acc-4k` accuracy --mode all on the 4k golden (prefill, decode, prefix reuse past 2 chunks of 2048) | full | FAIL | 1m31s | prefix reuse at chunk 1024: 3 judged, 0 skipped: prefix reuse differs from a cold run |
| 12 | `envelope` noise envelope and baseline for regression gating | full | SKIP | 0s | skipped by --skip |
| 13 | `bench-long` long-context bench up to 16k | full | SKIP | 0s | skipped by --skip |
| 14 | `ab` A/B against the reference server via localeval | full | SKIP | 0s | skipped by --skip |
| 15 | `tp` tensor-parallel variant | full | SKIP | 0s | skipped by --skip |

### Notes

- `acc-short`: gated against the recorded baseline /opt/code/intarsiaml/.worktrees/chunk4k/golden/baselines/qwen36-35b-a3b-nvfp4-vllm-top20.metal.baseline.json (commit unstamped) and its noise envelope
- `acc-oracle`: floor: mean KL(oracle \|\| engine) over the oracle's top-64 + tail, scored prefill rows, <= 2.0e-2 (measured, docs/results/2026-10-22-fx1-oracle-gate.md); top-1 not gated. The floor guards absolute accuracy; it does not see a 1 % scale bug, the drift check does
- `acc-oracle`: drift check: logits unchanged against the recorded dump /opt/code/intarsiaml/.worktrees/chunk4k/golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.metal.selfgolden.safetensors
- `bench`: bench warning: size 128: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 1024: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 4096: 0 graph replays during the measured reps (decode ran eagerly)
- `acc-long`: prefix reuse is gated in acc-4k, on the 4k golden
- `acc-4k`: gated against the recorded baseline golden/baselines/qwen36-35b-a3b-nvfp4-vllm-4k-top20.metal.baseline.json (commit unstamped), no noise envelope

## Speed

**quick bench** (floor: 2.25 GB/token at 614 GB/s; gb/token: estimated from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 614 GB/s, the Apple M5 Max (40-core GPU) spec figure of docs/decisions/0009-native-metal-backend.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 127 | 0.074 | 1761 | 123.74 | 1.76 | 272.3 | 45.4 |
| 1017 | 0.231 | 4409 | 116.97 | 2.14 | 272.3 | 43.0 |
| 4229 | 0.996 | 4248 | 114.62 | 1.75 | 272.3 | 42.1 |

## Accuracy

**acc-short** against `/opt/code/intarsiaml/.worktrees/chunk4k/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors` (given with --golden; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1441 | 97.85 | 91.34 | 7.93e-3 | 7.79e-3 | 1.03e0 | 1.76e-3 | 0 | 6.2 | pass |
| decode | 160 | 97.50 | 92.38 | 1.24e-2 | 1.28e-2 | 3.22e-1 | 2.95e-3 | 0 | 81.9 | pass |

**acc-long** against `golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 36887 | 94.20 | 90.34 | 4.19e-2 | 4.51e-2 | 2.55e1 | 7.59e-3 | 0 | 0.2 | pass |
| decode | 320 | 98.44 | 89.44 | 9.54e-3 | 8.72e-3 | 1.60e-1 | 2.77e-3 | 0 | 63.1 | pass |

**acc-4k** against `golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 12759 | 92.66 | 89.92 | 8.16e-2 | 9.46e-2 | 2.53e1 | 1.19e-2 | 0 | 0.3 | pass |
| decode | 192 | 98.96 | 90.10 | 9.69e-3 | 9.00e-3 | 1.31e-1 | 1.87e-3 | 0 | 93.8 | pass |

**Quick gate**: perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) (PASS)
- plan: 24 ops resolved, exactness deterministic, loaded in 4.6s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_bf16 | exact | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 31 |
| rmsnorm_add | rmsnorm_add_bf16 | deterministic | yes | 80 |
| attn_prefill | attn_prefill_bf16_flash_nax | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 10 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 70 |
| argmax | argmax_f32 | exact | yes | 1 |
| matmul_fp8 | matmul_fp8_gemv | deterministic | yes | 130 |
| matmul_fp8 | matmul_fp8_mma | deterministic | yes | 0 |
| matmul_fp8 | matmul_fp8_nax | deterministic | yes | 0 |
| matmul_nvfp4 | matmul_nvfp4_gemv | deterministic | yes | 121 |
| matmul_nvfp4 | matmul_nvfp4_mma | deterministic | yes | 0 |
| matmul_nvfp4 | matmul_nvfp4_nax | deterministic | yes | 0 |
| moe_topk | moe_topk_bf16 | deterministic | yes | 40 |
| moe_experts | moe_experts_bf16 | deterministic | yes | 40 |
| moe_combine | moe_combine_bf16 | deterministic | yes | 40 |
| gate_split | gate_split_bf16 | deterministic | yes | 10 |
| sigmoid_mul | sigmoid_mul_bf16 | deterministic | yes | 10 |
| gdn_conv | gdn_conv_bf16 | deterministic | yes | 30 |
| gdn_delta_rule | gdn_delta_rule_bf16 | deterministic | yes | 30 |
| gdn_delta_rule | gdn_delta_rule_bf16_pre | deterministic | yes | 0 |
| matmul_pair | matmul_pair_bf16_gemv | deterministic | yes | 70 |
| matmul_pair | matmul_pair_bf16_nax | deterministic | yes | 0 |
| qk_norm_rope_kv | qk_norm_rope_kv_bf16 | deterministic | yes | 10 |

- kernels test: 48 kernels, 924 cases in 30.8s (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | unstamped | endor.kuepker.haus | 1793298634 | `/opt/code/intarsiaml/.worktrees/chunk4k/.lane/intarsia-metal doctor --backend metal --device 0 --out /opt/code/intarsiaml/.worktrees/chunk4k/.lane/w1/q-1-verdict/doctor.json` |
| `recipe` | unstamped | endor.kuepker.haus | 1793298635 | `recipe::analyse(/Users/josef/Models/Qwen3.6-35B-A3B-NVFP4)` |
| `plan` | unstamped | endor.kuepker.haus | 1793298635 | `/opt/code/intarsiaml/.worktrees/chunk4k/.lane/intarsia-metal plan show --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 40 --admit --json` |
| `kernels` | unstamped | endor.kuepker.haus | 1793298640 | `/opt/code/intarsiaml/.worktrees/chunk4k/.lane/intarsia-metal kernels test --backend metal --device 0 --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4` |
| `gate` | unstamped | endor.kuepker.haus | 1793298671 | `/opt/code/intarsiaml/.worktrees/chunk4k/.lane/intarsia-metal accuracy --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 40 --quick --max-seq 2048` |
| `acc-format` | unstamped | endor.kuepker.haus | 1793298679 | `-` |
| `acc-short` | unstamped | endor.kuepker.haus | 1793298679 | `/opt/code/intarsiaml/.worktrees/chunk4k/.lane/intarsia-metal accuracy --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 40 --golden /opt/code/intarsiaml/.worktrees/chunk4k/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --decode-positions 16 --m` |
| `acc-oracle` | unstamped | endor.kuepker.haus | 1793298693 | `/opt/code/intarsiaml/.worktrees/chunk4k/.lane/intarsia-metal accuracy --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 40 --self-golden /opt/code/intarsiaml/.worktrees/chunk4k/golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.metal.selfgolden.safetensors --mod` |
| `bench` | unstamped | endor.kuepker.haus | 1793298732 | `/opt/code/intarsiaml/.worktrees/chunk4k/.lane/intarsia-metal bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3.6-35B-A3B-NVFP4 --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-Qwen3.6-35B-A3B-NVFP4-metal-bench --out-dir /opt/code/intarsiaml/.worktrees/ch` |
| `acc-long` | unstamped | endor.kuepker.haus | 1793298754 | `/opt/code/intarsiaml/.worktrees/chunk4k/.lane/intarsia-metal accuracy --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 40 --golden golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors --mode both --decode-positions 64 --max-seq 12288 --write-baseline /opt/` |
| `acc-4k` | unstamped | endor.kuepker.haus | 1793298841 | `/opt/code/intarsiaml/.worktrees/chunk4k/.lane/intarsia-metal accuracy --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 40 --golden golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors --mode reuse --decode-positions 64 --max-seq 8192 --write-baseline /opt/co` |
| `envelope` | unstamped | endor.kuepker.haus | 0 | `/opt/code/intarsiaml/.worktrees/chunk4k/.lane/intarsia-metal accuracy --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 40 --golden golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors --mode reuse --decode-positions 64 --max-seq 8192 --write-baseline /opt/co` |
| `bench-long` | unstamped | endor.kuepker.haus | 0 | `/opt/code/intarsiaml/.worktrees/chunk4k/.lane/intarsia-metal accuracy --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 40 --golden golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors --mode reuse --decode-positions 64 --max-seq 8192 --write-baseline /opt/co` |
| `ab` | unstamped | endor.kuepker.haus | 0 | `/opt/code/intarsiaml/.worktrees/chunk4k/.lane/intarsia-metal accuracy --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 40 --golden golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors --mode reuse --decode-positions 64 --max-seq 8192 --write-baseline /opt/co` |
| `tp` | unstamped | endor.kuepker.haus | 0 | `/opt/code/intarsiaml/.worktrees/chunk4k/.lane/intarsia-metal accuracy --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 40 --golden golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors --mode reuse --decode-positions 64 --max-seq 8192 --write-baseline /opt/co` |

Model fingerprint: `fnv1a64:9ca75aef5131bb81`.

Build stamp `unstamped` means a plain `cargo build`: baselines written here gate this binary, but `--pre-change` provenance checks refuse them as a "before".


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/kernels.out` (kernels)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `logs/acc-oracle.out` (acc-oracle)
- `baselines/acc-oracle.baseline.json` (acc-oracle)
- `baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.metal.selfgolden.safetensors` (acc-oracle)
- `bench/qualify-Qwen3.6-35B-A3B-NVFP4-metal-bench.json` (bench)
- `logs/bench-bench.out` (bench)
- `logs/bench-server.err` (bench)
- `logs/acc-long.out` (acc-long)
- `baselines/acc-long.baseline.json` (acc-long)
- `logs/acc-4k.out` (acc-4k)
- `baselines/acc-4k.baseline.json` (acc-4k)
- `logs/acc-4k-reuse.out` (acc-4k)
- `state.json` (stage records; `--resume` reads it)
