# Qualification report: Qwen3.6-35B-A3B-NVFP4 on metal

_Full (quick + in-depth) section, generated unix 1792679546, intarsia 0.0.1 build `head=ec603e1 code-tree=6eac9c190c06 dirty=no`._

## Headline

**Verdict: FAIL** (15 stages: 3 pass, 1 fail, 0 error, 11 skipped)

| | |
|---|---|
| Model | Qwen3.6-35B-A3B-NVFP4 (`/Users/josef/Models/Qwen3.6-35B-A3B-NVFP4`) |
| Backend / device | metal / Apple M5 Max (applegpu_g17s) |
| Recipe | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| Decode | not measured |
| Prefill | not measured |
| Accuracy (short golden) | skipped: skipped by --skip |
| Accuracy (format reference, same weights) | skipped: skipped by --skip |
| Accuracy (long golden) | skipped: skipped by --skip |
| Accuracy (4k golden) | **FAIL** (prefill top-1 93.14% KL 8.14e-2; decode top-1 98.96% KL 7.22e-3) |
| Quick gate | **SKIP**: skipped by --skip |
| Wall time | quick 9s, in-depth 2m06s, total 2m14s |

### Failures

- `acc-4k` **FAIL**: prefix reuse differs from a cold run

### Skipped

- `kernels`: skipped by --skip
- `gate`: skipped by --skip
- `acc-format`: skipped by --skip
- `acc-short`: skipped by --skip
- `acc-oracle`: skipped by --skip
- `bench`: skipped by --skip
- `acc-long`: skipped by --skip
- `envelope`: skipped by --skip
- `bench-long`: skipped by --skip
- `ab`: skipped by --skip
- `tp`: skipped by --skip

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | Apple M5 Max (applegpu_g17s), 39 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 8s | 21 ops resolved, exactness deterministic, loaded in 7.8s |
| 4 | `kernels` kernels test on the model's shapes | quick | SKIP | 0s | skipped by --skip |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | SKIP | 0s | skipped by --skip |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | skipped by --skip |
| 7 | `acc-short` short accuracy against a golden | quick | SKIP | 0s | skipped by --skip |
| 8 | `acc-oracle` short prefill accuracy against the f64 CPU oracle (same weights, deterministic reference) | quick | SKIP | 0s | skipped by --skip |
| 9 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | SKIP | 0s | skipped by --skip |
| 10 | `acc-long` accuracy --mode all on the long golden (prefill, decode, prefix reuse) | full | SKIP | 0s | skipped by --skip |
| 11 | `acc-4k` accuracy --mode all on the 4k golden (prefill, decode, prefix reuse past 2 chunks of 2048) | full | FAIL | 2m06s | prefix reuse differs from a cold run |
| 12 | `envelope` noise envelope and baseline for regression gating | full | SKIP | 0s | skipped by --skip |
| 13 | `bench-long` long-context bench up to 16k | full | SKIP | 0s | skipped by --skip |
| 14 | `ab` A/B against the reference server via localeval | full | SKIP | 0s | skipped by --skip |
| 15 | `tp` tensor-parallel variant | full | SKIP | 0s | skipped by --skip |

### Notes

- `acc-4k`: no baseline for backend metal next to the golden or in golden/baselines: KL, tail, prefix-reuse gates only, top-1 not gated

## Speed

No speed numbers.

## Accuracy

**acc-4k** against `golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 12759 | 93.14 | 90.18 | 8.14e-2 | 9.09e-2 | 2.21e1 | 1.15e-2 | 0 | 0.8 | pass |
| decode | 192 | 98.96 | 90.62 | 7.22e-3 | 6.90e-3 | 1.12e-1 | 1.79e-3 | 0 | 91.7 | pass |

**Quick gate**: skipped by --skip

## Plan and recipe

- recipe: builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) (PASS)
- plan: 21 ops resolved, exactness deterministic, loaded in 7.8s (PASS)

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
| `doctor` | head=ec603e1 code-tree=6eac9c190c06 dirty=no | endor.kuepker.haus | 1792679411 | `/opt/code/wt-gc-target/release/intarsia doctor --backend metal --device 0 --out /opt/code/wt-gc/docs/results/2026-10-22-gc-gate-coverage/raw/metal-acc4k/q-0-nobaseline/doctor.json` |
| `recipe` | head=ec603e1 code-tree=6eac9c190c06 dirty=no | endor.kuepker.haus | 1792679412 | `recipe::analyse(/Users/josef/Models/Qwen3.6-35B-A3B-NVFP4)` |
| `plan` | head=ec603e1 code-tree=6eac9c190c06 dirty=no | endor.kuepker.haus | 1792679412 | `/opt/code/wt-gc-target/release/intarsia plan show --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 45 --admit --json` |
| `kernels` | head=ec603e1 code-tree=6eac9c190c06 dirty=no | endor.kuepker.haus | 0 | `/opt/code/wt-gc-target/release/intarsia plan show --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 45 --admit --json` |
| `gate` | head=ec603e1 code-tree=6eac9c190c06 dirty=no | endor.kuepker.haus | 0 | `/opt/code/wt-gc-target/release/intarsia plan show --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 45 --admit --json` |
| `acc-format` | head=ec603e1 code-tree=6eac9c190c06 dirty=no | endor.kuepker.haus | 0 | `/opt/code/wt-gc-target/release/intarsia plan show --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 45 --admit --json` |
| `acc-short` | head=ec603e1 code-tree=6eac9c190c06 dirty=no | endor.kuepker.haus | 0 | `/opt/code/wt-gc-target/release/intarsia plan show --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 45 --admit --json` |
| `acc-oracle` | head=ec603e1 code-tree=6eac9c190c06 dirty=no | endor.kuepker.haus | 0 | `/opt/code/wt-gc-target/release/intarsia plan show --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 45 --admit --json` |
| `bench` | head=ec603e1 code-tree=6eac9c190c06 dirty=no | endor.kuepker.haus | 0 | `/opt/code/wt-gc-target/release/intarsia plan show --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 45 --admit --json` |
| `acc-long` | head=ec603e1 code-tree=6eac9c190c06 dirty=no | endor.kuepker.haus | 0 | `/opt/code/wt-gc-target/release/intarsia plan show --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 45 --admit --json` |
| `acc-4k` | head=ec603e1 code-tree=6eac9c190c06 dirty=no | endor.kuepker.haus | 1792679420 | `/opt/code/wt-gc-target/release/intarsia accuracy --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 45 --golden golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors --mode all --decode-positions 64 --max-seq 8192 --write-baseline /opt/code/wt-gc/docs/results/2` |
| `envelope` | head=ec603e1 code-tree=6eac9c190c06 dirty=no | endor.kuepker.haus | 0 | `/opt/code/wt-gc-target/release/intarsia accuracy --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 45 --golden golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors --mode all --decode-positions 64 --max-seq 8192 --write-baseline /opt/code/wt-gc/docs/results/2` |
| `bench-long` | head=ec603e1 code-tree=6eac9c190c06 dirty=no | endor.kuepker.haus | 0 | `/opt/code/wt-gc-target/release/intarsia accuracy --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 45 --golden golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors --mode all --decode-positions 64 --max-seq 8192 --write-baseline /opt/code/wt-gc/docs/results/2` |
| `ab` | head=ec603e1 code-tree=6eac9c190c06 dirty=no | endor.kuepker.haus | 0 | `/opt/code/wt-gc-target/release/intarsia accuracy --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 45 --golden golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors --mode all --decode-positions 64 --max-seq 8192 --write-baseline /opt/code/wt-gc/docs/results/2` |
| `tp` | head=ec603e1 code-tree=6eac9c190c06 dirty=no | endor.kuepker.haus | 0 | `/opt/code/wt-gc-target/release/intarsia accuracy --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --mem-budget 45 --golden golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors --mode all --decode-positions 64 --max-seq 8192 --write-baseline /opt/code/wt-gc/docs/results/2` |

Model fingerprint: `fnv1a64:9ca75aef5131bb81`.


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/acc-4k.out` (acc-4k)
- `baselines/acc-4k.baseline.json` (acc-4k)
- `state.json` (stage records; `--resume` reads it)
