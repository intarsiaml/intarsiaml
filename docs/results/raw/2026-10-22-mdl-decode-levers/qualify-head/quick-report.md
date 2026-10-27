# Qualification quick report: Qwen3.6-35B-A3B-NVFP4 on metal

_Quick section, generated unix 1792677100, intarsia 0.0.1 build `unstamped`._

## Headline

**Verdict: PASS** (9 stages: 5 pass, 0 fail, 0 error, 4 skipped)

| | |
|---|---|
| Model | Qwen3.6-35B-A3B-NVFP4 (`/Users/josef/Models/Qwen3.6-35B-A3B-NVFP4`) |
| Backend / device | metal / Apple M5 Max (applegpu_g17s) |
| Recipe | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| Decode | not measured |
| Prefill | not measured |
| Accuracy (short golden) | skipped: no golden found, and a golden for a MoE / hybrid / quantized model is made from a reference server: pass --reference <vLLM url> or --golden FILE; no short golden for 'Qwen3.6-35B-A3B-NVFP4' in /Users/josef/models/golden (give --golden FILE or --golden-dir DIR, then --resume) |
| Accuracy (format reference, same weights) | skipped: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| Quick gate | **PASS**: perplexity 1.21, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 62s |

### Failures

None.

### Skipped

- `acc-format`: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE)
- `acc-short`: no golden found, and a golden for a MoE / hybrid / quantized model is made from a reference server: pass --reference <vLLM url> or --golden FILE; no short golden for 'Qwen3.6-35B-A3B-NVFP4' in /Users/josef/models/golden (give --golden FILE or --golden-dir DIR, then --resume)
- `acc-oracle`: no CPU-oracle golden for 'Qwen3.6-35B-A3B-NVFP4' (--oracle-golden, or oracle/<model>-cpu-oracle*.safetensors in a golden directory)
- `bench`: skipped by --skip

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 1s | Apple M5 Max (applegpu_g17s), 46 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 29s | 28 ops resolved, exactness deterministic, loaded in 28.9s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 22s | 46 kernels, 845 cases in 21.9s; incomplete: 1 kernel(s) had no cases in this sweep (untested, not failed; plan-time admission tests them) |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 9s | perplexity 1.21, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| 7 | `acc-short` short accuracy against a golden | quick | SKIP | 0s | no golden found, and a golden for a MoE / hybrid / quantized model is made from a reference server: pass --reference <vLLM url> or --golden FILE; no short golden for 'Qwen3.6-35B-A3B-NVFP4' in /Users/josef/models/golden (give --golden FILE or --golden-dir DIR, then --resume) |
| 8 | `acc-oracle` short prefill accuracy against the f64 CPU oracle (same weights, deterministic reference) | quick | SKIP | 0s | no CPU-oracle golden for 'Qwen3.6-35B-A3B-NVFP4' (--oracle-golden, or oracle/<model>-cpu-oracle*.safetensors in a golden directory) |
| 9 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | SKIP | 0s | skipped by --skip |

### Notes

None.

## Speed

No speed numbers.

## Accuracy

**Quick gate**: perplexity 1.21, 5/5 sanity prompts ok, no non-finite rows

No golden comparison ran (see Skipped).

## Plan and recipe

- recipe: builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) (PASS)
- plan: 28 ops resolved, exactness deterministic, loaded in 28.9s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_bf16 | exact | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 1 |
| rmsnorm_add | rmsnorm_add_bf16 | deterministic | yes | 1 |
| attn_prefill | attn_prefill_bf16_flash | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 10 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 0 |
| argmax | argmax_f32 | exact | yes | 1 |
| matmul_fp8 | matmul_fp8_gemv | deterministic | yes | 10 |
| matmul_fp8 | matmul_fp8_mma | deterministic | yes | 0 |
| matmul_nvfp4 | matmul_nvfp4_gemv | deterministic | yes | 1 |
| matmul_nvfp4 | matmul_nvfp4_mma | deterministic | yes | 0 |
| moe_topk | moe_topk_bf16 | deterministic | yes | 40 |
| moe_experts | moe_experts_bf16 | deterministic | yes | 40 |
| moe_combine | moe_combine_bf16 | deterministic | yes | 0 |
| gate_split | gate_split_bf16 | deterministic | yes | 10 |
| sigmoid_mul | sigmoid_mul_bf16 | deterministic | yes | 0 |
| gdn_conv | gdn_conv_bf16 | deterministic | yes | 30 |
| gdn_delta_rule | gdn_delta_rule_bf16 | deterministic | yes | 30 |
| matmul_pair | matmul_pair_bf16_gemv | deterministic | yes | 1 |
| matmul_pair | matmul_pair_bf16_nax | deterministic | yes | 0 |
| matmul_fp8_pair | matmul_fp8_pair_gemv | deterministic | yes | 30 |
| matmul_nvfp4_glu | matmul_nvfp4_glu_gemv | deterministic | yes | 40 |
| qk_norm_rope_kv | qk_norm_rope_kv_bf16 | deterministic | yes | 10 |
| rmsnorm_add_matmul_pair | rmsnorm_add_matmul_pair_bf16_gemv | deterministic | yes | 69 |
| gated_rmsnorm_matmul_fp8 | gated_rmsnorm_matmul_fp8_gemv | deterministic | yes | 30 |
| rmsnorm_add_matmul_fp8_pair | rmsnorm_add_matmul_fp8_pair_gemv | deterministic | yes | 10 |
| sigmoid_mul_matmul_fp8 | sigmoid_mul_matmul_fp8_gemv | deterministic | yes | 10 |
| shared_down_combine | shared_down_combine_gemv | deterministic | yes | 40 |

- kernels test: 46 kernels, 845 cases in 21.9s; incomplete: 1 kernel(s) had no cases in this sweep (untested, not failed; plan-time admission tests them) (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | unstamped | endor.kuepker.haus | 1792677038 | `/opt/code/wt-mdl-bin/head doctor --backend metal --device 0 --out /opt/code/wt-mdl/docs/results/raw/2026-10-22-mdl-decode-levers/qualify-head/doctor.json` |
| `recipe` | unstamped | endor.kuepker.haus | 1792677039 | `recipe::analyse(/Users/josef/Models/Qwen3.6-35B-A3B-NVFP4)` |
| `plan` | unstamped | endor.kuepker.haus | 1792677039 | `/opt/code/wt-mdl-bin/head plan show --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --admit --json` |
| `kernels` | unstamped | endor.kuepker.haus | 1792677068 | `/opt/code/wt-mdl-bin/head kernels test --backend metal --device 0 --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4` |
| `gate` | unstamped | endor.kuepker.haus | 1792677090 | `/opt/code/wt-mdl-bin/head accuracy --model /Users/josef/Models/Qwen3.6-35B-A3B-NVFP4 --backend metal --device 0 --quick --max-seq 2048` |
| `acc-format` | unstamped | endor.kuepker.haus | 1792677099 | `-` |
| `acc-short` | unstamped | endor.kuepker.haus | 1792677099 | `-` |
| `acc-oracle` | unstamped | endor.kuepker.haus | 1792677099 | `-` |
| `bench` | unstamped | endor.kuepker.haus | 0 | `-` |

Model fingerprint: `fnv1a64:9ca75aef5131bb81`.

Build stamp `unstamped` means a plain `cargo build`: baselines written here gate this binary, but `--pre-change` provenance checks refuse them as a "before".


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/kernels.out` (kernels)
- `logs/gate.out` (gate)
- `state.json` (stage records; `--resume` reads it)
