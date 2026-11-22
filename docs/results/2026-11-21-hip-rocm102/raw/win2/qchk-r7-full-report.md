# Qualification report: Qwen3.6-35B-A3B-NVFP4 on hip (tp variant: 2)

_Full (quick + in-depth) section, generated unix 1795284782, intarsia 0.0.1 build `head=91f2b05 code-tree=f3db282f4435 dirty=no`._

## Headline

**Verdict: PASS** (15 stages: 7 pass, 0 fail, 0 error, 8 skipped)

| | |
|---|---|
| Model | Qwen3.6-35B-A3B-NVFP4 (`/models/Qwen3.6-35B-A3B-NVFP4`) |
| Backend / device | hip / AMD Radeon Graphics (gfx1201) |
| Recipe | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| Decode | not measured |
| Prefill | not measured |
| Accuracy (short golden) | **PASS** (prefill top-1 98.47% KL 6.00e-3; decode top-1 98.75% KL 1.11e-2) |
| Accuracy (format reference, same weights) | skipped: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| Accuracy (long golden) | skipped: skipped by --skip |
| Accuracy (4k golden) | skipped: skipped by --skip |
| Quick gate | **PASS**: perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 55s, in-depth 55s, total 1m50s |

### Failures

None.

### Skipped

- `kernels`: skipped by --skip
- `acc-format`: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE)
- `bench`: skipped by --skip
- `acc-long`: skipped by --skip
- `acc-4k`: skipped by --skip
- `envelope`: skipped by --skip
- `bench-long`: skipped by --skip
- `ab`: skipped by --skip

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | AMD Radeon Graphics (gfx1201), 95 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 9s | 31 ops resolved, exactness deterministic, loaded in 8.7s |
| 4 | `kernels` kernels test on the model's shapes | quick | SKIP | 0s | skipped by --skip |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 6s | perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| 7 | `acc-short` short accuracy against a golden | quick | PASS | 7s | prefill top-1 98.47% KL 6.00e-3; decode top-1 98.75% KL 1.11e-2 |
| 8 | `acc-oracle` short prefill accuracy against the f64 CPU oracle (same weights, deterministic reference) | quick | PASS | 32s | prefill top-1 95.77% KL 1.04e-2 |
| 9 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | SKIP | 0s | skipped by --skip |
| 10 | `acc-long` accuracy --mode all on the long golden (prefill, decode, prefix reuse) | full | SKIP | 0s | skipped by --skip |
| 11 | `acc-4k` accuracy --mode all on the 4k golden (prefill, decode, prefix reuse past 2 chunks of 2048) | full | SKIP | 0s | skipped by --skip |
| 12 | `envelope` noise envelope and baseline for regression gating | full | SKIP | 0s | skipped by --skip |
| 13 | `bench-long` long-context bench up to 16k | full | SKIP | 0s | skipped by --skip |
| 14 | `ab` A/B against the reference server via localeval | full | SKIP | 0s | skipped by --skip |
| 15 | `tp` tensor-parallel variant | full | PASS | 55s | decode 192.4 tok/s at 128 tokens (336% of the byte floor); oracle prefill top-1 96.13% KL 1.08e-2; accuracy prefill top-1 98.27% KL 7.08e-3; decode top-1 97.50% KL 8.21e-3 |

### Notes

- `acc-short`: gated against the recorded baseline /work/golden/baselines/qwen36-35b-a3b-nvfp4-vllm-top20.hip.baseline.json (commit 4379dd0) and its noise envelope
- `acc-oracle`: floor: mean KL(oracle \|\| engine) over the oracle's top-64 + tail, scored prefill rows, <= 2.0e-2 (measured, docs/results/2026-10-22-fx1-oracle-gate.md); top-1 not gated. The floor guards absolute accuracy; it does not see a 1 % scale bug, the drift check does
- `acc-oracle`: drift check: logits unchanged against the recorded dump /work/golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.hip.selfgolden.safetensors
- `tp`: size 128: 336% of the byte floor is above 100%: the --gb-per-token estimate is too high or the run is not a plain decode (check the floor inputs)
- `tp`: tp=2 drift check: logits unchanged against the recorded dump /work/golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.hip-tp2.selfgolden.safetensors

## Speed

**tensor-parallel bench** (floor: 10.94 GB/token at 627 GB/s; gb/token: busiest tp rank's weights from the plan; bandwidth: 627 GB/s, the R9700 figure of docs/results/2026-10-13-phase2a-hip.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 119 | 0.046 | 2599 | 192.44 | - | 57.3 | 335.8 |

## Accuracy

**acc-short** against `/models/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors` (given with --golden; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1441 | 98.47 | 91.70 | 6.00e-3 | 5.69e-3 | 6.19e-1 | 1.35e-3 | 0 | 6.7 | pass |
| decode | 160 | 98.75 | 93.38 | 1.11e-2 | 1.11e-2 | 7.31e-1 | 2.26e-3 | 0 | 88.8 | pass |

**tp** against `/models/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors` (given with --golden; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1441 | 98.27 | 91.23 | 7.08e-3 | 6.68e-3 | 1.11e0 | 1.63e-3 | 0 | 19.4 | pass |
| decode | 160 | 97.50 | 91.75 | 8.21e-3 | 8.32e-3 | 2.63e-1 | 1.95e-3 | 0 | 81.2 | pass |

**Quick gate**: perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) (PASS)
- plan: 31 ops resolved, exactness deterministic, loaded in 8.7s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_bf16 | exact | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 1 |
| rmsnorm_add | rmsnorm_add_bf16 | deterministic | yes | 1 |
| matmul | matmul_bf16 | deterministic | yes | 0 |
| matmul | matmul_bf16_gemv | deterministic | yes | 0 |
| attn_prefill | attn_prefill_bf16_flash_v2 | deterministic | yes | 0 |
| attn_prefill | attn_prefill_bf16_splitkv | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 10 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 0 |
| argmax | argmax_f32 | exact | yes | 1 |
| matmul_fp8 | matmul_fp8_gemv_v2 | deterministic | yes | 10 |
| matmul_fp8 | matmul_fp8_mma_f8 | deterministic | yes | 0 |
| matmul_nvfp4 | matmul_nvfp4_gemv_v2 | deterministic | yes | 1 |
| matmul_nvfp4 | matmul_nvfp4_mma | deterministic | yes | 0 |
| moe_topk | moe_topk_bf16_v2 | deterministic | yes | 0 |
| moe_experts | moe_experts_bf16_v2 | deterministic | yes | 0 |
| moe_combine | moe_combine_bf16 | deterministic | yes | 0 |
| sigmoid_mul | sigmoid_mul_bf16 | deterministic | yes | 0 |
| gdn_conv | gdn_conv_bf16 | deterministic | yes | 0 |
| gdn_delta_rule | gdn_delta_rule_bf16_chunk | deterministic | yes | 0 |
| gdn_delta_rule | gdn_delta_rule_bf16_v2 | deterministic | yes | 0 |
| matmul_pair | matmul_pair_bf16_gemv | deterministic | yes | 1 |
| matmul_fp8_pair | matmul_fp8_pair_gemv | deterministic | yes | 30 |
| gated_rmsnorm | gated_rmsnorm_bf16 | deterministic | yes | 0 |
| rmsnorm_add_matmul_pair | rmsnorm_add_matmul_pair_bf16_gemv | deterministic | yes | 69 |
| gated_rmsnorm_matmul_fp8 | gated_rmsnorm_matmul_fp8_gemv | deterministic | yes | 30 |
| rmsnorm_add_matmul_fp8_pair | rmsnorm_add_matmul_fp8_pair_gemv | deterministic | yes | 10 |
| sigmoid_mul_matmul_fp8 | sigmoid_mul_matmul_fp8_gemv | deterministic | yes | 10 |
| gate_split_qk_norm_rope_kv | gate_split_qk_norm_rope_kv_bf16 | deterministic | yes | 10 |
| gdn_conv_delta | gdn_conv_delta_bf16 | deterministic | yes | 30 |
| moe_block | moe_block_bf16 | deterministic | yes | 40 |

- kernels test: skipped by --skip (SKIP)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | head=91f2b05 code-tree=f3db282f4435 dirty=no | tatooine | 1795284672 | `/target/release/intarsia doctor --backend hip --device 0 --out /work/.rocm102-logs/1121-1307/qchk-r7/doctor.json` |
| `recipe` | head=91f2b05 code-tree=f3db282f4435 dirty=no | tatooine | 1795284673 | `recipe::analyse(/models/Qwen3.6-35B-A3B-NVFP4)` |
| `plan` | head=91f2b05 code-tree=f3db282f4435 dirty=no | tatooine | 1795284673 | `/target/release/intarsia plan show --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --admit --json` |
| `kernels` | head=91f2b05 code-tree=f3db282f4435 dirty=no | tatooine | 0 | `/target/release/intarsia plan show --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --admit --json` |
| `gate` | head=91f2b05 code-tree=f3db282f4435 dirty=no | tatooine | 1795284682 | `/target/release/intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --quick --max-seq 2048` |
| `acc-format` | head=91f2b05 code-tree=f3db282f4435 dirty=no | tatooine | 1795284687 | `-` |
| `acc-short` | head=91f2b05 code-tree=f3db282f4435 dirty=no | tatooine | 1795284687 | `/target/release/intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --golden /models/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --decode-positions 16 --max-seq 4096 --write-baseline /work/.rocm102-logs/1121-1307/qchk-r7/baselines/acc-s` |
| `acc-oracle` | head=91f2b05 code-tree=f3db282f4435 dirty=no | tatooine | 1795284695 | `/target/release/intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --self-golden /work/golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.hip.selfgolden.safetensors --mode prefill --decode-positions 16 --max-seq 4096 --model-id Qwen3.6-35B-A3B-NVFP4 --allo` |
| `bench` | head=91f2b05 code-tree=f3db282f4435 dirty=no | tatooine | 0 | `/target/release/intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --self-golden /work/golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.hip.selfgolden.safetensors --mode prefill --decode-positions 16 --max-seq 4096 --model-id Qwen3.6-35B-A3B-NVFP4 --allo` |
| `acc-long` | head=91f2b05 code-tree=f3db282f4435 dirty=no | tatooine | 0 | `/target/release/intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --self-golden /work/golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.hip.selfgolden.safetensors --mode prefill --decode-positions 16 --max-seq 4096 --model-id Qwen3.6-35B-A3B-NVFP4 --allo` |
| `acc-4k` | head=91f2b05 code-tree=f3db282f4435 dirty=no | tatooine | 0 | `/target/release/intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --self-golden /work/golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.hip.selfgolden.safetensors --mode prefill --decode-positions 16 --max-seq 4096 --model-id Qwen3.6-35B-A3B-NVFP4 --allo` |
| `envelope` | head=91f2b05 code-tree=f3db282f4435 dirty=no | tatooine | 0 | `/target/release/intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --self-golden /work/golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.hip.selfgolden.safetensors --mode prefill --decode-positions 16 --max-seq 4096 --model-id Qwen3.6-35B-A3B-NVFP4 --allo` |
| `bench-long` | head=91f2b05 code-tree=f3db282f4435 dirty=no | tatooine | 0 | `/target/release/intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --self-golden /work/golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.hip.selfgolden.safetensors --mode prefill --decode-positions 16 --max-seq 4096 --model-id Qwen3.6-35B-A3B-NVFP4 --allo` |
| `ab` | head=91f2b05 code-tree=f3db282f4435 dirty=no | tatooine | 0 | `/target/release/intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --self-golden /work/golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.hip.selfgolden.safetensors --mode prefill --decode-positions 16 --max-seq 4096 --model-id Qwen3.6-35B-A3B-NVFP4 --allo` |
| `tp` | head=91f2b05 code-tree=f3db282f4435 dirty=no | tatooine | 1795284727 | `/target/release/intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --tp 2 --devices 0,1 --mem-budget 29 --self-golden /work/golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.hip-tp2.selfgolden.safetensors --mode prefill --decode-positions 16 --max-seq 4096 --model-id Qwe` |

Model fingerprint: `fnv1a64:9ca75aef5131bb81`.


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `logs/acc-oracle.out` (acc-oracle)
- `baselines/acc-oracle.baseline.json` (acc-oracle)
- `baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.hip.selfgolden.safetensors` (acc-oracle)
- `bench/qualify-Qwen3.6-35B-A3B-NVFP4-hip-tp2.json` (tp)
- `logs/tp2-bench.out` (tp)
- `logs/tp2-server.err` (tp)
- `baselines/tp2-acc.baseline.json` (tp)
- `state.json` (stage records; `--resume` reads it)
