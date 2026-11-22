# Qualification report: Qwen3.6-35B-A3B-NVFP4 on hip (tp variant: 2)

_Full (quick + in-depth) section, generated unix 1795309445, intarsia 0.0.1 build `head=ec38aca code-tree=7b1c3dab64f0 dirty=no`._

## Headline

**Verdict: PASS** (15 stages: 12 pass, 0 fail, 0 error, 3 skipped)

| | |
|---|---|
| Model | Qwen3.6-35B-A3B-NVFP4 (`/models/Qwen3.6-35B-A3B-NVFP4`) |
| Backend / device | hip / AMD Radeon AI PRO R9700 (gfx1201) |
| Recipe | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| Decode | 128 tok: **163.3** tok/s (59% of floor) |
| Prefill | 128 tok: **1813** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 98.47% KL 6.00e-3; decode top-1 98.75% KL 1.11e-2) |
| Accuracy (format reference, same weights) | skipped: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| Accuracy (long golden) | **PASS** (prefill top-1 94.27% KL 3.99e-2; decode top-1 97.19% KL 1.02e-2) |
| Accuracy (4k golden) | **PASS** (prefill top-1 93.09% KL 7.95e-2; decode top-1 98.44% KL 8.92e-3) |
| Quick gate | **PASS**: perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 72s, in-depth 3m52s, total 5m05s |

### Failures

None.

### Skipped

- `kernels`: skipped by --skip
- `acc-format`: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE)
- `ab`: skipped by --skip

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 1s | AMD Radeon AI PRO R9700 (gfx1201), 95 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 12s | 31 ops resolved, exactness deterministic, loaded in 12.3s |
| 4 | `kernels` kernels test on the model's shapes | quick | SKIP | 0s | skipped by --skip |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 7s | perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| 7 | `acc-short` short accuracy against a golden | quick | PASS | 9s | prefill top-1 98.47% KL 6.00e-3; decode top-1 98.75% KL 1.11e-2 |
| 8 | `acc-oracle` short prefill accuracy against the f64 CPU oracle (same weights, deterministic reference) | quick | PASS | 35s | prefill top-1 95.77% KL 1.04e-2 |
| 9 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 9s | decode 163.3 tok/s at 128 tokens (59% of the byte floor) |
| 10 | `acc-long` accuracy --mode all on the long golden (prefill, decode, prefix reuse) | full | PASS | 81s | prefill top-1 94.27% KL 3.99e-2; decode top-1 97.19% KL 1.02e-2 |
| 11 | `acc-4k` accuracy --mode all on the 4k golden (prefill, decode, prefix reuse past 2 chunks of 2048) | full | PASS | 56s | prefill top-1 93.09% KL 7.95e-2; decode top-1 98.44% KL 8.92e-3 |
| 12 | `envelope` noise envelope and baseline for regression gating | full | PASS | 26s | baseline and envelope (chunk + attn-simt) written; they gate this build: PASS |
| 13 | `bench-long` long-context bench up to 16k | full | PASS | 13s | decode 163.0 tok/s at 1024 tokens (59% of the byte floor) |
| 14 | `ab` A/B against the reference server via localeval | full | SKIP | 0s | skipped by --skip |
| 15 | `tp` tensor-parallel variant | full | PASS | 57s | decode 191.1 tok/s at 128 tokens (334% of the byte floor); oracle prefill top-1 96.13% KL 1.08e-2; accuracy prefill top-1 98.27% KL 7.08e-3; decode top-1 97.50% KL 8.21e-3 |

### Notes

- `acc-short`: gated against the recorded baseline /work/golden/baselines/qwen36-35b-a3b-nvfp4-vllm-top20.hip.baseline.json (commit 4379dd0) and its noise envelope
- `acc-oracle`: floor: mean KL(oracle \|\| engine) over the oracle's top-64 + tail, scored prefill rows, <= 2.0e-2 (measured, docs/results/2026-10-22-fx1-oracle-gate.md); top-1 not gated. The floor guards absolute accuracy; it does not see a 1 % scale bug, the drift check does
- `acc-oracle`: drift check: logits unchanged against the recorded dump /work/golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.hip.selfgolden.safetensors
- `acc-long`: prefix reuse is gated in acc-4k, on the 4k golden
- `acc-4k`: prefix reuse at chunk 1024: 3 judged, 0 skipped
- `acc-4k`: gated against the recorded baseline golden/baselines/qwen36-35b-a3b-nvfp4-vllm-4k-top20.hip.baseline.json (commit e1c2f92), no noise envelope
- `tp`: size 128: 334% of the byte floor is above 100%: the --gb-per-token estimate is too high or the run is not a plain decode (check the floor inputs)
- `tp`: tp=2 drift check: logits unchanged against the recorded dump /work/golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.hip-tp2.selfgolden.safetensors

## Speed

**quick bench** (floor: 2.25 GB/token at 627 GB/s; gb/token: estimated from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 627 GB/s, the R9700 figure of docs/results/2026-10-13-phase2a-hip.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 129 | 0.071 | 1813 | 163.33 | - | 278.1 | 58.7 |

**long-context bench** (floor: 2.25 GB/token at 627 GB/s; gb/token: estimated from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 627 GB/s, the R9700 figure of docs/results/2026-10-13-phase2a-hip.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 997 | 0.104 | 9577 | 162.96 | - | 278.1 | 58.6 |
| 4085 | 0.319 | 12797 | 158.18 | - | 278.1 | 56.9 |
| 16699 | 1.526 | 10940 | 143.98 | - | 278.1 | 51.8 |

**tensor-parallel bench** (floor: 10.94 GB/token at 627 GB/s; gb/token: busiest tp rank's weights from the plan; bandwidth: 627 GB/s, the R9700 figure of docs/results/2026-10-13-phase2a-hip.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 125 | 0.051 | 2441 | 191.09 | - | 57.3 | 333.5 |

## Accuracy

**acc-short** against `/models/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors` (given with --golden; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1441 | 98.47 | 91.70 | 6.00e-3 | 5.69e-3 | 6.19e-1 | 1.35e-3 | 0 | 6.7 | pass |
| decode | 160 | 98.75 | 93.38 | 1.11e-2 | 1.11e-2 | 7.31e-1 | 2.26e-3 | 0 | 88.8 | pass |

**acc-long** against `/models/golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 36887 | 94.27 | 90.37 | 3.99e-2 | 4.27e-2 | 2.65e1 | 7.47e-3 | 0 | 0.3 | pass |
| decode | 320 | 97.19 | 89.50 | 1.02e-2 | 9.60e-3 | 2.02e-1 | 2.46e-3 | 0 | 81.6 | pass |

**acc-4k** against `golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 12759 | 93.09 | 89.90 | 7.95e-2 | 9.29e-2 | 2.73e1 | 1.19e-2 | 0 | 0.6 | pass |
| decode | 192 | 98.44 | 90.94 | 8.92e-3 | 8.69e-3 | 1.90e-1 | 1.29e-3 | 0 | 87.5 | pass |

**tp** against `/models/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors` (given with --golden; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1441 | 98.27 | 91.23 | 7.08e-3 | 6.68e-3 | 1.11e0 | 1.63e-3 | 0 | 19.4 | pass |
| decode | 160 | 97.50 | 91.75 | 8.21e-3 | 8.32e-3 | 2.63e-1 | 1.95e-3 | 0 | 81.2 | pass |

**Regression gating artifacts**: baseline and envelope (chunk + attn-simt) written; they gate this build: PASS

`intarsia accuracy --model <dir> --backend hip --golden /models/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --baseline baselines/acc-short.baseline.json --noise-envelope baselines/envelope.json  (copy both files next to the golden, e.g. golden/baselines/)`

**Quick gate**: perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) (PASS)
- plan: 31 ops resolved, exactness deterministic, loaded in 12.3s (PASS)

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
| `doctor` | head=ec38aca code-tree=7b1c3dab64f0 dirty=no | tatooine | 1795309140 | `/target/release/intarsia doctor --backend hip --device 0 --out /work/.rocm102-logs/1121-1958/full-r10/doctor.json` |
| `recipe` | head=ec38aca code-tree=7b1c3dab64f0 dirty=no | tatooine | 1795309141 | `recipe::analyse(/models/Qwen3.6-35B-A3B-NVFP4)` |
| `plan` | head=ec38aca code-tree=7b1c3dab64f0 dirty=no | tatooine | 1795309141 | `/target/release/intarsia plan show --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --admit --json` |
| `kernels` | head=ec38aca code-tree=7b1c3dab64f0 dirty=no | tatooine | 0 | `/target/release/intarsia plan show --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --admit --json` |
| `gate` | head=ec38aca code-tree=7b1c3dab64f0 dirty=no | tatooine | 1795309153 | `/target/release/intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --quick --max-seq 2048` |
| `acc-format` | head=ec38aca code-tree=7b1c3dab64f0 dirty=no | tatooine | 1795309160 | `-` |
| `acc-short` | head=ec38aca code-tree=7b1c3dab64f0 dirty=no | tatooine | 1795309160 | `/target/release/intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --golden /models/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --decode-positions 16 --max-seq 4096 --write-baseline /work/.rocm102-logs/1121-1958/full-r10/baselines/acc-` |
| `acc-oracle` | head=ec38aca code-tree=7b1c3dab64f0 dirty=no | tatooine | 1795309169 | `/target/release/intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --self-golden /work/golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.hip.selfgolden.safetensors --mode prefill --decode-positions 16 --max-seq 4096 --model-id Qwen3.6-35B-A3B-NVFP4 --allo` |
| `bench` | head=ec38aca code-tree=7b1c3dab64f0 dirty=no | tatooine | 1795309203 | `/target/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3.6-35B-A3B-NVFP4 --sizes 128 --gen 16 --reps 1 --warmup 1 --thinking off --label qualify-Qwen3.6-35B-A3B-NVFP4-hip-bench --out-dir /work/.rocm102-logs/1121-1958/full-r10/bench --tp 1 --fingerprint model=Qwen3.6-35B` |
| `acc-long` | head=ec38aca code-tree=7b1c3dab64f0 dirty=no | tatooine | 1795309212 | `/target/release/intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --golden /models/golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors --mode both --decode-positions 64 --max-seq 12288 --write-baseline /work/.rocm102-logs/1121-1958/full-r10/baseline` |
| `acc-4k` | head=ec38aca code-tree=7b1c3dab64f0 dirty=no | tatooine | 1795309294 | `/target/release/intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --golden golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors --mode reuse --decode-positions 64 --max-seq 8192 --write-baseline /work/.rocm102-logs/1121-1958/full-r10/baselines/acc-4k-r` |
| `envelope` | head=ec38aca code-tree=7b1c3dab64f0 dirty=no | tatooine | 1795309349 | `/target/release/intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --golden /models/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --decode-positions 16 --max-seq 4096 --baseline /work/.rocm102-logs/1121-1958/full-r10/baselines/acc-short.` |
| `bench-long` | head=ec38aca code-tree=7b1c3dab64f0 dirty=no | tatooine | 1795309375 | `/target/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3.6-35B-A3B-NVFP4 --sizes 1024,4096,16384 --gen 16 --reps 1 --warmup 1 --thinking off --label qualify-Qwen3.6-35B-A3B-NVFP4-hip-bench-long --out-dir /work/.rocm102-logs/1121-1958/full-r10/bench --tp 1 --fingerprint ` |
| `ab` | head=ec38aca code-tree=7b1c3dab64f0 dirty=no | tatooine | 0 | `/target/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3.6-35B-A3B-NVFP4 --sizes 1024,4096,16384 --gen 16 --reps 1 --warmup 1 --thinking off --label qualify-Qwen3.6-35B-A3B-NVFP4-hip-bench-long --out-dir /work/.rocm102-logs/1121-1958/full-r10/bench --tp 1 --fingerprint ` |
| `tp` | head=ec38aca code-tree=7b1c3dab64f0 dirty=no | tatooine | 1795309388 | `/target/release/intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --tp 2 --devices 0,1 --mem-budget 29 --self-golden /work/golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.hip-tp2.selfgolden.safetensors --mode prefill --decode-positions 16 --max-seq 4096 --model-id Qwe` |

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
- `bench/qualify-Qwen3.6-35B-A3B-NVFP4-hip-bench.json` (bench)
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
- `bench/qualify-Qwen3.6-35B-A3B-NVFP4-hip-bench-long.json` (bench-long)
- `logs/bench-long-bench.out` (bench-long)
- `logs/bench-long-server.err` (bench-long)
- `bench/qualify-Qwen3.6-35B-A3B-NVFP4-hip-tp2.json` (tp)
- `logs/tp2-bench.out` (tp)
- `logs/tp2-server.err` (tp)
- `baselines/tp2-acc.baseline.json` (tp)
- `state.json` (stage records; `--resume` reads it)
