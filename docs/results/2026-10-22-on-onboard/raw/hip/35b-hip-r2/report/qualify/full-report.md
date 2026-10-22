# Qualification report: Qwen3.6-35B-A3B-NVFP4 on hip

_Full (quick + in-depth) section, generated unix 1792640683, intarsia 0.0.1 build `head=a3288db code-tree=dc8cbb40ec42 dirty=yes`._

## Headline

**Verdict: ERROR** (14 stages: 8 pass, 0 fail, 2 error, 4 skipped)

| | |
|---|---|
| Model | Qwen3.6-35B-A3B-NVFP4 (`/models/Qwen3.6-35B-A3B-NVFP4`) |
| Backend / device | hip / AMD Radeon Graphics (gfx1201) |
| Recipe | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| Decode | 128 tok: **159.9** tok/s (58% of floor); 1024 tok: **159.3** tok/s (57% of floor); 4096 tok: **154.8** tok/s (56% of floor) |
| Prefill | 128 tok: **1945** tok/s; 1024 tok: **7902** tok/s; 4096 tok: **8642** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 98.61% KL 7.63e-3; decode top-1 98.75% KL 1.18e-2) |
| Accuracy (format reference, same weights) | skipped: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| Accuracy (long golden) | **ERROR** |
| Accuracy (4k golden) | skipped: no 4k golden for 'Qwen3.6-35B-A3B-NVFP4' in golden, /models/golden, /work/golden (give --golden-4k FILE or --golden-dir DIR, or --reference <vLLM url> to make one; build one with tools/golden.py --prompts tools/prompts/golden_4k.jsonl) |
| Quick gate | **PASS**: perplexity 1.21, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 80s, in-depth 67s, total 2m26s |

### Failures

- `acc-long` **ERROR**: accuracy exit 1: Error: long_docs_12k: bad request: sequence of 12064 tokens exceeds the planned context of 8192
- `envelope` **ERROR**: baseline self-check printed no baseline verdict: Error: the baseline was recorded from a modified tree (head a3288db code-tree dc8cbb40ec42 dirty): only a build of a3288db plus the harness files ["crates/intarsia-cli/src/cmd/accuracy.rs", "crates/intarsia-cli/src/cmd/accuracy/baseline.rs", "crates/i

### Skipped

- `acc-format`: no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE)
- `acc-4k`: no 4k golden for 'Qwen3.6-35B-A3B-NVFP4' in golden, /models/golden, /work/golden (give --golden-4k FILE or --golden-dir DIR, or --reference <vLLM url> to make one; build one with tools/golden.py --prompts tools/prompts/golden_4k.jsonl)
- `ab`: no --reference server given
- `tp`: --tp not given (or 1)

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | AMD Radeon Graphics (gfx1201), 54 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 9s | 30 ops resolved, exactness deterministic, loaded in 9.2s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 39s | 54 kernels, 995 cases in 38.8s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 6s | perplexity 1.21, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | no format-reference golden for 'Qwen3.6-35B-A3B-NVFP4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| 7 | `acc-short` short accuracy against a golden | quick | PASS | 8s | prefill top-1 98.61% KL 7.63e-3; decode top-1 98.75% KL 1.18e-2 |
| 8 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 17s | decode 159.9 tok/s at 128 tokens (58% of the byte floor) |
| 9 | `acc-long` accuracy --mode all on the long golden (prefill, decode, prefix reuse) | full | ERROR | 30s | accuracy exit 1: Error: long_docs_12k: bad request: sequence of 12064 tokens exceeds the planned context of 8192 |
| 10 | `acc-4k` accuracy --mode all on the 4k golden (prefill, decode, prefix reuse past 2 chunks of 2048) | full | SKIP | 0s | no 4k golden for 'Qwen3.6-35B-A3B-NVFP4' in golden, /models/golden, /work/golden (give --golden-4k FILE or --golden-dir DIR, or --reference <vLLM url> to make one; build one with tools/golden.py --prompts tools/prompts/golden_4k.jsonl) |
| 11 | `envelope` noise envelope and baseline for regression gating | full | ERROR | 19s | baseline self-check printed no baseline verdict: Error: the baseline was recorded from a modified tree (head a3288db code-tree dc8cbb40ec42 dirty): only a build of a3288db plus the harness files ["crates/intarsia-cli/src/cmd/accuracy.rs", "crates/intarsia-cli/src/cmd/accuracy/baseline.rs", "crates/i |
| 12 | `bench-long` long-context bench up to 16k | full | PASS | 18s | decode 159.3 tok/s at 1024 tokens (57% of the byte floor) |
| 13 | `ab` A/B against the reference server via localeval | full | SKIP | 0s | no --reference server given |
| 14 | `tp` tensor-parallel variant | full | SKIP | 0s | --tp not given (or 1) |

### Notes

None.

## Speed

**quick bench** (floor: 2.25 GB/token at 627 GB/s; gb/token: estimated from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 627 GB/s, the R9700 figure of docs/results/2026-10-13-phase2a-hip.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 130 | 0.067 | 1945 | 159.93 | 0.39 | 278.1 | 57.5 |
| 1023 | 0.129 | 7902 | 159.30 | 0.04 | 278.1 | 57.3 |
| 4242 | 0.491 | 8642 | 154.75 | 0.03 | 278.1 | 55.6 |

**long-context bench** (floor: 2.25 GB/token at 627 GB/s; gb/token: estimated from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 627 GB/s, the R9700 figure of docs/results/2026-10-13-phase2a-hip.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 1001 | 0.128 | 7845 | 159.28 | 0.19 | 278.1 | 57.3 |
| 4111 | 0.448 | 9206 | 155.60 | 0.03 | 278.1 | 56.0 |
| 16857 | 2.140 | 7876 | 141.18 | 0.10 | 278.1 | 50.8 |

## Accuracy

**acc-short** against `golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1441 | 98.61 | 91.30 | 7.63e-3 | 7.42e-3 | 7.11e-1 | 1.80e-3 | 0 | 31.0 | pass |
| decode | 160 | 98.75 | 92.88 | 1.18e-2 | 1.17e-2 | 7.31e-1 | 2.27e-3 | 0 | 88.8 | pass |

**Regression gating artifacts**: baseline self-check printed no baseline verdict: Error: the baseline was recorded from a modified tree (head a3288db code-tree dc8cbb40ec42 dirty): only a build of a3288db plus the harness files ["crates/intarsia-cli/src/cmd/accuracy.rs", "crates/intarsia-cli/src/cmd/accuracy/baseline.rs", "crates/intarsia-cli/src/cmd/accuracy/envelope.rs", "crates/intarsia-cli/src/cmd/accuracy/provenance.rs", "crates/intarsia-cli/src/cmd/accuracy/selfgolden.rs"] is a legitimate pre-change build

`intarsia accuracy --model <dir> --backend hip --golden golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --baseline baselines/acc-short.baseline.json --noise-envelope baselines/envelope.json  (copy both files next to the golden, e.g. golden/baselines/)`

**Quick gate**: perplexity 1.21, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) (PASS)
- plan: 30 ops resolved, exactness deterministic, loaded in 9.2s (PASS)

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

- kernels test: 54 kernels, 995 cases in 38.8s (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | head=a3288db code-tree=dc8cbb40ec42 dirty=yes | tatooine | 1792640537 | `/work/on-intarsia doctor --backend hip --device 0 --out /work/onboard-runs/35b-hip-r2/report/qualify/doctor.json` |
| `recipe` | head=a3288db code-tree=dc8cbb40ec42 dirty=yes | tatooine | 1792640537 | `recipe::analyse(/models/Qwen3.6-35B-A3B-NVFP4)` |
| `plan` | head=a3288db code-tree=dc8cbb40ec42 dirty=yes | tatooine | 1792640537 | `/work/on-intarsia plan show --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --admit --json` |
| `kernels` | head=a3288db code-tree=dc8cbb40ec42 dirty=yes | tatooine | 1792640546 | `/work/on-intarsia kernels test --backend hip --device 0 --model /models/Qwen3.6-35B-A3B-NVFP4` |
| `gate` | head=a3288db code-tree=dc8cbb40ec42 dirty=yes | tatooine | 1792640585 | `/work/on-intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --quick --max-seq 2048` |
| `acc-format` | head=a3288db code-tree=dc8cbb40ec42 dirty=yes | tatooine | 1792640617 | `-` |
| `acc-short` | head=a3288db code-tree=dc8cbb40ec42 dirty=yes | tatooine | 1792640591 | `/work/on-intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --golden golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --decode-positions 16 --max-seq 4096 --write-baseline /work/onboard-runs/35b-hip-r2/report/qualify/baselines/acc-short.bas` |
| `bench` | head=a3288db code-tree=dc8cbb40ec42 dirty=yes | tatooine | 1792640600 | `/work/on-intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3.6-35B-A3B-NVFP4 --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-Qwen3.6-35B-A3B-NVFP4-hip-bench --out-dir /work/onboard-runs/35b-hip-r2/report/qualify/bench --tp 1 --fingerprint model=Q` |
| `acc-long` | head=a3288db code-tree=dc8cbb40ec42 dirty=yes | tatooine | 1792640617 | `/work/on-intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --golden golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors --mode all --decode-positions 64 --max-seq 8192 --write-baseline /work/onboard-runs/35b-hip-r2/report/qualify/baselines/acc-long.` |
| `acc-4k` | head=a3288db code-tree=dc8cbb40ec42 dirty=yes | tatooine | 1792640647 | `-` |
| `envelope` | head=a3288db code-tree=dc8cbb40ec42 dirty=yes | tatooine | 1792640647 | `/work/on-intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --golden golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --decode-positions 16 --max-seq 4096 --baseline /work/onboard-runs/35b-hip-r2/report/qualify/baselines/acc-short.baseline.` |
| `bench-long` | head=a3288db code-tree=dc8cbb40ec42 dirty=yes | tatooine | 1792640665 | `/work/on-intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3.6-35B-A3B-NVFP4 --sizes 1024,4096,16384 --gen 64 --reps 2 --warmup 1 --thinking off --label qualify-Qwen3.6-35B-A3B-NVFP4-hip-bench-long --out-dir /work/onboard-runs/35b-hip-r2/report/qualify/bench --tp 1 --fingerprint m` |
| `ab` | head=a3288db code-tree=dc8cbb40ec42 dirty=yes | tatooine | 1792640683 | `-` |
| `tp` | head=a3288db code-tree=dc8cbb40ec42 dirty=yes | tatooine | 1792640683 | `-` |

Model fingerprint: `fnv1a64:9ca75aef5131bb81`.


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/kernels.out` (kernels)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `bench/qualify-Qwen3.6-35B-A3B-NVFP4-hip-bench.json` (bench)
- `logs/bench-bench.out` (bench)
- `logs/bench-server.err` (bench)
- `logs/acc-long.out` (acc-long)
- `baselines/acc-short.baseline.json` (envelope)
- `baselines/envelope.json` (envelope)
- `logs/baseline-selfcheck.out` (envelope)
- `bench/qualify-Qwen3.6-35B-A3B-NVFP4-hip-bench-long.json` (bench-long)
- `logs/bench-long-bench.out` (bench-long)
- `logs/bench-long-server.err` (bench-long)
- `state.json` (stage records; `--resume` reads it)
