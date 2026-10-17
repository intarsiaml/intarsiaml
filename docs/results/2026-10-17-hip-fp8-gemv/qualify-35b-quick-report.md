# Qualification quick report: Qwen3.6-35B-A3B-NVFP4 on hip

_Quick section, generated unix 1792253629, intarsia 0.0.1 build `unstamped`._

## Headline

**Verdict: PASS** (7 stages: 7 pass, 0 fail, 0 error, 0 skipped)

| | |
|---|---|
| Model | Qwen3.6-35B-A3B-NVFP4 (`/models/Qwen3.6-35B-A3B-NVFP4`) |
| Backend / device | hip / AMD Radeon Graphics (gfx1201) |
| Recipe | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| Decode | 128 tok: **136.8** tok/s (49% of floor); 1024 tok: **135.9** tok/s (49% of floor); 4096 tok: **132.3** tok/s (48% of floor) |
| Prefill | 128 tok: **1188** tok/s; 1024 tok: **2103** tok/s; 4096 tok: **2057** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 98.54% KL 6.56e-3; decode top-1 99.38% KL 9.96e-3) |
| Quick gate | **PASS**: perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 1m36s |

### Failures

None.

### Skipped

None.

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 1s | AMD Radeon Graphics (gfx1201), 38 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 13s | 28 ops resolved, exactness deterministic, loaded in 13.4s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 28s | 38 kernels, 613 cases in 27.6s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 8s | perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-short` short accuracy against a golden | quick | PASS | 9s | prefill top-1 98.54% KL 6.56e-3; decode top-1 99.38% KL 9.96e-3 |
| 7 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 38s | decode 136.8 tok/s at 128 tokens (49% of the byte floor) |

### Notes

None.

## Speed

**quick bench** (floor: 2.25 GB/token at 627 GB/s; gb/token: estimated from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 627 GB/s, the R9700 figure of docs/results/2026-10-13-phase2a-hip.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 133 | 0.113 | 1188 | 136.79 | 0.19 | 278.1 | 49.2 |
| 1027 | 0.488 | 2103 | 135.88 | 0.09 | 278.1 | 48.9 |
| 4224 | 2.050 | 2057 | 132.26 | 0.06 | 278.1 | 47.6 |

## Accuracy

**acc-short** against `/models/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1441 | 98.54 | 91.42 | 6.56e-3 | 6.35e-3 | 6.04e-1 | 1.71e-3 | 0 | 7.3 | pass |
| decode | 160 | 99.38 | 92.88 | 9.96e-3 | 1.07e-2 | 3.50e-1 | 3.17e-3 | 0 | 90.0 | pass |

**Quick gate**: perplexity 1.22, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3_5_moe covers this model (61/61 tensor templates resolve, quant nvfp4+fp8) (PASS)
- plan: 28 ops resolved, exactness deterministic, loaded in 13.4s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_bf16 | exact | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 1 |
| rmsnorm_add | rmsnorm_add_bf16 | deterministic | yes | 80 |
| qk_norm | qk_norm_bf16 | deterministic | yes | 20 |
| rope | rope_bf16 | deterministic | yes | 20 |
| matmul | matmul_bf16 | deterministic | yes | 0 |
| matmul | matmul_bf16_gemv | deterministic | yes | 0 |
| kv_append | kv_append_bf16 | exact | yes | 20 |
| attn_prefill | attn_prefill_bf16_flash | deterministic | yes | 0 |
| attn_prefill | attn_prefill_bf16_splitkv | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 10 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 0 |
| argmax | argmax_f32 | exact | yes | 1 |
| matmul_fp8 | matmul_fp8_gemv_v2 | deterministic | yes | 50 |
| matmul_fp8 | matmul_fp8_mma | deterministic | yes | 0 |
| matmul_nvfp4 | matmul_nvfp4_gemv_v2 | deterministic | yes | 41 |
| matmul_nvfp4 | matmul_nvfp4_mma | deterministic | yes | 0 |
| moe_topk | moe_topk_bf16_v2 | deterministic | yes | 40 |
| moe_experts | moe_experts_bf16_v2 | deterministic | yes | 40 |
| moe_combine | moe_combine_bf16 | deterministic | yes | 40 |
| gate_split | gate_split_bf16 | deterministic | yes | 10 |
| sigmoid_mul | sigmoid_mul_bf16 | deterministic | yes | 10 |
| gdn_conv | gdn_conv_bf16 | deterministic | yes | 30 |
| gdn_delta_rule | gdn_delta_rule_bf16 | deterministic | yes | 30 |
| matmul_pair | matmul_pair_bf16_gemv | deterministic | yes | 70 |
| matmul_fp8_pair | matmul_fp8_pair_gemv | deterministic | yes | 40 |
| matmul_nvfp4_glu | matmul_nvfp4_glu_gemv | deterministic | yes | 40 |
| gated_rmsnorm | gated_rmsnorm_bf16 | deterministic | yes | 30 |

- kernels test: 38 kernels, 613 cases in 27.6s (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | unstamped | tatooine | 1792253533 | `/target/release/intarsia doctor --backend hip --device 0 --out /work/bench-out/qualify-35b-pair1/doctor.json` |
| `recipe` | unstamped | tatooine | 1792253533 | `recipe::analyse(/models/Qwen3.6-35B-A3B-NVFP4)` |
| `plan` | unstamped | tatooine | 1792253534 | `/target/release/intarsia plan show --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --admit --json` |
| `kernels` | unstamped | tatooine | 1792253547 | `/target/release/intarsia kernels test --backend hip --device 0 --model /models/Qwen3.6-35B-A3B-NVFP4` |
| `gate` | unstamped | tatooine | 1792253575 | `/target/release/intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --quick --max-seq 2048` |
| `acc-short` | unstamped | tatooine | 1792253582 | `/target/release/intarsia accuracy --model /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --device 0 --mem-budget 29 --golden /models/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --decode-positions 16 --max-seq 4096 --write-baseline /work/bench-out/qualify-35b-pair1/baselines/acc-short` |
| `bench` | unstamped | tatooine | 1792253592 | `/target/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3.6-35B-A3B-NVFP4 --sizes 128,1024,4096 --gen 128 --reps 5 --warmup 1 --thinking off --label qualify-Qwen3.6-35B-A3B-NVFP4-hip-bench --out-dir /work/bench-out/qualify-35b-pair1/bench --tp 1 --fingerprint model=Qwen3` |

Model fingerprint: `fnv1a64:9ca75aef5131bb81`.

Build stamp `unstamped` means a plain `cargo build`: baselines written here gate this binary, but `--pre-change` provenance checks refuse them as a "before".


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
- `state.json` (stage records; `--resume` reads it)
