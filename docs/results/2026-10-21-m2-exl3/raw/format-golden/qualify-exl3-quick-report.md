# Qualification quick report: qwen3-8b-exl3-4.0bpw-bf16plain on cuda

_Quick section, generated unix 1792606418, intarsia 0.0.1 build `m2-fg-03232786`._

## Headline

**Verdict: PASS** (8 stages: 6 pass, 0 fail, 0 error, 2 skipped)

| | |
|---|---|
| Model | qwen3-8b-exl3-4.0bpw-bf16plain (`/home/nvidia/models-m2/qwen3-8b-exl3-4.0bpw-bf16plain`) |
| Backend / device | cuda / NVIDIA GB10 (sm_121) |
| Recipe | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant exl3) |
| Decode | not measured |
| Prefill | not measured |
| Accuracy (short golden) | **PASS** (prefill top-1 95.16% KL 2.32e-2; decode top-1 94.53% KL 3.26e-2) |
| Accuracy (format reference, same weights) | **PASS** (prefill top-1 98.85% KL 4.40e-4; decode top-1 99.22% KL 1.15e-3) |
| Quick gate | **PASS**: perplexity 2.01, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 1m34s |

### Failures

None.

### Skipped

- `kernels`: skipped by --skip
- `bench`: skipped by --skip

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 1s | NVIDIA GB10 (sm_121), 54 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant exl3) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 23s | 13 ops resolved, exactness deterministic, loaded in 23.4s |
| 4 | `kernels` kernels test on the model's shapes | quick | SKIP | 0s | skipped by --skip |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 26s | perplexity 2.01, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-short` short accuracy against a golden | quick | PASS | 22s | prefill top-1 95.16% KL 2.32e-2; decode top-1 94.53% KL 3.26e-2 |
| 7 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | PASS | 22s | prefill top-1 98.85% KL 4.40e-4; decode top-1 99.22% KL 1.15e-3 |
| 8 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | SKIP | 0s | skipped by --skip |

### Notes

None.

## Speed

No speed numbers.

## Accuracy

**acc-short** against `/home/nvidia/intarsiaml-golden/qwen3-8b-fp32.safetensors` (given with --golden; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 434 | 95.16 | 92.40 | 2.32e-2 | 2.00e-2 | 9.94e-1 | 2.69e-3 | 0 | 50.2 | pass |
| decode | 128 | 94.53 | 90.94 | 3.26e-2 | 2.73e-2 | 5.69e-1 | 3.71e-3 | 0 | 69.5 | pass |

**acc-format** against `/home/nvidia/m2-fg-gold/qwen3-8b-exl3-4.0bpw-bf16plain-formatref-exllamav3.safetensors` (found by model name; reference: exllamav3 1.6.0; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 434 | 98.85 | 98.66 | 4.40e-4 | 3.69e-4 | 2.55e-2 | 5.29e-4 | 0 | 69.6 | pass |
| decode | 128 | 99.22 | 98.44 | 1.15e-3 | 1.08e-3 | 6.71e-2 | 3.81e-4 | 0 | 97.7 | pass |

**Quick gate**: perplexity 2.01, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3 covers this model (25/25 tensor templates resolve, quant exl3) (PASS)
- plan: 13 ops resolved, exactness deterministic, loaded in 23.4s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_bf16 | exact | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 1 |
| rmsnorm_add | rmsnorm_add_bf16 | deterministic | yes | 72 |
| qk_norm | qk_norm_bf16 | deterministic | yes | 72 |
| rope | rope_bf16 | deterministic | yes | 72 |
| kv_append | kv_append_bf16 | exact | yes | 72 |
| attn_prefill | attn_prefill_bf16_flash | deterministic | yes | 0 |
| attn_prefill | attn_prefill_bf16_splitkv | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 36 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 36 |
| argmax | argmax_f32 | exact | yes | 1 |
| matmul_exl3 | matmul_exl3_gemm | deterministic | yes | 0 |
| matmul_exl3 | matmul_exl3_gemv | deterministic | yes | 253 |

- kernels test: skipped by --skip (SKIP)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | m2-fg-03232786 | spark-4e77 | 1792606324 | `/home/nvidia/intarsia-decfuse/m2-fg/target/acc/release/intarsia doctor --backend cuda --device 0 --out /home/nvidia/m2-fg-qualify/exl3/doctor.json` |
| `recipe` | m2-fg-03232786 | spark-4e77 | 1792606325 | `recipe::analyse(/home/nvidia/models-m2/qwen3-8b-exl3-4.0bpw-bf16plain)` |
| `plan` | m2-fg-03232786 | spark-4e77 | 1792606325 | `/home/nvidia/intarsia-decfuse/m2-fg/target/acc/release/intarsia plan show --model /home/nvidia/models-m2/qwen3-8b-exl3-4.0bpw-bf16plain --backend cuda --device 0 --admit --json` |
| `kernels` | m2-fg-03232786 | spark-4e77 | 0 | `/home/nvidia/intarsia-decfuse/m2-fg/target/acc/release/intarsia plan show --model /home/nvidia/models-m2/qwen3-8b-exl3-4.0bpw-bf16plain --backend cuda --device 0 --admit --json` |
| `gate` | m2-fg-03232786 | spark-4e77 | 1792606348 | `/home/nvidia/intarsia-decfuse/m2-fg/target/acc/release/intarsia accuracy --model /home/nvidia/models-m2/qwen3-8b-exl3-4.0bpw-bf16plain --backend cuda --device 0 --quick --max-seq 2048` |
| `acc-short` | m2-fg-03232786 | spark-4e77 | 1792606374 | `/home/nvidia/intarsia-decfuse/m2-fg/target/acc/release/intarsia accuracy --model /home/nvidia/models-m2/qwen3-8b-exl3-4.0bpw-bf16plain --backend cuda --device 0 --golden /home/nvidia/intarsiaml-golden/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq 4096 --write-baseline /home/n` |
| `acc-format` | m2-fg-03232786 | spark-4e77 | 1792606396 | `/home/nvidia/intarsia-decfuse/m2-fg/target/acc/release/intarsia accuracy --model /home/nvidia/models-m2/qwen3-8b-exl3-4.0bpw-bf16plain --backend cuda --device 0 --golden /home/nvidia/m2-fg-gold/qwen3-8b-exl3-4.0bpw-bf16plain-formatref-exllamav3.safetensors --mode both --decode-positions 16 --max-seq` |
| `bench` | m2-fg-03232786 | spark-4e77 | 0 | `/home/nvidia/intarsia-decfuse/m2-fg/target/acc/release/intarsia accuracy --model /home/nvidia/models-m2/qwen3-8b-exl3-4.0bpw-bf16plain --backend cuda --device 0 --golden /home/nvidia/m2-fg-gold/qwen3-8b-exl3-4.0bpw-bf16plain-formatref-exllamav3.safetensors --mode both --decode-positions 16 --max-seq` |

Model fingerprint: `fnv1a64:4640ad0b409e6c43`.


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `logs/acc-format.out` (acc-format)
- `baselines/acc-format.baseline.json` (acc-format)
- `state.json` (stage records; `--resume` reads it)
