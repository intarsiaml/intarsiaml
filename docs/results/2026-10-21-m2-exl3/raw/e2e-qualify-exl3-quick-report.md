# Qualification quick report: qwen3-8b-exl3-4.0bpw-bf16plain on cuda

_Quick section, generated unix 1792595512, intarsia 0.0.1 build `head=13f2eed code-tree=5fcde375616c dirty=yes`._

## Headline

**Verdict: FAIL** (7 stages: 6 pass, 1 fail, 0 error, 0 skipped)

| | |
|---|---|
| Model | qwen3-8b-exl3-4.0bpw-bf16plain (`/home/nvidia/models-m2/qwen3-8b-exl3-4.0bpw-bf16plain`) |
| Backend / device | cuda / NVIDIA GB10 (sm_121) |
| Recipe | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant exl3) |
| Decode | 128 tok: **47.2** tok/s (111% of floor); 1024 tok: **45.8** tok/s (108% of floor); 4096 tok: **37.9** tok/s (89% of floor) |
| Prefill | 128 tok: **1130** tok/s; 1024 tok: **1209** tok/s; 4096 tok: **1074** tok/s |
| Accuracy (short golden) | **FAIL** (prefill top-1 95.16% KL 2.32e-2; decode top-1 94.53% KL 3.26e-2) |
| Quick gate | **PASS**: perplexity 2.01, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 2m52s |

### Failures

- `acc-short` **FAIL**: decode: full-vocab KL 3.255e-2 (need <= 3.0e-2)

### Skipped

None.

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 1s | NVIDIA GB10 (sm_121), 52 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant exl3) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 19s | 12 ops resolved, exactness deterministic, loaded in 19.2s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 63s | 52 kernels, 964 cases in 62.3s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 15s | perplexity 2.01, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-short` short accuracy against a golden | quick | FAIL | 14s | decode: full-vocab KL 3.255e-2 (need <= 3.0e-2) |
| 7 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 60s | decode 47.2 tok/s at 128 tokens (111% of the byte floor) |

### Notes

- `bench`: size 128: 111% of the byte floor is above 100%: the --gb-per-token estimate is too high or the run is not a plain decode (check the floor inputs)
- `bench`: size 1024: 108% of the byte floor is above 100%: the --gb-per-token estimate is too high or the run is not a plain decode (check the floor inputs)

## Speed

**quick bench** (floor: 5.19 GB/token at 221 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 221 GB/s, the GB10 measured figure)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 122 | 0.108 | 1130 | 47.25 | 0.02 | 42.6 | 111.0 |
| 1008 | 0.834 | 1209 | 45.82 | 0.06 | 42.6 | 107.6 |
| 4204 | 3.889 | 1074 | 37.89 | 0.05 | 42.6 | 89.0 |

## Accuracy

**acc-short** against `/home/nvidia/intarsiaml-golden/qwen3-8b-fp32.safetensors` (given with --golden; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 434 | 95.16 | 92.40 | 2.32e-2 | 2.00e-2 | 9.94e-1 | 2.69e-3 | 0 | 50.2 | pass |
| decode | 128 | 94.53 | 90.94 | 3.26e-2 | 2.73e-2 | 5.69e-1 | 3.71e-3 | 0 | 69.5 | FAIL |

**Quick gate**: perplexity 2.01, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3 covers this model (25/25 tensor templates resolve, quant exl3) (PASS)
- plan: 12 ops resolved, exactness deterministic, loaded in 19.2s (PASS)

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
| matmul_exl3 | matmul_exl3_gemv | deterministic | yes | 253 |

- kernels test: 52 kernels, 964 cases in 62.3s (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | head=13f2eed code-tree=5fcde375616c dirty=yes | spark-4e77 | 1792595340 | `/home/nvidia/intarsia-decfuse/m2-exl3-e2e/target/acc/release/intarsia doctor --backend cuda --device 0 --out /home/nvidia/intarsia-decfuse/m2-exl3-e2e/reports-m2e2e/exl3/doctor.json` |
| `recipe` | head=13f2eed code-tree=5fcde375616c dirty=yes | spark-4e77 | 1792595341 | `recipe::analyse(/home/nvidia/models-m2/qwen3-8b-exl3-4.0bpw-bf16plain)` |
| `plan` | head=13f2eed code-tree=5fcde375616c dirty=yes | spark-4e77 | 1792595341 | `/home/nvidia/intarsia-decfuse/m2-exl3-e2e/target/acc/release/intarsia plan show --model /home/nvidia/models-m2/qwen3-8b-exl3-4.0bpw-bf16plain --backend cuda --device 0 --admit --json` |
| `kernels` | head=13f2eed code-tree=5fcde375616c dirty=yes | spark-4e77 | 1792595360 | `/home/nvidia/intarsia-decfuse/m2-exl3-e2e/target/acc/release/intarsia kernels test --backend cuda --device 0 --model /home/nvidia/models-m2/qwen3-8b-exl3-4.0bpw-bf16plain` |
| `gate` | head=13f2eed code-tree=5fcde375616c dirty=yes | spark-4e77 | 1792595423 | `/home/nvidia/intarsia-decfuse/m2-exl3-e2e/target/acc/release/intarsia accuracy --model /home/nvidia/models-m2/qwen3-8b-exl3-4.0bpw-bf16plain --backend cuda --device 0 --quick --max-seq 2048` |
| `acc-short` | head=13f2eed code-tree=5fcde375616c dirty=yes | spark-4e77 | 1792595438 | `/home/nvidia/intarsia-decfuse/m2-exl3-e2e/target/acc/release/intarsia accuracy --model /home/nvidia/models-m2/qwen3-8b-exl3-4.0bpw-bf16plain --backend cuda --device 0 --golden /home/nvidia/intarsiaml-golden/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq 4096 --write-baseline /` |
| `bench` | head=13f2eed code-tree=5fcde375616c dirty=yes | spark-4e77 | 1792595452 | `/home/nvidia/intarsia-decfuse/m2-exl3-e2e/target/acc/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-qwen3-8b-exl3-4.0bpw-bf16plain --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-qwen3-8b-exl3-4.0bpw-bf16plain-cuda-bench --out-dir /home/nv` |

Model fingerprint: `fnv1a64:4640ad0b409e6c43`.


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/kernels.out` (kernels)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `bench/qualify-qwen3-8b-exl3-4.0bpw-bf16plain-cuda-bench.json` (bench)
- `logs/bench-bench.out` (bench)
- `logs/bench-server.err` (bench)
- `state.json` (stage records; `--resume` reads it)
