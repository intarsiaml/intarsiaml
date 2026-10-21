# Qualification quick report: qwen3-8b-exl3-4.0bpw-bf16plain on cuda

_Quick section, generated unix 1792603582, intarsia 0.0.1 build `head=2a5298e code-tree=93f7717e89f5 dirty=no`._

## Headline

**Verdict: PASS** (7 stages: 7 pass, 0 fail, 0 error, 0 skipped)

| | |
|---|---|
| Model | qwen3-8b-exl3-4.0bpw-bf16plain (`/home/nvidia/models-m2/qwen3-8b-exl3-4.0bpw-bf16plain`) |
| Backend / device | cuda / NVIDIA GB10 (sm_121) |
| Recipe | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant exl3) |
| Decode | 128 tok: **47.3** tok/s (111% of floor); 1024 tok: **45.8** tok/s (107% of floor); 4096 tok: **42.1** tok/s (99% of floor) |
| Prefill | 128 tok: **2851** tok/s; 1024 tok: **3353** tok/s; 4096 tok: **3296** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 95.16% KL 2.32e-2; decode top-1 94.53% KL 3.26e-2) |
| Quick gate | **PASS**: perplexity 2.01, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 2m42s |

### Failures

None.

### Skipped

None.

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 1s | NVIDIA GB10 (sm_121), 54 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant exl3) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 22s | 13 ops resolved, exactness deterministic, loaded in 21.6s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 64s | 54 kernels, 1069 cases in 63.9s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 15s | perplexity 2.01, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-short` short accuracy against a golden | quick | PASS | 14s | prefill top-1 95.16% KL 2.32e-2; decode top-1 94.53% KL 3.26e-2 |
| 7 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 46s | decode 47.3 tok/s at 128 tokens (111% of the byte floor) |

### Notes

- `bench`: size 128: 111% of the byte floor is above 100%: the --gb-per-token estimate is too high or the run is not a plain decode (check the floor inputs)
- `bench`: size 1024: 107% of the byte floor is above 100%: the --gb-per-token estimate is too high or the run is not a plain decode (check the floor inputs)

## Speed

**quick bench** (floor: 5.19 GB/token at 221 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 221 GB/s, the GB10 measured figure)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 124 | 0.043 | 2851 | 47.26 | 0.02 | 42.6 | 111.0 |
| 1038 | 0.310 | 3353 | 45.75 | 0.07 | 42.6 | 107.4 |
| 4243 | 1.289 | 3296 | 42.15 | 0.05 | 42.6 | 99.0 |

## Accuracy

**acc-short** against `/home/nvidia/intarsiaml-golden/qwen3-8b-fp32.safetensors` (given with --golden; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 434 | 95.16 | 92.40 | 2.32e-2 | 2.00e-2 | 9.94e-1 | 2.69e-3 | 0 | 50.2 | pass |
| decode | 128 | 94.53 | 90.94 | 3.26e-2 | 2.73e-2 | 5.69e-1 | 3.71e-3 | 0 | 69.5 | pass |

**Quick gate**: perplexity 2.01, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3 covers this model (25/25 tensor templates resolve, quant exl3) (PASS)
- plan: 13 ops resolved, exactness deterministic, loaded in 21.6s (PASS)

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

- kernels test: 54 kernels, 1069 cases in 63.9s (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | head=2a5298e code-tree=93f7717e89f5 dirty=no | spark-4e77 | 1792603420 | `/home/nvidia/intarsia-decfuse/m2-exl3-pr/target/acc/release/intarsia doctor --backend cuda --device 0 --out /home/nvidia/m2-qualify-pr/exl3/doctor.json` |
| `recipe` | head=2a5298e code-tree=93f7717e89f5 dirty=no | spark-4e77 | 1792603421 | `recipe::analyse(/home/nvidia/models-m2/qwen3-8b-exl3-4.0bpw-bf16plain)` |
| `plan` | head=2a5298e code-tree=93f7717e89f5 dirty=no | spark-4e77 | 1792603421 | `/home/nvidia/intarsia-decfuse/m2-exl3-pr/target/acc/release/intarsia plan show --model /home/nvidia/models-m2/qwen3-8b-exl3-4.0bpw-bf16plain --backend cuda --device 0 --admit --json` |
| `kernels` | head=2a5298e code-tree=93f7717e89f5 dirty=no | spark-4e77 | 1792603443 | `/home/nvidia/intarsia-decfuse/m2-exl3-pr/target/acc/release/intarsia kernels test --backend cuda --device 0 --model /home/nvidia/models-m2/qwen3-8b-exl3-4.0bpw-bf16plain` |
| `gate` | head=2a5298e code-tree=93f7717e89f5 dirty=no | spark-4e77 | 1792603507 | `/home/nvidia/intarsia-decfuse/m2-exl3-pr/target/acc/release/intarsia accuracy --model /home/nvidia/models-m2/qwen3-8b-exl3-4.0bpw-bf16plain --backend cuda --device 0 --quick --max-seq 2048` |
| `acc-short` | head=2a5298e code-tree=93f7717e89f5 dirty=no | spark-4e77 | 1792603522 | `/home/nvidia/intarsia-decfuse/m2-exl3-pr/target/acc/release/intarsia accuracy --model /home/nvidia/models-m2/qwen3-8b-exl3-4.0bpw-bf16plain --backend cuda --device 0 --golden /home/nvidia/intarsiaml-golden/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq 4096 --write-baseline /h` |
| `bench` | head=2a5298e code-tree=93f7717e89f5 dirty=no | spark-4e77 | 1792603537 | `/home/nvidia/intarsia-decfuse/m2-exl3-pr/target/acc/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-qwen3-8b-exl3-4.0bpw-bf16plain --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-qwen3-8b-exl3-4.0bpw-bf16plain-cuda-bench --out-dir /home/nvi` |

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
