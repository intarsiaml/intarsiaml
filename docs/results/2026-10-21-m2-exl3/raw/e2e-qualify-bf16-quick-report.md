# Qualification quick report: Qwen3-8B on cuda

_Quick section, generated unix 1792596814, intarsia 0.0.1 build `head=13f2eed code-tree=e25c44f85845 dirty=yes`._

## Headline

**Verdict: PASS** (7 stages: 7 pass, 0 fail, 0 error, 0 skipped)

| | |
|---|---|
| Model | Qwen3-8B (`/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218`) |
| Backend / device | cuda / NVIDIA GB10 (sm_121) |
| Recipe | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| Decode | 128 tok: **15.6** tok/s (107% of floor); 1024 tok: **15.5** tok/s (106% of floor); 4096 tok: **15.1** tok/s (103% of floor) |
| Prefill | 128 tok: **1320** tok/s; 1024 tok: **2349** tok/s; 4096 tok: **2270** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 100.00% KL 2.90e-4; decode top-1 100.00% KL 3.24e-4) |
| Quick gate | **PASS**: perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 7m13s |

### Failures

None.

### Skipped

None.

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 1s | NVIDIA GB10 (sm_121), 52 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 1m47s | 16 ops resolved, exactness fast, loaded in 107.2s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 65s | 52 kernels, 964 cases in 64.3s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 54s | perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-short` short accuracy against a golden | quick | PASS | 43s | prefill top-1 100.00% KL 2.90e-4; decode top-1 100.00% KL 3.24e-4 |
| 7 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 2m43s | decode 15.6 tok/s at 128 tokens (107% of the byte floor) |

### Notes

- `bench`: size 128: 107% of the byte floor is above 100%: the --gb-per-token estimate is too high or the run is not a plain decode (check the floor inputs)
- `bench`: size 1024: 106% of the byte floor is above 100%: the --gb-per-token estimate is too high or the run is not a plain decode (check the floor inputs)
- `bench`: size 4096: 103% of the byte floor is above 100%: the --gb-per-token estimate is too high or the run is not a plain decode (check the floor inputs)

## Speed

**quick bench** (floor: 15.14 GB/token at 221 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 221 GB/s, the GB10 measured figure)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 124 | 0.094 | 1320 | 15.64 | 0.01 | 14.6 | 107.1 |
| 1010 | 0.430 | 2349 | 15.48 | 0.01 | 14.6 | 106.0 |
| 4221 | 1.863 | 2270 | 15.09 | 0.01 | 14.6 | 103.3 |

## Accuracy

**acc-short** against `/home/nvidia/intarsiaml-golden/qwen3-8b-fp32.safetensors` (given with --golden; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 434 | 100.00 | 98.89 | 2.90e-4 | 2.28e-4 | 1.48e-2 | 4.75e-4 | 0 | 100.0 | pass |
| decode | 128 | 100.00 | 99.22 | 3.24e-4 | 2.83e-4 | 5.85e-3 | 4.11e-4 | 0 | 100.0 | pass |

**Quick gate**: perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) (PASS)
- plan: 16 ops resolved, exactness fast, loaded in 107.2s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_bf16 | exact | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 1 |
| rmsnorm_add | rmsnorm_add_bf16 | deterministic | yes | 72 |
| qk_norm | qk_norm_bf16 | deterministic | yes | 72 |
| rope | rope_bf16 | deterministic | yes | 72 |
| matmul | matmul_bf16 | fast | yes | 0 |
| matmul | matmul_bf16_gemv | deterministic | yes | 108 |
| logits | logits_bf16 | fast | yes | 0 |
| logits | logits_bf16_gemv | deterministic | yes | 1 |
| kv_append | kv_append_bf16 | exact | yes | 72 |
| attn_prefill | attn_prefill_bf16_flash | deterministic | yes | 0 |
| attn_prefill | attn_prefill_bf16_splitkv | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 36 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 36 |
| argmax | argmax_f32 | exact | yes | 1 |
| matmul_pair | matmul_pair_bf16_gemv | deterministic | yes | 72 |

- kernels test: 52 kernels, 964 cases in 64.3s (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | head=13f2eed code-tree=e25c44f85845 dirty=yes | spark-4e77 | 1792596380 | `/home/nvidia/intarsia-decfuse/m2-exl3-e2e/target/acc/release/intarsia doctor --backend cuda --device 0 --out /home/nvidia/intarsia-decfuse/m2-exl3-e2e/reports-m2e2e/bf16/doctor.json` |
| `recipe` | head=13f2eed code-tree=e25c44f85845 dirty=yes | spark-4e77 | 1792596381 | `recipe::analyse(/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218)` |
| `plan` | head=13f2eed code-tree=e25c44f85845 dirty=yes | spark-4e77 | 1792596381 | `/home/nvidia/intarsia-decfuse/m2-exl3-e2e/target/acc/release/intarsia plan show --model /home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 --backend cuda --device 0 --admit --json` |
| `kernels` | head=13f2eed code-tree=e25c44f85845 dirty=yes | spark-4e77 | 1792596488 | `/home/nvidia/intarsia-decfuse/m2-exl3-e2e/target/acc/release/intarsia kernels test --backend cuda --device 0 --model /home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218` |
| `gate` | head=13f2eed code-tree=e25c44f85845 dirty=yes | spark-4e77 | 1792596553 | `/home/nvidia/intarsia-decfuse/m2-exl3-e2e/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 --backend cuda --device 0 --quick --max-seq 2048` |
| `acc-short` | head=13f2eed code-tree=e25c44f85845 dirty=yes | spark-4e77 | 1792596607 | `/home/nvidia/intarsia-decfuse/m2-exl3-e2e/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 --backend cuda --device 0 --golden /home/nvidia/intarsiaml-golden/qwen3-8b-fp32.safetensors --mode both` |
| `bench` | head=13f2eed code-tree=e25c44f85845 dirty=yes | spark-4e77 | 1792596651 | `/home/nvidia/intarsia-decfuse/m2-exl3-e2e/target/acc/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3-8B --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-Qwen3-8B-cuda-bench --out-dir /home/nvidia/intarsia-decfuse/m2-exl3-e2e/reports-m2` |

Model fingerprint: `fnv1a64:1e5ce918d9f0f30f`.


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/kernels.out` (kernels)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `bench/qualify-Qwen3-8B-cuda-bench.json` (bench)
- `logs/bench-bench.out` (bench)
- `logs/bench-server.err` (bench)
- `state.json` (stage records; `--resume` reads it)
