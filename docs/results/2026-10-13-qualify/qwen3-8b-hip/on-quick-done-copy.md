# Qualification quick report: Qwen3-8B on hip

_Quick section, generated unix 1791912552, intarsia 0.0.1 build `head=3d06ce8 code-tree=f83e8dda827f dirty=no`._

## Headline

**Verdict: PASS** (7 stages: 7 pass, 0 fail, 0 error, 0 skipped)

| | |
|---|---|
| Model | Qwen3-8B (`/models/Qwen3-8B`) |
| Backend / device | hip / AMD Radeon Graphics (gfx1201) |
| Recipe | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| Decode | 128 tok: **37.7** tok/s (91% of floor); 1024 tok: **37.1** tok/s (90% of floor); 4096 tok: **35.9** tok/s (87% of floor) |
| Prefill | 128 tok: **2033** tok/s; 1024 tok: **3473** tok/s; 4096 tok: **3009** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 100.00% KL 2.34e-4; decode top-1 100.00% KL 3.63e-4) |
| Quick gate | **PASS**: perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 3m08s |

### Failures

None.

### Skipped

None.

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | AMD Radeon Graphics (gfx1201), 30 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 50s | 15 ops resolved, exactness deterministic, loaded in 49.8s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 41s | 30 kernels, 479 cases in 41.3s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 9s | perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-short` short accuracy against a golden | quick | PASS | 8s | prefill top-1 100.00% KL 2.34e-4; decode top-1 100.00% KL 3.63e-4 |
| 7 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 80s | decode 37.7 tok/s at 128 tokens (91% of the byte floor) |

### Notes

- `bench`: bench warning: size 128: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 1024: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 4096: 0 graph replays during the measured reps (decode ran eagerly)

## Speed

**quick bench** (floor: 15.14 GB/token at 627 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 627 GB/s, the R9700 figure of docs/results/2026-10-13-phase2a-hip.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 124 | 0.061 | 2033 | 37.74 | 0.01 | 41.4 | 91.1 |
| 1015 | 0.292 | 3473 | 37.11 | 0.01 | 41.4 | 89.6 |
| 4163 | 1.384 | 3009 | 35.91 | 0.02 | 41.4 | 86.7 |

## Accuracy

**acc-short** against `/models/golden/qwen3-8b-fp32.safetensors` (given with --golden; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 434 | 100.00 | 98.80 | 2.34e-4 | 1.70e-4 | 6.16e-3 | 4.54e-4 | 0 | 100.0 | pass |
| decode | 128 | 100.00 | 98.12 | 3.63e-4 | 3.17e-4 | 8.26e-3 | 3.13e-4 | 0 | 100.0 | pass |

**Quick gate**: perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) (PASS)
- plan: 15 ops resolved, exactness deterministic, loaded in 49.8s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_bf16 | exact | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 1 |
| rmsnorm_add | rmsnorm_add_bf16 | deterministic | yes | 72 |
| qk_norm | qk_norm_bf16 | deterministic | yes | 72 |
| rope | rope_bf16 | deterministic | yes | 72 |
| matmul | matmul_bf16 | deterministic | yes | 0 |
| matmul | matmul_bf16_gemv | deterministic | yes | 252 |
| logits | logits_bf16 | deterministic | yes | 0 |
| logits | logits_bf16_gemv | deterministic | yes | 1 |
| kv_append | kv_append_bf16 | exact | yes | 72 |
| attn_prefill | attn_prefill_bf16_flash | deterministic | yes | 0 |
| attn_prefill | attn_prefill_bf16_splitkv | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 36 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 36 |
| argmax | argmax_f32 | exact | yes | 1 |

- kernels test: 30 kernels, 479 cases in 41.3s (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | head=3d06ce8 code-tree=f83e8dda827f dirty=no | tatooine | 1791912364 | `/target/release/intarsia doctor --backend hip --device 0 --out /work/reports/qwen3-8b/run3/doctor.json` |
| `recipe` | head=3d06ce8 code-tree=f83e8dda827f dirty=no | tatooine | 1791912364 | `recipe::analyse(/models/Qwen3-8B)` |
| `plan` | head=3d06ce8 code-tree=f83e8dda827f dirty=no | tatooine | 1791912364 | `/target/release/intarsia plan show --model /models/Qwen3-8B --backend hip --device 0 --mem-budget 28 --admit --json` |
| `kernels` | head=3d06ce8 code-tree=f83e8dda827f dirty=no | tatooine | 1791912414 | `/target/release/intarsia kernels test --backend hip --device 0 --model /models/Qwen3-8B` |
| `gate` | head=3d06ce8 code-tree=f83e8dda827f dirty=no | tatooine | 1791912455 | `/target/release/intarsia accuracy --model /models/Qwen3-8B --backend hip --device 0 --mem-budget 28 --quick --max-seq 2048` |
| `acc-short` | head=3d06ce8 code-tree=f83e8dda827f dirty=no | tatooine | 1791912465 | `/target/release/intarsia accuracy --model /models/Qwen3-8B --backend hip --device 0 --mem-budget 28 --golden /models/golden/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq 4096 --write-baseline /work/reports/qwen3-8b/run3/baselines/acc-short.baseline.json` |
| `bench` | head=3d06ce8 code-tree=f83e8dda827f dirty=no | tatooine | 1791912473 | `/target/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3-8B --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-Qwen3-8B-hip-bench --out-dir /work/reports/qwen3-8b/run3/bench --tp 1 --fingerprint model=Qwen3-8B --fingerprint backend=hip --` |

Model fingerprint: `fnv1a64:e6a70935b68a5561`.


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/kernels.out` (kernels)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `bench/qualify-Qwen3-8B-hip-bench.json` (bench)
- `logs/bench-bench.out` (bench)
- `logs/bench-server.err` (bench)
- `state.json` (stage records; `--resume` reads it)
