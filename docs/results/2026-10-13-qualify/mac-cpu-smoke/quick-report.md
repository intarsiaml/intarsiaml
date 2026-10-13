# Qualification quick report: Qwen3-Reranker-0.6B on cpu

_Quick section, generated unix 1791912845, intarsia 0.0.1 build `head=3d06ce8 code-tree=f83e8dda827f dirty=no`._

## Headline

**Verdict: FAIL** (7 stages: 4 pass, 2 fail, 0 error, 1 skipped)

| | |
|---|---|
| Model | Qwen3-Reranker-0.6B (`/Users/josef/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473`) |
| Backend / device | cpu / cpu-ref (Apple M5 Max) (any) |
| Recipe | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| Decode | 64 tok: **18.4** tok/s (35% of floor); 256 tok: **20.4** tok/s (39% of floor) |
| Prefill | 64 tok: **235** tok/s; 256 tok: **334** tok/s |
| Accuracy (short golden) | **FAIL** (prefill top-1 99.08% KL 5.50e-4; decode top-1 100.00% KL 1.09e-3) |
| Quick gate | **FAIL**: sanity FAIL: 48 tokens, max 4-gram repeat 43: non-empty, answers the fact ("paris"): NO |
| Wall time | quick 45s |

### Failures

- `gate` **FAIL**: sanity FAIL: 48 tokens, max 4-gram repeat 43: non-empty, answers the fact ("paris"): NO
- `acc-short` **FAIL**: prefill: top-1 99.08% (need >= 99.5); decode: full-vocab KL 1.089e-3 (need <= 1.0e-3)

### Skipped

- `kernels`: cpu-ref is the oracle every other backend is admitted against; admission has nothing to compare it with

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | cpu-ref (Apple M5 Max) (any), 46 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 5s | 12 ops resolved, exactness deterministic, loaded in 5.4s |
| 4 | `kernels` kernels test on the model's shapes | quick | SKIP | 0s | cpu-ref is the oracle every other backend is admitted against; admission has nothing to compare it with |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | FAIL | 18s | sanity FAIL: 48 tokens, max 4-gram repeat 43: non-empty, answers the fact ("paris"): NO |
| 6 | `acc-short` short accuracy against a golden | quick | FAIL | 7s | prefill: top-1 99.08% (need >= 99.5); decode: full-vocab KL 1.089e-3 (need <= 1.0e-3) |
| 7 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 14s | decode 18.4 tok/s at 64 tokens (35% of the byte floor) |

### Notes

- `bench`: bench warning: size 64: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 256: 0 graph replays during the measured reps (decode ran eagerly)

## Speed

**quick bench** (floor: 1.19 GB/token at 63 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: doctor's measured device copy (read+write bytes counted))

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 64 | 0.272 | 235 | 18.36 | - | 52.9 | 34.7 |
| 240 | 0.719 | 334 | 20.44 | - | 52.9 | 38.7 |

## Accuracy

**acc-short** against `golden/qwen3-reranker-0.6b-fp32.safetensors` (found by model name; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 434 | 99.08 | 98.29 | 5.50e-4 | 4.08e-4 | 3.58e-2 | 1.60e-3 | 0 | 78.3 | FAIL |
| decode | 32 | 100.00 | 97.50 | 1.09e-3 | 7.45e-4 | 6.57e-3 | 2.37e-3 | 0 | 100.0 | FAIL |

**Quick gate**: sanity FAIL: 48 tokens, max 4-gram repeat 43: non-empty, answers the fact ("paris"): NO

## Plan and recipe

- recipe: builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) (PASS)
- plan: 12 ops resolved, exactness deterministic, loaded in 5.4s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | cpu_f32_embedding | exact | yes | 1 |
| rmsnorm | cpu_f32_rmsnorm | deterministic | yes | 1 |
| rmsnorm_add | cpu_f32_rmsnorm_add | deterministic | yes | 56 |
| qk_norm | cpu_f32_qk_norm | deterministic | yes | 56 |
| rope | cpu_f32_rope | deterministic | yes | 56 |
| matmul | cpu_f32_matmul | deterministic | yes | 196 |
| logits | cpu_f32_logits | deterministic | yes | 1 |
| kv_append | cpu_f32_kv_append | exact | yes | 56 |
| attn_prefill | cpu_f32_attn_prefill | deterministic | yes | 0 |
| attn_decode | cpu_f32_attn_decode | deterministic | yes | 28 |
| silu_glu | cpu_f32_silu_glu | deterministic | yes | 28 |
| argmax | cpu_f32_argmax | exact | yes | 1 |

- kernels test: cpu-ref is the oracle every other backend is admitted against; admission has nothing to compare it with (SKIP)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | head=3d06ce8 code-tree=f83e8dda827f dirty=no | endor.kuepker.haus | 1791912800 | `/opt/code/intarsiaml/target/release/intarsia doctor --backend cpu --device 0 --out docs/results/2026-10-13-qualify/mac-cpu-smoke/doctor.json` |
| `recipe` | head=3d06ce8 code-tree=f83e8dda827f dirty=no | endor.kuepker.haus | 1791912800 | `recipe::analyse(/Users/josef/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473)` |
| `plan` | head=3d06ce8 code-tree=f83e8dda827f dirty=no | endor.kuepker.haus | 1791912800 | `/opt/code/intarsiaml/target/release/intarsia plan show --model /Users/josef/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473 --backend cpu --device 0 --chat-template-file docs/results/2026-10-13-qualify/chatml.jinja --admit --json` |
| `kernels` | head=3d06ce8 code-tree=f83e8dda827f dirty=no | endor.kuepker.haus | 1791912806 | `-` |
| `gate` | head=3d06ce8 code-tree=f83e8dda827f dirty=no | endor.kuepker.haus | 1791912806 | `/opt/code/intarsiaml/target/release/intarsia accuracy --model /Users/josef/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473 --backend cpu --device 0 --chat-template-file docs/results/2026-10-13-qualify/chatml.jinja --quick --max-seq 2048` |
| `acc-short` | head=3d06ce8 code-tree=f83e8dda827f dirty=no | endor.kuepker.haus | 1791912823 | `/opt/code/intarsiaml/target/release/intarsia accuracy --model /Users/josef/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473 --backend cpu --device 0 --chat-template-file docs/results/2026-10-13-qualify/chatml.jinja --golden golden/qwen3-rera` |
| `bench` | head=3d06ce8 code-tree=f83e8dda827f dirty=no | endor.kuepker.haus | 1791912830 | `/opt/code/intarsiaml/target/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3-Reranker-0.6B --sizes 64,256 --gen 16 --reps 1 --warmup 1 --thinking off --label qualify-Qwen3-Reranker-0.6B-cpu-bench --out-dir docs/results/2026-10-13-qualify/mac-cpu-smoke/bench --tp 1 --fin` |

Model fingerprint: `fnv1a64:19caa4b9743b467d`.


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `bench/qualify-Qwen3-Reranker-0.6B-cpu-bench.json` (bench)
- `logs/bench-bench.out` (bench)
- `logs/bench-server.err` (bench)
- `state.json` (stage records; `--resume` reads it)
