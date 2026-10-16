# Qualification quick report: Qwen3-8B on metal

_Quick section, generated unix 1791983699, intarsia 0.0.1 build `unstamped`._

## Headline

**Verdict: PASS** (7 stages: 7 pass, 0 fail, 0 error, 0 skipped)

| | |
|---|---|
| Model | Qwen3-8B (`/Users/josef/Models/Qwen3-8B`) |
| Backend / device | metal / Apple M5 Max (applegpu_g17s) |
| Recipe | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| Decode | 128 tok: **33.3** tok/s (82% of floor); 1024 tok: **32.1** tok/s (79% of floor); 4096 tok: **30.3** tok/s (75% of floor) |
| Prefill | 128 tok: **311** tok/s; 1024 tok: **326** tok/s; 4096 tok: **213** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 100.00% KL 2.61e-4; decode top-1 100.00% KL 3.64e-4) |
| Quick gate | **PASS**: perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 4m11s |

### Failures

None.

### Skipped

None.

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | Apple M5 Max (applegpu_g17s), 16 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 34s | 14 ops resolved, exactness deterministic, loaded in 33.7s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 22s | 16 kernels, 181 cases in 21.9s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 16s | perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-short` short accuracy against a golden | quick | PASS | 11s | prefill top-1 100.00% KL 2.61e-4; decode top-1 100.00% KL 3.64e-4 |
| 7 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 2m48s | decode 33.3 tok/s at 128 tokens (82% of the byte floor) |

### Notes

- `bench`: bench warning: size 128: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 1024: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 4096: 0 graph replays during the measured reps (decode ran eagerly)

## Speed

**quick bench** (floor: 15.14 GB/token at 614 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: --bandwidth-gbs)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 125 | 0.402 | 311 | 33.31 | 0.10 | 40.6 | 82.1 |
| 1042 | 3.200 | 326 | 32.06 | 0.42 | 40.6 | 79.0 |
| 4311 | 20.022 | 213 | 30.34 | 0.22 | 40.6 | 74.8 |

## Accuracy

**acc-short** against `golden/qwen3-8b-fp32.safetensors` (found by model name; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 434 | 100.00 | 98.85 | 2.61e-4 | 1.98e-4 | 6.69e-3 | 4.11e-4 | 0 | 100.0 | pass |
| decode | 128 | 100.00 | 98.59 | 3.64e-4 | 3.33e-4 | 8.35e-3 | 3.83e-4 | 0 | 100.0 | pass |

**Quick gate**: perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) (PASS)
- plan: 14 ops resolved, exactness deterministic, loaded in 33.7s (PASS)

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
| attn_prefill | attn_prefill_bf16 | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16 | deterministic | yes | 36 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 36 |
| argmax | argmax_f32 | exact | yes | 1 |

- kernels test: 16 kernels, 181 cases in 21.9s (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | unstamped | endor.kuepker.haus | 1791983447 | `/opt/code/intarsiaml/target/release/intarsia doctor --backend metal --device 0 --out docs/results/raw/2026-10-14-metal-phase-a/qualify/doctor.json` |
| `recipe` | unstamped | endor.kuepker.haus | 1791983448 | `recipe::analyse(/Users/josef/Models/Qwen3-8B)` |
| `plan` | unstamped | endor.kuepker.haus | 1791983448 | `/opt/code/intarsiaml/target/release/intarsia plan show --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --admit --json` |
| `kernels` | unstamped | endor.kuepker.haus | 1791983481 | `/opt/code/intarsiaml/target/release/intarsia kernels test --backend metal --device 0 --model /Users/josef/Models/Qwen3-8B` |
| `gate` | unstamped | endor.kuepker.haus | 1791983503 | `/opt/code/intarsiaml/target/release/intarsia accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --quick --max-seq 2048` |
| `acc-short` | unstamped | endor.kuepker.haus | 1791983520 | `/opt/code/intarsiaml/target/release/intarsia accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --golden golden/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq 4096 --write-baseline docs/results/raw/2026-10-14-metal-phase-a/qualify/baselines/acc-short.base` |
| `bench` | unstamped | endor.kuepker.haus | 1791983531 | `/opt/code/intarsiaml/target/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3-8B --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-Qwen3-8B-metal-bench --out-dir docs/results/raw/2026-10-14-metal-phase-a/qualify/bench --tp 1 --fingerprint` |

Model fingerprint: `fnv1a64:e6a70935b68a5561`.

Build stamp `unstamped` means a plain `cargo build`: baselines written here gate this binary, but `--pre-change` provenance checks refuse them as a "before".


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/kernels.out` (kernels)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `bench/qualify-Qwen3-8B-metal-bench.json` (bench)
- `logs/bench-bench.out` (bench)
- `logs/bench-server.err` (bench)
- `state.json` (stage records; `--resume` reads it)
