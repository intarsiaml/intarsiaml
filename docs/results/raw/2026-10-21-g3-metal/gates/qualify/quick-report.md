# Qualification quick report: Qwen3-8B on metal

_Quick section, generated unix 1792596672, intarsia 0.0.1 build `unstamped`._

## Headline

**Verdict: PASS** (7 stages: 7 pass, 0 fail, 0 error, 0 skipped)

| | |
|---|---|
| Model | Qwen3-8B (`/Users/josef/Models/Qwen3-8B`) |
| Backend / device | metal / Apple M5 Max (applegpu_g17s) |
| Recipe | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| Decode | 128 tok: **36.2** tok/s (89% of floor); 1024 tok: **36.0** tok/s (89% of floor); 4096 tok: **34.3** tok/s (84% of floor) |
| Prefill | 128 tok: **2479** tok/s; 1024 tok: **4024** tok/s; 4096 tok: **3226** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 100.00% KL 2.83e-4; decode top-1 100.00% KL 3.31e-4) |
| Quick gate | **PASS**: perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 3m05s |

### Failures

None.

### Skipped

None.

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 1s | Apple M5 Max (applegpu_g17s), 27 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 44s | 14 ops resolved, exactness deterministic, loaded in 43.8s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 43s | 27 kernels, 321 cases in 43.2s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 10s | perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-short` short accuracy against a golden | quick | PASS | 7s | prefill top-1 100.00% KL 2.83e-4; decode top-1 100.00% KL 3.31e-4 |
| 7 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 80s | decode 36.2 tok/s at 128 tokens (89% of the byte floor) |

### Notes

- `bench`: bench warning: size 128: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 1024: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 4096: 0 graph replays during the measured reps (decode ran eagerly)

## Speed

**quick bench** (floor: 15.14 GB/token at 614 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 614 GB/s, the Apple M5 Max (40-core GPU) spec figure of docs/decisions/0009-native-metal-backend.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 128 | 0.052 | 2479 | 36.25 | 0.43 | 40.6 | 89.4 |
| 1010 | 0.251 | 4024 | 35.96 | 0.18 | 40.6 | 88.7 |
| 4185 | 1.294 | 3226 | 34.28 | 0.17 | 40.6 | 84.5 |

## Accuracy

**acc-short** against `/opt/code/intarsiaml/golden/qwen3-8b-fp32.safetensors` (found by model name; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 434 | 100.00 | 98.89 | 2.83e-4 | 2.20e-4 | 1.51e-2 | 4.46e-4 | 0 | 100.0 | pass |
| decode | 128 | 100.00 | 98.59 | 3.31e-4 | 2.98e-4 | 7.94e-3 | 3.07e-4 | 0 | 100.0 | pass |

**Quick gate**: perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) (PASS)
- plan: 14 ops resolved, exactness deterministic, loaded in 43.8s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_bf16 | exact | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 1 |
| rmsnorm_add | rmsnorm_add_bf16 | deterministic | yes | 72 |
| matmul | matmul_bf16_gemv | deterministic | yes | 108 |
| matmul | matmul_bf16_nax | deterministic | yes | 0 |
| logits | logits_bf16 | deterministic | yes | 0 |
| logits | logits_bf16_gemv | deterministic | yes | 1 |
| attn_prefill | attn_prefill_bf16_flash_nax | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 36 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 36 |
| argmax | argmax_f32 | exact | yes | 1 |
| matmul_pair | matmul_pair_bf16_gemv | deterministic | yes | 72 |
| matmul_pair | matmul_pair_bf16_nax | deterministic | yes | 0 |
| qk_norm_rope_kv | qk_norm_rope_kv_bf16 | deterministic | yes | 36 |

- kernels test: 27 kernels, 321 cases in 43.2s (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | unstamped | endor.kuepker.haus | 1792596487 | `/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/intarsia-dev doctor --backend metal --device 0 --out /opt/code/intarsiaml/.claude/worktrees/agent-a34836167f72ea687/docs/results/raw/2026-10-21-g3-metal/gates/qualify/doctor.json` |
| `recipe` | unstamped | endor.kuepker.haus | 1792596487 | `recipe::analyse(/Users/josef/Models/Qwen3-8B)` |
| `plan` | unstamped | endor.kuepker.haus | 1792596487 | `/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/intarsia-dev plan show --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --admit --json` |
| `kernels` | unstamped | endor.kuepker.haus | 1792596531 | `/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/intarsia-dev kernels test --backend metal --device 0 --model /Users/josef/Models/Qwen3-8B` |
| `gate` | unstamped | endor.kuepker.haus | 1792596575 | `/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/intarsia-dev accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --quick --max-seq 2048` |
| `acc-short` | unstamped | endor.kuepker.haus | 1792596585 | `/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/intarsia-dev accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --golden /opt/code/intarsiaml/golden/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq 4096 --write-b` |
| `bench` | unstamped | endor.kuepker.haus | 1792596592 | `/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/intarsia-dev bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3-8B --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-Qwen3-8B-metal-bench --out-dir /opt/code/intar` |

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
