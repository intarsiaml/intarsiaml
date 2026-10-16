# Qualification report: Qwen3-8B on metal

_Full (quick + in-depth) section, generated unix 1792028603, intarsia 0.0.1 build `unstamped`._

## Headline

**Verdict: PASS** (12 stages: 10 pass, 0 fail, 0 error, 2 skipped)

| | |
|---|---|
| Model | Qwen3-8B (`/Users/josef/Models/Qwen3-8B`) |
| Backend / device | metal / Apple M5 Max (applegpu_g17s) |
| Recipe | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| Decode | 128 tok: **35.9** tok/s (89% of floor); 1024 tok: **35.9** tok/s (89% of floor); 4096 tok: **33.2** tok/s (82% of floor) |
| Prefill | 128 tok: **2150** tok/s; 1024 tok: **1495** tok/s; 4096 tok: **432** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 100.00% KL 3.17e-4; decode top-1 100.00% KL 3.74e-4) |
| Accuracy (long golden) | **PASS** (prefill top-1 99.79% KL 3.76e-4; decode top-1 99.60% KL 3.16e-4) |
| Quick gate | **PASS**: perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 3m11s, in-depth 9m28s, total 12m39s |

### Failures

None.

### Skipped

- `ab`: no --reference server given
- `tp`: --tp not given (or 1)

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | Apple M5 Max (applegpu_g17s), 21 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 31s | 15 ops resolved, exactness deterministic, loaded in 31.4s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 30s | 21 kernels, 251 cases in 30.1s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 11s | perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-short` short accuracy against a golden | quick | PASS | 7s | prefill top-1 100.00% KL 3.17e-4; decode top-1 100.00% KL 3.74e-4 |
| 7 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 1m51s | decode 35.9 tok/s at 128 tokens (89% of the byte floor) |
| 8 | `acc-long` accuracy --mode all on the long golden (prefill, decode, prefix reuse) | full | PASS | 36s | prefill top-1 99.79% KL 3.76e-4; decode top-1 99.60% KL 3.16e-4 |
| 9 | `envelope` noise envelope and baseline for regression gating | full | PASS | 15s | baseline and envelope (chunk) written; they gate this build: PASS |
| 10 | `bench-long` long-context bench up to 16k | full | PASS | 8m37s | decode 35.5 tok/s at 1024 tokens (88% of the byte floor) |
| 11 | `ab` A/B against the reference server via localeval | full | SKIP | 0s | no --reference server given |
| 12 | `tp` tensor-parallel variant | full | SKIP | 0s | --tp not given (or 1) |

### Notes

- `bench`: bench warning: size 128: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 1024: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 4096: 0 graph replays during the measured reps (decode ran eagerly)
- `bench-long`: bench warning: size 1024: 0 graph replays during the measured reps (decode ran eagerly)
- `bench-long`: bench warning: size 4096: 0 graph replays during the measured reps (decode ran eagerly)
- `bench-long`: bench warning: size 16384: 0 graph replays during the measured reps (decode ran eagerly)

## Speed

**quick bench** (floor: 15.14 GB/token at 614 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: --bandwidth-gbs)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 124 | 0.058 | 2150 | 35.92 | 0.12 | 40.6 | 88.6 |
| 1017 | 0.680 | 1495 | 35.94 | 0.09 | 40.6 | 88.6 |
| 4214 | 9.800 | 432 | 33.24 | 0.41 | 40.6 | 82.0 |

**long-context bench** (floor: 15.14 GB/token at 614 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: --bandwidth-gbs)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 993 | 0.766 | 1308 | 35.49 | 0.21 | 40.6 | 87.5 |
| 4085 | 9.198 | 444 | 34.69 | 0.31 | 40.6 | 85.5 |
| 16563 | 143.736 | 115 | 30.52 | 0.17 | 40.6 | 75.2 |

## Accuracy

**acc-short** against `/opt/code/intarsiaml/golden/qwen3-8b-fp32.safetensors` (found by model name; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 434 | 100.00 | 98.85 | 3.17e-4 | 2.53e-4 | 2.02e-2 | 4.17e-4 | 0 | 100.0 | pass |
| decode | 128 | 100.00 | 98.12 | 3.74e-4 | 3.34e-4 | 1.30e-2 | 3.58e-4 | 0 | 100.0 | pass |

**acc-long** against `/opt/code/intarsiaml/golden/qwen3-8b-long-fp32.safetensors` (found by model name; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1434 | 99.79 | 98.68 | 3.76e-4 | 3.43e-4 | 2.02e-2 | 3.29e-4 | 0 | 37.7 | pass |
| decode | 500 | 99.60 | 99.12 | 3.16e-4 | 2.95e-4 | 1.84e-2 | 2.10e-4 | 0 | 85.8 | pass |

**Regression gating artifacts**: baseline and envelope (chunk) written; they gate this build: PASS

`intarsia accuracy --model <dir> --backend metal --golden /opt/code/intarsiaml/golden/qwen3-8b-fp32.safetensors --baseline baselines/acc-short.baseline.json --noise-envelope baselines/envelope.json  (copy both files next to the golden, e.g. golden/baselines/)`

**Quick gate**: perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) (PASS)
- plan: 15 ops resolved, exactness deterministic, loaded in 31.4s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_bf16 | exact | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 1 |
| rmsnorm_add | rmsnorm_add_bf16 | deterministic | yes | 72 |
| matmul | matmul_bf16 | deterministic | yes | 0 |
| matmul | matmul_bf16_gemv | deterministic | yes | 108 |
| matmul | matmul_bf16_nax | deterministic | yes | 0 |
| logits | logits_bf16 | deterministic | yes | 0 |
| logits | logits_bf16_gemv | deterministic | yes | 1 |
| attn_prefill | attn_prefill_bf16 | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 36 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 36 |
| argmax | argmax_f32 | exact | yes | 1 |
| matmul_pair | matmul_pair_bf16_gemv | deterministic | yes | 72 |
| matmul_pair | matmul_pair_bf16_nax | deterministic | yes | 0 |
| qk_norm_rope_kv | qk_norm_rope_kv_bf16 | deterministic | yes | 36 |

- kernels test: 21 kernels, 251 cases in 30.1s (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | unstamped | endor.kuepker.haus | 1792027843 | `/opt/code/intarsiaml/.claude/worktrees/agent-a1247931a45f1aeb7/target/release/intarsia doctor --backend metal --device 0 --out /opt/code/intarsiaml/.claude/worktrees/agent-a1247931a45f1aeb7/docs/results/raw/2026-10-14-metal-prefill-p1/full/final/qualify-golden/doctor.json` |
| `recipe` | unstamped | endor.kuepker.haus | 1792027844 | `recipe::analyse(/Users/josef/Models/Qwen3-8B)` |
| `plan` | unstamped | endor.kuepker.haus | 1792027844 | `/opt/code/intarsiaml/.claude/worktrees/agent-a1247931a45f1aeb7/target/release/intarsia plan show --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --admit --json` |
| `kernels` | unstamped | endor.kuepker.haus | 1792027875 | `/opt/code/intarsiaml/.claude/worktrees/agent-a1247931a45f1aeb7/target/release/intarsia kernels test --backend metal --device 0 --model /Users/josef/Models/Qwen3-8B` |
| `gate` | unstamped | endor.kuepker.haus | 1792027905 | `/opt/code/intarsiaml/.claude/worktrees/agent-a1247931a45f1aeb7/target/release/intarsia accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --quick --max-seq 2048` |
| `acc-short` | unstamped | endor.kuepker.haus | 1792027916 | `/opt/code/intarsiaml/.claude/worktrees/agent-a1247931a45f1aeb7/target/release/intarsia accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --golden /opt/code/intarsiaml/golden/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq 4096 --write-baseline /opt/code/i` |
| `bench` | unstamped | endor.kuepker.haus | 1792027923 | `/opt/code/intarsiaml/.claude/worktrees/agent-a1247931a45f1aeb7/target/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3-8B --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-Qwen3-8B-metal-bench --out-dir /opt/code/intarsiaml/.claude/workt` |
| `acc-long` | unstamped | endor.kuepker.haus | 1792028034 | `/opt/code/intarsiaml/.claude/worktrees/agent-a1247931a45f1aeb7/target/release/intarsia accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --golden /opt/code/intarsiaml/golden/qwen3-8b-long-fp32.safetensors --mode all --decode-positions 64 --max-seq 8192 --write-baseline /opt/co` |
| `envelope` | unstamped | endor.kuepker.haus | 1792028070 | `/opt/code/intarsiaml/.claude/worktrees/agent-a1247931a45f1aeb7/target/release/intarsia accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --golden /opt/code/intarsiaml/golden/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq 4096 --baseline /opt/code/intarsi` |
| `bench-long` | unstamped | endor.kuepker.haus | 1792028085 | `/opt/code/intarsiaml/.claude/worktrees/agent-a1247931a45f1aeb7/target/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3-8B --sizes 1024,4096,16384 --gen 64 --reps 2 --warmup 1 --thinking off --label qualify-Qwen3-8B-metal-bench-long --out-dir /opt/code/intarsiaml/.claude` |
| `ab` | unstamped | endor.kuepker.haus | 1792028603 | `-` |
| `tp` | unstamped | endor.kuepker.haus | 1792028603 | `-` |

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
- `logs/acc-long.out` (acc-long)
- `baselines/acc-long.baseline.json` (acc-long)
- `baselines/acc-short.baseline.json` (envelope)
- `baselines/envelope.json` (envelope)
- `logs/baseline-selfcheck.out` (envelope)
- `bench/qualify-Qwen3-8B-metal-bench-long.json` (bench-long)
- `logs/bench-long-bench.out` (bench-long)
- `logs/bench-long-server.err` (bench-long)
- `state.json` (stage records; `--resume` reads it)
