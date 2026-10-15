# Qualification report: Qwen3-8B on metal

_Full (quick + in-depth) section, generated unix 1792041674, intarsia 0.0.1 build `unstamped`._

## Headline

**Verdict: PASS** (12 stages: 10 pass, 0 fail, 0 error, 2 skipped)

| | |
|---|---|
| Model | Qwen3-8B (`/Users/josef/Models/Qwen3-8B`) |
| Backend / device | metal / Apple M5 Max (applegpu_g17s) |
| Recipe | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| Decode | 128 tok: **36.7** tok/s (90% of floor); 1024 tok: **36.5** tok/s (90% of floor); 4096 tok: **35.1** tok/s (86% of floor) |
| Prefill | 128 tok: **2414** tok/s; 1024 tok: **3515** tok/s; 4096 tok: **2437** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 99.77% KL 2.69e-4; decode top-1 100.00% KL 3.74e-4) |
| Accuracy (long golden) | **PASS** (prefill top-1 99.65% KL 4.34e-4; decode top-1 99.80% KL 3.16e-4) |
| Quick gate | **PASS**: perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 2m40s, in-depth 2m19s, total 5m00s |

### Failures

None.

### Skipped

- `ab`: no --reference server given
- `tp`: --tp not given (or 1)

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | Apple M5 Max (applegpu_g17s), 22 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 36s | 15 ops resolved, exactness deterministic, loaded in 35.8s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 32s | 22 kernels, 265 cases in 31.7s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 12s | perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-short` short accuracy against a golden | quick | PASS | 7s | prefill top-1 99.77% KL 2.69e-4; decode top-1 100.00% KL 3.74e-4 |
| 7 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 74s | decode 36.7 tok/s at 128 tokens (90% of the byte floor) |
| 8 | `acc-long` accuracy --mode all on the long golden (prefill, decode, prefix reuse) | full | PASS | 33s | prefill top-1 99.65% KL 4.34e-4; decode top-1 99.80% KL 3.16e-4 |
| 9 | `envelope` noise envelope and baseline for regression gating | full | PASS | 15s | baseline and envelope (chunk) written; they gate this build: PASS |
| 10 | `bench-long` long-context bench up to 16k | full | PASS | 1m32s | decode 36.2 tok/s at 1024 tokens (89% of the byte floor) |
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
| 124 | 0.051 | 2414 | 36.69 | 0.12 | 40.6 | 90.5 |
| 1002 | 0.286 | 3515 | 36.47 | 0.08 | 40.6 | 89.9 |
| 4172 | 1.701 | 2437 | 35.06 | 0.05 | 40.6 | 86.4 |

**long-context bench** (floor: 15.14 GB/token at 614 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: --bandwidth-gbs)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 974 | 0.303 | 3220 | 36.17 | 0.21 | 40.6 | 89.2 |
| 4033 | 1.864 | 2176 | 35.37 | 0.39 | 40.6 | 87.2 |
| 16547 | 12.986 | 1274 | 32.16 | 0.09 | 40.6 | 79.3 |

## Accuracy

**acc-short** against `/opt/code/intarsiaml/golden/qwen3-8b-fp32.safetensors` (found by model name; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 434 | 99.77 | 98.76 | 2.69e-4 | 2.04e-4 | 5.69e-3 | 4.36e-4 | 0 | 88.5 | pass |
| decode | 128 | 100.00 | 98.91 | 3.74e-4 | 3.34e-4 | 1.08e-2 | 3.14e-4 | 0 | 100.0 | pass |

**acc-long** against `/opt/code/intarsiaml/golden/qwen3-8b-long-fp32.safetensors` (found by model name; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1434 | 99.65 | 98.61 | 4.34e-4 | 4.01e-4 | 4.27e-2 | 3.32e-4 | 0 | 30.6 | pass |
| decode | 500 | 99.80 | 98.88 | 3.16e-4 | 2.95e-4 | 1.94e-2 | 2.22e-4 | 0 | 95.8 | pass |

**Regression gating artifacts**: baseline and envelope (chunk) written; they gate this build: PASS

`intarsia accuracy --model <dir> --backend metal --golden /opt/code/intarsiaml/golden/qwen3-8b-fp32.safetensors --baseline baselines/acc-short.baseline.json --noise-envelope baselines/envelope.json  (copy both files next to the golden, e.g. golden/baselines/)`

**Quick gate**: perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) (PASS)
- plan: 15 ops resolved, exactness deterministic, loaded in 35.8s (PASS)

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
| attn_prefill | attn_prefill_bf16_flash | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 36 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 36 |
| argmax | argmax_f32 | exact | yes | 1 |
| matmul_pair | matmul_pair_bf16_gemv | deterministic | yes | 72 |
| matmul_pair | matmul_pair_bf16_nax | deterministic | yes | 0 |
| qk_norm_rope_kv | qk_norm_rope_kv_bf16 | deterministic | yes | 36 |

- kernels test: 22 kernels, 265 cases in 31.7s (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | unstamped | endor.kuepker.haus | 1792041374 | `/opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00/target/release/intarsia doctor --backend metal --device 0 --out /opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00/docs/results/raw/2026-10-15-metal-prefill-p45/gates/qualify/doctor.json` |
| `recipe` | unstamped | endor.kuepker.haus | 1792041374 | `recipe::analyse(/Users/josef/Models/Qwen3-8B)` |
| `plan` | unstamped | endor.kuepker.haus | 1792041374 | `/opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00/target/release/intarsia plan show --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --admit --json` |
| `kernels` | unstamped | endor.kuepker.haus | 1792041410 | `/opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00/target/release/intarsia kernels test --backend metal --device 0 --model /Users/josef/Models/Qwen3-8B` |
| `gate` | unstamped | endor.kuepker.haus | 1792041442 | `/opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00/target/release/intarsia accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --quick --max-seq 2048` |
| `acc-short` | unstamped | endor.kuepker.haus | 1792041453 | `/opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00/target/release/intarsia accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --golden /opt/code/intarsiaml/golden/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq 4096 --write-baseline /opt/code/i` |
| `bench` | unstamped | endor.kuepker.haus | 1792041461 | `/opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00/target/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3-8B --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-Qwen3-8B-metal-bench --out-dir /opt/code/intarsiaml/.claude/workt` |
| `acc-long` | unstamped | endor.kuepker.haus | 1792041534 | `/opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00/target/release/intarsia accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --golden /opt/code/intarsiaml/golden/qwen3-8b-long-fp32.safetensors --mode all --decode-positions 64 --max-seq 8192 --write-baseline /opt/co` |
| `envelope` | unstamped | endor.kuepker.haus | 1792041567 | `/opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00/target/release/intarsia accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --golden /opt/code/intarsiaml/golden/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq 4096 --baseline /opt/code/intarsi` |
| `bench-long` | unstamped | endor.kuepker.haus | 1792041582 | `/opt/code/intarsiaml/.claude/worktrees/agent-a03ff605ef605cf00/target/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3-8B --sizes 1024,4096,16384 --gen 64 --reps 2 --warmup 1 --thinking off --label qualify-Qwen3-8B-metal-bench-long --out-dir /opt/code/intarsiaml/.claude` |
| `ab` | unstamped | endor.kuepker.haus | 1792041674 | `-` |
| `tp` | unstamped | endor.kuepker.haus | 1792041674 | `-` |

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
