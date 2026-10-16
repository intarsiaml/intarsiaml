# Qualification quick report: Qwen3-8B on metal

_Quick section, generated unix 1792026747, intarsia 0.0.1 build `unstamped`._

## Headline

**Verdict: PASS** (7 stages: 6 pass, 0 fail, 0 error, 1 skipped)

| | |
|---|---|
| Model | Qwen3-8B (`/Users/josef/Models/Qwen3-8B`) |
| Backend / device | metal / Apple M5 Max (applegpu_g17s) |
| Recipe | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| Decode | 128 tok: **36.1** tok/s (89% of floor); 1024 tok: **35.9** tok/s (88% of floor); 4096 tok: **34.6** tok/s (85% of floor) |
| Prefill | 128 tok: **2180** tok/s; 1024 tok: **1502** tok/s; 4096 tok: **446** tok/s |
| Accuracy (short golden) | skipped: no golden found, and 16.4 GB of weights is too large to make an fp32 HF golden quickly (limit 5 GB, --make-golden-max-gb): pass --golden FILE or --reference <vLLM url> |
| Accuracy (long golden) | skipped: no golden found, and 16.4 GB of weights is too large to make an fp32 HF golden quickly (limit 5 GB, --make-golden-max-gb): pass --golden FILE or --reference <vLLM url> |
| Quick gate | **PASS**: perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 3m07s |

### Failures

None.

### Skipped

- `acc-short`: no golden found, and 16.4 GB of weights is too large to make an fp32 HF golden quickly (limit 5 GB, --make-golden-max-gb): pass --golden FILE or --reference <vLLM url>

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | Apple M5 Max (applegpu_g17s), 21 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 36s | 15 ops resolved, exactness deterministic, loaded in 36.1s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 30s | 21 kernels, 251 cases in 30.1s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 12s | perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-short` short accuracy against a golden | quick | SKIP | 0s | no golden found, and 16.4 GB of weights is too large to make an fp32 HF golden quickly (limit 5 GB, --make-golden-max-gb): pass --golden FILE or --reference <vLLM url> |
| 7 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 1m48s | decode 36.1 tok/s at 128 tokens (89% of the byte floor) |

### Notes

- `bench`: bench warning: size 128: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 1024: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 4096: 0 graph replays during the measured reps (decode ran eagerly)

## Speed

**quick bench** (floor: 15.14 GB/token at 614 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: --bandwidth-gbs)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 128 | 0.059 | 2180 | 36.08 | 0.27 | 40.6 | 88.9 |
| 1015 | 0.681 | 1502 | 35.91 | 0.10 | 40.6 | 88.5 |
| 4199 | 9.368 | 446 | 34.56 | 0.14 | 40.6 | 85.2 |

## Accuracy

**Quick gate**: perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows

No golden comparison ran (see Skipped).

## Plan and recipe

- recipe: builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) (PASS)
- plan: 15 ops resolved, exactness deterministic, loaded in 36.1s (PASS)

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
| `doctor` | unstamped | endor.kuepker.haus | 1792024828 | `/opt/code/intarsiaml/.claude/worktrees/agent-a1247931a45f1aeb7/target/release/intarsia doctor --backend metal --device 0 --out /opt/code/intarsiaml/.claude/worktrees/agent-a1247931a45f1aeb7/docs/results/raw/2026-10-14-metal-prefill-p1/full/final/qualify/doctor.json` |
| `recipe` | unstamped | endor.kuepker.haus | 1792024829 | `recipe::analyse(/Users/josef/Models/Qwen3-8B)` |
| `plan` | unstamped | endor.kuepker.haus | 1792024829 | `/opt/code/intarsiaml/.claude/worktrees/agent-a1247931a45f1aeb7/target/release/intarsia plan show --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --admit --json` |
| `kernels` | unstamped | endor.kuepker.haus | 1792024865 | `/opt/code/intarsiaml/.claude/worktrees/agent-a1247931a45f1aeb7/target/release/intarsia kernels test --backend metal --device 0 --model /Users/josef/Models/Qwen3-8B` |
| `gate` | unstamped | endor.kuepker.haus | 1792024895 | `/opt/code/intarsiaml/.claude/worktrees/agent-a1247931a45f1aeb7/target/release/intarsia accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --quick --max-seq 2048` |
| `acc-short` | unstamped | endor.kuepker.haus | 1792024907 | `-` |
| `bench` | unstamped | endor.kuepker.haus | 1792024907 | `/opt/code/intarsiaml/.claude/worktrees/agent-a1247931a45f1aeb7/target/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3-8B --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-Qwen3-8B-metal-bench --out-dir /opt/code/intarsiaml/.claude/workt` |

Model fingerprint: `fnv1a64:e6a70935b68a5561`.

Build stamp `unstamped` means a plain `cargo build`: baselines written here gate this binary, but `--pre-change` provenance checks refuse them as a "before".


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/kernels.out` (kernels)
- `logs/gate.out` (gate)
- `bench/qualify-Qwen3-8B-metal-bench.json` (bench)
- `logs/bench-bench.out` (bench)
- `logs/bench-server.err` (bench)
- `state.json` (stage records; `--resume` reads it)
