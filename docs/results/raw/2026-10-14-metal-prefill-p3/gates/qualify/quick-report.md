# Qualification quick report: Qwen3-8B on metal

_Quick section, generated unix 1792031426, intarsia 0.0.1 build `head=e8be186 code-tree=6c2d16b008e2 dirty=no`._

## Headline

**Verdict: PASS** (7 stages: 7 pass, 0 fail, 0 error, 0 skipped)

| | |
|---|---|
| Model | Qwen3-8B (`/Users/josef/Models/Qwen3-8B`) |
| Backend / device | metal / Apple M5 Max (applegpu_g17s) |
| Recipe | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| Decode | 128 tok: **36.0** tok/s (89% of floor); 1024 tok: **36.0** tok/s (89% of floor); 4096 tok: **34.7** tok/s (86% of floor) |
| Prefill | 128 tok: **2407** tok/s; 1024 tok: **3621** tok/s; 4096 tok: **2659** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 99.77% KL 2.69e-4; decode top-1 100.00% KL 3.74e-4) |
| Quick gate | **PASS**: perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 2m39s |

### Failures

None.

### Skipped

None.

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | Apple M5 Max (applegpu_g17s), 22 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 36s | 15 ops resolved, exactness deterministic, loaded in 36.1s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 32s | 22 kernels, 265 cases in 32.2s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 10s | perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-short` short accuracy against a golden | quick | PASS | 7s | prefill top-1 99.77% KL 2.69e-4; decode top-1 100.00% KL 3.74e-4 |
| 7 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 73s | decode 36.0 tok/s at 128 tokens (89% of the byte floor) |

### Notes

- `bench`: bench warning: size 128: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 1024: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 4096: 0 graph replays during the measured reps (decode ran eagerly)

## Speed

**quick bench** (floor: 15.14 GB/token at 614 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: --bandwidth-gbs)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 129 | 0.053 | 2407 | 36.02 | 0.25 | 40.6 | 88.8 |
| 1004 | 0.277 | 3621 | 36.02 | 0.10 | 40.6 | 88.8 |
| 4201 | 1.580 | 2659 | 34.68 | 0.31 | 40.6 | 85.5 |

## Accuracy

**acc-short** against `/opt/code/intarsiaml/golden/qwen3-8b-fp32.safetensors` (found by model name; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 434 | 99.77 | 98.76 | 2.69e-4 | 2.04e-4 | 5.69e-3 | 4.36e-4 | 0 | 88.5 | pass |
| decode | 128 | 100.00 | 98.91 | 3.74e-4 | 3.34e-4 | 1.08e-2 | 3.14e-4 | 0 | 100.0 | pass |

**Quick gate**: perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows

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
| attn_prefill | attn_prefill_bf16_flash | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 36 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 36 |
| argmax | argmax_f32 | exact | yes | 1 |
| matmul_pair | matmul_pair_bf16_gemv | deterministic | yes | 72 |
| matmul_pair | matmul_pair_bf16_nax | deterministic | yes | 0 |
| qk_norm_rope_kv | qk_norm_rope_kv_bf16 | deterministic | yes | 36 |

- kernels test: 22 kernels, 265 cases in 32.2s (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | head=e8be186 code-tree=6c2d16b008e2 dirty=no | endor.kuepker.haus | 1792031267 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/p3/intarsia-p3 doctor --backend metal --device 0 --out /opt/code/intarsiaml/.claude/worktrees/agent-af7ec7dd89422c6a9/docs/results/raw/2026-10-14-metal-prefill-p3/gates/qualify/doctor.json` |
| `recipe` | head=e8be186 code-tree=6c2d16b008e2 dirty=no | endor.kuepker.haus | 1792031267 | `recipe::analyse(/Users/josef/Models/Qwen3-8B)` |
| `plan` | head=e8be186 code-tree=6c2d16b008e2 dirty=no | endor.kuepker.haus | 1792031267 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/p3/intarsia-p3 plan show --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --admit --json` |
| `kernels` | head=e8be186 code-tree=6c2d16b008e2 dirty=no | endor.kuepker.haus | 1792031303 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/p3/intarsia-p3 kernels test --backend metal --device 0 --model /Users/josef/Models/Qwen3-8B` |
| `gate` | head=e8be186 code-tree=6c2d16b008e2 dirty=no | endor.kuepker.haus | 1792031336 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/p3/intarsia-p3 accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --quick --max-seq 2048` |
| `acc-short` | head=e8be186 code-tree=6c2d16b008e2 dirty=no | endor.kuepker.haus | 1792031346 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/p3/intarsia-p3 accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --golden /opt/code/intarsiaml/golden/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq 4096 --write` |
| `bench` | head=e8be186 code-tree=6c2d16b008e2 dirty=no | endor.kuepker.haus | 1792031353 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/p3/intarsia-p3 bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3-8B --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-Qwen3-8B-metal-bench --out-dir /opt/code/int` |

Model fingerprint: `fnv1a64:e6a70935b68a5561`.


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
