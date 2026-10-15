# Qualification quick report: Qwen3-8B on metal

_Quick section, generated unix 1792043578, intarsia 0.0.1 build `head=0b2e3a2 code-tree=4757814f12d4 dirty=no`._

## Headline

**Verdict: PASS** (7 stages: 7 pass, 0 fail, 0 error, 0 skipped)

| | |
|---|---|
| Model | Qwen3-8B (`/Users/josef/Models/Qwen3-8B`) |
| Backend / device | metal / Apple M5 Max (applegpu_g17s) |
| Recipe | explicit recipe /opt/code/intarsiaml/.claude/worktrees/agent-a0a61a602516c3238/docs/results/raw/2026-10-15-metal-candidate/gates/qwen3-c256.toml (family qwen3) |
| Decode | 128 tok: **36.5** tok/s (90% of floor); 1024 tok: **36.6** tok/s (90% of floor); 4096 tok: **35.6** tok/s (88% of floor) |
| Prefill | 128 tok: **2418** tok/s; 1024 tok: **3775** tok/s; 4096 tok: **3356** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 99.54% KL 2.60e-4; decode top-1 100.00% KL 3.09e-4) |
| Quick gate | **PASS**: perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 2m33s |

### Failures

None.

### Skipped

None.

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | Apple M5 Max (applegpu_g17s), 23 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | explicit recipe /opt/code/intarsiaml/.claude/worktrees/agent-a0a61a602516c3238/docs/results/raw/2026-10-15-metal-candidate/gates/qwen3-c256.toml (family qwen3) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 32s | 15 ops resolved, exactness deterministic, loaded in 31.5s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 33s | 23 kernels, 272 cases in 32.6s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 10s | perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-short` short accuracy against a golden | quick | PASS | 7s | prefill top-1 99.54% KL 2.60e-4; decode top-1 100.00% KL 3.09e-4 |
| 7 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 72s | decode 36.5 tok/s at 128 tokens (90% of the byte floor) |

### Notes

- `bench`: bench warning: size 128: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 1024: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 4096: 0 graph replays during the measured reps (decode ran eagerly)

## Speed

**quick bench** (floor: 15.14 GB/token at 614 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: --bandwidth-gbs)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 122 | 0.051 | 2418 | 36.51 | 0.22 | 40.6 | 90.0 |
| 1027 | 0.268 | 3775 | 36.60 | 0.08 | 40.6 | 90.2 |
| 4214 | 1.255 | 3356 | 35.61 | 0.15 | 40.6 | 87.8 |

## Accuracy

**acc-short** against `/opt/code/intarsiaml/golden/qwen3-8b-fp32.safetensors` (found by model name; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 434 | 99.54 | 99.03 | 2.60e-4 | 1.96e-4 | 7.50e-3 | 4.56e-4 | 0 | 78.1 | pass |
| decode | 128 | 100.00 | 99.06 | 3.09e-4 | 2.74e-4 | 4.41e-3 | 2.72e-4 | 0 | 100.0 | pass |

**Quick gate**: perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: explicit recipe /opt/code/intarsiaml/.claude/worktrees/agent-a0a61a602516c3238/docs/results/raw/2026-10-15-metal-candidate/gates/qwen3-c256.toml (family qwen3) (PASS)
- plan: 15 ops resolved, exactness deterministic, loaded in 31.5s (PASS)

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
| attn_prefill | attn_prefill_bf16_flash_nax | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 36 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 36 |
| argmax | argmax_f32 | exact | yes | 1 |
| matmul_pair | matmul_pair_bf16_gemv | deterministic | yes | 72 |
| matmul_pair | matmul_pair_bf16_nax | deterministic | yes | 0 |
| qk_norm_rope_kv | qk_norm_rope_kv_bf16 | deterministic | yes | 36 |

- kernels test: 23 kernels, 272 cases in 32.6s (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | head=0b2e3a2 code-tree=4757814f12d4 dirty=no | endor.kuepker.haus | 1792043425 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/intarsia-cand doctor --backend metal --device 0 --out /opt/code/intarsiaml/.claude/worktrees/agent-a0a61a602516c3238/docs/results/raw/2026-10-15-metal-candidate/gates/qualify/c256/doctor.json` |
| `recipe` | head=0b2e3a2 code-tree=4757814f12d4 dirty=no | endor.kuepker.haus | 1792043425 | `-` |
| `plan` | head=0b2e3a2 code-tree=4757814f12d4 dirty=no | endor.kuepker.haus | 1792043425 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/intarsia-cand plan show --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --recipe /opt/code/intarsiaml/.claude/worktrees/agent-a0a61a602516c3238/docs/results/raw/2026-10-15-metal-candidate/gat` |
| `kernels` | head=0b2e3a2 code-tree=4757814f12d4 dirty=no | endor.kuepker.haus | 1792043457 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/intarsia-cand kernels test --backend metal --device 0 --model /Users/josef/Models/Qwen3-8B` |
| `gate` | head=0b2e3a2 code-tree=4757814f12d4 dirty=no | endor.kuepker.haus | 1792043489 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/intarsia-cand accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --recipe /opt/code/intarsiaml/.claude/worktrees/agent-a0a61a602516c3238/docs/results/raw/2026-10-15-metal-candidate/gate` |
| `acc-short` | head=0b2e3a2 code-tree=4757814f12d4 dirty=no | endor.kuepker.haus | 1792043499 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/intarsia-cand accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --recipe /opt/code/intarsiaml/.claude/worktrees/agent-a0a61a602516c3238/docs/results/raw/2026-10-15-metal-candidate/gate` |
| `bench` | head=0b2e3a2 code-tree=4757814f12d4 dirty=no | endor.kuepker.haus | 1792043506 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/intarsia-cand bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3-8B --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-Qwen3-8B-metal-bench --out-dir /opt/code/inta` |

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
