# Qualification report: Qwen3-8B on metal

_Full (quick + in-depth) section, generated unix 1792070365, intarsia 0.0.1 build `head=cd4c7d3 code-tree=42a65ddc0934 dirty=no`._

## Headline

**Verdict: PASS** (13 stages: 11 pass, 0 fail, 0 error, 2 skipped)

| | |
|---|---|
| Model | Qwen3-8B (`/Users/josef/Models/Qwen3-8B`) |
| Backend / device | metal / Apple M5 Max (applegpu_g17s) |
| Recipe | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| Decode | 128 tok: **36.9** tok/s (91% of floor); 1024 tok: **36.4** tok/s (90% of floor); 4096 tok: **34.8** tok/s (86% of floor) |
| Prefill | 128 tok: **2440** tok/s; 1024 tok: **3669** tok/s; 4096 tok: **2945** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 100.00% KL 2.83e-4; decode top-1 100.00% KL 3.31e-4) |
| Accuracy (long golden) | **PASS** (prefill top-1 99.72% KL 3.70e-4; decode top-1 99.80% KL 3.54e-4) |
| Accuracy (4k golden) | **PASS** (prefill top-1 99.28% KL 2.44e-4; decode top-1 98.96% KL 2.57e-4) |
| Quick gate | **PASS**: perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 2m47s, in-depth 3m44s, total 6m31s |

### Failures

None.

### Skipped

- `ab`: no --reference server given
- `tp`: --tp not given (or 1)

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | Apple M5 Max (applegpu_g17s), 23 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 39s | 14 ops resolved, exactness deterministic, loaded in 38.8s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 33s | 23 kernels, 291 cases in 32.9s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 10s | perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-short` short accuracy against a golden | quick | PASS | 7s | prefill top-1 100.00% KL 2.83e-4; decode top-1 100.00% KL 3.31e-4 |
| 7 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 79s | decode 36.9 tok/s at 128 tokens (91% of the byte floor) |
| 8 | `acc-long` accuracy --mode all on the long golden (prefill, decode, prefix reuse) | full | PASS | 23s | prefill top-1 99.72% KL 3.70e-4; decode top-1 99.80% KL 3.54e-4 |
| 9 | `acc-4k` accuracy --mode all on the 4k golden (prefill, decode, prefix reuse past 2 chunks of 2048) | full | PASS | 1m43s | prefill top-1 99.28% KL 2.44e-4; decode top-1 98.96% KL 2.57e-4 |
| 10 | `envelope` noise envelope and baseline for regression gating | full | PASS | 14s | baseline and envelope (chunk) written; they gate this build: PASS |
| 11 | `bench-long` long-context bench up to 16k | full | PASS | 84s | decode 36.2 tok/s at 1024 tokens (89% of the byte floor) |
| 12 | `ab` A/B against the reference server via localeval | full | SKIP | 0s | no --reference server given |
| 13 | `tp` tensor-parallel variant | full | SKIP | 0s | --tp not given (or 1) |

### Notes

- `bench`: bench warning: size 128: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 1024: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 4096: 0 graph replays during the measured reps (decode ran eagerly)
- `acc-long`: prefix reuse is gated in acc-4k, on the 4k golden
- `bench-long`: bench warning: size 1024: 0 graph replays during the measured reps (decode ran eagerly)
- `bench-long`: bench warning: size 4096: 0 graph replays during the measured reps (decode ran eagerly)
- `bench-long`: bench warning: size 16384: 0 graph replays during the measured reps (decode ran eagerly)

## Speed

**quick bench** (floor: 15.14 GB/token at 614 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: --bandwidth-gbs)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 126 | 0.051 | 2440 | 36.87 | 0.03 | 40.6 | 90.9 |
| 1016 | 0.275 | 3669 | 36.40 | 0.05 | 40.6 | 89.7 |
| 4235 | 1.433 | 2945 | 34.81 | 0.15 | 40.6 | 85.8 |

**long-context bench** (floor: 15.14 GB/token at 614 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: --bandwidth-gbs)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 995 | 0.306 | 3256 | 36.17 | 0.04 | 40.6 | 89.2 |
| 4103 | 1.617 | 2562 | 35.34 | 0.20 | 40.6 | 87.1 |
| 16764 | 8.455 | 1984 | 31.96 | 0.03 | 40.6 | 78.8 |

## Accuracy

**acc-short** against `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/gold/qwen3-8b-fp32.safetensors` (found by model name; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 434 | 100.00 | 98.89 | 2.83e-4 | 2.20e-4 | 1.51e-2 | 4.46e-4 | 0 | 100.0 | pass |
| decode | 128 | 100.00 | 98.59 | 3.31e-4 | 2.98e-4 | 7.94e-3 | 3.07e-4 | 0 | 100.0 | pass |

**acc-long** against `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/gold/qwen3-8b-long-fp32.safetensors` (found by model name; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1434 | 99.72 | 98.68 | 3.70e-4 | 3.38e-4 | 2.16e-2 | 3.34e-4 | 0 | 56.3 | pass |
| decode | 500 | 99.80 | 98.84 | 3.54e-4 | 3.33e-4 | 4.49e-2 | 2.27e-4 | 0 | 90.0 | pass |

**acc-4k** against `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/gold/qwen3-8b-4k-fp32-top20.safetensors` (found by model name; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 12685 | 99.28 | 98.94 | 2.44e-4 | 2.27e-4 | 2.78e-1 | 6.81e-4 | 0 | 2.1 | pass |
| decode | 192 | 98.96 | 98.96 | 2.57e-4 | 2.55e-4 | 8.05e-3 | 3.22e-4 | 0 | 71.9 | pass |

**Regression gating artifacts**: baseline and envelope (chunk) written; they gate this build: PASS

`intarsia accuracy --model <dir> --backend metal --golden /private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/gold/qwen3-8b-fp32.safetensors --baseline baselines/acc-short.baseline.json --noise-envelope baselines/envelope.json  (copy both files next to the golden, e.g. golden/baselines/)`

**Quick gate**: perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) (PASS)
- plan: 14 ops resolved, exactness deterministic, loaded in 38.8s (PASS)

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

- kernels test: 23 kernels, 291 cases in 32.9s (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | head=cd4c7d3 code-tree=42a65ddc0934 dirty=no | endor.kuepker.haus | 1792069973 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-a2 doctor --backend metal --device 0 --out /opt/code/intarsiaml-golden-4k/docs/results/raw/2026-10-15-metal-golden-4k/round2/safe-qualify/doctor.json` |
| `recipe` | head=cd4c7d3 code-tree=42a65ddc0934 dirty=no | endor.kuepker.haus | 1792069973 | `recipe::analyse(/Users/josef/Models/Qwen3-8B)` |
| `plan` | head=cd4c7d3 code-tree=42a65ddc0934 dirty=no | endor.kuepker.haus | 1792069973 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-a2 plan show --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --admit --json` |
| `kernels` | head=cd4c7d3 code-tree=42a65ddc0934 dirty=no | endor.kuepker.haus | 1792070012 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-a2 kernels test --backend metal --device 0 --model /Users/josef/Models/Qwen3-8B` |
| `gate` | head=cd4c7d3 code-tree=42a65ddc0934 dirty=no | endor.kuepker.haus | 1792070045 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-a2 accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --quick --max-seq 2048` |
| `acc-short` | head=cd4c7d3 code-tree=42a65ddc0934 dirty=no | endor.kuepker.haus | 1792070054 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-a2 accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --golden /private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/gold/qwen3-8` |
| `bench` | head=cd4c7d3 code-tree=42a65ddc0934 dirty=no | endor.kuepker.haus | 1792070061 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-a2 bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3-8B --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-Qwen3-8B-metal-bench --out-dir /opt/code/in` |
| `acc-long` | head=cd4c7d3 code-tree=42a65ddc0934 dirty=no | endor.kuepker.haus | 1792070140 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-a2 accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --golden /private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/gold/qwen3-8` |
| `acc-4k` | head=cd4c7d3 code-tree=42a65ddc0934 dirty=no | endor.kuepker.haus | 1792070163 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-a2 accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --golden /private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/gold/qwen3-8` |
| `envelope` | head=cd4c7d3 code-tree=42a65ddc0934 dirty=no | endor.kuepker.haus | 1792070267 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-a2 accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --golden /private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/gold/qwen3-8` |
| `bench-long` | head=cd4c7d3 code-tree=42a65ddc0934 dirty=no | endor.kuepker.haus | 1792070281 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/g4k/intarsia-a2 bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3-8B --sizes 1024,4096,16384 --gen 64 --reps 2 --warmup 1 --thinking off --label qualify-Qwen3-8B-metal-bench-long --out-dir /opt/c` |
| `ab` | head=cd4c7d3 code-tree=42a65ddc0934 dirty=no | endor.kuepker.haus | 1792070365 | `-` |
| `tp` | head=cd4c7d3 code-tree=42a65ddc0934 dirty=no | endor.kuepker.haus | 1792070365 | `-` |

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
- `logs/acc-long.out` (acc-long)
- `baselines/acc-long.baseline.json` (acc-long)
- `logs/acc-4k.out` (acc-4k)
- `baselines/acc-4k.baseline.json` (acc-4k)
- `baselines/acc-short.baseline.json` (envelope)
- `baselines/envelope.json` (envelope)
- `logs/baseline-selfcheck.out` (envelope)
- `bench/qualify-Qwen3-8B-metal-bench-long.json` (bench-long)
- `logs/bench-long-bench.out` (bench-long)
- `logs/bench-long-server.err` (bench-long)
- `state.json` (stage records; `--resume` reads it)
