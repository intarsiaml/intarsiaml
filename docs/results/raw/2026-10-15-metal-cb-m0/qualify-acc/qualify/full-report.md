# Qualification report: Qwen3-8B on metal

_Full (quick + in-depth) section, generated unix 1792094143, intarsia 0.0.1 build `head=8254311 code-tree=f2cc8867e6c9 dirty=no`._

## Headline

**Verdict: PASS** (13 stages: 11 pass, 0 fail, 0 error, 2 skipped)

| | |
|---|---|
| Model | Qwen3-8B (`/Users/josef/Models/Qwen3-8B`) |
| Backend / device | metal / Apple M5 Max (applegpu_g17s) |
| Recipe | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| Decode | 128 tok: **36.5** tok/s (90% of floor); 1024 tok: **36.3** tok/s (90% of floor); 4096 tok: **34.4** tok/s (85% of floor) |
| Prefill | 128 tok: **2531** tok/s; 1024 tok: **4006** tok/s; 4096 tok: **3245** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 100.00% KL 2.83e-4; decode top-1 100.00% KL 3.31e-4) |
| Accuracy (long golden) | **PASS** (prefill top-1 99.72% KL 3.70e-4; decode top-1 99.80% KL 3.54e-4) |
| Accuracy (4k golden) | **PASS** (prefill top-1 99.28% KL 2.44e-4; decode top-1 98.96% KL 2.57e-4) |
| Quick gate | **PASS**: perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 3m04s, in-depth 2m53s, total 5m57s |

### Failures

None.

### Skipped

- `ab`: no --reference server given
- `tp`: --tp not given (or 1)

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | Apple M5 Max (applegpu_g17s), 27 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 46s | 14 ops resolved, exactness deterministic, loaded in 46.3s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 43s | 27 kernels, 321 cases in 42.5s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 10s | perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-short` short accuracy against a golden | quick | PASS | 7s | prefill top-1 100.00% KL 2.83e-4; decode top-1 100.00% KL 3.31e-4 |
| 7 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 78s | decode 36.5 tok/s at 128 tokens (90% of the byte floor) |
| 8 | `acc-long` accuracy --mode all on the long golden (prefill, decode, prefix reuse) | full | PASS | 24s | prefill top-1 99.72% KL 3.70e-4; decode top-1 99.80% KL 3.54e-4 |
| 9 | `acc-4k` accuracy --mode all on the 4k golden (prefill, decode, prefix reuse past 2 chunks of 2048) | full | PASS | 59s | prefill top-1 99.28% KL 2.44e-4; decode top-1 98.96% KL 2.57e-4 |
| 10 | `envelope` noise envelope and baseline for regression gating | full | PASS | 14s | baseline and envelope (chunk) written; they gate this build: PASS |
| 11 | `bench-long` long-context bench up to 16k | full | PASS | 77s | decode 36.4 tok/s at 1024 tokens (90% of the byte floor) |
| 12 | `ab` A/B against the reference server via localeval | full | SKIP | 0s | no --reference server given |
| 13 | `tp` tensor-parallel variant | full | SKIP | 0s | --tp not given (or 1) |

### Notes

- `bench`: bench warning: size 128: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 1024: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 4096: 0 graph replays during the measured reps (decode ran eagerly)
- `acc-long`: prefix reuse is gated in acc-4k, on the 4k golden
- `acc-4k`: gated against the recorded baseline /opt/code/intarsiaml/golden/baselines/qwen3-8b-4k-fp32-top20.metal.baseline.json (commit 9eeb96c) and its noise envelope
- `bench-long`: bench warning: size 1024: 0 graph replays during the measured reps (decode ran eagerly)
- `bench-long`: bench warning: size 4096: 0 graph replays during the measured reps (decode ran eagerly)
- `bench-long`: bench warning: size 16384: 0 graph replays during the measured reps (decode ran eagerly)

## Speed

**quick bench** (floor: 15.14 GB/token at 614 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 614 GB/s, the Apple M5 Max (40-core GPU) spec figure of docs/decisions/0009-native-metal-backend.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 129 | 0.051 | 2531 | 36.55 | 0.02 | 40.6 | 90.1 |
| 1015 | 0.253 | 4006 | 36.31 | 1.02 | 40.6 | 89.5 |
| 4235 | 1.302 | 3245 | 34.41 | 0.41 | 40.6 | 84.8 |

**long-context bench** (floor: 15.14 GB/token at 614 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 614 GB/s, the Apple M5 Max (40-core GPU) spec figure of docs/decisions/0009-native-metal-backend.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 1000 | 0.258 | 3876 | 36.36 | 0.01 | 40.6 | 89.6 |
| 4051 | 1.234 | 3298 | 35.21 | 0.04 | 40.6 | 86.8 |
| 16574 | 6.873 | 2411 | 32.05 | 0.02 | 40.6 | 79.0 |

## Accuracy

**acc-short** against `/opt/code/intarsiaml/golden/qwen3-8b-fp32.safetensors` (found by model name; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 434 | 100.00 | 98.89 | 2.83e-4 | 2.20e-4 | 1.51e-2 | 4.46e-4 | 0 | 100.0 | pass |
| decode | 128 | 100.00 | 98.59 | 3.31e-4 | 2.98e-4 | 7.94e-3 | 3.07e-4 | 0 | 100.0 | pass |

**acc-long** against `/opt/code/intarsiaml/golden/qwen3-8b-long-fp32.safetensors` (found by model name; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1434 | 99.72 | 98.68 | 3.70e-4 | 3.38e-4 | 2.16e-2 | 3.34e-4 | 0 | 56.3 | pass |
| decode | 500 | 99.80 | 98.84 | 3.54e-4 | 3.33e-4 | 4.49e-2 | 2.27e-4 | 0 | 90.0 | pass |

**acc-4k** against `/opt/code/intarsiaml/golden/qwen3-8b-4k-fp32-top20.safetensors` (found by model name; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 12685 | 99.28 | 98.94 | 2.44e-4 | 2.27e-4 | 2.78e-1 | 6.81e-4 | 0 | 2.1 | pass |
| decode | 192 | 98.96 | 98.96 | 2.57e-4 | 2.55e-4 | 8.05e-3 | 3.22e-4 | 0 | 71.9 | pass |

**Regression gating artifacts**: baseline and envelope (chunk) written; they gate this build: PASS

`intarsia accuracy --model <dir> --backend metal --golden /opt/code/intarsiaml/golden/qwen3-8b-fp32.safetensors --baseline baselines/acc-short.baseline.json --noise-envelope baselines/envelope.json  (copy both files next to the golden, e.g. golden/baselines/)`

**Quick gate**: perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) (PASS)
- plan: 14 ops resolved, exactness deterministic, loaded in 46.3s (PASS)

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

- kernels test: 27 kernels, 321 cases in 42.5s (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | head=8254311 code-tree=f2cc8867e6c9 dirty=no | endor.kuepker.haus | 1792093785 | `/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/bin/intarsia-cbm0-8254311 doctor --backend metal --device 0 --out /opt/code/intarsiaml-cb-m0/docs/results/raw/2026-10-15-metal-cb-m0/qualify-acc/qualify/doctor.json` |
| `recipe` | head=8254311 code-tree=f2cc8867e6c9 dirty=no | endor.kuepker.haus | 1792093786 | `recipe::analyse(/Users/josef/Models/Qwen3-8B)` |
| `plan` | head=8254311 code-tree=f2cc8867e6c9 dirty=no | endor.kuepker.haus | 1792093786 | `/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/bin/intarsia-cbm0-8254311 plan show --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --admit --json` |
| `kernels` | head=8254311 code-tree=f2cc8867e6c9 dirty=no | endor.kuepker.haus | 1792093832 | `/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/bin/intarsia-cbm0-8254311 kernels test --backend metal --device 0 --model /Users/josef/Models/Qwen3-8B` |
| `gate` | head=8254311 code-tree=f2cc8867e6c9 dirty=no | endor.kuepker.haus | 1792093875 | `/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/bin/intarsia-cbm0-8254311 accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --quick --max-seq 2048` |
| `acc-short` | head=8254311 code-tree=f2cc8867e6c9 dirty=no | endor.kuepker.haus | 1792093884 | `/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/bin/intarsia-cbm0-8254311 accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --golden /opt/code/intarsiaml/golden/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq 4` |
| `bench` | head=8254311 code-tree=f2cc8867e6c9 dirty=no | endor.kuepker.haus | 1792093891 | `/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/bin/intarsia-cbm0-8254311 bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3-8B --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-Qwen3-8B-metal-bench --out-dir /o` |
| `acc-long` | head=8254311 code-tree=f2cc8867e6c9 dirty=no | endor.kuepker.haus | 1792093969 | `/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/bin/intarsia-cbm0-8254311 accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --golden /opt/code/intarsiaml/golden/qwen3-8b-long-fp32.safetensors --mode both --decode-positions 64 --max-` |
| `acc-4k` | head=8254311 code-tree=f2cc8867e6c9 dirty=no | endor.kuepker.haus | 1792093994 | `/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/bin/intarsia-cbm0-8254311 accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --golden /opt/code/intarsiaml/golden/qwen3-8b-4k-fp32-top20.safetensors --mode all --decode-positions 64 --m` |
| `envelope` | head=8254311 code-tree=f2cc8867e6c9 dirty=no | endor.kuepker.haus | 1792094052 | `/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/bin/intarsia-cbm0-8254311 accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --golden /opt/code/intarsiaml/golden/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq 4` |
| `bench-long` | head=8254311 code-tree=f2cc8867e6c9 dirty=no | endor.kuepker.haus | 1792094066 | `/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/bin/intarsia-cbm0-8254311 bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3-8B --sizes 1024,4096,16384 --gen 64 --reps 2 --warmup 1 --thinking off --label qualify-Qwen3-8B-metal-bench-long --out-` |
| `ab` | head=8254311 code-tree=f2cc8867e6c9 dirty=no | endor.kuepker.haus | 1792094143 | `-` |
| `tp` | head=8254311 code-tree=f2cc8867e6c9 dirty=no | endor.kuepker.haus | 1792094143 | `-` |

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
