# Qualification report: Qwen3-8B on metal

_Full (quick + in-depth) section, generated unix 1792093283, intarsia 0.0.1 build `unstamped`._

## Headline

**Verdict: ERROR** (13 stages: 5 pass, 2 fail, 1 error, 5 skipped)

| | |
|---|---|
| Model | Qwen3-8B (`/Users/josef/Models/Qwen3-8B`) |
| Backend / device | metal / Apple M5 Max (applegpu_g17s) |
| Recipe | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| Decode | not measured |
| Prefill | not measured |
| Accuracy (short golden) | **FAIL** (prefill top-1 97.93% KL 5.93e-3; decode top-1 96.88% KL 7.78e-3) |
| Accuracy (long golden) | **FAIL** (prefill top-1 98.33% KL 9.79e-3; decode top-1 97.00% KL 9.18e-3) |
| Accuracy (4k golden) | **ERROR** |
| Quick gate | **PASS**: perplexity 1.86, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 1m43s, in-depth 24s, total 2m07s |

### Failures

- `acc-short` **FAIL**: prefill: full-vocab KL 5.927e-3 (need <= 5.0e-3); decode: full-vocab KL 7.781e-3 (need <= 5.0e-3)
- `acc-long` **FAIL**: prefill: full-vocab KL 9.795e-3 (need <= 5.0e-3); decode: full-vocab KL 9.176e-3 (need <= 5.0e-3)
- `acc-4k` **ERROR**: accuracy exit 1: WARNING: KV dtype mismatch: the reference ran with a f32 KV cache, we run fp8_e4m3. Part of the KL / top-1 gap is KV quantisation noise of the reference, not model error; rebuild the golden with a matching KV dtype to separate them

### Skipped

- `bench`: skipped by --skip
- `envelope`: skipped by --skip
- `bench-long`: skipped by --skip
- `ab`: skipped by --skip
- `tp`: skipped by --skip

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 0s | Apple M5 Max (applegpu_g17s), 33 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 39s | 14 ops resolved, exactness deterministic, loaded in 38.9s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 47s | 33 kernels, 388 cases in 46.8s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 10s | perplexity 1.86, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-short` short accuracy against a golden | quick | FAIL | 8s | prefill: full-vocab KL 5.927e-3 (need <= 5.0e-3); decode: full-vocab KL 7.781e-3 (need <= 5.0e-3) |
| 7 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | SKIP | 0s | skipped by --skip |
| 8 | `acc-long` accuracy --mode all on the long golden (prefill, decode, prefix reuse) | full | FAIL | 23s | prefill: full-vocab KL 9.795e-3 (need <= 5.0e-3); decode: full-vocab KL 9.176e-3 (need <= 5.0e-3) |
| 9 | `acc-4k` accuracy --mode all on the 4k golden (prefill, decode, prefix reuse past 2 chunks of 2048) | full | ERROR | 0s | accuracy exit 1: WARNING: KV dtype mismatch: the reference ran with a f32 KV cache, we run fp8_e4m3. Part of the KL / top-1 gap is KV quantisation noise of the reference, not model error; rebuild the golden with a matching KV dtype to separate them |
| 10 | `envelope` noise envelope and baseline for regression gating | full | SKIP | 0s | skipped by --skip |
| 11 | `bench-long` long-context bench up to 16k | full | SKIP | 0s | skipped by --skip |
| 12 | `ab` A/B against the reference server via localeval | full | SKIP | 0s | skipped by --skip |
| 13 | `tp` tensor-parallel variant | full | SKIP | 0s | skipped by --skip |

### Notes

- `acc-long`: prefix reuse is gated in acc-4k, on the 4k golden
- `acc-4k`: gated against the recorded baseline /opt/code/intarsiaml/golden/baselines/qwen3-8b-4k-fp32-top20.metal.baseline.json (commit 9eeb96c) and its noise envelope

## Speed

No speed numbers.

## Accuracy

**acc-short** against `/opt/code/intarsiaml/golden/qwen3-8b-fp32.safetensors` (found by model name; gate class fp8)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 434 | 97.93 | 95.90 | 5.93e-3 | 5.51e-3 | 6.95e-1 | 1.27e-3 | 0 | 55.3 | FAIL |
| decode | 128 | 96.88 | 95.31 | 7.78e-3 | 6.66e-3 | 8.96e-2 | 2.38e-3 | 0 | 71.1 | FAIL |

**acc-long** against `/opt/code/intarsiaml/golden/qwen3-8b-long-fp32.safetensors` (found by model name; gate class fp8)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1434 | 98.33 | 94.27 | 9.79e-3 | 9.45e-3 | 7.28e-1 | 1.22e-3 | 0 | 37.4 | FAIL |
| decode | 500 | 97.00 | 95.20 | 9.18e-3 | 8.74e-3 | 7.09e-1 | 1.20e-3 | 0 | 49.2 | FAIL |

**Quick gate**: perplexity 1.86, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) (PASS)
- plan: 14 ops resolved, exactness deterministic, loaded in 38.9s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_bf16 | exact | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 1 |
| rmsnorm_add | rmsnorm_add_bf16 | deterministic | yes | 72 |
| matmul | matmul_bf16_gemv | deterministic | yes | 108 |
| matmul | matmul_bf16_nax | deterministic | yes | 0 |
| logits | logits_bf16 | deterministic | yes | 0 |
| logits | logits_bf16_gemv | deterministic | yes | 1 |
| attn_prefill | attn_prefill_f8_flash_nax | deterministic | yes | 0 |
| attn_decode | attn_decode_f8_splitkv | deterministic | yes | 36 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 36 |
| argmax | argmax_f32 | exact | yes | 1 |
| matmul_pair | matmul_pair_bf16_gemv | deterministic | yes | 72 |
| matmul_pair | matmul_pair_bf16_nax | deterministic | yes | 0 |
| qk_norm_rope_kv | qk_norm_rope_kv_f8 | deterministic | yes | 36 |

- kernels test: 33 kernels, 388 cases in 46.8s (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | unstamped | endor.kuepker.haus | 1792093156 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/intarsia-new doctor --backend metal --device 0 --out /opt/code/intarsiaml-kvq/docs/results/raw/2026-10-15-metal-kv-quant/qualify-fp8/doctor.json` |
| `recipe` | unstamped | endor.kuepker.haus | 1792093156 | `recipe::analyse(/Users/josef/Models/Qwen3-8B)` |
| `plan` | unstamped | endor.kuepker.haus | 1792093156 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/intarsia-new plan show --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --kv-dtype fp8 --admit --json` |
| `kernels` | unstamped | endor.kuepker.haus | 1792093195 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/intarsia-new kernels test --backend metal --device 0 --model /Users/josef/Models/Qwen3-8B` |
| `gate` | unstamped | endor.kuepker.haus | 1792093242 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/intarsia-new accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --kv-dtype fp8 --quick --max-seq 2048` |
| `acc-short` | unstamped | endor.kuepker.haus | 1792093252 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/intarsia-new accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --kv-dtype fp8 --golden /opt/code/intarsiaml/golden/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq` |
| `bench` | unstamped | endor.kuepker.haus | 0 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/intarsia-new accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --kv-dtype fp8 --golden /opt/code/intarsiaml/golden/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq` |
| `acc-long` | unstamped | endor.kuepker.haus | 1792093259 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/intarsia-new accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --kv-dtype fp8 --golden /opt/code/intarsiaml/golden/qwen3-8b-long-fp32.safetensors --mode both --decode-positions 64 --ma` |
| `acc-4k` | unstamped | endor.kuepker.haus | 1792093282 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/intarsia-new accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --kv-dtype fp8 --golden /opt/code/intarsiaml/golden/qwen3-8b-4k-fp32-top20.safetensors --mode all --decode-positions 64 -` |
| `envelope` | unstamped | endor.kuepker.haus | 0 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/intarsia-new accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --kv-dtype fp8 --golden /opt/code/intarsiaml/golden/qwen3-8b-4k-fp32-top20.safetensors --mode all --decode-positions 64 -` |
| `bench-long` | unstamped | endor.kuepker.haus | 0 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/intarsia-new accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --kv-dtype fp8 --golden /opt/code/intarsiaml/golden/qwen3-8b-4k-fp32-top20.safetensors --mode all --decode-positions 64 -` |
| `ab` | unstamped | endor.kuepker.haus | 0 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/intarsia-new accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --kv-dtype fp8 --golden /opt/code/intarsiaml/golden/qwen3-8b-4k-fp32-top20.safetensors --mode all --decode-positions 64 -` |
| `tp` | unstamped | endor.kuepker.haus | 0 | `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/intarsia-new accuracy --model /Users/josef/Models/Qwen3-8B --backend metal --device 0 --kv-dtype fp8 --golden /opt/code/intarsiaml/golden/qwen3-8b-4k-fp32-top20.safetensors --mode all --decode-positions 64 -` |

Model fingerprint: `fnv1a64:e6a70935b68a5561`.

Build stamp `unstamped` means a plain `cargo build`: baselines written here gate this binary, but `--pre-change` provenance checks refuse them as a "before".


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/kernels.out` (kernels)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `logs/acc-long.out` (acc-long)
- `baselines/acc-long.baseline.json` (acc-long)
- `logs/acc-4k.out` (acc-4k)
- `state.json` (stage records; `--resume` reads it)
