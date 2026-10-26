# Qualification quick report: Qwen3.8-27B-MXFP4-mtpfp8 on hip

_Quick section, generated unix 1792982318, intarsia 0.0.1 build `head=6c470ef code-tree=526c6e845aab dirty=no`._

## Headline

**Verdict: MISSING-BASELINE** (9 stages: 7 pass, 0 fail, 1 missing baseline, 0 error, 1 skipped)

| | |
|---|---|
| Model | Qwen3.8-27B-MXFP4-mtpfp8 (`/models/radiance/Qwen3.8-27B-MXFP4-mtpfp8`) |
| Backend / device | hip / AMD Radeon Graphics (gfx1201) |
| Recipe | builtin:qwen3_5_vl covers this model (45/45 tensor templates resolve, quant mxfp4) |
| Decode | 128 tok: **33.1** tok/s (82% of floor); 1024 tok: **32.9** tok/s (81% of floor); 4096 tok: **32.3** tok/s (80% of floor) |
| Prefill | 128 tok: **1055** tok/s; 1024 tok: **1804** tok/s; 4096 tok: **1854** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 91.95% KL 1.72e-1; decode top-1 88.12% KL 2.11e-1) |
| Accuracy (format reference, same weights) | skipped: no format-reference golden for 'Qwen3.8-27B-MXFP4-mtpfp8' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| Quick gate | **PASS**: perplexity 1.59, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 4m47s |

### Failures

- `acc-oracle` **MISSING-BASELINE**: MISSING-BASELINE: no CPU-oracle golden for 'Qwen3.8-27B-MXFP4-mtpfp8' (--oracle-golden, or oracle/<model>-cpu-oracle*.safetensors in a golden directory). Qwen3.8-27B-MXFP4-mtpfp8 has no full-precision golden, so it is gated against the f64 CPU oracle; make it once (intarsia accuracy --model /models/

### Skipped

- `acc-format`: no format-reference golden for 'Qwen3.8-27B-MXFP4-mtpfp8' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE)

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 1s | AMD Radeon Graphics (gfx1201), 80 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_vl covers this model (45/45 tensor templates resolve, quant mxfp4) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 33s | 20 ops resolved, exactness deterministic, loaded in 33.5s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 2m35s | 76 kernels, 1465 cases in 154.1s; incomplete: 10 kernel(s) had no cases in this sweep (untested, not failed; plan-time admission tests them) |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 12s | perplexity 1.59, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | no format-reference golden for 'Qwen3.8-27B-MXFP4-mtpfp8' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| 7 | `acc-short` short accuracy against a golden | quick | PASS | 13s | prefill top-1 91.95% KL 1.72e-1; decode top-1 88.12% KL 2.11e-1 |
| 8 | `acc-oracle` short prefill accuracy against the f64 CPU oracle (same weights, deterministic reference) | quick | MISSING-BASELINE | 0s | MISSING-BASELINE: no CPU-oracle golden for 'Qwen3.8-27B-MXFP4-mtpfp8' (--oracle-golden, or oracle/<model>-cpu-oracle*.safetensors in a golden directory). Qwen3.8-27B-MXFP4-mtpfp8 has no full-precision golden, so it is gated against the f64 CPU oracle; make it once (intarsia accuracy --model /models/ |
| 9 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 73s | decode 33.1 tok/s at 128 tokens (82% of the byte floor) |

### Notes

- `acc-short`: gated against the recorded baseline /work/golden/baselines/qwen3.8-27b-mxfp4-hfquark-top20.hip.baseline.json (commit 092c961) and its noise envelope

## Speed

**quick bench** (floor: 15.48 GB/token at 627 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 627 GB/s, the R9700 figure of docs/results/2026-10-13-phase2a-hip.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 125 | 0.116 | 1055 | 33.06 | 0.02 | 40.5 | 81.6 |
| 981 | 0.544 | 1804 | 32.90 | 0.01 | 40.5 | 81.3 |
| 4014 | 2.164 | 1854 | 32.34 | 0.01 | 40.5 | 79.9 |

## Accuracy

**acc-short** against `/models/intarsia/golden/qwen3.8-27b-mxfp4-hfquark-top20.safetensors` (given with --golden; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1441 | 91.95 | 77.72 | 1.72e-1 | 1.71e-1 | 1.46e1 | 9.28e-3 | 0 | 3.5 | pass |
| decode | 160 | 88.12 | 82.38 | 2.11e-1 | 2.27e-1 | 1.43e1 | 2.07e-2 | 0 | 51.9 | FAIL |

**Quick gate**: perplexity 1.59, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3_5_vl covers this model (45/45 tensor templates resolve, quant mxfp4) (PASS)
- plan: 20 ops resolved, exactness deterministic, loaded in 33.5s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_bf16 | exact | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 1 |
| rmsnorm_add | rmsnorm_add_bf16 | deterministic | yes | 128 |
| logits | logits_bf16 | deterministic | yes | 0 |
| logits | logits_bf16_gemv | deterministic | yes | 1 |
| attn_prefill | attn_prefill_bf16_flash_v2 | deterministic | yes | 0 |
| attn_prefill | attn_prefill_bf16_splitkv | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 16 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 64 |
| argmax | argmax_f32 | exact | yes | 1 |
| sigmoid_mul | sigmoid_mul_bf16 | deterministic | yes | 16 |
| gdn_conv | gdn_conv_bf16 | deterministic | yes | 0 |
| gdn_delta_rule | gdn_delta_rule_bf16_v2 | deterministic | yes | 0 |
| gdn_delta_rule | gdn_delta_rule_bf16_v3 | deterministic | yes | 0 |
| gated_rmsnorm | gated_rmsnorm_bf16 | deterministic | yes | 48 |
| gate_split_qk_norm_rope_kv | gate_split_qk_norm_rope_kv_bf16 | deterministic | yes | 16 |
| gdn_conv_delta | gdn_conv_delta_bf16 | deterministic | yes | 48 |
| matmul_mxfp4 | matmul_mxfp4_gemv | deterministic | yes | 144 |
| matmul_mxfp4 | matmul_mxfp4_mma_fold | deterministic | yes | 0 |
| matmul_mxfp4_pair | matmul_mxfp4_pair_gemv | deterministic | yes | 176 |

- kernels test: 76 kernels, 1465 cases in 154.1s; incomplete: 10 kernel(s) had no cases in this sweep (untested, not failed; plan-time admission tests them) (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | head=6c470ef code-tree=526c6e845aab dirty=no | tatooine | 1792982030 | `/work/od-intarsia doctor --backend hip --device 0 --out /work/onboard-runs/hip1/report/qualify/doctor.json` |
| `recipe` | head=6c470ef code-tree=526c6e845aab dirty=no | tatooine | 1792982032 | `recipe::analyse(/models/radiance/Qwen3.8-27B-MXFP4-mtpfp8)` |
| `plan` | head=6c470ef code-tree=526c6e845aab dirty=no | tatooine | 1792982032 | `/work/od-intarsia plan show --model /models/radiance/Qwen3.8-27B-MXFP4-mtpfp8 --backend hip --device 0 --mem-budget 29 --admit --json` |
| `kernels` | head=6c470ef code-tree=526c6e845aab dirty=no | tatooine | 1792982065 | `/work/od-intarsia kernels test --backend hip --device 0 --model /models/radiance/Qwen3.8-27B-MXFP4-mtpfp8` |
| `gate` | head=6c470ef code-tree=526c6e845aab dirty=no | tatooine | 1792982220 | `/work/od-intarsia accuracy --model /models/radiance/Qwen3.8-27B-MXFP4-mtpfp8 --backend hip --device 0 --mem-budget 29 --quick --max-seq 2048` |
| `acc-format` | head=6c470ef code-tree=526c6e845aab dirty=no | tatooine | 1792982318 | `-` |
| `acc-short` | head=6c470ef code-tree=526c6e845aab dirty=no | tatooine | 1792982233 | `/work/od-intarsia accuracy --model /models/radiance/Qwen3.8-27B-MXFP4-mtpfp8 --backend hip --device 0 --mem-budget 29 --golden /models/intarsia/golden/qwen3.8-27b-mxfp4-hfquark-top20.safetensors --mode both --decode-positions 16 --max-seq 4096 --write-baseline /work/onboard-runs/hip1/report/qualify/` |
| `acc-oracle` | head=6c470ef code-tree=526c6e845aab dirty=no | tatooine | 1792982318 | `-` |
| `bench` | head=6c470ef code-tree=526c6e845aab dirty=no | tatooine | 1792982246 | `/work/od-intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3.8-27B-MXFP4-mtpfp8 --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-Qwen3.8-27B-MXFP4-mtpfp8-hip-bench --out-dir /work/onboard-runs/hip1/report/qualify/bench --tp 1 --fingerprint model=Q` |

Model fingerprint: `fnv1a64:5985627d6885f76d`.


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/kernels.out` (kernels)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `bench/qualify-Qwen3.8-27B-MXFP4-mtpfp8-hip-bench.json` (bench)
- `logs/bench-bench.out` (bench)
- `logs/bench-server.err` (bench)
- `state.json` (stage records; `--resume` reads it)
