# Qualification quick report: Qwen3.8-27B-MXFP4-mtpfp8 on cuda

> **Onboard verdict: ERROR.** recipe: PASS; smoke: PASS; quick: ERROR (qualify exit 2: acc-short FAIL: decode: 21-way KL (scored rows, n=118) 2.968e-1 (need <= 2.0e-1); acc-oracle MISSING-BASELINE: MISSING-BASELINE: no CPU-oracle golden for 'Qwen3.8-27B-MXFP4-mtpfp8' (--oracle-golden, or oracle/<model>-cpu-oracle*.safetensors in a golden directory). Qwen3.8-27B-MXFP4-mtpfp8 has no full-precision golden, so it is gated against the f64 CPU oracle; make it once (intarsia accuracy --model /home/nvidia/intarsiaml-od/models/Qwen3.8-27B-MXFP4-mtpfp8 --backend cpu --golden /home/nvidia/intarsiaml-od/goldens/qwen3.8-27b-mxfp4-hfquark-top20.safetensors --mode prefill --max-seq 8192 --prefill-chunk 256 --model-id Qwen3.8-27B-MXFP4-mtpfp8 --write-self-golden golden/oracle/qwen3.8-27b-mxfp4-mtpfp8-cpu-oracle.safetensors), commit it, and record this backend's dump with a qualify run on clean main; bench ERROR: bench speed timed out after 1800s and wrote no result:   size    128 rep: prompt=130 gen=128 ttft=13.76s prefill=9 tok/s decode=6.7 tok/s [qualify    844s] server up in 58s on http://127.0.0.1:8090 (max-seq 4608));

_Quick section, generated unix 1792979069, intarsia 0.0.1 build `head=d3b2380 tree=78239dd0d154 backend=cuda`._

## Headline

**Verdict: ERROR** (9 stages: 5 pass, 1 fail, 1 missing baseline, 1 error, 1 skipped)

| | |
|---|---|
| Model | Qwen3.8-27B-MXFP4-mtpfp8 (`/home/nvidia/intarsiaml-od/models/Qwen3.8-27B-MXFP4-mtpfp8`) |
| Backend / device | cuda / NVIDIA GB10 (sm_121) |
| Recipe | builtin:qwen3_5_vl covers this model (45/45 tensor templates resolve, quant mxfp4) |
| Decode | not measured |
| Prefill | not measured |
| Accuracy (short golden) | **FAIL** (prefill top-1 91.19% KL 1.97e-1; decode top-1 85.62% KL 2.48e-1) |
| Accuracy (format reference, same weights) | skipped: no format-reference golden for 'Qwen3.8-27B-MXFP4-mtpfp8' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| Quick gate | **PASS**: perplexity 1.58, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 44m04s |
| Reference | /home/nvidia/intarsiaml-od/goldens/qwen3.8-27b-mxfp4-hfquark-top20.safetensors |

### Failures

- `acc-short` **FAIL**: decode: 21-way KL (scored rows, n=118) 2.968e-1 (need <= 2.0e-1)
- `acc-oracle` **MISSING-BASELINE**: MISSING-BASELINE: no CPU-oracle golden for 'Qwen3.8-27B-MXFP4-mtpfp8' (--oracle-golden, or oracle/<model>-cpu-oracle*.safetensors in a golden directory). Qwen3.8-27B-MXFP4-mtpfp8 has no full-precision golden, so it is gated against the f64 CPU oracle; make it once (intarsia accuracy --model /home/nv
- `bench` **ERROR**: bench speed timed out after 1800s and wrote no result:   size    128 rep: prompt=130 gen=128 ttft=13.76s prefill=9 tok/s decode=6.7 tok/s

### Skipped

- `acc-format`: no format-reference golden for 'Qwen3.8-27B-MXFP4-mtpfp8' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE)

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 1s | NVIDIA GB10 (sm_121), 81 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3_5_vl covers this model (45/45 tensor templates resolve, quant mxfp4) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 73s | 19 ops resolved, exactness fast, loaded in 73.3s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 2m27s | 77 kernels, 1514 cases in 146.8s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 3m27s | perplexity 1.58, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | no format-reference golden for 'Qwen3.8-27B-MXFP4-mtpfp8' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| 7 | `acc-short` short accuracy against a golden | quick | FAIL | 5m56s | decode: 21-way KL (scored rows, n=118) 2.968e-1 (need <= 2.0e-1) |
| 8 | `acc-oracle` short prefill accuracy against the f64 CPU oracle (same weights, deterministic reference) | quick | MISSING-BASELINE | 0s | MISSING-BASELINE: no CPU-oracle golden for 'Qwen3.8-27B-MXFP4-mtpfp8' (--oracle-golden, or oracle/<model>-cpu-oracle*.safetensors in a golden directory). Qwen3.8-27B-MXFP4-mtpfp8 has no full-precision golden, so it is gated against the f64 CPU oracle; make it once (intarsia accuracy --model /home/nv |
| 9 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | ERROR | 30m58s | bench speed timed out after 1800s and wrote no result:   size    128 rep: prompt=130 gen=128 ttft=13.76s prefill=9 tok/s decode=6.7 tok/s |

### Notes

- `doctor`: warning disk_free:.: 270.1 GB free of 4032 GB (7% free)
- `doctor`: warning disk_free:/home/nvidia: 270.1 GB free of 4032 GB (7% free)
- `acc-short`: no baseline for backend cuda next to the golden or in golden/baselines: KL, tail, prefix-reuse gates only, top-1 not gated

## Speed

No speed numbers.

## Accuracy

**acc-short** against `/home/nvidia/intarsiaml-od/goldens/qwen3.8-27b-mxfp4-hfquark-top20.safetensors` (given with --golden; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1441 | 91.19 | 77.54 | 1.97e-1 | 1.96e-1 | 1.53e1 | 9.48e-3 | 0 | 9.5 | pass |
| decode | 160 | 85.62 | 81.25 | 2.48e-1 | 2.50e-1 | 7.14e0 | 1.30e-2 | 0 | 42.5 | FAIL |

**Quick gate**: perplexity 1.58, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3_5_vl covers this model (45/45 tensor templates resolve, quant mxfp4) (PASS)
- plan: 19 ops resolved, exactness fast, loaded in 73.3s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_bf16 | exact | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 1 |
| rmsnorm_add | rmsnorm_add_bf16 | deterministic | yes | 128 |
| qk_norm | qk_norm_bf16 | deterministic | yes | 32 |
| rope | rope_bf16 | deterministic | yes | 32 |
| logits | logits_bf16 | fast | yes | 0 |
| logits | logits_bf16_gemv | deterministic | yes | 1 |
| kv_append | kv_append_bf16 | exact | yes | 32 |
| attn_prefill | attn_prefill_bf16_flash | deterministic | yes | 0 |
| attn_prefill | attn_prefill_bf16_splitkv | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 16 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 64 |
| argmax | argmax_f32 | exact | yes | 1 |
| gate_split | gate_split_bf16 | deterministic | yes | 16 |
| sigmoid_mul | sigmoid_mul_bf16 | deterministic | yes | 16 |
| gdn_conv | gdn_conv_bf16 | deterministic | yes | 48 |
| gdn_delta_rule | gdn_delta_rule_bf16_split | deterministic | yes | 48 |
| gated_rmsnorm | gated_rmsnorm_bf16 | deterministic | yes | 48 |
| matmul_mxfp4 | matmul_mxfp4_gemv | deterministic | yes | 496 |

- kernels test: 77 kernels, 1514 cases in 146.8s (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | head=d3b2380 tree=78239dd0d154 backend=cuda | spark-4e77 | 1792976424 | `/home/nvidia/intarsiaml-od/tree/target/release/intarsia doctor --backend cuda --device 0 --out /home/nvidia/intarsiaml-od/runs/cuda1/report/qualify/doctor.json` |
| `recipe` | head=d3b2380 tree=78239dd0d154 backend=cuda | spark-4e77 | 1792976425 | `recipe::analyse(/home/nvidia/intarsiaml-od/models/Qwen3.8-27B-MXFP4-mtpfp8)` |
| `plan` | head=d3b2380 tree=78239dd0d154 backend=cuda | spark-4e77 | 1792976425 | `/home/nvidia/intarsiaml-od/tree/target/release/intarsia plan show --model /home/nvidia/intarsiaml-od/models/Qwen3.8-27B-MXFP4-mtpfp8 --backend cuda --device 0 --mem-budget 45 --admit --json` |
| `kernels` | head=d3b2380 tree=78239dd0d154 backend=cuda | spark-4e77 | 1792976499 | `/home/nvidia/intarsiaml-od/tree/target/release/intarsia kernels test --backend cuda --device 0 --model /home/nvidia/intarsiaml-od/models/Qwen3.8-27B-MXFP4-mtpfp8` |
| `gate` | head=d3b2380 tree=78239dd0d154 backend=cuda | spark-4e77 | 1792976646 | `/home/nvidia/intarsiaml-od/tree/target/release/intarsia accuracy --model /home/nvidia/intarsiaml-od/models/Qwen3.8-27B-MXFP4-mtpfp8 --backend cuda --device 0 --mem-budget 45 --quick --max-seq 2048` |
| `acc-format` | head=d3b2380 tree=78239dd0d154 backend=cuda | spark-4e77 | 1792976853 | `-` |
| `acc-short` | head=d3b2380 tree=78239dd0d154 backend=cuda | spark-4e77 | 1792976853 | `/home/nvidia/intarsiaml-od/tree/target/release/intarsia accuracy --model /home/nvidia/intarsiaml-od/models/Qwen3.8-27B-MXFP4-mtpfp8 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsiaml-od/goldens/qwen3.8-27b-mxfp4-hfquark-top20.safetensors --mode both --decode-positions 16 --ma` |
| `acc-oracle` | head=d3b2380 tree=78239dd0d154 backend=cuda | spark-4e77 | 1792977210 | `-` |
| `bench` | head=d3b2380 tree=78239dd0d154 backend=cuda | spark-4e77 | 1792977210 | `/home/nvidia/intarsiaml-od/tree/target/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3.8-27B-MXFP4-mtpfp8 --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-Qwen3.8-27B-MXFP4-mtpfp8-cuda-bench --out-dir /home/nvidia/intarsiaml-od/runs/cu` |

Model fingerprint: `fnv1a64:5985627d6885f76d`.


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/kernels.out` (kernels)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `state.json` (stage records; `--resume` reads it)

## Stage timeline

All times UTC. The qualify stages ran inside the `quick` / `in-depth` stages. A stage that was killed and resumed shows its last attempt; the qualify rows show every sub-stage with the time it actually ran.

| Stage | Tier | Verdict | Start | End | Duration | Note |
|---|---|---|---|---|---|---|
| recipe | prepare | PASS | 2026-10-26T00:59:37Z | 2026-10-26T00:59:37Z | 0s |  |
| smoke | prepare | PASS | 2026-10-26T00:59:37Z | 2026-10-26T01:00:24Z | 47s |  |
| quick | quick | ERROR | 2026-10-26T01:00:24Z | 2026-10-26T01:44:29Z | 44m04s |  |
| doctor | qualify quick | PASS | 2026-10-26T01:00:24Z | 2026-10-26T01:00:25Z | 1s |  |
| recipe | qualify quick | PASS | 2026-10-26T01:00:25Z | 2026-10-26T01:00:25Z | 0s |  |
| plan | qualify quick | PASS | 2026-10-26T01:00:25Z | 2026-10-26T01:01:38Z | 73s |  |
| kernels | qualify quick | PASS | 2026-10-26T01:01:39Z | 2026-10-26T01:04:06Z | 2m27s |  |
| gate | qualify quick | PASS | 2026-10-26T01:04:06Z | 2026-10-26T01:07:33Z | 3m27s |  |
| acc-format | qualify quick | SKIP | 2026-10-26T01:07:33Z | 2026-10-26T01:07:33Z | 0s |  |
| acc-short | qualify quick | FAIL | 2026-10-26T01:07:33Z | 2026-10-26T01:13:30Z | 5m56s |  |
| acc-oracle | qualify quick | MISSING-BASELINE | 2026-10-26T01:13:30Z | 2026-10-26T01:13:30Z | 0s |  |
| bench | qualify quick | ERROR | 2026-10-26T01:13:30Z | 2026-10-26T01:44:29Z | 30m58s |  |

_Published 2026-10-26T01:44:29Z by `intarsia onboard` (report directory `/home/nvidia/intarsiaml-od/runs/cuda1/report`)._
