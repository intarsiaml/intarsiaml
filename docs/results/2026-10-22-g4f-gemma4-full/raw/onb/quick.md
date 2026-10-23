# Qualification quick report: gemma4-26b-a4b-nvfp4 on cuda

> **Onboard verdict: FAIL.** recipe: PASS; smoke: PASS; quick: FAIL (qualify exit 1 (a gate failed): gate FAIL: perplexity: 533.484 over 1166 tokens (need <= 40), non-finite rows 0 FAIL | sanity FAIL: 48 tokens, max 4-gram repeat 19: non-empty | sanity FAIL: 3 tokens, max 4-gram repeat 0: non-empty, answers the fact ("blue"): NO | sanity FAIL: 48 tokens, max 4-gram repeat 45: non-empty, answers the fact ("paris"): NO | sanity FAIL: 48 tokens, max 4-gram repeat 12: non-empty, answers the fact ("oxygen"): NO; acc-short FAIL: prefill: 21-way KL (scored rows, n=1468) 2.953e-1 (need <= 3.0e-2); decode: 21-way KL (scored rows, n=160) 3.350e-1 (need <= 3.0e-2); acc-oracle MISSING-BASELINE: MISSING-BASELINE: no CPU-oracle golden for 'gemma4-26b-a4b-nvfp4' (--oracle-golden, or oracle/<model>-cpu-oracle*.safetensors in a golden directory). gemma4-26b-a4b-nvfp4 has no full-precision golden, so it is gated against the f64 CPU oracle; make it once (intarsia accuracy --model /home/nvidia/models/Gemma-4-26B-A4B-NVFP4 --backend cpu --golden /home/nvidia/g4-full/gemma4-26b.golden.safetensors --mode prefill --max-seq 8192 --prefill-chunk 256 --model-id gemma4-26b-a4b-nvfp4 --write-self-golden golden/oracle/gemma4-26b-a4b-nvfp4-cpu-oracle.safetensors), commit it, and record this backend's dump with a qualify run on clean main);

_Quick section, generated unix 1792713785, intarsia 0.0.1 build `unstamped`._

## Headline

**Verdict: FAIL** (9 stages: 5 pass, 2 fail, 1 missing baseline, 0 error, 1 skipped)

| | |
|---|---|
| Model | gemma4-26b-a4b-nvfp4 (`/home/nvidia/models/Gemma-4-26B-A4B-NVFP4`) |
| Backend / device | cuda / NVIDIA GB10 (sm_121) |
| Recipe | builtin:gemma4 covers this model (41/41 tensor templates resolve, quant nvfp4) |
| Decode | 128 tok: **34.3** tok/s (274% of floor); 1024 tok: **31.1** tok/s (248% of floor); 4096 tok: **24.6** tok/s (196% of floor) |
| Prefill | 128 tok: **634** tok/s; 1024 tok: **536** tok/s; 4096 tok: **366** tok/s |
| Accuracy (short golden) | **FAIL** (prefill top-1 82.97% KL 2.86e-1; decode top-1 80.62% KL 3.36e-1) |
| Accuracy (format reference, same weights) | skipped: no format-reference golden for 'gemma4-26b-a4b-nvfp4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| Quick gate | **FAIL**: perplexity: 533.484 over 1166 tokens (need <= 40), non-finite rows 0 FAIL \| sanity FAIL: 48 tokens, max 4-gram repeat 19: non-empty \| sanity FAIL: 3 tokens, max 4-gram repeat 0: non-empty, answers the fact ("blue"): NO \| sanity FAIL: 48 tokens, max 4-gram repeat 45: non-empty, answers the fact ("par |
| Wall time | quick 6m33s |
| Reference | /home/nvidia/g4-full/gemma4-26b.golden.safetensors |

### Failures

- `gate` **FAIL**: perplexity: 533.484 over 1166 tokens (need <= 40), non-finite rows 0 FAIL \| sanity FAIL: 48 tokens, max 4-gram repeat 19: non-empty \| sanity FAIL: 3 tokens, max 4-gram repeat 0: non-empty, answers the fact ("blue"): NO \| sanity FAIL: 48 tokens, max 4-gram repeat 45: non-empty, answers the fact ("par
- `acc-short` **FAIL**: prefill: 21-way KL (scored rows, n=1468) 2.953e-1 (need <= 3.0e-2); decode: 21-way KL (scored rows, n=160) 3.350e-1 (need <= 3.0e-2)
- `acc-oracle` **MISSING-BASELINE**: MISSING-BASELINE: no CPU-oracle golden for 'gemma4-26b-a4b-nvfp4' (--oracle-golden, or oracle/<model>-cpu-oracle*.safetensors in a golden directory). gemma4-26b-a4b-nvfp4 has no full-precision golden, so it is gated against the f64 CPU oracle; make it once (intarsia accuracy --model /home/nvidia/mod

### Skipped

- `acc-format`: no format-reference golden for 'gemma4-26b-a4b-nvfp4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE)

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 1s | NVIDIA GB10 (sm_121), 71 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:gemma4 covers this model (41/41 tensor templates resolve, quant nvfp4) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 72s | 21 ops resolved, exactness fast, loaded in 72.4s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 84s | 71 kernels, 1425 cases in 83.0s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | FAIL | 38s | perplexity: 533.484 over 1166 tokens (need <= 40), non-finite rows 0 FAIL \| sanity FAIL: 48 tokens, max 4-gram repeat 19: non-empty \| sanity FAIL: 3 tokens, max 4-gram repeat 0: non-empty, answers the fact ("blue"): NO \| sanity FAIL: 48 tokens, max 4-gram repeat 45: non-empty, answers the fact ("par |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | no format-reference golden for 'gemma4-26b-a4b-nvfp4' (make one with the format's reference implementation, e.g. tools/golden_exl3.py for EXL3, and give --format-golden FILE) |
| 7 | `acc-short` short accuracy against a golden | quick | FAIL | 46s | prefill: 21-way KL (scored rows, n=1468) 2.953e-1 (need <= 3.0e-2); decode: 21-way KL (scored rows, n=160) 3.350e-1 (need <= 3.0e-2) |
| 8 | `acc-oracle` short prefill accuracy against the f64 CPU oracle (same weights, deterministic reference) | quick | MISSING-BASELINE | 0s | MISSING-BASELINE: no CPU-oracle golden for 'gemma4-26b-a4b-nvfp4' (--oracle-golden, or oracle/<model>-cpu-oracle*.safetensors in a golden directory). gemma4-26b-a4b-nvfp4 has no full-precision golden, so it is gated against the f64 CPU oracle; make it once (intarsia accuracy --model /home/nvidia/mod |
| 9 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 2m32s | decode 34.3 tok/s at 128 tokens (274% of the byte floor) |

### Notes

- `doctor`: warning disk_free:.: 60.6 GB free of 4032 GB (2% free)
- `doctor`: warning disk_free:/home/nvidia: 60.6 GB free of 4032 GB (2% free)
- `acc-short`: no baseline for backend cuda next to the golden or in golden/baselines: KL, tail, prefix-reuse gates only, top-1 not gated
- `bench`: size 128: 274% of the byte floor is above 100%: the --gb-per-token estimate is too high or the run is not a plain decode (check the floor inputs)
- `bench`: size 1024: 248% of the byte floor is above 100%: the --gb-per-token estimate is too high or the run is not a plain decode (check the floor inputs)
- `bench`: size 4096: 196% of the byte floor is above 100%: the --gb-per-token estimate is too high or the run is not a plain decode (check the floor inputs)
- `bench`: bench warning: size 128: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 1024: 0 graph replays during the measured reps (decode ran eagerly)
- `bench`: bench warning: size 4096: 0 graph replays during the measured reps (decode ran eagerly)

## Speed

**quick bench** (floor: 17.64 GB/token at 221 GB/s; gb/token: estimated from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 221 GB/s, the GB10 measured figure)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 133 | 0.210 | 634 | 34.32 | 0.02 | 12.5 | 273.9 |
| 1014 | 1.892 | 536 | 31.14 | 0.01 | 12.5 | 248.5 |
| 4266 | 11.600 | 366 | 24.60 | 0.03 | 12.5 | 196.3 |

## Accuracy

**acc-short** against `/home/nvidia/g4-full/gemma4-26b.golden.safetensors` (given with --golden; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1468 | 82.97 | 79.20 | 2.86e-1 | 2.95e-1 | 1.91e1 | 2.46e-2 | 0 | 2.2 | FAIL |
| decode | 160 | 80.62 | 80.62 | 3.36e-1 | 3.35e-1 | 1.32e1 | 3.13e-2 | 0 | 30.0 | FAIL |

**Quick gate**: perplexity: 533.484 over 1166 tokens (need <= 40), non-finite rows 0 FAIL \| sanity FAIL: 48 tokens, max 4-gram repeat 19: non-empty \| sanity FAIL: 3 tokens, max 4-gram repeat 0: non-empty, answers the fact ("blue"): NO \| sanity FAIL: 48 tokens, max 4-gram repeat 45: non-empty, answers the fact ("par

## Plan and recipe

- recipe: builtin:gemma4 covers this model (41/41 tensor templates resolve, quant nvfp4) (PASS)
  - TODO model.norm_offset: family "gemma4" is of a lineage that uses zero-centred RMSNorm (x * (1 + w)) but no existing recipe of it was found; check the model's RMSNorm in transformers and set norm_offset = 1.0 if so
  - TODO layer: sliding-window attention (text_config.sliding_window = 1024 (use_sliding_window absent, so it applies)) is not implemented, a new attention block kind is needed
  - TODO layer: sliding-window attention (text_config.layer_types contains sliding_attention) is not implemented, a new attention block kind is needed
  - TODO layer: no attention or linear-attention block recognised (module groups in the layer: ["experts.0", "experts.1", "experts.10", "experts.100", "experts.101", "experts.102", "experts.103", "experts.104", "experts.105", "experts.106", "experts.107", "experts.108", "experts.109", "experts.11", "experts.110", "experts.111", "experts.112", "experts.113", "experts.114", "experts.115", "experts.116", "experts.117", "experts.118", "experts.119", "experts.12", "experts.120", "experts.121", "experts.122", "experts.123", "experts.124", "experts.125", "experts.126", "experts.127", "experts.13", "experts.14", "experts.15", "experts.16", "experts.17", "experts.18", "experts.19", "experts.2", "experts.20", "experts.21", "experts.22", "experts.23", "experts.24", "experts.25", "experts.26", "experts.27", "experts.28", "experts.29", "experts.3", "experts.30", "experts.31", "experts.32", "experts.33", "experts.34", "experts.35", "experts.36", "experts.37", "experts.38", "experts.39", "experts.4", "experts.40", "experts.41", "experts.42", "experts.43", "experts.44", "experts.45", "experts.46", "experts.47", "experts.48", "experts.49", "experts.5", "experts.50", "experts.51", "experts.52", "experts.53", "experts.54", "experts.55", "experts.56", "experts.57", "experts.58", "experts.59", "experts.6", "experts.60", "experts.61", "experts.62", "experts.63", "experts.64", "experts.65", "experts.66", "experts.67", "experts.68", "experts.69", "experts.7", "experts.70", "experts.71", "experts.72", "experts.73", "experts.74", "experts.75", "experts.76", "experts.77", "experts.78", "experts.79", "experts.8", "experts.80", "experts.81", "experts.82", "experts.83", "experts.84", "experts.85", "experts.86", "experts.87", "experts.88", "experts.89", "experts.9", "experts.90", "experts.91", "experts.92", "experts.93", "experts.94", "experts.95", "experts.96", "experts.97", "experts.98", "experts.99", "mlp", "router", "self_attn"]); a new block or a different tensor layout is needed
  - TODO layer.ffn.act: text_config.hidden_act = "gelu_pytorch_tanh"; only silu (silu_glu) is implemented, a new activation is needed in ops.rs and the ffn block
  - TODO top: 1 checkpoint tensor(s) outside the layer stack are claimed by no block, e.g. ["model.embed_vision.embedding_projection.weight"]; decide whether the model needs them
  - TODO layer: 1542 per-layer tensor(s) are claimed by no block, e.g. ["experts.1.down_proj.input_scale", "experts.1.down_proj.weight", "experts.1.down_proj.weight_scale", "experts.1.down_proj.weight_scale_2", "experts.1.gate_proj.input_scale", "experts.1.gate_proj.weight"] ... (relative to model.language_model.layers.<i>.); extra norms, biases or unknown modules need a block or an explicit decision to ignore them
  - TODO quant: quant_algo "NVFP4" (NVFP4 weights AND 4-bit activations) declared in hf_quant_config.json; the engine implements only weight-only W4A16_NVFP4
  - TODO quant: unsupported: 'model.language_model.layers.0.experts.0.gate_proj' is declared quant_algo 'NVFP4' (NVFP4 weights AND 4-bit activations) in hf_quant_config.json; this engine implements the weight-only W4A16_NVFP4 scheme and would run it with 16-bit activations, different from the checkpoint's declared scheme and from vLLM
  - TODO quant: unsupported: 'model.language_model.layers.0.experts.0.up_proj' is declared quant_algo 'NVFP4' (NVFP4 weights AND 4-bit activations) in hf_quant_config.json; this engine implements the weight-only W4A16_NVFP4 scheme and would run it with 16-bit activations, different from the checkpoint's declared scheme and from vLLM
  - TODO quant: unsupported: 'model.language_model.layers.0.experts.0.down_proj' is declared quant_algo 'NVFP4' (NVFP4 weights AND 4-bit activations) in hf_quant_config.json; this engine implements the weight-only W4A16_NVFP4 scheme and would run it with 16-bit activations, different from the checkpoint's declared scheme and from vLLM
- plan: 21 ops resolved, exactness fast, loaded in 72.4s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_scaled_bf16 | deterministic | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 241 |
| add | add_bf16 | deterministic | yes | 60 |
| add | add_scaled_bf16 | deterministic | yes | 30 |
| qk_norm | qk_norm_bf16 | deterministic | yes | 90 |
| rope | rope_bf16 | deterministic | yes | 50 |
| rope | rope_prop_bf16 | deterministic | yes | 10 |
| matmul | matmul_bf16 | fast | yes | 0 |
| matmul | matmul_bf16_gemv | deterministic | yes | 235 |
| logits | logits_bf16 | fast | yes | 0 |
| logits | logits_bf16_gemv | deterministic | yes | 1 |
| kv_append | kv_append_bf16 | exact | yes | 60 |
| attn_prefill | attn_prefill_bf16_ext | deterministic | yes | 0 |
| attn_prefill | attn_prefill_bf16_splitkv_ext | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_ext | deterministic | yes | 5 |
| attn_decode | attn_decode_bf16_splitkv_ext | deterministic | yes | 25 |
| argmax | argmax_f32 | exact | yes | 1 |
| moe_topk | moe_topk_bf16 | deterministic | yes | 30 |
| moe_experts | moe_experts_decode_v2_gelu_bf16 | deterministic | yes | 30 |
| moe_experts | moe_experts_gelu_bf16 | deterministic | yes | 0 |
| gelu_glu | gelu_glu_bf16 | deterministic | yes | 30 |

- kernels test: 71 kernels, 1425 cases in 83.0s (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | unstamped | spark-fa00 | 1792713391 | `/home/nvidia/g4-main-target/release/intarsia doctor --backend cuda --device 0 --out /home/nvidia/g4-full/onb/qualify/doctor.json` |
| `recipe` | unstamped | spark-fa00 | 1792713392 | `recipe::analyse(/home/nvidia/models/Gemma-4-26B-A4B-NVFP4)` |
| `plan` | unstamped | spark-fa00 | 1792713392 | `/home/nvidia/g4-main-target/release/intarsia plan show --model /home/nvidia/models/Gemma-4-26B-A4B-NVFP4 --backend cuda --device 0 --mem-budget 40 --admit --json` |
| `kernels` | unstamped | spark-fa00 | 1792713464 | `/home/nvidia/g4-main-target/release/intarsia kernels test --backend cuda --device 0 --model /home/nvidia/models/Gemma-4-26B-A4B-NVFP4` |
| `gate` | unstamped | spark-fa00 | 1792713548 | `/home/nvidia/g4-main-target/release/intarsia accuracy --model /home/nvidia/models/Gemma-4-26B-A4B-NVFP4 --backend cuda --device 0 --mem-budget 40 --quick --max-seq 2048` |
| `acc-format` | unstamped | spark-fa00 | 1792713586 | `-` |
| `acc-short` | unstamped | spark-fa00 | 1792713586 | `/home/nvidia/g4-main-target/release/intarsia accuracy --model /home/nvidia/models/Gemma-4-26B-A4B-NVFP4 --backend cuda --device 0 --mem-budget 40 --golden /home/nvidia/g4-full/gemma4-26b.golden.safetensors --mode both --decode-positions 16 --max-seq 4096 --write-baseline /home/nvidia/g4-full/onb/qua` |
| `acc-oracle` | unstamped | spark-fa00 | 1792713632 | `-` |
| `bench` | unstamped | spark-fa00 | 1792713632 | `/home/nvidia/g4-main-target/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-gemma4-26b-a4b-nvfp4 --sizes 128,1024,4096 --gen 128 --reps 3 --warmup 1 --thinking off --label qualify-gemma4-26b-a4b-nvfp4-cuda-bench --out-dir /home/nvidia/g4-full/onb/qualify/bench --tp 1 --finge` |

Model fingerprint: `fnv1a64:18477c8fdb281732`.

Build stamp `unstamped` means a plain `cargo build`: baselines written here gate this binary, but `--pre-change` provenance checks refuse them as a "before".


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/kernels.out` (kernels)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `bench/qualify-gemma4-26b-a4b-nvfp4-cuda-bench.json` (bench)
- `logs/bench-bench.out` (bench)
- `logs/bench-server.err` (bench)
- `state.json` (stage records; `--resume` reads it)

## Stage timeline

All times UTC. The qualify stages ran inside the `quick` / `in-depth` stages. A stage that was killed and resumed shows its last attempt; the qualify rows show every sub-stage with the time it actually ran.

| Stage | Tier | Verdict | Start | End | Duration | Note |
|---|---|---|---|---|---|---|
| recipe | prepare | PASS | 2026-10-22T23:56:03Z | 2026-10-22T23:56:03Z | 0s |  |
| smoke | prepare | PASS | 2026-10-22T23:56:03Z | 2026-10-22T23:56:31Z | 28s |  |
| quick | quick | FAIL | 2026-10-22T23:56:31Z | 2026-10-23T00:03:05Z | 6m34s |  |
| doctor | qualify quick | PASS | 2026-10-22T23:56:31Z | 2026-10-22T23:56:32Z | 1s |  |
| recipe | qualify quick | PASS | 2026-10-22T23:56:32Z | 2026-10-22T23:56:32Z | 0s |  |
| plan | qualify quick | PASS | 2026-10-22T23:56:32Z | 2026-10-22T23:57:44Z | 72s |  |
| kernels | qualify quick | PASS | 2026-10-22T23:57:44Z | 2026-10-22T23:59:08Z | 84s |  |
| gate | qualify quick | FAIL | 2026-10-22T23:59:08Z | 2026-10-22T23:59:46Z | 38s |  |
| acc-format | qualify quick | SKIP | 2026-10-22T23:59:46Z | 2026-10-22T23:59:46Z | 0s |  |
| acc-short | qualify quick | FAIL | 2026-10-22T23:59:46Z | 2026-10-23T00:00:32Z | 46s |  |
| acc-oracle | qualify quick | MISSING-BASELINE | 2026-10-23T00:00:32Z | 2026-10-23T00:00:32Z | 0s |  |
| bench | qualify quick | PASS | 2026-10-23T00:00:32Z | 2026-10-23T00:03:05Z | 2m32s |  |

_Published 2026-10-23T00:03:05Z by `intarsia onboard` (report directory `/home/nvidia/g4-full/onb`)._
