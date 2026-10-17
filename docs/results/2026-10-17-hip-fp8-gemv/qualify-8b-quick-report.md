# Qualification quick report: Qwen3-8B on hip

_Quick section, generated unix 1792253854, intarsia 0.0.1 build `unstamped`._

## Headline

**Verdict: PASS** (7 stages: 7 pass, 0 fail, 0 error, 0 skipped)

| | |
|---|---|
| Model | Qwen3-8B (`/models/Qwen3-8B`) |
| Backend / device | hip / AMD Radeon Graphics (gfx1201) |
| Recipe | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| Decode | 128 tok: **38.0** tok/s (92% of floor); 1024 tok: **37.4** tok/s (90% of floor); 4096 tok: **36.2** tok/s (87% of floor) |
| Prefill | 128 tok: **2081** tok/s; 1024 tok: **3436** tok/s; 4096 tok: **3005** tok/s |
| Accuracy (short golden) | **PASS** (prefill top-1 100.00% KL 2.34e-4; decode top-1 100.00% KL 3.63e-4) |
| Quick gate | **PASS**: perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows |
| Wall time | quick 3m44s |

### Failures

None.

### Skipped

None.

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 1s | AMD Radeon Graphics (gfx1201), 38 kernels registered |
| 2 | `recipe` recipe resolve | quick | PASS | 0s | builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | PASS | 54s | 16 ops resolved, exactness deterministic, loaded in 53.6s |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 48s | 38 kernels, 626 cases in 47.8s |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | PASS | 9s | perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows |
| 6 | `acc-short` short accuracy against a golden | quick | PASS | 8s | prefill top-1 100.00% KL 2.34e-4; decode top-1 100.00% KL 3.63e-4 |
| 7 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | PASS | 1m45s | decode 38.0 tok/s at 128 tokens (92% of the byte floor) |

### Notes

None.

## Speed

**quick bench** (floor: 15.14 GB/token at 627 GB/s; gb/token: from checkpoint bytes read per token (experts scaled by top_k/experts, input embedding excluded when the head is separate); bandwidth: 627 GB/s, the R9700 figure of docs/results/2026-10-13-phase2a-hip.md)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| 126 | 0.060 | 2081 | 37.97 | 0.01 | 41.4 | 91.7 |
| 1022 | 0.294 | 3436 | 37.41 | 0.01 | 41.4 | 90.3 |
| 4174 | 1.386 | 3005 | 36.22 | 0.01 | 41.4 | 87.4 |

## Accuracy

**acc-short** against `/models/golden/qwen3-8b-fp32.safetensors` (found by model name; gate class bf16)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 434 | 100.00 | 98.80 | 2.34e-4 | 1.70e-4 | 6.16e-3 | 4.54e-4 | 0 | 100.0 | pass |
| decode | 128 | 100.00 | 98.12 | 3.63e-4 | 3.17e-4 | 8.26e-3 | 3.13e-4 | 0 | 100.0 | pass |

**Quick gate**: perplexity 1.85, 5/5 sanity prompts ok, no non-finite rows

## Plan and recipe

- recipe: builtin:qwen3 covers this model (25/25 tensor templates resolve, quant bf16) (PASS)
- plan: 16 ops resolved, exactness deterministic, loaded in 53.6s (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| embedding | embedding_bf16 | exact | yes | 1 |
| rmsnorm | rmsnorm_bf16 | deterministic | yes | 1 |
| rmsnorm_add | rmsnorm_add_bf16 | deterministic | yes | 72 |
| qk_norm | qk_norm_bf16 | deterministic | yes | 72 |
| rope | rope_bf16 | deterministic | yes | 72 |
| matmul | matmul_bf16 | deterministic | yes | 0 |
| matmul | matmul_bf16_gemv | deterministic | yes | 108 |
| logits | logits_bf16 | deterministic | yes | 0 |
| logits | logits_bf16_gemv | deterministic | yes | 1 |
| kv_append | kv_append_bf16 | exact | yes | 72 |
| attn_prefill | attn_prefill_bf16_flash | deterministic | yes | 0 |
| attn_prefill | attn_prefill_bf16_splitkv | deterministic | yes | 0 |
| attn_decode | attn_decode_bf16_splitkv | deterministic | yes | 36 |
| silu_glu | silu_glu_bf16 | deterministic | yes | 36 |
| argmax | argmax_f32 | exact | yes | 1 |
| matmul_pair | matmul_pair_bf16_gemv | deterministic | yes | 72 |

- kernels test: 38 kernels, 626 cases in 47.8s (PASS)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | unstamped | tatooine | 1792253630 | `/target/release/intarsia doctor --backend hip --device 0 --out /work/bench-out/qualify-8b-pair1/doctor.json` |
| `recipe` | unstamped | tatooine | 1792253630 | `recipe::analyse(/models/Qwen3-8B)` |
| `plan` | unstamped | tatooine | 1792253630 | `/target/release/intarsia plan show --model /models/Qwen3-8B --backend hip --device 0 --mem-budget 29 --admit --json` |
| `kernels` | unstamped | tatooine | 1792253684 | `/target/release/intarsia kernels test --backend hip --device 0 --model /models/Qwen3-8B` |
| `gate` | unstamped | tatooine | 1792253732 | `/target/release/intarsia accuracy --model /models/Qwen3-8B --backend hip --device 0 --mem-budget 29 --quick --max-seq 2048` |
| `acc-short` | unstamped | tatooine | 1792253741 | `/target/release/intarsia accuracy --model /models/Qwen3-8B --backend hip --device 0 --mem-budget 29 --golden /models/golden/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq 4096 --write-baseline /work/bench-out/qualify-8b-pair1/baselines/acc-short.baseline.json` |
| `bench` | unstamped | tatooine | 1792253748 | `/target/release/intarsia bench speed --url http://127.0.0.1:8090 --model qualify-Qwen3-8B --sizes 128,1024,4096 --gen 128 --reps 5 --warmup 1 --thinking off --label qualify-Qwen3-8B-hip-bench --out-dir /work/bench-out/qualify-8b-pair1/bench --tp 1 --fingerprint model=Qwen3-8B --fingerprint backend=h` |

Model fingerprint: `fnv1a64:e6a70935b68a5561`.

Build stamp `unstamped` means a plain `cargo build`: baselines written here gate this binary, but `--pre-change` provenance checks refuse them as a "before".


## Artifacts

- `doctor.json` (doctor)
- `plan.json` (plan)
- `logs/kernels.out` (kernels)
- `logs/gate.out` (gate)
- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `bench/qualify-Qwen3-8B-hip-bench.json` (bench)
- `logs/bench-bench.out` (bench)
- `logs/bench-server.err` (bench)
- `state.json` (stage records; `--resume` reads it)
