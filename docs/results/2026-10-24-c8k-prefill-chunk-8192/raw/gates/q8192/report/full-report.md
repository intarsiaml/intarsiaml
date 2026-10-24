# Qualification report: qwen36-35b-a3b-nvfp4 on cuda

_Full (quick + in-depth) section, generated unix 1792835644, intarsia 0.0.1 build `head=6189466 code-tree=9cac1d016898 dirty=yes`._

## Headline

**Verdict: FAIL** (15 stages: 2 pass, 1 fail, 0 error, 12 skipped)

| | |
|---|---|
| Model | qwen36-35b-a3b-nvfp4 (`/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5`) |
| Backend / device | cuda / ? (?) |
| Recipe | skipped by --skip |
| Decode | not measured |
| Prefill | not measured |
| Accuracy (short golden) | **PASS** (prefill top-1 98.27% KL 6.83e-3; decode top-1 99.38% KL 6.99e-3) |
| Accuracy (format reference, same weights) | skipped: skipped by --skip |
| Accuracy (long golden) | **PASS** (prefill top-1 94.35% KL 4.28e-2; decode top-1 97.50% KL 1.04e-2) |
| Accuracy (4k golden) | **FAIL** (prefill top-1 92.88% KL 8.76e-2; decode top-1 97.92% KL 1.01e-2) |
| Quick gate | **SKIP**: skipped by --skip |
| Wall time | quick 38s, in-depth 3m58s, total 4m36s |

### Failures

- `acc-4k` **FAIL**: prefix reuse: nothing ran: no prompt can cross a chunk boundary at prefill chunk 8192, so no restore was tested and this is not a pass; prefix reuse differs from a cold run

### Skipped

- `doctor`: skipped by --skip
- `recipe`: skipped by --skip
- `plan`: skipped by --skip
- `kernels`: skipped by --skip
- `gate`: skipped by --skip
- `acc-format`: skipped by --skip
- `acc-oracle`: skipped by --skip
- `bench`: skipped by --skip
- `envelope`: skipped by --skip
- `bench-long`: skipped by --skip
- `ab`: skipped by --skip
- `tp`: skipped by --skip

## Stages

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | SKIP | 0s | skipped by --skip |
| 2 | `recipe` recipe resolve | quick | SKIP | 0s | skipped by --skip |
| 3 | `plan` plan show: load the model, resolve a kernel per op | quick | SKIP | 0s | skipped by --skip |
| 4 | `kernels` kernels test on the model's shapes | quick | SKIP | 0s | skipped by --skip |
| 5 | `gate` quick gate: sanity prompts, perplexity, non-finite | quick | SKIP | 0s | skipped by --skip |
| 6 | `acc-format` short accuracy against the format's own reference (same quantized weights) | quick | SKIP | 0s | skipped by --skip |
| 7 | `acc-short` short accuracy against a golden | quick | PASS | 38s | prefill top-1 98.27% KL 6.83e-3; decode top-1 99.38% KL 6.99e-3 |
| 8 | `acc-oracle` short prefill accuracy against the f64 CPU oracle (same weights, deterministic reference) | quick | SKIP | 0s | skipped by --skip |
| 9 | `bench` bench speed 128 / 1024 / 4096 with the byte floor | quick | SKIP | 0s | skipped by --skip |
| 10 | `acc-long` accuracy --mode all on the long golden (prefill, decode, prefix reuse) | full | PASS | 1m46s | prefill top-1 94.35% KL 4.28e-2; decode top-1 97.50% KL 1.04e-2 |
| 11 | `acc-4k` accuracy --mode all on the 4k golden (prefill, decode, prefix reuse past 2 chunks of 2048) | full | FAIL | 2m12s | prefix reuse: nothing ran: no prompt can cross a chunk boundary at prefill chunk 8192, so no restore was tested and this is not a pass; prefix reuse differs from a cold run |
| 12 | `envelope` noise envelope and baseline for regression gating | full | SKIP | 0s | skipped by --skip |
| 13 | `bench-long` long-context bench up to 16k | full | SKIP | 0s | skipped by --skip |
| 14 | `ab` A/B against the reference server via localeval | full | SKIP | 0s | skipped by --skip |
| 15 | `tp` tensor-parallel variant | full | SKIP | 0s | skipped by --skip |

### Notes

- `acc-short`: baseline /home/nvidia/intarsia-decfuse/pf-golden-c8k/baselines/qwen36-35b-a3b-nvfp4-vllm-top20.cuda.baseline.json not used (recorded at prefill chunk 2048, this run uses ?): KL, tail, prefix-reuse gates only, top-1 not gated
- `acc-long`: prefix reuse is gated in acc-4k, on the 4k golden
- `acc-4k`: prefix reuse at chunk 1024: 3 judged, 0 skipped
- `acc-4k`: no baseline for backend cuda next to the golden or in golden/baselines: KL, tail, prefix-reuse gates only, top-1 not gated

## Speed

No speed numbers.

## Accuracy

**acc-short** against `/home/nvidia/intarsia-decfuse/pf-golden-c8k/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 1441 | 98.27 | 91.56 | 6.83e-3 | 6.65e-3 | 1.04e0 | 1.62e-3 | 0 | 4.8 | pass |
| decode | 160 | 99.38 | 93.50 | 6.99e-3 | 7.34e-3 | 1.10e-1 | 2.89e-3 | 0 | 90.0 | pass |

**acc-long** against `/home/nvidia/intarsia-decfuse/pf-golden-c8k/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 36887 | 94.35 | 90.38 | 4.28e-2 | 4.59e-2 | 2.51e1 | 7.53e-3 | 0 | 0.3 | pass |
| decode | 320 | 97.50 | 90.31 | 1.04e-2 | 9.67e-3 | 1.88e-1 | 2.43e-3 | 0 | 73.4 | pass |

**acc-4k** against `/home/nvidia/intarsia-decfuse/pf-golden-c8k/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors` (found by model name; gate class 4bit)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | |tail diff| | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 12759 | 92.88 | 90.00 | 8.76e-2 | 1.02e-1 | 2.56e1 | 1.23e-2 | 0 | 0.7 | pass |
| decode | 192 | 97.92 | 90.94 | 1.01e-2 | 9.72e-3 | 1.81e-1 | 1.60e-3 | 0 | 57.3 | pass |

**Quick gate**: skipped by --skip

## Plan and recipe

- recipe: skipped by --skip (SKIP)
- plan: skipped by --skip (SKIP)

- kernels test: skipped by --skip (SKIP)

## Provenance

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `doctor` | head=6189466 code-tree=9cac1d016898 dirty=yes | spark-4e77 | 0 | `-` |
| `recipe` | head=6189466 code-tree=9cac1d016898 dirty=yes | spark-4e77 | 0 | `-` |
| `plan` | head=6189466 code-tree=9cac1d016898 dirty=yes | spark-4e77 | 0 | `-` |
| `kernels` | head=6189466 code-tree=9cac1d016898 dirty=yes | spark-4e77 | 0 | `-` |
| `gate` | head=6189466 code-tree=9cac1d016898 dirty=yes | spark-4e77 | 0 | `-` |
| `acc-format` | head=6189466 code-tree=9cac1d016898 dirty=yes | spark-4e77 | 0 | `-` |
| `acc-short` | head=6189466 code-tree=9cac1d016898 dirty=yes | spark-4e77 | 1792835367 | `/home/nvidia/intarsia-decfuse/c8k-r8192/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsia-decfuse/pf-golden` |
| `acc-oracle` | head=6189466 code-tree=9cac1d016898 dirty=yes | spark-4e77 | 0 | `/home/nvidia/intarsia-decfuse/c8k-r8192/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsia-decfuse/pf-golden` |
| `bench` | head=6189466 code-tree=9cac1d016898 dirty=yes | spark-4e77 | 0 | `/home/nvidia/intarsia-decfuse/c8k-r8192/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsia-decfuse/pf-golden` |
| `acc-long` | head=6189466 code-tree=9cac1d016898 dirty=yes | spark-4e77 | 1792835405 | `/home/nvidia/intarsia-decfuse/c8k-r8192/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsia-decfuse/pf-golden` |
| `acc-4k` | head=6189466 code-tree=9cac1d016898 dirty=yes | spark-4e77 | 1792835511 | `/home/nvidia/intarsia-decfuse/c8k-r8192/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsia-decfuse/pf-golden` |
| `envelope` | head=6189466 code-tree=9cac1d016898 dirty=yes | spark-4e77 | 0 | `/home/nvidia/intarsia-decfuse/c8k-r8192/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsia-decfuse/pf-golden` |
| `bench-long` | head=6189466 code-tree=9cac1d016898 dirty=yes | spark-4e77 | 0 | `/home/nvidia/intarsia-decfuse/c8k-r8192/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsia-decfuse/pf-golden` |
| `ab` | head=6189466 code-tree=9cac1d016898 dirty=yes | spark-4e77 | 0 | `/home/nvidia/intarsia-decfuse/c8k-r8192/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsia-decfuse/pf-golden` |
| `tp` | head=6189466 code-tree=9cac1d016898 dirty=yes | spark-4e77 | 0 | `/home/nvidia/intarsia-decfuse/c8k-r8192/target/acc/release/intarsia accuracy --model /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 --backend cuda --device 0 --mem-budget 45 --golden /home/nvidia/intarsia-decfuse/pf-golden` |

Model fingerprint: `fnv1a64:25b7234fc8ebc6ee`.


## Artifacts

- `logs/acc-short.out` (acc-short)
- `baselines/acc-short.baseline.json` (acc-short)
- `logs/acc-long.out` (acc-long)
- `baselines/acc-long.baseline.json` (acc-long)
- `logs/acc-4k.out` (acc-4k)
- `baselines/acc-4k.baseline.json` (acc-4k)
- `logs/acc-4k-reuse.out` (acc-4k)
- `state.json` (stage records; `--resume` reads it)
