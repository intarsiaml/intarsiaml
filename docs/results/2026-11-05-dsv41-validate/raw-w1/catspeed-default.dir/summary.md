# localeval: adhoc — 2026-11-05T15:37

- model: `dsv41-intarsia` on ? (?), url `http://100.120.88.104:8899/v1`
- thinking: off; run kind: quality; label: dsv41validate-cat-default
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git a43b846
- command: `/opt/code/deepeval/.venv/bin/localeval quality --dataset /opt/code/deepeval/datasets/categories-rcp.jsonl --dataset /opt/code/intarsiaml/.worktrees/dsv41-validate/scripts/dsv41-tp2/chat-rcp.jsonl --judge none --max-tokens 256 --label dsv41validate-cat-default --url http://100.120.88.104:8899/v1 --model dsv41-intarsia --thinking off --recipe intarsia 1faac7bf5 dirty=no 1105-1503 scripts/dsv41-tp2/validate-test.sh: rank 0 spark1 serve --tp 2 --tp-remote, rank 1 spark2 tp-worker, rank caches r0of2/r1of2, NCCL CX7, cap 66000, chunk 2048, engram hot 1024 MiB`
- result file: `results/20261105-153702-adhoc-dsv41validate-cat-default/result.json`

## Launch recipe

from inline command (secrets redacted):

````
intarsia 1faac7bf5 dirty=no 1105-1503 scripts/dsv41-tp2/validate-test.sh: rank 0 spark1 serve --tp 2 --tp-remote, rank 1 spark2 tp-worker, rank caches r0of2/r1of2, NCCL CX7, cap 66000, chunk 2048, engram hot 1024 MiB
````

## Dataset `/opt/code/deepeval/datasets/categories-rcp.jsonl` (24 rows, judge none)

| metric | mean | pass rate | n | n/a | errors |
|---|---:|---:|---:|---:|---:|
| Contains expected | - | - | 0 | 24 | 0 |
| TTFT (s) | 0.186 | 1.0 | 24 | 0 | 0 |
| Prefill (tok/s) | - | - | 0 | 24 | 0 |
| Decode (tok/s) | 59.4 | 1.0 | 24 | 0 | 0 |

Per-row speed is incidental: dataset prompts are short, so prefill here is mostly request overhead. Speed metrics pass unless --min-prefill/--min-decode/--max-ttft is set.

| row | pass | correctness | TTFT (s) | decode tok/s | output |
|---|---|---:|---:|---:|---|
| recall-tcp-handshake | yes | 0.0 | 0.22 | 59.2 | # TCP Three-Way Handshake and Four-Way Teardown TCP is a con… |
| recall-photosynthesis | yes | 0.0 | 0.16 | 64.7 | # Photosynthesis: The Two Stages Photosynthesis converts lig… |
| recall-french-revolution | yes | 0.0 | 0.17 | 56.2 | # The French Revolution, 1789–1799: Causes, Phases, and Cons… |
| recall-git-internals | yes | 0.0 | 0.18 | 62.1 | # How Git Stores Data Internally Git is fundamentally a **co… |
| recall-vaccines | yes | 0.0 | 0.17 | 54.8 | # How Three Vaccine Platforms Train the Immune System ## mRN… |
| recall-cpu-caches | yes | 0.0 | 0.19 | 54.8 | # How CPU Caches Work ## The Basic Problem Main memory (DRAM… |
| recall-plate-tectonics | yes | 0.0 | 0.18 | 55.6 | # Plate Tectonics: A Complete Overview Plate tectonics is th… |
| recall-http2 | yes | 0.0 | 0.19 | 60.5 | # HTTP/1.1 vs HTTP/2 vs HTTP/3 ## HTTP/1.1 (1997) **Connecti… |
| code-lru-cache | yes | 0.0 | 0.20 | 82.1 | # LRU Cache Implementation ```python from __future__ import … |
| code-rate-limiter | yes | 0.0 | 0.23 | 69.3 | # Thread-Safe Token-Bucket Rate Limiter Below is a complete … |
| code-json-parser | yes | 0.0 | 0.20 | 80.2 | # Recursive-Descent JSON Parser ```python class JSONParseErr… |
| code-rust-cli | yes | 0.0 | 0.18 | 69.6 | # Word Frequency Counter in Rust Below is a complete, well-c… |
| code-sql-schema | yes | 0.0 | 0.18 | 82.4 | # PostgreSQL Library System Schema ## Schema Design ```sql -… |
| code-ts-debounce | yes | 0.0 | 0.18 | 67.6 | # Debounce & Throttle in TypeScript Below are production-rea… |
| code-go-worker-pool | yes | 0.0 | 0.18 | 76.7 | # Go Worker Pool with Context Cancellation and Error Collect… |
| code-bash-backup | yes | 0.0 | 0.21 | 67.1 | # Robust Directory Backup Script ```bash #!/usr/bin/env bash… |
| prose-lighthouse | yes | 0.0 | 0.15 | 46.5 | The salt spray had become a permanent perfume, and the const… |
| prose-city-essay | yes | 0.0 | 0.15 | 43.2 | # The Geography of Belonging I knew my city had become home … |
| prose-product-launch | yes | 0.0 | 0.15 | 42.8 | # Fresh from Our Oven to Your Doorstep: Introducing Neighbor… |
| prose-mountain-letter | yes | 0.0 | 0.16 | 47.1 | Dear Sam, It’s just past nine, and the tent is glowing orang… |
| prose-ocean-poem | yes | 0.0 | 0.17 | 43.7 | **Nocturne for the Unlit Sea** The sun has gone down like a … |
| prose-robot-fable | yes | 0.0 | 0.16 | 42.1 | In a city of gleaming chrome and ceaseless motion, there liv… |
| prose-travel | yes | 0.0 | 0.16 | 48.7 | The morning sun, still soft and buttery, spills over the lip… |
| prose-history-scene | yes | 0.0 | 0.35 | 48.6 | The air in the workshop was thick with the smell of lamp oil… |

## Dataset `/opt/code/intarsiaml/.worktrees/dsv41-validate/scripts/dsv41-tp2/chat-rcp.jsonl` (8 rows, judge none)

| metric | mean | pass rate | n | n/a | errors |
|---|---:|---:|---:|---:|---:|
| Contains expected | - | - | 0 | 8 | 0 |
| TTFT (s) | 0.259 | 1.0 | 8 | 0 | 0 |
| Prefill (tok/s) | - | - | 0 | 8 | 0 |
| Decode (tok/s) | 50.7 | 1.0 | 8 | 0 | 0 |

Per-row speed is incidental: dataset prompts are short, so prefill here is mostly request overhead. Speed metrics pass unless --min-prefill/--min-decode/--max-ttft is set.

| row | pass | correctness | TTFT (s) | decode tok/s | output |
|---|---|---:|---:|---:|---|
| chat-weekend-trip | yes | 0.0 | 0.29 | 51.8 | Here's a relaxed 2-day/2-night plan built around a small coa… |
| chat-job-offer | yes | 0.0 | 0.27 | 48.2 | Given that time with two young kids is your priority, the 15… |
| chat-sourdough | yes | 0.0 | 0.26 | 54.9 | Four hours at 19°C is almost certainly your problem. That's … |
| chat-roommate | yes | 0.0 | 0.25 | 50.8 | Yes — let's make this concrete. Here's a plan that keeps it … |
| chat-learn-piano | yes | 0.0 | 0.24 | 46.6 | Here's a realistic sketch of six months, assuming ~20–30 min… |
| chat-houseplants | yes | 0.0 | 0.25 | 55.4 | With a north-facing window and frequent travel, you want pla… |
| chat-budget | yes | 0.0 | 0.26 | 53.8 | Great — rent at about a third of your pay is on the higher s… |
| chat-dog-training | yes | 0.0 | 0.26 | 44.4 | Good — that treat-taking tells you a lot. If she'll still ea… |

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
