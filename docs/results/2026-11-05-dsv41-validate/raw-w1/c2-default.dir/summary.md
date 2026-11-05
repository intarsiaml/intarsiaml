# localeval: adhoc — 2026-11-05T15:27

- model: `dsv41-intarsia` on ? (?), url `http://100.120.88.104:8899/v1`
- thinking: off; run kind: quality; label: dsv41validate-c2-default
- harness: localeval 0.4.0, deepeval 4.2.6, runner endor.kuepker.haus, git a43b846
- command: `/opt/code/deepeval/.venv/bin/localeval quality --dataset /opt/code/intarsiaml/.worktrees/dsv41-validate/.scratch/c2-qfirst12.jsonl --judge none --max-tokens 400 --label dsv41validate-c2-default --url http://100.120.88.104:8899/v1 --model dsv41-intarsia --thinking off --recipe intarsia 1faac7bf5 dirty=no 1105-1503 scripts/dsv41-tp2/validate-test.sh: rank 0 spark1 serve --tp 2 --tp-remote, rank 1 spark2 tp-worker, rank caches r0of2/r1of2, NCCL CX7, cap 66000, chunk 2048, engram hot 1024 MiB`
- result file: `results/20261105-152717-adhoc-dsv41validate-c2-default/result.json`

## Launch recipe

from inline command (secrets redacted):

````
intarsia 1faac7bf5 dirty=no 1105-1503 scripts/dsv41-tp2/validate-test.sh: rank 0 spark1 serve --tp 2 --tp-remote, rank 1 spark2 tp-worker, rank caches r0of2/r1of2, NCCL CX7, cap 66000, chunk 2048, engram hot 1024 MiB
````

## Dataset `/opt/code/intarsiaml/.worktrees/dsv41-validate/.scratch/c2-qfirst12.jsonl` (12 rows, judge none)

| metric | mean | pass rate | n | n/a | errors |
|---|---:|---:|---:|---:|---:|
| Contains expected | 1.000 | 1.0 | 12 | 0 | 0 |
| TTFT (s) | 17.7 | 1.0 | 12 | 0 | 0 |
| Prefill (tok/s) | 1758.0 | 1.0 | 12 | 0 | 0 |
| Decode (tok/s) | 76.3 | 1.0 | 12 | 0 | 0 |

Per-row speed is incidental: dataset prompts are short, so prefill here is mostly request overhead. Speed metrics pass unless --min-prefill/--min-decode/--max-ttft is set.

| row | pass | correctness | TTFT (s) | decode tok/s | output |
|---|---|---:|---:|---:|---|
| hop2-L3-32k-000-qfirst | yes | 1.0 | 16.80 | 70.8 | The document states: "Ops ledger: the release nicknamed Tory… |
| hop3-L3-32k-000-qfirst | yes | 1.0 | 16.84 | 79.3 | The document states: "Release ledger: the release nicknamed … |
| agg_sum-L3-32k-000-qfirst | yes | 1.0 | 16.82 | 77.2 | The Pelmar cluster storage shards mentioned: - AR-6283-PN: 3… |
| agg_sum-L4-32k-001-qfirst | yes | 1.0 | 16.88 | 81.2 | Storage shards of the Wynbri cluster: UC-8590-WQ 602, GF-246… |
| agg_max-L3-32k-000-qfirst | yes | 1.0 | 19.70 | 58.0 | The document contains many tracker notes and ledgers for the… |
| repo_sum-L4-32k-000-qfirst | yes | 1.0 | 18.82 | 86.9 | RING = 4096 (coll_trace.rs) WARMUP = 5 (icb_check.rs) PRE_MI… |
| hop2-L3-32k-001-qfirst | yes | 1.0 | 19.33 | 76.1 | The document states: "Tracker note: the release nicknamed El… |
| hop3-L3-32k-001-qfirst | yes | 1.0 | 19.19 | 72.2 | The release nicknamed Vexfal is tracked in this ledger as th… |
| agg_sum-L3-32k-001-qfirst | yes | 1.0 | 16.96 | 79.3 | The Rostor cluster storage shards mentioned: DX-8430-GW hold… |
| agg_sum-L4-32k-002-qfirst | yes | 1.0 | 16.86 | 82.4 | Storage shards of the Barfal cluster: - SP-2710-CW: 977 - NN… |
| agg_max-L3-32k-001-qfirst | yes | 1.0 | 16.95 | 75.3 | The document lists several releases of the Bribri line with … |
| repo_sum-L4-32k-001-qfirst | yes | 1.0 | 16.84 | 76.7 | EXIT_FAILED = 87 (health.rs); HOLD = 16 (feedback.rs); CONV_… |

## Method

Streamed OpenAI-compatible calls, temperature 0. TTFT = request start to first content/reasoning token; prefill tok/s = uncached prompt_tokens / TTFT; decode tok/s = (completion_tokens - 1) / (last - first token), counts from server usage. Speed sweep: fresh nonce first in every prompt (no prefix-cache hits), ignore_eos+min_tokens forces the full length, 1 discarded warmup, median of reps. Other clients' in-flight requests checked before every sweep call. Quality-run speed is incidental (varying prompts), not a sweep.
