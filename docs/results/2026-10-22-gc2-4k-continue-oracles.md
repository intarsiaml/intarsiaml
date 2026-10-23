# 4k-golden continue oracles: Metal 35B `qualify` PASSES on #104 and #109 (gc2)

**Result: with continue oracles for the 4k golden's `long_docs_4k` (splits 2000 and 3072), Metal 35B `qualify` PASSES on #104 alone
(chunk 256) and on #109 (chunk 2048), exit 0, no tolerance loosened.** The earlier failure (docs/results/2026-10-22-gc-gate-coverage.md,
2026-10-22-gc2-metal-35b-chunk2048.md) was the absolute 0.25 nats / 1e-2 KL `continue` tolerance applied to a near-tie
(cold margin 0.13 nats) because no committed oracle matched the 4k golden's prompt ids.

## 1. Oracle files (committed on #104, `golden/oracle/`)

Same recipe as fx5c (`scripts/fx5c`, `make_continue_oracle.py`): freeze the engine's cold greedy stream for `long_docs_4k`
(4k golden ids, fingerprint 9c90e2739690c12b), score one prefill over turn2 + 32 stream tokens with the f64 CPU oracle
(`accuracy --backend cpu --mode prefill`, chunk 256, top-64), keep the 32 x 64 rows. Names carry `-g4k` so they do not collide with
the long golden's files of the same model / prompt / split.

| file | split | frozen at chunk | sha256 |
|---|---|---|---|
| `qwen36-35b-a3b-nvfp4-continue-long_docs_4k-s2000-g4k.continue-oracle.safetensors` | 2000 (chunks 256, 1024) | 1024 | see sidecar `.json` |
| `qwen36-35b-a3b-nvfp4-continue-long_docs_4k-s3072-g4k.continue-oracle.safetensors` | 3072 (chunk 2048) | 2048 | see sidecar `.json` |

The set does not depend on the run's chunk (a frozen token list scored by a CPU oracle), so one set serves both branches; it is
committed on #104 and merged into #109. Differences from fx5c: the frozen stream comes from the Metal cold engine on endor, not CUDA
(any engine's stream works because the gate substitutes the frozen turn 2 into the real runs), and the oracle ran on endor's CPU
(Apple M5 Max, 14 and 18 min, 38.7 GB max RSS) not spark1 (spark1 was down for a kernel update). fx5c measured the two machines'
oracles differ by KL 3.1e-3 on shared rows. The oracle tool exits 1 on the filler golden by design.
Raw: `2026-10-22-gc2-4k-continue-oracles/raw/` (job.log, seq.json, oracle logs; fleetq #4316, non-exclusive, endor).

## 2. qualify verdicts (fleetq #4380, `--quiet gpu,cpu`, non-exclusive, endor services up; no envelope / bench-long / ab / tp)

| | #104 alone (chunk 256) | #109 (chunk 2048) |
|---|---|---|
| exit code | 0 | 0 |
| doctor, recipe, plan, kernels, gate | PASS | PASS |
| acc-short | PASS (top-1 98.33 %, KL 7.04e-3) | PASS (same) |
| acc-oracle | PASS (KL 1.09e-2) | PASS |
| bench | PASS (decode 125.0 tok/s at 128) | PASS (125.5) |
| acc-long | PASS | PASS |
| acc-4k | PASS (prefill top-1 93.14 %, KL 8.14e-2; decode KL 7.22e-3) | PASS (same) |
| `long_docs_4k` continue | split 2000: oracle rule PASS, KL(o\|\|cold) 1.26e-2, KL(o\|\|warm) 1.25e-2, 0.99x (limit 1.5x + 3e-3 = 2.20e-2) | split 2000 (qualify's 1024 run): same 0.99x; split 3072 (2048 run): 1.00e-2 / 1.17e-2, 1.16x (limit 1.81e-2) PASS |
| `restore`, `aligned` | ok, aligned bit-identical | ok, aligned bit-identical |

Rule applied: `warm <= 1.5 x cold + 3e-3` on the oracle KL (fx5c-oracle-gate), unchanged. Raw: `2026-10-22-gc2-4k-continue-oracles/raw-qualify/`.
Note the stream is frozen from one engine: at 3072 the ratio is 1.16x, inside the 1.5x margin but not far (n = 32 rows, fx5c's caveat).
