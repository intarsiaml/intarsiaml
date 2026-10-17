# Metal serve release check of integrated `main` at b02a5f4

Date 2026-10-17. Machine endor (Apple M5 Max), Qwen3-8B BF16, `serve --backend metal --max-seq 8192`. Branch `metal-relcheck` = `origin/main` b02a5f4 (merge of PR #7), worktree `/opt/code/intarsiaml-relcheck`, no source change. Since the last Metal serve validation, main gained the HIP/CUDA norm-fusion merges (#5, #6; backend-specific, not compiled into the Metal build) and the telemetry fix of #7 (`requests_total` keeps queued requests that fail before starting; `once idle, requests_total == sum(finished{reason})`).

Binary: `cargo build --release -p intarsia-cli --features metal` at b02a5f4, copied to a private path and used by every run, sha256 prefix `47e4a0feb4b09e39`. CPU tests of the two crates pass on this tree (`cargo test -p intarsia-server -p intarsia-core`, rc 0, 127 passed, 0 failed, 1 ignored; no GPU, run for the record).

All GPU work ran in **one** exclusive fleetq slot: **#2378** (`--host endor --exclusive --mem 24 --expect 150`), released after the last run (state `done`; `fleetq status` shows endor with jobs 0 and only the two standing services). No `--force`, no `--no-queue`, standing services untouched, no `intarsia serve` left behind after any script.

## Verdict

| Gate | Result | Numbers (this run vs previous) |
|---|---|---|
| 1a. Batched identity, B=8, reuse off and on | **PASS** | 203/203 + 203/203 = **406/406** bitwise identical to solo, 0 mismatches (previous 406/406) |
| 1b. Spec + prefix-reuse identity, B=2/3/8 | **PASS** | 268 + 268 + 265 = **801/801** comparable, 0 mismatches, 0 errors (previous 799 comparable of 804, 0 mismatches; same chains, 120) |
| 2a. B=4 steady ITL / ITL during the 4k prefill | **PASS, no regression** | steady p95 29.8 before / 32.7 after (p50 29.3 / 31.7), during-prefill p95 127.2 (limit 150); previous soak3 31.6 / 34.1 and 128.2 |
| 2b. B=8 ITL p95 during the 4k prefill | **PASS, 7% headroom, higher than the previous closeout run** | **139.7** ms (max 148.0), limit 150; previous closeout 133.2 (max 143.5), earlier soak-batched 140.8 (max 148.5) |
| 2c. B=8 4k TTFT vs solo | **PASS** | **2.16x** (limit 2.5x); previous 2.10x |
| 3. Keep-warm idle TTFT, on vs off | **keep-warm helps at 10 s and 40 s; gain is smaller than before** | 10 s: -34 ms mean (-30 median, 6 of 6 rounds negative); 40 s: -56 ms (6 of 6 negative). 40 s holds. |
| 4. `requests_total == sum(finished{reason})` when idle | **PASS** | holds on all 9 end-of-run scrapes with `running 0`, `waiting 0` |

Nothing regressed in the shipped `serve` path. Two things to read carefully: the B=8 during-prefill ITL moved toward the limit (section 2.2), and the keep-warm gain at 10 s is 30 to 34 ms, not the 58 ms of the previous final build (section 4).

## 1. Identity (gate 1)

Scripts are the previous ones with only paths changed (`raw/2026-10-17-metal-relcheck-main/`: `slot_identity.sh` from `2026-10-16-metal-cb-m2-closeout/slot_identity.sh`, `slot_partB3.sh` + `end_sample.sh` from `2026-10-16-metal-cb-soak3/`, `env.sh` rewritten for these paths). Run in slot #2378:

```
FLEETQ_JOB=2378 BIN=<private copy of the b02a5f4 binary> slot_identity.sh OUT/identity           # 14:49 to 15:12
FLEETQ_JOB=2378 slot_partB3.sh OUT/partB                    # CHAINS default 100, 666 requests
FLEETQ_JOB=2378 CHAINS=120 slot_partB3.sh OUT/partB120      # the previous soak3 Part B setting, 804 requests
```

### 1a. B=8, 203 cases (`tools/cbidentity.py`, `tools/prompts/cb_identity_cases.jsonl`)

Solo references on `--max-batch 1` at `--prefill-chunk 2048` and `256`; batched servers `--max-batch 8` with `INTARSIA_CB_TRACE=1`, 8 concurrent clients, `--stream mixed`, `--arrival-scale 0.25`, reuse off (`INTARSIA_CB_NO_REUSE=1`) and on.

| Run | Cases | Pass | Mismatch | Token ids compared | Cancels (with output) | Started from a different cached length | FIFO (`tools/cbfifo.py`) |
|---|---|---|---|---|---|---|---|
| reuse off | 203 | **203** | **0** | 201 | 20 (18) | 0 | OK: 204 requests, 202 admitted, 1085 chunks (664 contested), 0 violations |
| reuse on | 203 | **203** | **0** | 201 | 20 (18) | 26 (all still identical) | OK: 28 admissions from reused KV, 1085 chunks (661 contested) |

Same numbers as the M2 closeout (203/203 both settings, 201 token-id comparisons, 20 cancels, 113 streamed / 90 not). The `WARNING ... server plan differs from the reference server on ['max_batch','ops']` in the compare logs is expected (batched versus solo servers).

### 1b. `--spec ngram` + prefix reuse (decode-written KV), B=2/3/8 (`tools/cbreuse.py`, `tools/cbspectrace.py`)

Reference `--max-batch 1 --spec ngram --prefill-chunk 256`, test `--max-batch 8 --prefill-chunk 256 --spec ngram`, reuse on, `INTARSIA_CB_TRACE=1`, `INTARSIA_SPEC_TRACE=1`, `--copy-frac 0.5`.

| Run | B | Requests | Comparable | Pass | Mismatch | Errors | Started from reused KV (decode-written) | Not comparable |
|---|---|---|---|---|---|---|---|---|
| `partB120` (120 chains, = previous Part B) | 2 | 268 | 268 | **268** | **0** | 0 | 148 (55%), 17098 tokens | 0 |
| | 3 | 268 | 268 | **268** | **0** | 0 | 148 (55%), 17098 tokens | 0 |
| | 8 | 268 | 265 | **265** | **0** | 0 | 145 (55%), 16721 tokens | 3 (idle slot taken by another chain, as before) |
| `partB` (100 chains) | 2 / 3 / 8 | 222 each | 222 / 222 / 218 | all | **0** | 0 | 122 / 122 / 118 | 0 / 0 / 4 |

801 = 268 + 268 + 265. The 100-chain run was the first run (script default); it was repeated at the previous 120 chains so the counts line up with soak3 Part B (268 requests per B). Drafting and switch-off behaved as before (`spectrace.txt`, 120 chains): 22782 decode phases, spec-off phases 2904, **0** phases with a verify pass while off, gate value wrong in 0 phases (drafting on exactly when fewer than 4 sequences decode), 1 on-to-off and 1 off-to-on transition, 13891 of 17488 drafted tokens accepted (79%), per b: drafting passes at b=1,2,3 only, none at b=4..8.

Not rerun: the negative control (planted token-id flips) of soak3 Part B; the harness is unchanged and 0 mismatches is not the result of a disabled comparison (the same `compare` produced the 26 "cached length differs" annotations in 1a).

## 2. Soak regression (gate 2)

### 2.1 B=4: `slot_soak3.sh OUT/b4 4 0.33 3 600` (copy of the soak3 script, paths only; `cbsoak_serve.sh` with `--max-batch 4`)

Scenario x5 (3 streams of ~1k tokens, a ~4k prompt at t=5 s, `--max-steady-itl-p95-ms 60 --max-itl-prefill-p95-ms 150 --max-long-ttft-x-solo 2.5`) then a Poisson soak at 0.33 req/s (58% of the measured B=4 saturation of the previous run) with 10% cancels, drain 120 s. **The soak ran 600 s, not the 30 min of soak3**; the numbers below that depend on duration (memory growth, tails) are therefore weaker evidence than soak3's.

| | This run (b02a5f4) | Previous (soak3, 66feda6 + keep-warm) | Limit |
|---|---|---|---|
| steady ITL p50 / p95, before the 4k | 29.3 / 29.8 ms (1917 gaps) | 29.7 / 30.6 to 31.6 | about 60 |
| steady ITL p50 / p95, after (4 rows) | 31.7 / 32.7 ms (3758 gaps) | 31.6 / 34.1 | about 60 |
| ITL during the 4k prefill p50 / p95 / max | 109.6 / **127.2** / 132.5 ms (295 gaps) | p95 128.2 | p95 150 |
| per-rep p95 during the prefill | 109.6, 128.9, 127.2, 119.2, 116.9 | 111.3, 120.2, 128.6, 110.5, 129.3 | |
| 4k TTFT under load p50 (max) / warm solo | 1969.7 (2290.1) / 972.7 ms = **2.02x** | 2.06x | 2.5x |
| soak: sent / ok / cancelled / shed / errors | 206 / 194 / 12 / 0 / 0 | 0 shed, 0 errors | shed <= 1%, errors <= 0.1% |
| soak TTFT p50 / p95 / p99 | 405.6 / 2515 / 3678 ms | 401 / 1976 / 3426 | 1500 / 5000 / 8000 |
| soak ITL p95 overall / prefill-overlap (>=3000-token prompts, 19 of them) | 94.2 / 117.0 ms | 128.2 (overlap) | 150 |
| memory growth (fitted) / device | 7.2 MB / 0 MB | 20 MB | 100 / 100 |

`cbsoak` SLO block: **PASS**, rc 0. The scenario limits passed too (rc 0 for run 1).

### 2.2 B=8 scenario: `slot_b8.sh OUT/b8` (`cbsoak_serve.sh`, `SERVE_EXTRA="--max-batch 8"`, exactly the scenario command of `slot_perf.sh` (a) of the M2 closeout: 7 streams, 5 reps, 3 solo reps)

| | This run (b02a5f4) | M2 closeout rerun (8ce01d2, #2208) | Limit |
|---|---|---|---|
| steady ITL p50 / p95, before the 4k | 36.2 / 37.8 ms (1893 gaps) | 34.6 / 37.5 | about 60 |
| steady ITL p50 / p95, after (8 rows) | 43.5 / **50.0** ms (6283 gaps) | 42.6 / 45.6 | about 60 |
| ITL during the 4k prefill p50 / p95 / max | 126.4 / **139.7** / 148.0 ms (693 gaps) | 111.5 / 133.2 / 143.5 (686 gaps) | p95 150 |
| per-rep p95 during the prefill | 118.5, 144.3, 144.6, 137.8, 145.8 | 113.4, 124.8, 122.6, 141.7, 122.4 | |
| 4k TTFT p50 (max) | 2097.0 (2601.4) ms | 2039.6 (2480.4) | |
| solo 4k TTFT p50 (3 reps) | 970.7 ms | 970.5 | |
| ratio of medians | **2.16x** | 2.10x | 2.5x |
| per request, worst rep | 2601 / 970.7 = 2.68x raw (rep 3, 4955-token prompt) | 2.56x raw | the limit is read as ratio of medians (Q13) |

All limits pass, violations list empty. Compared with the closeout rerun the B=8 numbers that involve a prefill are worse: during-prefill p50 +15 ms (+13%), p95 +6.5 ms, after-prefill steady p95 +4.4 ms, TTFT ratio +0.06. The pre-prefill steady ITL (pure decode) is unchanged (37.8 vs 37.5), so this is not a decode regression, and the 139.7 ms sits inside the range already seen on earlier builds (140.8 ms p95, max 148.5, in `2026-10-16-metal-cb-soak-batched.md`; the closeout's own 133.2 was the best of the three B=8 runs). Headroom to the 150 ms limit is 7% here (11% in the closeout rerun, 6% in soak-batched), and the max gap of 148.0 ms is 2 ms below the limit. Candidate cause, not tested: the GPU ran slower in this run. macmon, samples with GPU utilisation above 0.5, mean 1399 MHz (min 1109, max 1617), mean GPU temperature 86 C, against 1483 MHz and 84 C in the closeout scenario (-5.7% clock), and prefill chunks are GPU-bound. Nothing in the b02a5f4 diff touches the Metal path (#5, #6 are HIP/CUDA, #7 is telemetry), so a code cause is not expected, but no A/B against the closeout binary was run to rule one out.

## 3. Burst (shedding, 429s) with the invariant on a refusing server

`slot_b8.sh` second part (as `slot_perf.sh` (c)): `--max-batch 4 --max-queue 4`, bursts of 16 x 512 and 24 x 1024 tokens. 16 simultaneous: 8 ok, 8 x 429 with `Retry-After: 1`; 24 simultaneous: 8 ok, 16 x 429 with `Retry-After: 1`; `intarsia_requests_rejected_total` 24; no 503; no violations. The 24 refused submits are **not** in `requests_total` (22 accepted, all finished `length`), which is what the new help text says.

## 4. Keep-warm idle TTFT (gate 3)

`slot_idle.sh` (adaptation of `2026-10-16-metal-weight-residency/serve-idle2.sh`: paths, and the two arms are on **one binary**: `kw` = serve default, `base` = `INTARSIA_METAL_KEEPWARM=0`), `BIN=<b02a5f4 copy> NOPROF=1 IDLES="10 40" slot_idle.sh OUT/idle 6 kw base`. Per sample: start `intarsia serve` (port 8102, no cb trace, as #2203), wait for `/health`, warm-up 4k request (1 token), sleep 10 s, one streamed 4062-token request (33 tokens, unique nonce), sleep 40 s, the same again; the next sample starts after the GPU is idle again (ioreg utilisation <= 12% in 3 consecutive 1 s samples). The arm order rotates per round: kw,base / base,kw / kw,base / base,kw / kw,base / base,kw (ABBA, 3 rounds each way), 6 rounds = 6 reps per arm and idle point. About 16:14 to 16:35.

Validity checks: every `kw` server log has the five keep-warm registrations (4.00, 3.19, 1.24, 3.99, 3.96 GB), no `base` log has any; generated text identical across arms for every round (0 differing groups); decode after the idle unchanged (mean ITL 29.8 vs 29.2 ms at 10 s, 29.2 vs 28.8 at 40 s: keep-warm costs 0.4 to 0.6 ms/token of decode here, within the 28.5 to 30.4 ms round spread).

TTFT ms of the first 4k request after the idle (median [min..max] over 6 rounds):

| Idle | keep-warm off | keep-warm on | paired on minus off, median [min..max] | mean +- sd | Rounds negative |
|---|---|---|---|---|---|
| 10 s | 1116.8 [1110.1..1147.9] | 1096.8 [1068.3..1105.4] | **-29.6 [-71.3..-15.7]** | -34.3 +- 22.5 | 6 / 6 |
| 40 s | 1126.1 [1115.8..1152.7] | 1069.3 [1064.9..1092.0] | **-56.4 [-87.8..-23.9]** | -56.4 +- 20.2 | 6 / 6 |

Paired differences per round (ms): 10 s: -71.3, -15.7, -16.7, -15.8, -44.1, -42.5; 40 s: -55.9, -56.2, -23.9, -56.6, -87.8, -58.1. By order: 10 s kw-first rounds -44.0, base-first -24.7; 40 s -55.9 and -57.0. The warm-up request (first request after startup, no idle) also shows the gain: kw 1073.7 / 1078.3 ms against base 1130.4 / 1108.5 ms in rounds 1 and 2 (only those two were looked at).

* **40 s holds**: -56 ms, same sign in all six rounds, and not smaller than at 10 s. The keep-warm gain does not decay between 10 s and 40 s.
* **Against the earlier numbers**: #2178 (pre-final build, cb trace on): -110.6 ms at 10 s, base 1173.5 vs keep-warm 1061.4. #2203 (final build against base 6d99a6f, no trace): -58.5 ms at 10 s, base 1126.2 [1121.6..1139.5], keep-warm 1066.5 [1060.7..1078.4], back-to-back request 1128.6 vs 1076.1. This run: off 1116.8 (10 s) and 1126.1 (40 s), on 1096.8 (10 s) and 1069.3 (40 s). So the off arm is where #2203 had base (about 1120), and the **on arm at 40 s equals #2203's on arm (1069 vs 1066.5)**, as does the on-arm warm-up request here (1074, 1078 vs 1076.1). Only the **on arm at 10 s is slower** than in #2203, by about 30 ms (1096.8 [1068.3..1105.4] against 1066.5 [1060.7..1078.4]; the fastest round here, 1068.3, just overlaps the old range). That is the whole difference between -58 and -30 at 10 s. The 110 ms of #2178 is not reproduced at any point: its base arm (1173.5) was slower than any off arm measured since (1110 to 1153).
* **Why the on arm at 10 s is slower, not explained.** The 10 s on-arm rounds are 1068.4, 1094.4, 1099.1, 1102.0, 1068.3, 1105.4: two rounds (kw-first rounds 1 and 5) are at the old level, four are 25 to 35 ms above it, and the same four server starts are not distinguishable from the two fast ones in the logs I looked at (registration lines identical). The 40 s on-arm rounds are tight (1064.9 to 1072.3, plus one at 1092.0). Nothing was run to test a cause, and no cb-trace-on arm was run (the earlier doc notes #2178 had the trace on, #2203 did not). Treat the 10 s gain as "between 16 and 71 ms in 6 of 6 rounds, median 30", not as 58.
* Server-side explanation not attempted (no cb trace in this run).

## 5. `/metrics` invariant (gate 4)

For each scrape: `intarsia_requests_total`, `intarsia_requests_finished_total{reason}` summed, `intarsia_requests_running`, `intarsia_requests_waiting`, `intarsia_requests_rejected_total`. Taken with nothing in flight (scrape after the clients finished or after the last `cbsoak` drain, before the server is killed). Computed by `inv.py` from the files in this directory (`invariant.txt`).

| Scrape | total | stop | length | cancelled | sum | running / waiting | rejected | Equal |
|---|---|---|---|---|---|---|---|---|
| identity, reuse off | 204 | 55 | 129 | 20 | 204 | 0 / 0 | 0 | yes |
| identity, reuse on | 204 | 55 | 129 | 20 | 204 | 0 / 0 | 0 | yes |
| Part B (100 chains) after B=8 | 669 | 0 | 669 | 0 | 669 | 0 / 0 | 0 | yes |
| Part B (120 chains) after B=2 | 269 | 1 | 268 | 0 | 269 | 0 / 0 | 0 | yes |
| Part B (120 chains) after B=3 | 538 | 2 | 536 | 0 | 538 | 0 / 0 | 0 | yes |
| Part B (120 chains) after B=8 | 807 | 3 | 804 | 0 | 807 | 0 / 0 | 0 | yes |
| B=4 scenario + 600 s soak (12 client cancels in the soak, 11 reached the server as `cancelled`) | 242 | 0 | 231 | 11 | 242 | 0 / 0 | 0 | yes |
| B=8 scenario (5 reps) | 52 | 0 | 52 | 0 | 52 | 0 / 0 | 0 | yes |
| bursts at `--max-queue 4` | 22 | 0 | 22 | 0 | 22 | 0 / 0 | **24** | yes (refused submits stay out) |

**PASS** on 9 of 9. No `error` finish reason occurred in any run, so the "queued request that fails before starting stays counted" path of #7 was not exercised on hardware (it is a unit-test path; fault injection on the GPU is `2026-10-17-metal-cb-fault-inject.md`). The invariant is checked here only where it should hold trivially after normal traffic and cancels; the `cancelled` reason (20 in identity, 11 in the soak) and the refused-at-submit path are the non-trivial parts covered. One cancel difference: the soak client reports 12 cancelled, the server counts 11; not examined (a cancel arriving after the last token would finish as `length`).

## 6. What was not done, caveats

* **Soak length**: 600 s at B=4, not the 30 min of soak3; the 0.8 req/s B=8 saturation soak of the closeout was not rerun (only its scenario and bursts).
* **B=8 steady and during-prefill ITL are worse than the closeout rerun** (section 2.2), within the earlier range, no A/B run against 8ce01d2 or `spike/metal` to separate noise or thermal state from code. The box had run ~1.5 h of GPU work before the B=8 scenario (the slot began with identity, Part B x2 and the B=4 soak); GPU clock was 5.7% lower in that run.
* **Part B was run twice**: 100 chains (script default) and 120 chains (previous setting); both pass. Per-request jsonl, chains, macmon and most server logs are not committed (size).
* Keep-warm: one binary, two arms, **no** trace-on arm (not requested); 10 s gain not explained (section 4); the earlier builds were not rebuilt.
* The B=4 soak, the identity runs and the idle run are single runs on one machine.
* `--version` prints only `0.0.1`; the binary is tied to b02a5f4 only by having been built from that worktree (HEAD b02a5f4, clean tree) and by its sha256 prefix.
* Port check before the slot: 8099, 8101, 8102, 8107, 8142, 8143 had no listener.

## 7. Reproduce

Everything below inside one slot (`fleetq acquire --host endor --exclusive --mem 24 --expect 150 --label "..." --wait`, then `FLEETQ_JOB=<id>` on each line, `fleetq release <id>` at the end). `$B` = the release binary, `$R` = `docs/results/raw/2026-10-17-metal-relcheck-main`.

```
BIN=$B $R/slot_identity.sh OUT/identity
$R/slot_partB3.sh OUT/partB ; CHAINS=120 $R/slot_partB3.sh OUT/partB120      # env.sh in $R sets W and HEAD
$R/slot_soak3.sh OUT/b4 4 0.33 3 600
$R/slot_b8.sh OUT/b8
BIN=$B NOPROF=1 IDLES="10 40" $R/slot_idle.sh OUT/idle 6 kw base
python3 docs/results/raw/2026-10-16-metal-weight-residency/idlesum.py OUT/idle    # arm "base" is the reference
python3 $R/inv.py <metrics files>                                               # section 5
```

Raw outputs: `docs/results/raw/2026-10-17-metal-relcheck-main/` (about 0.9 MB): slot scripts, `*.console` slot logs, `identity/` (compare logs, FIFO checker output, metrics, plans, traced server logs gzipped), `partB/` and `partB120/` (compare logs, `spectrace.txt/json`, metrics, gzipped server logs for 120), `b4/` and `b8/` (`scenario.json/md`, `report.json/md`, `burst.json/md`, metrics, gzipped server logs), `idle/` (per-sample `*.idle10.json`, `*.idle40.json`, `*.warm.json`, `idlesum.txt`), `invariant.txt`, `inv.py`. Not committed: per-request jsonl, chains, `macmon.jsonl`, per-case reports, untraced idle server logs (scratch: `/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/out/`).
