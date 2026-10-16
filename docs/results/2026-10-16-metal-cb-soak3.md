# Metal continuous batching: soak3 (B=4 soak, B=1 A/B/A guard, spec + prefix reuse identity)

Date 2026-10-16. Branch `metal-cb-soak3` off `spike/metal` 66feda6. Tools, docs and raw results only; nothing under `crates/` changed. Machine endor (Apple M5 Max), Qwen3-8B BF16, `serve --backend metal --max-seq 8192`. Binary `intarsia-head-66feda6` (sha256 prefix `d979ae2ceb9a57e6`, `cargo build --release -p intarsia-cli --features metal` at 66feda6). Baseline `intarsia-base28e5c5a` (`10abb4203759a958`, the M1-era 28e5c5a). **Keep-warm is on in `serve` by default in the head binary** (residency set re-requested every 400 ms); the baseline has no keep-warm.

This closes the three runs that the 2026-10-16 soak2 doc (`2026-10-16-metal-cb-soak-batched.md`) left undone. Limits are the adopted design-doc Q13 SLOs.

## Result per run

| run | verdict | headline | fleetq |
|---|---|---|---|
| 1. B=4 soak (ramp, scenario x5, 30 min Poisson at 0.33 req/s = 58% of measured saturation) | **PASS** all Q13 rows | steady ITL p95 31.6 / 34.1 ms (scenario before / after), prefill-overlap ITL p95 128.2 ms, TTFT 401 / 1976 / 3426 ms, 0 errors, 0 shed, memory growth 20 MB | #2191 |
| 2. B=1 A/B/A guard, plus keep-warm-off arm (base, head, head-kwoff, base; 6 min each) | **PASS** | head / base ITL p50 1.009, p95 1.004 (limit 1.1); base to base drift 1.4% (p50), 0.3% (p95) | #2193 |
| 3. Part B, `--spec ngram` + prefix reuse, solo-reference identity at B=2, 3, 8 | **PASS** | 268 requests per B, 0 mismatches in 799 comparable comparisons, 0 errors; drafting and decode-written KV reuse both happened; negative control gives exactly the 3 planted mismatches | #2194 (pilot and full run in one slot) |

(#2192 was a run-3 lease queued by mistake ahead of run 2 and cancelled before it started.) No engine bug was found. No server or runner of mine is left.

## 1. B=4 soak

### 1.1 Capacity ramp (`b4-ramp/`, same slot, one server `--max-batch 4`)

100 s segments, 10% cancel plan, default prompt / max_tokens mix, seeds 41..45, `slot_ramp4.sh`. A 0.8 req/s segment was added to the suggested 0.2 / 0.3 / 0.45 / 0.6 to be sure of seeing saturation.

| planned rps | sent | completed rps | out tok/s | TTFT p50 / p95 / p99 ms | ITL p50 / p95 ms | queue max | mean rows | shed / err |
|---|---|---|---|---|---|---|---|---|
| 0.2 | 22 | 0.231 | 37.9 | 306 / 1714 / 2667 | 28.5 / 32.6 | 0 | 1.65 | 0 / 0 |
| 0.3 | 29 | 0.303 | 38.7 | 291 / 1697 / 2357 | 28.9 / 33.2 | 1 | 1.78 | 0 / 0 |
| 0.45 | 52 | 0.468 | 53.8 | 355 / 2477 / 2630 | 30.4 / 98.2 | 2 | 2.09 | 0 / 0 |
| 0.6 | 47 | 0.415 | 60.5 | 755 / 7421 / 9883 | 31.1 / 122.6 | 4 | 2.43 | 0 / 0 |
| 0.8 | 94 | 0.565 | 80.9 | 17337 / 46219 / 48981 | 32.0 / 123.9 | 26 | 3.62 | 0 / 0 |

Saturation: the 0.8 segment is saturated (queue 26 and growing, TTFT p50 17 s) and delivered 0.565 req/s, 81 tok/s; at about 143 output tokens per request (the B=8 ramp had 135) that is a saturation of about 0.57 req/s. The Poisson draw in the 0.6 segment sent only 47 requests (0.47 realised), and even that already breaks TTFT p95 (7.4 s) and p99 (9.9 s). B=4 saturates at about two thirds of B=8 (0.86 req/s). Soak rate = 0.33 req/s = 58% of 0.57. The process footprint rises from 4.0 to 5.15 GB during the first segment (end minus start 1156 MB; the preallocated KV pool is 4.83 GB at B=4, touched on first use, the same effect as at B=8); the memory-growth column in the table is the post-warmup fit (1 to 4 MB per segment).

### 1.2 Scenario x5 (3 streams decoding + one 4k prompt, `b4-soak/soak/scenario.md`)

Chunk 256 (server default busy chunk), 5 valid repetitions, warm-up request before each, fresh nonces.

| window | n gaps | p50 ms | p95 ms | p99 ms | max ms |
|---|---|---|---|---|---|
| steady before (3 rows) | 1868 | 30.1 | 31.6 | 32.3 | 63.0 |
| during the long prefill | 294 | 108.5 | **128.2** | 131.2 | 133.5 |
| steady after (4 rows, 4k stream decoding) | 3722 | 32.0 | 34.1 | 34.8 | 64.4 |

Per repetition p95 during the prefill: 111.3, 120.2, 128.6, 110.5, 129.3. 4k TTFT under load p50 2002 ms (max 2166) against warm solo 973 ms: 2.06x (limit 2.5x). Decode passes by rows in the scenario: 3: 1951, 4: 1995. Compared with the soak2 B=4 scenario on the 6d99a6f binary (before 29.7 / 30.6, during p95 130.8, after 32.3 / 33.5, 4k 2.04x) this is the same within noise; keep-warm and the busy-chunk flag did not change it.

### 1.3 Poisson soak (`b4-soak/soak2/`)

0.33 req/s for 1800 s, seed 21, default mix, reuse on, 10% cancel plan (6.7% realised), `--max-batch 4`, default `--max-queue`. 586 sent, 547 ok, 39 cancelled, 0 shed, 0 error. Completed 0.304 req/s, 41.4 tok/s. Per-stream decode 31.5 tok/s mean. 1.000 chunks per token.

### 1.4 Q13 scorecard, B=4

| Q13 row | limit | measured | verdict |
|---|---|---|---|
| steady-state ITL p95 (scenario before / after windows) | <= about 60 ms | 31.6 / 34.1 ms | pass |
| steady-state ITL p95, soak-derived cross-check (gaps with no other request in its prefill phase, first 360 s skipped) | <= about 60 ms | 30.8 ms (n 50223, p50 28.6) | pass (cross-check only) |
| prefill-overlap ITL p95, judged on the scripted scenario | <= 150 ms | 128.2 ms (5 reps, 294 gaps, max 133.5) | pass |
| prefill-overlap ITL p95, soak (gaps overlapping a 3000+ token prefill, 51 prompts, 1923 gaps) | informational | 112.9 ms (p50 96.1, max 127.4) | informational |
| TTFT p50 / p95 / p99 (soak, 58% of saturation) | 1500 / 5000 / 8000 ms | 401 / 1976 / 3426 ms (max 4741) | pass |
| error rate | <= 0.001 | 0 / 586 | pass |
| shed rate | <= 0.01 | 0 / 586 | pass |
| memory growth after warmup (footprint, Theil-Sen fit after 360 s, 24 min) | <= 100 MB | 20.5 MB (end minus start 28.5, slope 0.85 MB/min, r2 0.05, leak_suspect false) | pass |
| device memory per `/metrics` | <= 100 MB | 0 MB | pass (engine accounting only) |
| 4k TTFT under scenario load against warm solo | <= 2.5x | 2.06x | pass |
| B=1 regression | <= 1.1x | section 2 | pass |

Other soak numbers. Pooled ITL p50 / p95 / p99 / max 28.8 / 84.5 / 102.1 / 127.7 ms (74000 gaps; the old pooled 150 ms limit would also pass). Per 60 s window (windows with more than 3 requests) ITL p95 29.7 to 98.1 ms (p99 92.9 to 113.9), TTFT p95 726 to 4000 ms (max in window 29), no drift. TTFT p50 by prompt bucket: 128: 103 ms, 512: 273, 1024: 471, 2048: 810, 4096: 1639 ms. Cold-start TTFT (arrival with nothing open for more than 3 s, n 55): p50 278, p95 1009 ms; warm arrivals (n 525) p50 464, p95 2020 ms. Server side: batch-size histogram of decode passes by rows 1: 49.2%, 2: 29.0%, 3: 13.6%, 4: 8.1% (mean 1.81 rows); `intarsia_batch_queue_depth` max 2 (mean 0.08), server-reported `queue_depth_max` 4; step EWMA mean 32.5 ms (max 101); KV used max 1.64 GB of 4.83 GB; 2361 prefill chunks; rejected 0.

Limit of this soak: at 58% of saturation the mean batch is 1.8 rows and only 8% of the decode passes have 4 rows, so the soak says little about full-width behaviour; the scenario (3 and 4 rows) covers that. The soak-derived steady-state ITL is a cross-check, not the Q13 measure: its "prefill phase" is the client-visible [send, TTFT] window, which includes queue wait, so it removes more gaps than a pure prefill window would (`tools/cbsoak_steady.py`; checked on the soak2 B=8 data: steady p95 34.9 ms, n 79295).

GPU clock (macmon 1 s, whole slot): GPU MHz mean 1157 / 1333 / 1129 / 1191 / 1192 / 1220 / 1388 for the seven 5-minute bins of the soak slot (ramp excluded), utilisation mean 63 to 83%, GPU power mean 22 to 32 W, max temperature 97 C (first bin) then 84 to 93 C. No clock collapse; the lower mean clock than at B=8 (1.26 to 1.47 GHz) is the lower utilisation of a 58%-of-saturation B=4 run.

## 2. `--max-batch 1` A/B/A guard (+ keep-warm off)

Same slot (#2193), same seed 31, 0.3 req/s for 360 s, `--max-tokens-mix 16:2,48:3,96:1`, default prompt mix, no `--max-batch` flag, 80 requests and 3214 ITL gaps per arm. Arm order base, head, head with `INTARSIA_METAL_KEEPWARM=0`, base (the coordinator's addition). Each arm is a fresh server start with a warm-up before the window.

| arm | binary | keep-warm | requests | ITL n | ITL p50 | ITL p95 | ITL p99 | TTFT p50 | TTFT p95 | decode tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 base | 28e5c5a | none | 80 | 3214 | 27.9 | 28.6 | 28.9 | 387 | 2500 | 35.8 |
| 2 head | 66feda6 | on (default) | 80 | 3214 | 28.0 | 28.7 | 29.1 | 302 | 2421 | 35.7 |
| 3 head-kwoff | 66feda6 | off | 80 | 3214 | 27.5 | 28.2 | 28.5 | 386 | 2435 | 36.3 |
| 4 base | 28e5c5a | none | 80 | 3214 | 27.5 | 28.7 | 29.6 | 385 | 2447 | 36.2 |

| comparison (ITL, candidate / baseline) | p50 | p95 | limit | verdict |
|---|---|---|---|---|
| head / base pooled (arms 1 and 4) | 1.009 | 1.004 | 1.1 | **pass** |
| head / the earlier base run (arm 1) | 1.004 | 1.003 | 1.1 | pass |
| head / the later base run (arm 4) | 1.018 | 1.000 | 1.1 | pass |
| head-kwoff / base pooled | 0.991 | 0.984 | 1.1 | pass |
| head / head-kwoff | 1.018 | 1.018 | informational | |

Drift between the two base runs (arm 4 against arm 1, with head and head-kwoff between them): ITL p50 27.5 vs 27.9 ms (-1.4%), p95 28.7 vs 28.6 (+0.3%), p99 29.6 vs 28.9, decode 36.2 vs 35.8 tok/s. So slot drift is about 1.4% on p50 and 0.3% on p95. Everything the guard measures is within the 1.1x limit with at least 8 points of headroom, and the whole spread (0.984 to 1.018) is the same size as the base-to-base drift.

Does keep-warm move ITL? head is 0.5 ms (1.8%) above head-kwoff on both p50 and p95, which is larger than the p95 drift (0.3%) but about the size of the p50 drift (1.4%). One 6 min run per arm cannot separate a 1.8% effect from drift; there is no evidence that keep-warm costs decode ITL, and the 1.1x guard passes with or without it. TTFT p50 is the clear keep-warm signal: 302 ms with keep-warm against 385 to 387 ms for both baselines and for head-kwoff (-22%, about 85 ms, the re-wire stall of an arrival after idle); the TTFT p95 (2.42 to 2.50 s) is dominated by long prompts and does not show it.

GPU state per arm (macmon): mean utilisation 23 / 24 / 24 / 25%, mean GPU power 7.7 / 8.0 / 8.1 / 8.2 W; mean clock 573 / 639 / 651 / 645 MHz (low because a B=1 stream leaves the GPU 75% idle, same order for all arms; the baseline's first arm is the lowest, 10% below the others, which does not show in ITL). Idle-GPU sample before each server: 0 to 2.5% utilisation, 0.1 W (arm 4: 2.5%, 0.2 W, a single idle sample, see hygiene).

## 3. Part B: `--spec ngram` + prefix reuse identity (`partB/`, `partB-pilot/`, job #2194)

### 3.1 Setup

`slot_partB3.sh` (the soak2 `slot_partB.sh` with paths for this branch and an end-of-slot idle sample). Reference: `--max-batch 1 --spec ngram --prefill-chunk 256`, prefix cache on; `cbreuse.py gen --chains 120 --copy-frac 0.5` builds the chains (P, F = P + output + new tokens, F2) and records the solo results, 268 requests, 0 errors, 410 s. Test: one batched server `--max-batch 8 --prefill-chunk 256 --spec ngram` (spec-k default, reuse on, `INTARSIA_CB_TRACE=1`, `INTARSIA_SPEC_TRACE=1`), B concurrent clients (2, 3, 8), one chain per client at a time, the same 268 requests in each run, 15% of follow-ups preceded by a random pause. `cbreuse.py compare` checks each request against the reproduced predecessor: token ids, text, deltas, finish_reason, usage. Spec switches off at 4 or more decoding rows (`spec_off_at` 4, hard default), so B=2 and 3 are the spec-active runs and B=8 is mostly spec-off; this is the same design as the gates doc (`2026-10-16-metal-cb-gates-reuse-spec.md`).

Pilot first (2 min of work in the slot; the `--copy-frac` change had only been tested against a stub server): `CHAINS=8 BS=2`, 19 requests, gen exit 0, 19 / 19 pass, 0 mismatches, spec proposed 1163 accepted 658 in 291 verify passes, 11 requests started from decode-written KV (1526 tokens). Go; full run with CHAINS=120 (the 100 of the plan would give about 220 requests, a thin margin over 200 comparable).

### 3.2 Identity table

| run | sent | comparable | pass | mismatch | errors | not comparable | started from reused KV | of which decode-written KV (tokens) |
|---|---|---|---|---|---|---|---|---|
| B=2 (spec active) | 268 | 268 | **268** | **0** | 0 | 0 | 148 (55%) | 148 (17098) |
| B=3 (spec active) | 268 | 268 | **268** | **0** | 0 | 0 | 148 (55%) | 148 (17098) |
| B=8 (spec off at 4+ rows) | 268 | 263 | **263** | **0** | 0 | 5 | 143 (54%) | 143 (16460) |
| total | 804 | 799 | 799 | 0 | 0 | 5 | | |

Chain kinds, B=2 and B=3: ext 98, ext2 84, branch 50, exact 36 all pass; B=8: ext 97, ext2 82, branch 48, exact 36 pass, 5 not comparable. Prompts 74 to 2493 tokens, 147 streamed, 52 sampled (seeded), the rest greedy non-stream.

The 5 not-comparable requests at B=8 (c019s2, c033s1, c036s1, c072s1, c093s2) all started with cached 0 where the reference had cached 213 to 445: the idle slot that held their predecessor's KV had been reassigned to another chain in the meantime, so they were re-prefilled cold and the contract (same cached length) does not apply. They are counted and listed, not dropped (the gates doc saw the same effect, 6 of 328). Their output is not compared.

### 3.3 Drafting happened (counter deltas of `/metrics`, each run's before and after)

| run | spec verify passes | tokens proposed | tokens accepted | accept rate |
|---|---|---|---|---|
| B=2 | 3901 | 15577 | 9745 | 62.6% |
| B=3 | 3901 | 15577 | 9745 | 62.6% |
| B=8 | 25 | 100 | 96 | 96% |
| reference server (solo, whole `gen`) | 3901 | 15577 | 9745 | 62.6% |

The batched B=2 and B=3 runs drafted exactly as often as the solo reference (same verify passes, same proposed and accepted tokens), which is what token identity with spec predicts: the draft sequence is a function of the token stream. `tools/cbspectrace.py` on the trace (per `b` = decode rows): B=2: 11873 decode phases, 3901 verify passes, 15577 drafted, 0 spec-off phases; B=3: 7898 phases, 3565 verify passes at b=3, 0 spec-off; B=8: 3060 phases, 2906 spec-off phases, 25 passes, switch-off at b=4..8 (24 + 28 + 125 + 542 + 2187 phases with 0 verify passes, gate value b<4 violated 0 times), one on-to-off and one off-to-on transition. The decode-written KV reuse is the "from reused DECODE-WRITTEN KV" column: 55% of the comparable requests started from KV that the predecessor's decode wrote (in a spec-active run, so the KV written by verify passes), 143 to 148 requests and 16 to 17 thousand tokens per run, and all of them match the reproduced predecessor token for token.

### 3.4 Negative control

Three token ids were bumped by one (requests c008s0, c043s1, c089s1) in a copy of `conc-b2.jsonl` (the copy is not kept; `partB/negative-control.txt`). `compare` reports exactly those 3 (`MISMATCH 3`, `pass 265`, `TOKEN IDS differ at token 23/46`, 17/35 and 31/63) and exits 1. So the comparator sees a one-token difference in a spec-active run.

## Hygiene

* GPU idle before the server, from the 6 s macmon sample of each slot: 0 to 2.5% utilisation, 0.01 to 0.2 W in all runs (ramp 0.4 to 1.1%, soak 0.4 to 0.5%, B=1 arms 0.1 to 2.5%, Part B 0.7 to 3.6% of five samples). End-of-slot idle sample (5 s, after every server of the slot had stopped): soak 0.2%, Part B 0.4 to 2.3%, B=1 slot 0.7 to 2.9% (0.04 to 0.26 W); `leftover servers: none` in all three. The #2191 end sample was taken by hand after the soak (`end_sample.sh`), the other two by the slot scripts.
* `endor-kev` (`kev.serve`, pid 44338) and `endor-tensorfold` (`tensorfold serve`, pid 33226 plus its proxy) were up in every end snapshot and idle on the GPU in every idle sample. Nothing else used the GPU at the start of any slot.
* Foreign load on the machine (CPU, not GPU, flagged not quantified): `mediaanalysisd` at 205 to 216% CPU in the soak and B=1 slots (up 17 h); `Syrtis.app` at 48% at the start of Part B; a stray `macmon pipe -i 200` (pid 25718, up 17 h, not mine) at 15 to 20% CPU for the whole session; Claude Code processes. Swap 10.4 GB used of 11.3 GB in every snapshot.
* Warm-up before each measured window: the harness' warm-up requests (soak, ramp, B=1) and a warm-up request before each scenario repetition. Part B has no timing and needs none.
* `intarsia_spec_*` counters, `ps` and macmon logs: in each run directory (`hygiene.txt`, `hygiene-end-idle.txt`, `macmon.jsonl`, `macmon-end-idle.jsonl`).

## Engine bugs found

None. Observations, not bugs: Part B at B=8 loses reuse for 5 of 268 requests because an idle slot was taken by another chain (expected `pick_slot` behaviour); `slot_partB.sh` prints `no intarsia serve left` via `pgrep -fl`, which matches any shell whose command line mentions "intarsia serve" (cosmetic).

## Tools added or changed

* `tools/cbsoak_steady.py` (new): soak-derived steady-state ITL (gaps with no other request in its prefill phase), `--skip-s`, `--max-p95`.
* Slot scripts in `docs/results/raw/2026-10-16-metal-cb-soak3/`: `slot_ramp4.sh`, `slot_soak3.sh` (scenario with `--max-steady-itl-p95-ms 60` then the soak), `slot_b1.sh` (base, head, head-kwoff, base), `slot_partB3.sh`, `end_sample.sh`, `env.sh`.

## Reproduce

```
cargo build --release -p intarsia-cli --features metal   # copy target/release/intarsia to a scratch dir, set it in env.sh
D=docs/results/raw/2026-10-16-metal-cb-soak3
fleetq acquire --host endor --exclusive --mem 24 --label "cbsoak3 B=4" --wait   # then FLEETQ_JOB=<id> for the two lines below
$D/slot_ramp4.sh $PWD/$D/b4-ramp
$D/slot_soak3.sh $PWD/$D/b4-soak 4 0.33 3 1800
fleetq run --host endor --exclusive --mem 24 --label "cbsoak3 B=1" -- $PWD/$D/slot_b1.sh $PWD/$D/b1
FLEETQ_JOB=<id> CHAINS=120 BS="2 3 8" $D/slot_partB3.sh $PWD/$D/partB
uv run tools/cbsoak_ab.py --base $D/b1/b1-base1 $D/b1/b1-base2 --cand $D/b1/b1-head
uv run tools/cbsoak_steady.py $D/b4-soak --skip-s 360 --max-p95 60
```

Raw: `docs/results/raw/2026-10-16-metal-cb-soak3/{b4-ramp,b4-soak,b1,partB-pilot,partB}/` (each: `raw.json`, `requests.jsonl`, `report.{json,md}`, `scenario.{json,md}`, `server.log`, `macmon.jsonl`, `hygiene*.txt`, `metrics*.txt`; Part B: `chains/`, `conc-b*.jsonl`, `compare-b*.log`, `report-b*.json`, `spectrace.{txt,json}`, `negative-control.txt`) and the `*.out` slot logs.

`uv run tools/cbsoak.py selftest` still passes (nothing in `cbsoak.py` changed; run after adding `cbsoak_steady.py`).
