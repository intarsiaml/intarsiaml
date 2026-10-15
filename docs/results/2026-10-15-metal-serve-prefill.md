# Metal serve-path prefill: why it looks slower and noisier than plain `generate`

Date: 2026-10-15 (03:23-04:55 EDT), endor (MacBook Pro Mac17,6, Apple M5 Max, 140 W adapter, `pmset powermode 2`),
Qwen3-8B BF16, branch `metal-serve-prefill` from `spike/metal` b1f15ad (SAFE recipe: prefill chunk 256).
Raw data and scripts: `docs/results/raw/2026-10-15-metal-serve-prefill/`. fleetq jobs 1967-1973 (all released).

## Outcome

**No code fix. The slowdown is not in intarsia's serve path; it is the GPU running the same kernels at a lower clock.**

1. At a matched GPU state, serve and plain `generate` give the same prefill. Interleaved in one slot (pvs1, 16
   iterations, 4062-token prompt): plain `generate` 1308 ms median (1284-1336), serve after 5 s idle 1289 ms
   (1275-1302), serve back-to-back 1172 ms (1147-1193). Plain is "tight" because every plain run starts from the same
   state (GPU idle during model load, so it pays the GPU wake-up); it never follows a decode burst or another prefill.
2. The serve path adds nothing measurable: client TTFT minus the engine's own prefill time is 1-8 ms (tokenization,
   chat template, HTTP, SSE all included), the engine thread burns 4-5 ms of CPU per 4k prefill, and the GPU is idle
   between command buffers for 0.4% of the prefill (about 5 ms of 1120 ms), including in the slowest prefills.
3. The variance (server-side prefill 1147 ms to 2096 ms for the identical 4k prompt on one server) is the GPU
   executing the same command buffers more slowly: per-command-buffer GPU time 23.8 ms in fast prefills, 29-36 ms in
   slow ones, with no added gaps. macmon shows ~1550-1600 MHz / ~85-90 W GPU in fast prefills and ~1100-1300 MHz /
   35-50 W in slow ones. Across 136 requests the correlation of prefill time with mean GPU clock during the request is
   -0.96 (GPU power -0.93); with P-core utilisation +0.04, E-core utilisation -0.06.
4. Why the clock drops is **not identified** (no thermal/performance warning in `pmset -g therm`, CPU load does not
   explain it, GPU temperature 63-92 C without a clear threshold). Which preceding load is "slow" changes between
   runs: after a 256-token decode was slow in base1/trace1/long1 and fast in mac1/cb1/more2, while back-to-back prefill
   was slow in mac1/more2 and fast elsewhere. So a build-vs-build comparison on the serve path cannot be read from a
   single localeval pass; the lead's 1712-3501 tok/s swing is this effect, not a SAFE/CAND difference.

Only fixed, reproducible serve-path cost found: a GPU wake-up of about 120-140 ms when the GPU has been idle for 2 s or
more (not at 1 s). Plain `generate` pays the same cost. It is hardware power-state latency, not something the engine
can remove without keeping the GPU busy while idle, which I did not consider acceptable.

## Experiments

All on one fresh `intarsia serve` (SAFE, `--max-seq 8192 --no-supervisor --no-ui`), nonce-prefixed 4062-token prompts
(`prompt-4k.txt`), `exp.py` records client TTFT and the engine's own prefill ms (from `/metrics`
`intarsia_last_prefill_tokens_per_second`), macmon at 100 ms for GPU clock/power. Conditions per round, rotated: A =
4k immediately after a 256-token decode; B = decode, 5 s idle, 4k; C = 4k right after a 4k (max_tokens 1); E = the
localeval pattern (4k+256 decode, then 4k+256 decode; the second is measured); G1/G2 = decode then 1 s / 2 s idle;
H = 32-token decode then 4k. Server-side prefill ms, median (min-max):

| run (time) | A after decode | B decode+5 s | C b2b prefill | E localeval | G1 1 s | G2 2 s | H decode32 |
|---|---|---|---|---|---|---|---|
| base1 (03:25) n=12 | 1380 (1147-1779) | 1430 (1277-1527) | 1358 (1148-1555) | 1418 (1277-1813) | | | |
| trace1 (03:35) n=12 | 1712 (1140-2064) | 1308 (1279-1581) | 1196 (1144-1547) | 1550 (1174-2096) | | | |
| mac1 (03:45) n=14 | 1152 (1140-1294) | 1480 (1275-1529) | 1373 (1149-1476) | 1248 (1182-1373) | 1161 (1158-1348) | 1293 (1282-1538) | 1150 (1149-1297) |
| cb1 (04:00) n=12 | 1151 (1141-1733) | 1311 (1298-1355) | 1150 (1142-1339) | 1151 (1150-1296) | | | |
| long1 (04:15) n=22 | 1527 (1150-2067) | 1318 (1280-1593) | 1363 (1149-1580) | 1432 (1189-1847) | | | |
| more2 (04:32) n=14 | 1155 (1140-1250) | 1368 (1304-1476) | 1410 (1146-1480) | 1176 (1149-1263) | 1161 (1159-1376) | 1365 (1302-1504) | 1186 (1150-1426) |

Floor (GPU at full clock, no wake-up) is ~1148 ms; GPU-idle wake-up adds ~120-140 ms (B, G2).

| hypothesis | experiment | result |
|---|---|---|
| tokenization / chat template / HTTP counted in TTFT | client TTFT - engine prefill (all runs, ~400 requests) | 1-8 ms median per condition. Not a factor. |
| serve uses a different chunk size | server log | `prefill chunk 256 (default)`, same as plain SAFE. |
| prefix-cache lookup/copy | nonce-prefixed prompts, `cached_tokens` | 0 cached; engine path (`generate_one`) is shared with `generate`. Not a factor. |
| host thread starving GPU submission (QoS, E-core) | `INTARSIA_PREFILL_TRACE` patch (`prefill-trace.patch`): engine-thread CPU time per prefill, trace1 | 4-5 ms CPU per 1.1-2.1 s prefill (chunk calls wait on the GPU, 65-76 ms each in fast prefills, up to 100-140 ms in slow ones). Thread is blocked, not busy. Not a factor. |
| command-buffer commit cadence / sync gaps | `INTARSIA_METAL_PROFILE=cb:` GPU timestamps, cb1, 74 prefills | median idle between CBs 5 ms of 1124 ms (0.4%, max gap 0.4 ms) in fast prefills and 6 ms of 1383 ms in slow ones. The only gaps (17-49 ms) are a first gap after GPU idle. Slow prefills = busy per CB 29.3 ms median (max 35.8) vs 23.8 ms. Not a factor. |
| GPU clock/power state after decode, idle ramp | macmon, long1 + more2 + pvs1 (136 requests) | corr(prefill ms, GPU MHz) -0.96, GPU power -0.93. Bins: <1200 ms: 1554 MHz / 81 W; 1200-1350: 1432 / 60 W; 1350-1500: 1269 / 43 W; >=1500: 1097 / 32 W. This is the cause of the slowdown. |
| CPU contention from other processes | macmon P/E-core utilisation (same 136 requests), `ps` top-6 sampling | corr +0.04 / -0.06; foreign CPU ~70% of a core in both fast and slow requests (66 vs 71 in long1, 72 vs 70 in more2). Not the trigger. |
| decode burst causes it | conditions A/E vs C/B/G/H across six runs | regime flips between runs (table above); G1 (1 s) is as fast as the floor in both mac1 and more2. Not a stable cause. |
| serve vs plain at matched state | pvs1 interleaved | 1308 / 1289 / 1172 ms (plain / serve idle 5 s / serve b2b). Same. |
| localeval's own measurement | `leval-ttft.txt` (remeasure runs 03:08-03:20) | see below. |

Slow-prefill time series (more2, C at 04:36:39, 1480 ms): GPU 1617 MHz / 90 W until the preceding prefill ended, then
1120-1230 MHz at a flat 33-38 W for the whole request while `sys_power` read 116-166 W and the CPU was ~22% busy.
`series.py more2 C_b2b 0` reproduces it. Similar plateau at 38 W in long1 (A at 04:17:26). One request (long1 A at
04:18:35, 2067 ms) fell to 680 MHz / 42% utilisation. At the start of more2 `ioreg` read "Device Utilization 99%"
before any request was sent to the fresh server, so other GPU clients do exist on this machine (WindowServer, Safari,
BambuStudio and others hold GPU contexts); I did not isolate them.

### How this explains localeval

localeval computes marginal prefill as (4096-1024 tokens) / (median 4k TTFT - median 1k TTFT) over 3 reps per size.
The remeasure passes (`leval-ttft.txt`) show the 1k TTFT itself moving 0.26 s to 0.74 s and 4k 1.15 s to 2.46 s:

| pass | 1k TTFT min/med/max | 4k TTFT min/med/max | marginal tok/s |
|---|---|---|---|
| safe-1 | 0.33/0.51/0.67 | 2.21/2.31/2.41 | 1720 |
| safe-1-warm | 0.54/0.64/0.74 | 2.36/2.45/2.46 | 1712 |
| cand-1 | 0.52/0.56/0.59 | 1.97/2.02/2.07 | 2126 |
| cand-1-warm | 0.42/0.63/0.65 | 1.73/1.74/1.83 | 2795 |
| cand-2 | 0.26/0.26/0.27 | 1.51/1.90/1.90 | 1900 |
| cand-2-warm | 0.28/0.56/0.58 | 2.09/2.11/2.16 | 1990 |
| safe-2 | 0.27/0.27/0.27 | 1.15/1.15/1.20 | 3501 |
| safe-2-warm | 0.30/0.30/0.30 | 1.72/1.78/1.84 | 2092 |

A fast-state pass (0.27 s / 1.15 s) gives about 3500 tok/s; a slow-state pass gives 1700-2100 tok/s. Medians of three
reps cannot tell the two states apart from a build difference, and the 0.9-1.8 s denominator amplifies it. safe-2
(3501) and safe-1 (1720) are the same binary. The Metal-specific piece is that Apple's GPU clock/power state moves
by up to ~2x within minutes on this laptop, independent of the inference process.

## Fix and gates

None applied. The branch tip is `spike/metal` b1f15ad plus this document and raw data; no source files changed, so
the code gates (metal/workspace tests, `cargo check`, `kernels test`, goldens, greedy ids, `qualify`, localeval ABBA)
were not run; they would test nothing. The instrumented binary (`prefill-trace.patch`: an `INTARSIA_PREFILL_TRACE`
per-chunk wall/CPU print in `intarsia-core/src/engine.rs` plus a `libc` dependency for `intarsia-core`) was used only
for experiments and is not committed, because it touches core/shared code and would need the full gate suite.
`cargo build --release -p intarsia-cli --features metal` succeeded on the clean branch (before and after the patch).

Options considered and rejected: keeping the GPU busy while idle to avoid the 120-140 ms wake-up (power cost, does not
address the clock cap); changing commit cadence / chunk size / thread QoS (all shown not to be the bottleneck);
a Metal command-queue GPU priority (no such API in objc2-metal 0.3.2, and the cause is the clock, not scheduling).

## Recommendations for measurement

1. Compare builds on the serve path by the engine's own prefill time, not by a single localeval pass, and only
   within one state: log `macmon pipe -i 100` (GPU MHz, GPU W) during each pass and discard prefills whose mean GPU
   clock is below ~1500 MHz (`mac_an.py`, `cpu_an.py` show how). Or interleave builds in time (plain `generate` ABBA as
   done in remeasure is the right shape, and it is robust because each run is a fresh process).
2. Use more reps and report min or a trimmed median; a 5 s idle gap does not normalise the state (B ranged 1275-1593).
3. Expect plain `generate` prefill ~120-140 ms above the serve back-to-back figure for the same reason (GPU wake-up).
4. To find why the clock drops, someone with sudo can run `powermetrics --samplers gpu_power,thermal -i 200` during a
   slow stretch (GPU "performance limit"/throttle reason); I did not use sudo. Running mlx-lm or llama.cpp through the
   same sequence of conditions would confirm the effect is engine-independent.

## Files

- `run-exp.sh`, `exp.py` (conditions A/B/C/E/G1/G2/H), `run-pvs.sh`, `one.py` (plain vs serve), `macmon_log.py`,
  `psample.sh`: drivers. Results: `{base1,trace1,mac1,cb1,long1,more2,pvs1}.tsv`, `*-macmon.tsv`, `*-ps.txt`,
  `*-server.err`, `cbprof1.jsonl.gz` (GPU command-buffer trace), `summary.txt`, `cb1-analysis.txt`, `cpu-analysis.txt`,
  `*-analysis.txt`, `leval-ttft.txt`. Analysis: `summary.py`, `cb_an.py`, `mac_an.py`, `corr.py`, `cpu_an.py`,
  `ps_an.py`, `series.py`, `leval_ttft.py`. Experiment-only patch: `prefill-trace.patch`.
- Binaries: SAFE b1f15ad (`intarsia-base`) and the same plus the trace patch (`intarsia-trace`), built from this tree.
