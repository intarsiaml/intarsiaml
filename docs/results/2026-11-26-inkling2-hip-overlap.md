# Inkling-2-Mini prefill / decode overlap on HIP, one R9700 (lane ink-hip-overlap, 2026-11-26)

Branch `ink/hip-overlap` from the `ink/hip-bpf` head 2561d5957 (PR #538: packed prefill, default-on with `--max-batch`), in
`.worktrees/ink-hip-overlap`. User (via the coordinator, 2026-11-26): "build the prefill overlap too". Goal: cut the ITL p99 at 8 and
16 streams (0.59-0.65 s today, about one 2048-row prefill chunk) while every stream stays bitwise equal to its solo run.

## Overlap stalls (lane ink-hip-overlap2)

### Brief (coordinator, 2026-11-26 23:38 EST; written here 23:36 by `date`; the user is asleep and said "keep working")

In substance, every rule kept:
- Background: PR #542 (draft, `ink/hip-overlap` @ 2b3550800) failed gate 4 at 8 streams (ITL p99 ratio 0.422 [-0.02, 0.863], limit
  0.70): S3's B had three stalls of 0.34-0.41 s, cause unknown. Also against overlap: ITL p90 1.3-2.1x, TTFT max 1.07-1.21x, ITL p99
  at 4 streams worse in 2 of 3 windows. Parked as a draft: ~350 lines of shared `scheduler.rs` / `engine/batch.rs` plus HIP
  `device.rs` and `ik_runtime.{h,hip}`, no Metal or Qwen-on-HIP no-op check, gate failed.
- Where: this worktree, scratch under `.scratch/`; this doc, new top section (this one); raw under `raw/` (`s1/`-`s5/`,
  `speed_sum.py`); tatooine tree `~/intarsiaml-ink-hip-bpf` only (o4, main5 in `.scratch/bin/`); fleet / window rules in
  `.worktrees/hip-swarm/COMMON-1126.md` (this brief wins where they differ).
- Step 1 (Mac only, no GPU): line up S3's three stalls against chunk boundaries and sizes, admission / plan changes, allocations and
  workspace growth, the minimum decode interval while a chunk is in flight, the stream waits at both handoffs, the per-stream
  counters, graph capture / replay misses, the request arrival pattern; check S1 / S2 / S5 for smaller versions. One hypothesis and
  how to test it; if the logs cannot decide, the cheapest opt-in instrumentation (no change to the default path).
- Step 2: merge `origin/main` (a merge, never a rebase) only once `gh pr view 544 --json state` is MERGED. Expect conflicts: #543
  `model/batch.rs`; #541 and #544 `model/inkling2.rs`; #544 HIP `device.rs`. Then the full cpu suite on endor (`fleetq run --host
  endor --load cpu`): core lib, `tests/scheduler.rs`, models tests, CLI `docs_env_vars`.
- Step 3: fix the cause if found; regate with gate 4 exactly as section 4 pre-registered (no threshold change): paired windows vs
  main, S1-S3's shapes, n >= 3; report ITL p90 and TTFT max next to the gated metrics.
- Step 4, no-op gates before any merge: Metal (greedy md5 main vs branch on Inkling + the Metal tiny suite, one exclusive endor
  slot <= 15 min, queued behind the other Metal lane); Qwen on HIP (greedy md5 main vs branch for the 35B in intarsiaml's own
  container, in a radiance window, `INTARSIA_OVERLAP` unset and `=1`; batched serve at 4 streams equals solo); Inkling HIP (every
  check of section "Bitwise" on the merged head).
- Step 5: update #542, keep it a **draft**, `INK_OVERLAP_DEFAULT = false`; results here, speed table first; LOG.md row; unanimis
  capture (no `project` / `scope`, `intarsiaml` in `source`). If gate 4 and the no-op gates pass, say so; default-on is the user's
  call in the morning; nothing default-on merged tonight. PR body keeps the Claude Code line last; commits end with the
  Co-Authored-By line.
- Budget: radiance <= 15 min in total, each window <= 5 min, none after 06:00 EST; lane ink-hip-specflip has priority on radiance
  (queue behind its windows, never start ahead of its queued jobs). Metal: one exclusive slot <= 15 min. Every fleet command through
  fleetq with an honest `--expect`; builds / tests `--load cpu`; windows `--host tatooine-r9700 --takes-down radiance --quiet
  gpu,cpu`. Never `fleetq release` a running window job (abort = kill the driver's tree, then confirm `curl -s
  http://tatooine:8080/v1/models`); confirm radiance answers after every window. No tool call or wait over 4 min. No new folders,
  none deleted.
- Stop rules: no testable hypothesis from step 1 -> report that, no blind reruns. Two failed fix attempts -> stop and name the
  assumption that might be wrong.
- Final report <= 15 lines: stall cause with evidence; speed table (gate-4 metrics with 95 % CI, ITL p90, TTFT max); gate-4
  verdict; each no-op gate; PR head sha; radiance and Metal minutes; not verified; fleetq ids.

Later go-aheads from the coordinator are written below as they arrive.

- **Coordinator, 23:59 EST 11-26** (received 23:56 by this lane's `date`; as written):
  "The gate-4 amendment is OK: regate S6-S8 with the real-prompt client running on tatooine. Thresholds, shapes and order stay
  unchanged, and the endor-client FAIL stays on record next to the new result.
  Conditions:
  1. **Report both clients' views.** For each window and both arms, also report the server's own ITL histogram and p99. That way the
     client move can be seen not to change the engine-side picture.
  2. **Re-report the ungated metrics with the new client:** ITL p90 and TTFT max, which the endor client showed at 1.3-2.1x and
     1.07-1.21x. Report them even if they still look bad. The user decides default-on in the morning, and those numbers matter to
     that decision.
  3. **Keep the client off the GPU cards.** Pin it to a few CPU cores away from the server's busy threads if that is easy, and record
     where it ran. Keep the same client placement for both arms.
  The rest of your brief still applies: radiance <= 15 min, behind the flip lane; the Metal and Qwen-HIP no-op gates; #542 stays a
  draft with `INK_OVERLAP_DEFAULT = false`."
- **Coordinator go-ahead, 00:06 EST 11-27** (sent as 00:10, corrected by the coordinator to 00:06; in substance, every condition
  kept), answering this lane's 00:08 question (W0's card-1 main6 priming failed: `SENV` unbound in `stage_lib.sh` under `set -u`):
  **option (a)**: keep AB / BA / AB; this lane's radiance cap raised to **16:00 total**; S6 <= 5:00 by itself. Conditions:
  (1) main6's ~100 s admission priming must finish before S6's measured A run starts, in the same warm state o6 gets; if priming
  would fall inside the measured period (an unprimed main6 would make main look slow and bias toward overlap), switch to (b)
  BA / AB / BA and note why here; (2) fix the SENV-unbound stage bug before S6 so S7 / S8 cannot hit it, and check card 1's main6
  server is actually up before the window starts; (3) hard stop: if the total would pass 16:00, finish the current window and stop,
  no S9; never cancel a running window job; (4) write this go-ahead here now. Everything else unchanged: thresholds as
  pre-registered, the endor-client FAIL stays on record, overlap stays opt-in.
- **This lane's choice under condition 1 (00:14): (b), order BA / AB / BA.** Why: under AB, main6 cannot be primed *and* start in
  the state o6 gets (a fresh process on a cached admission) inside S6's 5:00. Priming it before A (on either card) puts S6 at ~5:13
  (4 s stop + ~105 s admission + 2 x ~89 s visits + ~26 s end / restart). Letting A's own first start admit (it answers /health only
  after admission, so no measured request overlaps it) gives ~4:53 with <= 7 s left before the 270 s visit cap would cut B's filler,
  and A's server would have run 134 admission cases before serving while o6's had not. Under BA, card 1 primes main6 while card 0
  measures B (o6, primed in W0); B's visit ends with `wait:` for card 1's marker (server idle, after its measured cells) and
  `WAIT_STRICT=1` skips A if the marker is missing (the window is then void), so both arms start from a cached admission. S6 ~3:50,
  S7 / S8 ~3:20: about 14:25 in total with W0. Order balance: B goes first in 2 of 3 windows (A did under AB / BA / AB).
- Validity rule correction (before S6): the engine prints "prefill/decode overlap: on ..." only when overlap is on **by default**
  (`engine/batch.rs` ~1412, `opts.overlap.is_none()`); with `INTARSIA_OVERLAP=1` it prints nothing (W0P's log: none). The pace is
  still 150 ms (`engine/batch.rs` ~1358: `overlap_pace_default_ms` applies whenever `INTARSIA_OVERLAP_PACE_MS` is unset). Valid B
  visit therefore: "stream 1 (decode lane) at priority -1" in its log, no "overlap unavailable" line, and
  `intarsia_batch_overlap_steps_total` > 0 at its end.
- SENV fix (condition 2): `stage_lib2.sh` sets `SENV=${SENV:-}` before any `srv`; dry check on tatooine under `set -u` (no GPU):
  sourced, `SENV=[]`. Card 1's main6 server coming up is checked inside S6 by the marker (written only once it answered).

### Step 1: S3's three stalls are in the client path (endor, Wi-Fi, Tailscale), not in the engine (Mac only, 23:36-23:46)

Scripts and outputs in `raw/`: `stalls.py` (multi-stream gap events vs arrivals, first tokens, prefill spans), `steps.py` (the decode
step timeline), `srvitl.py` (the server's own ITL / step histograms per real level, from the 0.2 s poll), `coalesce.py`;
`stall-s3-B8.txt`, `stall-server-itl.txt`, `stall-coalesce.txt`.

1. **The server never saw the stalls.** `intarsia_batch_itl_seconds` is recorded in `scheduler.rs` `token_done` at each token's
   emission (one `now` per decode step), before `on_token` hands it to the HTTP stream. Over S3 B's 8-stream level it has **0 of
   1016 gaps above 0.25 s** (buckets: <=20 ms 30, <=35 ms 817, <=100 ms 11, <=150 ms 108, <=250 ms 50), while the client saw 19 gaps
   of 0.34-0.41 s. The same holds for B at 8 and 16 streams in S1, S2 and S3 (server ITL p99 in the 150-250 ms bucket every time);
   A's 0.5-0.66 s chunk waits do show in the server histogram ((0.25, 1] s: 30-48 at 8 streams).
2. **Each stall is a "late pair".** The stalled token arrives 0.04-2 ms before the next one, in every decoding stream at the same
   instant (S3 B: 5, 6, 7 streams at 1.73, 2.71, 3.74 s into the run; gaps 0.368 / 0.344 / 0.410 s, each followed by 0.002 / 4e-5 /
   0.001 s). A decode step takes at least ~10 ms (the histogram's smallest bucket), so two tokens of one request 2 ms apart cannot
   both come from the scheduler at those times: the first was held after emission and delivered with the second.
3. **Not overlap-specific.** S3 A (main5, no overlap) has the same signature (one 7-stream late pair of 0.31 s), and so has O3's N3
   (b13, no overlap: 15 late pairs at 16 streams). No other S1 / S2 visit has one.
4. **Only the Mac client shows it.** Card 0's real prompts are sent by `serve_bitwise.py` on endor; card 1's (S1 poison, S3 pace
   arms, S4) by the same script on tatooine. Gaps under 5 ms: **2-28 % of all gaps in every endor-client run** (O1-O4, S1-S3, A and
   B), **0 in every tatooine-client run** (O2 poison, S1 card 1, S3 card 1, S4), and 0 late pairs there. Card 1's o4 pace-150
   server (B's binary and config) ran its 16 real prompts at 21:31:31-21:31:47, the seconds of card 0's three stalls: its local
   client saw 0 late pairs, 0 gaps under 5 ms, max gap 0.37 s (the ordinary step beside a chunk).
5. **What differs on endor.** endor reaches tatooine through Tailscale (`utun4`, direct over the LAN) on Wi-Fi (`en0`); a ping now
   (endor running two cpu jobs) is 2.8-76 ms. S3 ran inside the exclusive Metal slot #12360 (ink-sampled-adapt, seeds / calibration /
   speed, 21:17:04-21:42:55) and O3 beside cpu tests #12306 / #12308; S1 ran with endor idle. S2 also fell in #12360 and O4 beside
   cpu tests #12330 without a late pair, so the load raises the chance; it does not cause one every time.
6. Smaller versions: S1 B's single 0.375 s gap (one stream, 4.39 s in) and S2 B's 0.293 s are absent from the server histogram too.
   Nothing in the engine lines up with the three events: they fall mid-pass (passes end at 1.60, 2.24, 3.58, 5.02 s), no arrival or
   admission is near them (all 8 arrive within 2 ms at t = 0), and the steps on either side are ordinary (0.11-0.16 s beside a chunk).

**Hypothesis.** S3's 8-stream ITL p99 for B (0.368 s, ratio 0.627, the one window that broke the interval) is client-path
delay on endor (Wi-Fi + Tailscale + a loaded laptop), not the overlap; the engine's tail at 8 streams is the 150-250 ms bucket in
all three windows. **Test.** Run the real prompts with the client on tatooine (the script and prompts unchanged, as the card-1 stages
do) for both arms: the gaps under 5 ms must drop to 0 and the 8-stream B tail must match the server histogram. Recorded before
any GPU run: 2026-11-26 23:46 EST. No engine instrumentation is needed: the server histogram plus the client's gaps decide it.

The endor path has one more hop: the Mac reaches the server through `fwd.py` on tatooine (a one-thread asyncio TCP relay,
tatooine:18132 -> the container's 8132), started by `serve_ctl.sh`. The card-1 clients connect to the container's address directly.
So "client path" here means endor's Python client + Wi-Fi + Tailscale + `fwd.py`; the test below bypasses all four.

What the client path does **not** explain (engine costs, from the server histogram and section 3): at 4 streams B has 32-44 of
~480 gaps in the (75, 250] ms buckets per window and A 0-3 (A instead has 7 chunk waits in (0.25, 1] s); a step beside a chunk is
110-150 ms, so B's 4-stream ITL p99 (0.16-0.18 s) is above A's (0.11 s) when A's few chunk waits stay under the 1 % line. ITL p90
1.3-2.1x and TTFT max 1.07-1.21x come from the same mechanism (decode steps share the GPU with the chunk). Only S3's 8-stream
outliers come from the client path.

Bound from the existing data (not the gate): at 8 and 16 streams, in all three windows, the server's B p99 sits in the
(0.15, 0.25] s bucket and A's in (0.5, 1] s, so the server-side per-window ratio is at most 0.25 / 0.5 = 0.5.

### Gate 4 amendment (2026-11-26 23:55 EST, written before any regate window)

Gate 4 as pre-registered (endor client, S1-S3, o4 vs main5): **FAIL** at 8 streams; that verdict stands as recorded. The regate
changes one thing, where the real-prompt client runs: `serve_bitwise.py` on tatooine's host against card 0's container address
(`http://<ip>:8132/v1`), not on endor through `fwd.py`. Unchanged: the thresholds (a)-(c), the shapes (real 4 / 8 / 16, nonce per
window, `REAL_OFF` = window index 0 / 1 / 2 as S1-S3; filler 1k x 1 / 4 / 8 / 16 by localeval from endor), card 0, `serve
--max-batch 16 --max-seq 4608 --kv-fraction 0.9`, order AB / BA / AB, n = 3, mean and 95 % t interval of B / A per window.
- A = main6 (origin/main 1bf40e29f); B = o6 (the merged head 408e902b9 or its doc-only descendants) with `INTARSIA_OVERLAP=1` (the PR's
  default is off; `=1` is the measured config: pace 150 ms, the decode lane at priority -1). A B visit with no env would be A again.
- Valid B visit: its log has "prefill/decode overlap: on" and "stream 1 (decode lane) at priority -1", and
  `intarsia_batch_overlap_steps_total` > 0 at its end. Otherwise the window is void.
- Kill rule for the step-1 hypothesis: if the tatooine-client B at 8 streams still shows a late pair or any gap under 5 ms, the
  hypothesis is dead; then stop (no further speed windows) and report.
- Reported side by side: ITL p90 and TTFT max (not gated), and the server-histogram p99 bucket per level.

Radiance plan (15 min total, each window <= 5 min, none after 06:00, behind ink-hip-specflip's queued windows):
- W0 (~4 min, checks + admission priming): card 0 o6 `=1` batched (primes o6) then the Inkling bitwise set (real 4 x 4/4, 2 x 8/8,
  16/16 vs W4's solo replies; o6 unset 8/8 and 16/16); card 1 main6 batched (primes main6) 16/16, then o6 poisoned `=1` 16/16 and
  4 x 4/4.
- S6 (AB), S7 (BA), S8 (AB), ~3:10 each: card 0 the speed visits; card 1 in turn: Inkling greedy md5 o6 vs main6 (spec off and
  the DSpark default, short / 4k, 64 / 256) and G2; the 35B no-op (Qwen3.6-35B-A3B-NVFP4, `generate --spec off` 256 tokens short /
  4k: main6 and o6 each with `INTARSIA_OVERLAP` unset and `=1`; serve `--max-batch 4` at 4 streams vs the solo server, o6 unset /
  `=1`, main6 `=1`); hip tiny `inkling2_batch` / `inkling2_packed` / `inkling2` on the o6 tree, the solo server if time is left.
  (Superseded by the 00:06 go-ahead and the (b) choice above: S6 BA with card 1 priming main6 and Inkling md5; S7 AB with the 35B
  md5 / solo / `--max-batch 4` unset; S8 BA with the 35B `=1` arms, G2, hip tiny.)

Binaries (tatooine `~/intarsiaml-ink-hip-bpf/.scratch/bin/`, fleetq #12477, `raw/build6.sh`): **o6** = merged head 408e902b9
(`--features hip`, md5 d6fa5dffedfd1488c908c1c7dfd3305a), **main6** = origin/main 1bf40e29f (`.scratch/main-src`, md5
0aa06968fe983af300747aa4bd01b1bc); hip test builds `inkling2_batch` / `inkling2_packed` / `inkling2` for both trees. Later commits on
the branch are doc / script only.

### W0: checks + o6 priming (fleetq #12484; radiance down 00:00:33 - 00:04:26 = 3:53)

`raw/window_r.sh w0`, raw `raw/w0/` (card 1 `raw/w0/card1/`). Real-prompt client on tatooine pinned to cores 10, 11, 22, 23
(`taskset`; card 1's own client on 4, 5, 16, 17).
- Card 0, o6 `INTARSIA_OVERLAP=1` (first start: admission 23 kernels, 134 cases run, up after 103 s; "stream 1 (decode lane) at
  priority -1"; 216 decode steps beside chunks in flight): **conc:4 waves 0-3 4/4 each, conc:8 waves 0-1 8/8 each, 16 at once 16/16
  equal to W4's solo replies.**
- Card 0, o6 with `INTARSIA_OVERLAP` unset (the PR default = main's batching path; 0 overlap steps): **8/8 and 16/16 equal.**
- Card 1, **o6 poisoned** (`INTARSIA_HIP_POISON=ff INTARSIA_OVERLAP=1`, cached admission, decode lane at -1, 146 overlap steps):
  **16/16 at once and 4 x 4/4 equal.**
- Card 1, main6's priming did **not** run: `stage_lib.sh` reads `SENV` under `set -u` and the first `srv` call had none (fixed).
- Client path: all 14 tatooine-client runs above have **0 gaps under 5 ms and 0 late pairs**. Not gated, single runs: o6 `=1` real 8
  ITL p99 0.192 / 0.190 s (waves 0 / 1), 16 at once 0.215 s; o6 unset 0.584 / 0.625 s.

### S6: regate rep 1, order BA (fleetq #12492; radiance down 00:11:20 - 00:14:49 = 3:29)

`raw/window_r.sh 6` (BGCAP 195), raw `raw/s6/`, card 1 `raw/s6/card1/`; summary `raw/regate_sum.sh`.
- B (o6 `=1`): up after 18 s (cached), decode lane at -1, 206 overlap steps, no "unavailable"; after its filler it waited 20 s
  (server idle) for card 1's marker. A (main6): started at 00:13:11 after the marker, **up after 18 s from the cached admission**
  (card 1's prime: 134 cases run, up after 105 s, KV 73728); both arms KV 73728, context 4608.
- Complete: real 4 / 8 / 16 (nonce s6) and filler 1 / 4 / 8 / 16 for both, every cell exit 0. **B = A replies 4/4, 8/8, 16/16.**
- Client path (kill rule): **0 gaps under 5 ms, 0 late pairs** in all six real levels (A and B). Hypothesis survives.
- Server histogram (p99 bucket, A -> B): real 4 (0.5, 1] s -> (150, 250] ms; real 8 (0.5, 1] s -> (150, 250] ms; real 16 (0.5, 1] s
  -> (150, 250] ms. Client ITL p99 A -> B: 0.097 -> 0.156 (4), 0.584 -> 0.195 (8), 0.625 -> 0.269 (16).
- Card 1: main6 primed; **o6 greedy md5 8 / 8 = main's** (`--spec off` and DSpark: short 64 04cdb28a, short 256 9a27c21b, 4k 64
  2252566a, 4k 256 9675bbb9, the values main3 / main5 printed in W3 / O4 / S1); decode tok/s spec off 83.1 / 83.6 / 79.6 / 80.3,
  DSpark 97.1 / 94.6 / 86.8 / 108.7 (o5's). main6's md5 cells skipped (12 s left).
- One cell to watch: filler 1-stream TTFT p50 (n = 1 request per visit) B 0.455 s vs A 0.391 s (S1-S3: 0.389-0.418 both arms);
  B ran first in this window.

## Result (2026-11-26 ~21:45): overlap works and is bitwise, but gate 4 fails at 8 streams, so it ships opt-in

Speed, gate 4's paired windows S1 (AB), S2 (BA), S3 (AB): A = main5 (main a4c8f3b19, #538 packing on), B = o4 (overlap on, pace
150 ms, stream 1 at priority -1; = `INTARSIA_OVERLAP=1` on the PR head), card 0, `serve --max-batch 16 --max-seq 4608 --kv-fraction
0.9`. Ratio B / A, mean over the 3 windows with the 95 % t interval (n = 3, t = 4.303); TTFT / ITL: lower is better. Raw
`raw/s1..s3`, summary `raw/speed-s1-s3.txt` (`speed_sum.py s1 s2 s3`).

| streams | TTFT p50 | aggregate tok/s | ITL p99 | ITL p50 | ITL p90 | TTFT max |
|---|---|---|---|---|---|---|
| real 4 | 0.797 [0.752, 0.842] | 0.996 [0.979, 1.014] | 1.224 [0.15, 2.30] (1.46, 1.49, 0.73) | 1.003 | 2.13 | 1.071 |
| real 8 | 0.786 [0.509, 1.063] | 0.987 [0.973, 1.001] | **0.422 [-0.02, 0.863]** (0.31, 0.33, 0.63) | 1.010 | 1.77 | 1.109 |
| real 16 | 0.932 [0.819, 1.044] | 0.990 [0.966, 1.014] | **0.383 [0.326, 0.440]** | 1.033 | 1.31 | 1.205 |
| filler 1 | 0.987 [0.912, 1.061] | 1.002 [0.995, 1.009] | - | - | - | 0.987 |
| filler 4 | 0.886 [0.874, 0.899] | 1.008 [0.990, 1.027] | - | - | - | 1.061 |
| filler 8 | 0.964 [0.936, 0.993] | 0.994 [0.967, 1.020] | - | - | - | 1.068 |
| filler 16 | 1.013 [0.997, 1.028] | 0.995 [0.989, 1.000] | - | - | - | 1.159 |

Filler ITL (server histograms over each level, `itl_seg.py`, bucketed, `raw/filler-itl-s1-s3.txt`): the same in all 3 windows: ITL
p99 bucket A 250-500 ms -> B 150-250 ms at 4 / 8 / 16 streams, 10-20 ms both at 1; ITL p50 the same bucket for A and B at every N.

Absolute (mean of the 3 windows, A -> B): real ITL p99 at 8 0.597 -> 0.251 s, at 16 0.637 -> 0.244 s, at 4 0.157 -> 0.169 s;
aggregate real 16 125.8 -> 124.5, filler 16 154.2 -> 153.3 tok/s.

- **Gate 4 (section 4's rule, written before the windows): FAIL.** (a) ITL p99 upper bound <= 0.70: 16 streams 0.440 pass, **8
  streams 0.863 fail** (S3's B had three stalls of 0.34-0.41 s seen by 7-8 streams at once; S1 / S2's B had 0-1 gaps above 0.3 s,
  A has 27-34 in every window; cause not diagnosed). (b) aggregate upper bound >= 0.99 at every N: pass (lowest 1.000, filler 16;
  real 8 1.001). (c) TTFT p50 lower bound <= 1.01 at every N: pass.
- Not gated, against overlap: ITL p90 is 1.3-2.1x main's at every N (the long chunk stalls become many medium gaps: a decode step
  beside a chunk takes ~110-150 ms, O3's trace), ITL p99 at 4 streams is worse in 2 of 3 windows, TTFT max is 1.07-1.21x (the last prompt of a
  burst waits longer: its chunks share the GPU with decode steps), ITL p50 at 16 +3.3 %.
- **Default left: opt-in.** `INK_OVERLAP_DEFAULT = false` (`model/inkling2.rs`); `INTARSIA_OVERLAP=1` turns on exactly the
  measured config (pace default 150 ms for Inkling-2, the decode lane at high priority); `docs/tuning.md` row updated.
- **Bitwise**: every serve run with overlap matched: S1-S3 card 0 B = A (4/4, 8/8, 16/16 each window, nonce'd prompts); S3 card 1
  o4 at pace 0 / 150 / 300 and main5, 8/8 and 16/16 each vs W4's solo replies; poisoned o4 (`INTARSIA_HIP_POISON=ff`) 16/16 and
  4 x 4/4; O1-O4 before. Greedy md5 8 / 8 = main for o4 `--spec off`, main5 DSpark and the merged head o5 (both modes). G2 short
  4.491e-3 / long 4.083e-3, o4 = main5. The merged head o5 (S4 / S5): overlap on 8/8 and 16/16, `INTARSIA_OVERLAP=0` 8/8 and 16/16, solo server 16/16 (o4 16/16
too); hip tiny `inkling2_batch` 10/10 and `inkling2_packed` 6/6 on the PR-head tree; cpu tests of the PR head pass (endor).
- **Not verified**: speed of the PR head itself (gate 4 measured o4; the merge adds only opt-in sampled-DSpark code, and o5's
  md5 / decode tok/s match o4's); the PR head binary with the flipped constant was built (`--features hip`) but not run on the GPU:
  its unset default takes the branch S4 ran with `INTARSIA_OVERLAP=0`. The cause of S3's three 0.34-0.41 s stalls with overlap on.
  The 35B on HIP (batched serve) was not run on this branch: unchanged by code reading (overlap stays off for it, its stream 1
  never gets the decode lane) and the cpu scheduler tests.
- Files outside the three kernel files brief 3 assigned: `ink_attn_rows.hip` (one line: the ticket slot), `ik_runtime.{h,hip}`
  (`ik_rt_stream_create_prio`), and the shared `device.rs` (side staging, decode lane) and `backend.rs` (`decode_lane`, default
  no-op), plus the scheduler / engine / model files this lane owns.
- unanimis capture a5bd873f-8f41-4794-85f4-5bd4dbdb2477; PR #542.

## Briefs received

- 2026-11-26 ~18:30 EST (coordinator, brief 3; in substance): start only after brief 2 (#538's window 9, default decision, PR update,
  report). Goal as above (R3d). Branch `ink/hip-overlap` from the `ink/hip-bpf` head in `.worktrees/ink-hip-overlap`, rebased onto
  main once #538 merges; this lane is the only writer in scheduler.rs, engine/batch.rs and model/inkling2.rs.
  - Step 0: a design note here, before any kernel or scheduler code, picking one mechanism: (a) concurrent HIP streams (preferred:
    the batched decode step on a second stream while a prefill chunk runs; same kernels and inputs, so no bit risk; the work is
    per-stream scratch and workspaces including static / global scratch in the kernel TUs, an event sync at the prefill-to-decode
    handoff, no shared mutable host state between the two passes); (b) chunk-length-invariant prefill (per-op dumps at chunk 2048
    vs 512 / 256 first; costs prefill throughput; only if (a) is blocked or the dump shows a cheap fix); (c) a literal mixed pass,
    only if (a) and (b) are blocked, with every row-count-chosen form (flash above 16 rows, fused blend up to 8 rows, op 75 at 1 row,
    MoE WMMA tiles) proven by per-op dumps.
  - Scope guard: `scheduler.rs:30-31` (decode rows and prefill chunks never share a pass) and `busy_chunk` are core code shared with
    the Qwen3.6-35B on HIP (packing default-on there), Metal and CUDA. Gate the new behaviour to Inkling-2 on HIP, off everywhere else;
    the 35B HIP serve path stays byte-for-byte unchanged in behaviour. `crates/intarsia-core/tests/scheduler.rs` joins the gates.
  - Gates: (1) the 16 real prompts equal the solo server 16/16 at 4 / 8 / 16 streams with the overlap on, plus a poisoned-scratch
    run (new allocations NaN-filled) with the same result; (2) solo `generate` md5 and the solo server equal to main, G2 unchanged;
    (3) hip tiny `inkling2_batch` and `inkling2_packed`, `intarsia-core` including tests/scheduler.rs, admission of any touched op;
    (4) speed, paired against main's batching server with the packing config: ITL p99 at 8 and 16 streams at least 30 % lower beyond
    the paired spread; aggregate tok/s and TTFT p50 no more than 1 % worse at any stream count (1 / 4 / 8 / 16); report ITL p50 / p99,
    TTFT p50 / p95 and aggregate for filler and real prompts; (5) default-on with `--max-batch` if 1-4 pass, otherwise opt-in with the
    numbers.
  - Radiance: 30 min in total for brief 3, every window under 5 min. Its own PR; do not merge. fleetq for every job; work folders
    only under `.worktrees/`.

- 2026-11-26 ~19:10 EST (coordinator answers for #538 and brief 3; in substance):
  - **#538:** default-on at 2048 rows accepted, on one condition before the merge (b13 has not run on a GPU; packing-on serving
    on the merged base not checked bitwise): one window, at most 5 min down, an extra 6 min on top of the 30-min day budget for
    this window only. (1) packing on, no env, the recipe defaults (2048 rows): the 16 real prompts at 4 / 8 / 16 streams, every
    reply equal to solo; (2) hip tiny `inkling2_packed` on b13; (3) if time is left, one ABBA pair at 2048 rows vs main, 4-stream
    filler TTFT p50 and aggregate, pooled with any 2048-row data already measured; rule for the 2048 default: the pooled TTFT p50
    interval's lower bound <= 1.01; if it does not fit, say which 2048 numbers the default rests on and whether they had an interval
    or were single runs; (4) if 1 or 2 fails, stop and report, no flip; if they pass, push and report the head (the coordinator
    merges on green CI); (5) 4096 rows stay opt-in, documented as `INTARSIA_INK_PACK_ROWS=4096` plus `--kv-fraction 0.9`.
  - **Brief 3, Q1:** yes, edit `kernels/hip/ink_mix.hip`, `ink_attn.hip` and `ink_route_fused.hip` for per-stream counters (this
    lane owns the three files now; lane ink-dflash2 keeps off them). Minimal and add-only where possible; solo and batched paths stay
    bitwise: admission for every touched op, greedy md5 vs main at short / 4k, 64 / 256; the same-ISA check where the default path's
    code should not change. No refuse-when-counters-are-in-use fallback.
  - **Brief 3, Q2:** gate 4's baseline is main's default after #538 (2048-row packing on), not the 4096-row config. Overlap must
    keep packing (the packed chunk is the stream-0 pass); accept the larger scheduler work. Report back before building if keeping
    packing plus a twin workspace does not fit the default KV budget, or if the scheduler work grows past about one day. The minimum
    interval between decode steps during a chunk is accepted: a knob, aiming at about 2 steps per chunk, with the TTFT p50 vs ITL
    p99 trade-off recorded at 2 or 3 settings. Radiance for brief 3: 30 min, windows <= 5 min, after the #538 window.

- 2026-11-26 ~20:10 EST (coordinator; in substance): (1) #538 is merged as `b507ab7ad`; main is now `a4c8f3b19` (also #540, a
  cpu-only 8-bit study, opt-in). Merge `origin/main` into `ink/hip-overlap` at the next convenient point, a merge, not a rebase.
  Overlap's gate-4 baseline is now main itself. (2) Cleanup: say in the next report which of these can go now or are still used:
  the Mac worktree `.worktrees/ink-hip-bpf`, the remote branch `ink/hip-bpf`, the tatooine folder `~/intarsiaml-ink-hip-bpf`, its
  docker volume. Nothing is touched until answered.

- 2026-11-26 ~21:00 EST (coordinator, resume brief to the new agent after the Desktop restart; in substance): (1) analyse S1 with
  `speed_sum.py s1`; if complete and valid (every rep, card-1 md5, G2, poison) count it as rep 1 and run only S2 (BA) and S3 (AB),
  else rerun S1. (2) Radiance: about 10 min left of the 30, **plus one extra window of up to 5 min approved** (about 15 min, 3
  windows); use the S windows' card 1 for the unverified items (poisoned o4, md5 o4 `--spec off` and main5 DSpark, G2 short / long,
  the solo server on o4). (3) Merge `origin/main` (aee06201c, #535 sampled DSpark, opt-in), a merge, not a rebase; cpu tests of the
  merged head on endor (`fleetq run --host endor --load cpu`); if the merge touches files this change touches, md5 of the merged
  head in a window, within budget. (4) Speed table first, gate verdict, LOG.md row, unanimis capture. (5) Push, open a PR to main,
  do not merge. Overlap stays default-on (`INK_OVERLAP_DEFAULT`) only if gate 4 passes on paired reps, else opt-in with the
  numbers. Windows: `--host tatooine-r9700 --takes-down radiance --quiet gpu,cpu`, each <= 5 min, none after 06:00 EST; never
  `fleetq release` a running window job (abort = kill the driver's tree, then check `/v1/models`).

## 1. Design (step 0)

### 1.1 Mechanism: (a) concurrent HIP streams, through the existing R3d decode twin

Pick **(a)**. It needs no new numerics: each pass is the same call it is today (the prefill chunk or packed pass on the main model,
stream 0; the batched decode step on a twin, stream 1), so bitwise equality to the solo run follows from "no state shared between
the two in-flight passes". It is also already built generically (R3d, `INTARSIA_OVERLAP`, CUDA only so far):
- `Model::decode_twin` (`model/batch.rs:1171`): a second `Model` over the same weights, kernels and plan, with its own workspace,
  kernel scratch (`plan::resolve` sizes it by a dry run), graphs, row tables and sampling buffers, launching on stream 1.
- `SlotModel::prefill_begin / prefill_done / prefill_end` (`engine/batch.rs:425-460`): the chunk runs on its own host thread with
  the slot's state moved out; the scheduler (`scheduler.rs:1363`) starts the chunk first and keeps stepping decode rows until it is
  done. The chunk schedule (which chunk, how long, which head) is the one `prefill_one` would run.

(b) is rejected: a smaller busy chunk lowers ITL p99 only in proportion and costs prefill throughput (-25 % at B8 for 256 in the
batch lane), and it needs every chunk-length dependent form found and made invariant. (c) is rejected: a decode row inside a
prefill call changes the row-count-chosen forms (flash above 16 rows, fused blend up to 8, op 75 at 1, the MoE WMMA tiles from 16 x
experts pairs) unless each is split out again, which is (a) with extra steps.

### 1.2 What is shared today, and the fix for each (code audit, nothing run)

| # | Shared between the two passes | Where | Effect if both run at once | Fix |
|---|---|---|---|---|
| 1 | The Inkling workspace `InkWs` (s, p, bank, q / k / v, rel, mlpin, moe, route tables ...) | `model/inkling2.rs:299-325` | the twin has none (`decode_twin` sets `ink: None`), so a twin cannot step Inkling at all | an Inkling twin: the same layers / visits / weights (shared, `Arc`), its own `InkWs` of `BATCH_MAX_ROWS` rows |
| 2 | Device-global tickets: `g_gate_ctr[3*16]` (the gate kernel's fused norm, last-block election) | `kernels/hip/ink_mix.hip:60` | blend chunks of <= 2 rows (`blend_fnorm_rows`) on both streams at once corrupt each other's counts (a hang or a wrong norm). A prefill call's LAST blend chunk can be 1-2 rows (call rows mod 256) | one ticket set per lane: `[2][3*16]`, the lane (0 = stream 0, 1 = any other) from the launch's stream |
| 3 | `ink_dec_tick[MAXROWS*256]` (InkAttnDecode split-merge) | `ink_attn.hip:659` | a prefill call of 1-16 rows takes InkAttnDecode (a final chunk of a prompt whose rest is <= 16 rows) | per lane, as 2 |
| 4 | `g_route_ctr` (fused route, `T <= 16`) | `ink_route_fused.hip:17` | as 3 for a <= 16-row prefill call | per lane, as 2 |
| 5 | `capturing: AtomicBool` on the HIP backend (a graph capture in progress) | `crates/intarsia-backend-hip/src/device.rs:192` | the twin's graph capture on stream 1 makes stream-0 launches skip the span / prequant fast paths (`:588`, `:679`) while it runs: same math by the code read, but a cross-thread effect | record WHICH stream captures; only that stream skips |
| 6 | Packed prefill turns off with a twin (`pack_rows_max` returns 0) | `engine/batch.rs:566-573` | overlap would lose #538's packing | a packed pass may be the in-flight prefill: `prefill_begin_packed` runs `ink_prefill_pack` on the thread |
| 7 | Host-side state otherwise (streams table, fork events, staging ring, span / prequant caches) | `device.rs:181-204` | mutexed; span / prequant are stream-0 only | none needed |

Not shared: per-sequence rings and KV live in `SequenceState` (a sequence is either prefilling or decoding); the twin's row tables,
`batch_pos`, sampling buffers, scratch and graphs are its own (`enable_batch`, `plan::resolve`); MoE syncs keyed by (device, stream)
(`moe.hip:1323`, `moe_block.hip:53`) are not on the Inkling path. The prefill-to-decode handoff: `prefill_end` joins the thread,
whose last op synchronizes stream 0 (the head's download), so a sequence's KV / ring writes are complete before the twin's first
decode read of it; no extra event is needed while that holds (checked in the build: the packed and plain paths both end in a host
read of the head on stream 0).

### 1.3 Scope guard

- The overlap stays behind `BatchedOptions::overlap`. New: its default is on only for an Inkling-2 model on the HIP backend with
  `--max-batch` > 1 (`INTARSIA_OVERLAP=0` turns it off, `=1` as before elsewhere). Every other model and backend keeps
  `INTARSIA_OVERLAP` unset = off, and nothing in the scheduler's non-overlap path changes, so the 35B HIP serve path is unchanged.
- `decode_twin` builds an Inkling twin only on HIP / cpu; the existing twin code path is unchanged for other models.
- The ticket arrays and the capture flag change kernel / backend code shared with the 35B only where Inkling kernels live
  (`ink_*.hip`); `device.rs`'s capture flag change keeps a single-stream run identical.

### 1.4 Expected gain and risk

- ITL p99 today is a decode row waiting for a whole 2048-row chunk (~0.6 s at 16 streams, ~0.66 s per 2048-row chunk alone). With
  the decode step on stream 1 beside the chunk, a decode row waits at most for the kernels in flight; a step should take about its
  plain time plus the share of the GPU the prefill kernels hold. Expect ITL p99 near 2-3x the ITL p50 (0.1-0.2 s at 16), that is
  -60..-80 %, if the R9700 runs the two streams concurrently.
- Risk 1, TTFT: the prefill chunk now shares the GPU with decode steps that used to wait. At 16 streams a decode step is ~58 ms of
  mostly bandwidth-bound work; if it takes ~30 % of the GPU during a chunk, chunks slow by up to that much, and TTFT p50 would fail the
  "no more than 1 % worse" gate. A high-priority stream 1 lowers ITL further but slows prefill more. Measured, not assumed: the first
  window measures both.
- Risk 2, aggregate: no worse expected (decode no longer stalls during chunks; total work is the same).
- Risk 3: HIP runs two streams concurrently only if neither uses the legacy null stream and no allocation syncs the device mid-pass;
  both streams are `hipStreamCreate`d (`device.rs:776`), and the twin's buffers are allocated before serving.

### 1.5 Design points from review (before code)

- **TTFT is the gate most likely to fail.** Today a decoding stream gets one step per ~0.66 s chunk. Unpaced overlap runs about 11
  decode steps during that chunk, whose contention slows the chunk. A 30 % lower ITL p99 needs only a step every ~0.42 s (about 2
  per chunk). So a **pacing knob** from the start: a minimum interval between decode steps while a chunk is in flight, swept in the
  first speed window (0 / 150 / 300 ms).
- **Order:** the Inkling twin, then the per-lane tickets, then pacing, then a feasibility window; the packed in-flight scheduler work
  (1.2 row 6) only after the numbers say overlap can pass, and only if gate 4's baseline is the packing config (question to the
  coordinator).
- **Tickets without kernel edits (fallback):** refuse overlap when the in-flight pass would take a ticketed form (any call of <= 16
  rows, or rows mod 256 within the fused-norm limit). Fragile; only if the coordinator says no to edits of `ink_mix.hip`,
  `ink_attn.hip`, `ink_route_fused.hip` (not assigned to this lane by brief 3; asked).
- **Handoffs need explicit ordering, not a sync side effect:** (1) slot reuse: a slot's last decode step on stream 1 must finish
  before a new request's first chunk on stream 0 writes that slot's rings; (2) a non-final chunk has no head download to synchronize
  on. Use `stream_wait` (event record / wait) at both handoffs.
- **The `device.rs` capture flag (1.2 row 5):** leave it; it changes no bits and is shared with the 35B. Gate 1 and the poison run
  decide.
- **Poison run:** `INTARSIA_HIP_POISON=ff` (every new device allocation 0xff: NaN in bf16 and f32).
- **DSpark:** `decode_twin` refuses a model with a DSpark head (`self.dflash`); confirm a batched Inkling load has none, or the twin
  fails quietly ("overlap unavailable").
- **First window, before overlap:** #538's open item, a packing-default server's 16 real prompts at 4 / 8 / 16 on the merged base.

### 1.6 Plan

1. Inkling twin (`InkModel` weights behind `Arc`, `InkModel::twin` allocating its own `InkWs`), `decode_twin` for Inkling; cpu test:
   a twin's batched decode step == the main model's, bitwise.
2. Per-lane tickets (ink_mix, ink_attn, ink_route_fused), lane from the launch stream; per-stream capture flag.
3. Packed in-flight prefill (`prefill_begin_packed`), scheduler: the overlap branch picks the packed set `prefill_packed` would.
4. Default: on for Inkling-2 on HIP with `--max-batch` > 1 only if gates 1-4 pass; opt-in (`INTARSIA_OVERLAP=1`) until then.
5. Gates on tatooine in windows of < 5 min (30 min in total).

## 2. Build (branch `ink/hip-overlap`)

- **Inkling decode twin** (`model/inkling2.rs` `InkModel::twin`, `model/batch.rs` `decode_twin`): the layers are behind an `Arc`
  (shared, read-only once loaded; the batched load drops the P12 copies before any twin exists), visits / queries / norms cloned,
  its own `InkWs` of `max(rows, BATCH_MAX_ROWS)` rows (zero-filled on stream 0, then a stream-0 sync before the twin launches on
  stream 1). A batched Inkling load has no DSpark head (the DSpark default needs `--max-batch 1`), so `decode_twin`'s refusals do
  not apply.
- **Per-stream ticket slots** (`kernels/hip/ink_tick.h`, new): `g_gate_ctr`, `ink_dec_tick` and `g_route_ctr` hold
  `kInkTickSlots = 8` copies; a launch uses the copy of its (device, HIP stream), handed out in first-launch order per device (as
  `moe_sync_slot` does for `g_moe_sync`). The gate kernel gets the slot in `GateNorm.slot`, InkAttnDecode in `flags` bits 24..27 of
  its launch's copy of `InkP` (0 for the first stream of a device, so the solo and the cache-write passes' arguments stay main's
  bytes), the fused route as a new last argument. More than 8 streams on a device (a long test process) share slot 0 (main's
  single array) with a warning.
- **Packed pass in flight** (`engine/batch.rs` `prefill_begin_pack` / `prefill_end_pack`, `ink_pack_run`; `scheduler.rs`
  `pack_pick` / `pack_finish` / `overlap_decode`): with overlap, the packed pass `prefill_one` would run (same pick, same chunks) runs
  on the prefill thread with its slots' states moved out, as a single chunk does; only where `busy_chunk == solo_chunk` (a first
  chunk's size then does not depend on the decode rows). `pack_rows_max` no longer turns packing off when a twin exists (Inkling).
- **Pace knob** `INTARSIA_OVERLAP_PACE_MS` (default 0): the least time between the starts of two decode steps while a chunk is in
  flight. **Metric** `intarsia_batch_overlap_steps_total`.
- cpu (tatooine, fleetq #12278 / #12280): `intarsia-core` lib 270 + `tests/scheduler.rs` 54 (new
  `batched_equals_solo_randomized_with_overlap_and_packing`: pace 0 / 300 us, B 2 / 4 / 8, packed passes in flight), `inkling2` 17,
  `inkling2_batch` 10 (new `inkling2_decode_twin_equals_solo_and_overlaps_prefill`), `inkling2_packed` 6 (new
  `inkling2_overlap_engine_equals_solo_engine`, pace 0 / 2 ms), `docs_env_vars` 2, `intarsia-server` lib 71: all pass.

## 3. Windows

### O1 (fleetq #12281; radiance down 19:34:35 - 19:38:39 = 4:04)

Card 0: N1 = b13 (#538's head, packing default, no overlap), X1 = o1 `INTARSIA_OVERLAP=1` (pace 0), X2 = o1 overlap pace 150 ms;
all `--max-batch 16 --max-seq 4608 --kv-fraction 0.9`. Card 1: o1's admission (overlap on: 23 kernels, 134 cases, 0 cached, pass),
hip tiny `inkling2_batch` 10 / 10 (the twin test: rows == solo, and a prefill on the model beside the twin's steps, calls of 1 .. 16
rows on stream 0 running the ticketed kernels) and `inkling2_packed` 6 / 6 (overlap engine: 138 / 60 decode phases beside chunks
in flight at pace 0 / 2 ms, packed passes in flight).

Bitwise vs W4's solo replies: X1 conc:8:0 8/8, conc:8:1 8/8, conc (16) 16/16; X2 conc:8:0 8/8, conc 16/16; N1 conc 16/16.

| arm (1 rep each) | real 8 (conc:8:0) agg / TTFT p50 / ITL p99 | real 16 agg / TTFT p50 / TTFT max / ITL p99 | filler 1k x 8 agg |
|---|---|---|---|
| N1 b13 | 117.1 / 1.93 s / 0.581 s | 125.7 / 2.81 / 9.66 / 0.622 | 140.3 |
| X1 overlap pace 0 | 103.2 / 2.20 / 0.301 | 105.2 / 4.49 / 16.15 / 0.336 | 89.6 |
| X2 overlap pace 150 | 118.4 / 1.94 / 0.570 | 123.2 / 2.83 / 10.50 / 0.568 | - |

Reading: unpaced overlap halves ITL p99 (-46 % at 16, -48 % at 8) but costs 16 % aggregate and +60 % TTFT p50 at 16 real, and 36 %
filler aggregate (decode steps 676 -> 1460 for the same tokens: many small steps beside each chunk, each reading the weights). At
pace 150 the cost is gone and so is most of the gain (ITL p99 -9 %). The poll log (decode steps every 0.2 s) and the replies' gaps
show why the knob does little: a decode step beside a chunk takes ~0.3 s (vs 33-58 ms alone), longer than the pace, so pace 150
leaves ~2 steps per chunk and each waits behind the chunk's kernels. HIP streams here are created at the default priority
(`hipStreamCreateWithFlags(.., NonBlocking)`), not high as on CUDA, so the twin's kernels queue with the chunk's.

### O2 (fleetq #12293; radiance down 19:51:21 - 19:54:53 = 3:32)

o2 = o1 + `INTARSIA_HIP_STREAM1_PRIORITY=high` (opt-in: stream 1 created by `hipStreamCreateWithPriority` at the greatest priority,
-1 on the R9700) + a CB-trace line per decode step beside a chunk (`OverlapStep { step_us }`). Card 0: N2 = b13; P0 = o2 overlap,
priority high, pace 0; P3 = the same at pace 300 ms (both with `INTARSIA_CB_TRACE=1`). Card 1: o2's admission (23 kernels, 134
cases, pass), then gate 1's poisoned run: o2 overlap + priority + `INTARSIA_HIP_POISON=ff`.

Bitwise vs W4's solo replies: P0 8/8 and 16/16, P3 8/8 and 16/16, N2 16/16; **poisoned overlap: 16/16 at once and 4 x 4/4 at 4**.

| arm | real 8 agg / TTFT p50 / ITL p99 | real 16 agg / TTFT p50 / ITL p99 | filler 1k x 16 agg |
|---|---|---|---|
| N2 b13 | 118.8 / 1.94 / 0.585 | 125.7 / 2.81 / 0.629 | 155.1 |
| P0 priority, pace 0 | 119.9 / 1.91 / 0.857 | 125.1 / 2.88 / 0.741 | - |
| P3 priority, pace 300 | 118.9 / 1.93 / 0.872 | 125.7 / 2.80 / 0.728 | - |

The trace shows what holds the twin back: per chunk the first step beside it takes 85-165 ms, the next one 0.3-1.1 s, i.e. until
the chunk ends. Not the GPU: the HIP backend's host staging (`device.rs` `Staging`, round 9) is one ring of 8 page-locked upload
slots plus ONE download slot for all streams, under one mutex, and a staged download holds that mutex across its stream's sync. The
prefill thread's head read at the end of a pass (stream 0) therefore blocks the twin's next upload or token download (stream 1)
until the whole chunk is done; an upload slot last used by stream 0 mid-pass also waits for stream 0 to reach it. With priority the
first step gets through quickly and the second waits, so ITL p99 got worse, not better. Fix (o3): every stream other than 0 gets
its own staging (ring + download slot, `side_staging`); stream 0 keeps the old one, so a single-stream run is unchanged.

Cleanup asked by the coordinator (answer): the Mac worktree `.worktrees/ink-hip-bpf` (clean at 4cb9ba97b, merged) and the remote
branch `ink/hip-bpf` can go; this lane copied what it uses (window scripts, W4's solo replies as `raw/ref-solo-w4.json`). The
tatooine folder `~/intarsiaml-ink-hip-bpf` and the volume `intarsia-cargo-target-ink-hip-bpf` are still in use: the overlap lane
builds and serves there (binaries b13, o1 .. o3, `.scratch` scripts, logs); free them when this lane ends.

### O3 (fleetq #12303; radiance down 20:03:24 - 20:07:41 = 4:17)

o3 = merged main a4c8f3b19 + per-stream staging (every stream other than 0 has its own upload ring and download slot). Card 0:
N3 = b13 (#538's head = main's serving default); Y0 = o3 overlap, default priority, pace 0; Y1 = o3 overlap, stream 1 at high
priority, pace 0; Y2 = Y1 at pace 150 ms (all `--max-batch 16 --max-seq 4608 --kv-fraction 0.9`, CB trace on for Y*). Card 1: o3's
admission (overlap + priority: 23 kernels, 134 cases, pass), hip tiny `inkling2_batch` 10 / 10, `inkling2_packed` 6 / 6 (150 / 60
decode phases beside chunks in flight at pace 0 / 2 ms), `inkling2` 15 / 17 (the 2 screened-head tests: no HIP HeadScreen kernel,
the same 2 that fail on main, docs/results/2026-11-26-inkling2-hip-bpf.md section 3).

Bitwise vs W4's solo replies: N3, Y0, Y1, Y2 each 8/8 (conc:8:0) and 16/16 (16 at once).

| arm (1 rep) | 8 real: agg / TTFT p50 / ITL p50 / ITL p99 | 16 real: agg / TTFT p50 / ITL p50 / ITL p99 | overlap steps: n / step p50 / p90 |
|---|---|---|---|
| N3 b13 (main default) | 118.8 / 1.94 / 0.033 / 0.584 | 125.1 / 2.57 / 0.058 / 0.633 | - |
| Y0 overlap, default priority | 118.2 / 1.82 / 0.033 / 0.585 | 123.9 / 2.59 / 0.058 / 0.614 | 45 / 288 ms / 614 ms |
| Y1 overlap, priority high, pace 0 | 113.7 / 1.58 / 0.034 / **0.155** | 122.3 / 2.72 / 0.087 / **0.235** | 190 / 90 ms / 141 ms |
| Y2 overlap, priority high, pace 150 | 116.5 / **1.45** / 0.033 / **0.203** | 124.7 / **2.34** / 0.072 / **0.241** | 113 / 110 ms / 147 ms |

Reading: the staging fix alone (Y0) does not help: at the default priority the twin's kernels queue behind the chunk's (a step
beside a chunk takes 0.29 s at p50). With stream 1 at high priority (Y1, Y2) a step beside a chunk takes 90-110 ms and ITL p99 falls
62-73 %. Pace 150 keeps that and gives the chunk more of the GPU: TTFT p50 is better than main at 8 and 16, aggregate -1.9 % at 8
and -0.3 % at 16 (single runs; the 8-stream wall time ends on one long request, so it is noisy). Y2 is the candidate; it needs
paired reps against main5 (a4c8f3b19, built: #12304, md5 42fdf7a7d871d6694625fba108e086c7) at 1 / 4 / 8 / 16, filler and real.

## 4. o4 (the candidate) and the gate-4 plan

o4 (8ed77739d, md5 38475e68394d85800f5e970d39006508) is Y2 wired as the default, scoped as the brief's guard asks:
- `INTARSIA_OVERLAP` unset: on for Inkling-2 on HIP in a batching engine without a prefix store (`Model::overlap_default`,
  `inkling2::INK_OVERLAP_DEFAULT`); off for every other model and backend (CUDA keeps its opt-in). `=0` off, `=1` on.
- `INTARSIA_OVERLAP_PACE_MS` unset: 150 for Inkling-2 (`INK_OVERLAP_PACE_MS_DEFAULT`), 0 elsewhere.
- `Backend::decode_lane(stream)` (new, default no-op): the Inkling twin calls it for stream 1 before its first use; the HIP backend
  then creates that stream at the greatest priority and gives it its own host staging. No other stream changes: the 35B's
  single-card serve path never calls it (stream 1 there is only the TP all-reduce lane, `model.rs` `stream_wait(1, ..)`, and the
  twin; the shared-expert side stream is `self.stream + 2`), so its streams and staging are main's. `INTARSIA_HIP_DECODE_LANE=0`
  ignores the hook (replaces o2's `INTARSIA_HIP_STREAM1_PRIORITY`).
- cpu (#12319): `intarsia-core` lib 274 + `tests/scheduler.rs` 54, `inkling2` 17, `inkling2_batch` 10, `inkling2_packed` 6,
  `docs_env_vars` 2, `intarsia-server` lib 71: pass.

VRAM (O3, `--kv-fraction 0.9`): the twin adds 62 MiB (card 0 peak 28530 -> 28592 MiB). The KV plan is fixed at load, before the
engine makes the twin, so the twin cannot lower the context; window O4 checks it at the recipe default (`--max-batch 16` alone).

**Gate-4 rule (written before the paired windows).** A = main5 (origin/main a4c8f3b19), B = o4 with no env; card 0, `--max-batch 16
--max-seq 4608 --kv-fraction 0.9`; per window one A visit and one B visit (order AB, BA, AB), each: the real prompts at 4 / 8 / 16
(tagged, a nonce per window) and the 1k filler sweep at 1 / 4 / 8 / 16 (localeval, 128 tokens). Ratios B / A per window, mean and
95 % t interval over the 3 windows (n = 3, t = 4.303). Pass: (a) real-prompt ITL p99 at 8 and 16: interval upper bound <= 0.70;
(b) aggregate at every N, real and filler: interval upper bound >= 0.99 (not shown more than 1 % worse); (c) TTFT p50 at every N,
real and filler: interval lower bound <= 1.01. Filler ITL comes from the server's histograms (bucketed, reported, not gated).
1 stream has nothing to overlap (no decode beside a chunk): its rows are reported from the same visits. Budget: O1-O3 used 11:53 of
brief 3's 30:00; O4 (gates) plus 3 speed windows at <= 4:30 each fit in the rest.

### O4: gates on o4 (fleetq #12322; radiance down 20:26:17 - 20:30:19 = 4:02)

| check | result |
|---|---|
| default budget (`serve --max-batch 16`, no other flag) | main5: context 3285 tokens, `intarsia_kv_tokens_capacity` 52560, VRAM 28098 MiB after load (card 0). o4: context 3285, 52560, 28130 MiB (card 1), overlap on by default. **The twin costs no context or KV.** |
| speed config (`--kv-fraction 0.9 --max-seq 4608`) | o4: context 4608, 73728 KV tokens (main's 73728); VRAM peak of the G4 visit 28593 MiB (b13, main's packing default, peaked at 28530 in O3's N3) |
| gate 1, real prompts, o4 overlap default-on (G4) | conc:4 waves 0-3: 4/4 each; conc:8 waves 0-1: 8/8 each; 16 at once: 16/16 (vs W4's solo replies); 219 of 1108 decode steps ran beside a chunk in flight; stream 1 created at priority -1 |
| gate 2, greedy md5 (`generate --ignore-eos`) | main5 `--spec off`: short 64 04cdb28a, short 256 9a27c21b, 4k 64 2252566a, 4k 256 9675bbb9 (= W3's main3); o4 DSpark default: the same four md5. Decode tok/s o4 DSpark 98.2 / 94.5 / 86.5 / 108.8 vs W3 main3 98.1 / 94.6 / 85.1 / 105.9 (the per-launch ticket-slot lookup costs nothing measurable) |
| gate 3, hip tiny on the o4 tree | `inkling2_batch` 10/10, `inkling2_packed` 6/6 (46 / 32 decode phases beside chunks in flight) |
| gate 3, admission (`kernels test --op`, o4) | conv_mix_bf16 532 cases ADMITTED, loop_blend_bf16 27 ADMITTED, ink_attn_decode_bf16 70 ADMITTED, ink_route_bf16 40 ADMITTED |

Left for S1-S3's card 1: o4 `--spec off` and main5 DSpark md5, G2 short / long both, the poisoned run on o4, the solo server.

### Same-ISA check (fleetq #12325, cpu slot; `raw/isa_check.sh`, `raw/isa_cmp.py`, log `raw/isa-check.log`)

Device assembly (`hipcc --cuda-device-only -S`, build.rs's flags, gfx1201) of the 7 changed translation units, origin/main
a4c8f3b19 vs o4, compared per kernel (instructions only): **168 kernels identical; the only differences are the three kernels whose
ticket index changed**: `ink_blend_gate_kernel` (4 instantiations in each of ink_mix / ink_mix_rows / ink_mix_segs, +2..4
instructions: the slot term of the counter address), `ink_attn_dec_k` (6 instantiations in ink_attn.hip and 6 in ink_attn_rows.hip,
+5..6 instructions: the slot from `flags` bits 24..27), and `ink_route_kernel` (one more argument, so its mangled name changed; its
body differs by the slot index). `ik_runtime.hip` is host code only (3 identical, the new `ik_rt_stream_create_prio` is host).
`__hip_cuid_*` are per-compile ids.

### S1: paired speed rep 1, order AB (fleetq #12331; radiance down 20:38:19 - 20:42:17 = 3:58)

fleetq started #12331 at 20:38:19; the parked agent released it at 20:38:23, which ended the fleetq job but not the Mac driver
(`serve_window.sh`), so the window ran to its 250 s cap without a slot. `fleetq log` shows nothing else started on tatooine-r9700
between 20:38:23 and radiance's return at 20:42:19 except this window's own localeval jobs (#12337 A filler, #12338 B filler), so
the visits ran on a quiet box. Checked by the resumed agent (21:05):
- **Complete**: A (main5) and B (o4) each ran the real prompts at 4 / 8 / 16 (nonce s1, exit 0) and the filler sweep 1 / 4 / 8 / 16
  (localeval exit 0); `speed_sum.py s1` reads every cell.
- **Bitwise, card 0**: the nonce'd replies of B equal A's (`serve_bitwise.py compare`): 4/4, 8/8, 16/16.
- **Card 1** (`stage_s1bg.sh`, logs copied to `raw/s1/card1/`):
  - md5 (`generate --ignore-eos`): o4 `--spec off` short 64 04cdb28a, short 256 9a27c21b, 4k 64 2252566a, 4k 256 9675bbb9; main5
    DSpark the same four (= W3 / O4). Decode tok/s o4 spec off 83.2 / 83.6 / 79.7 / 80.3, main5 DSpark 97.9 / 94.6 / 85.8 / 108.6.
  - G2 (`accuracy --mode prefill`, oracle W4A16): short KL 4.491e-3, top-1 99.47 %, long KL 4.083e-3, top-1 98.98 %, PASS, **o4 and
    main5 identical** to the printed digits.
  - **Poisoned run on o4** (`INTARSIA_HIP_POISON=ff`, overlap on by default, stream 1 at priority -1, `--max-batch 16 --max-seq 4608
    --kv-fraction 0.9`): 16 at once 16/16 and 4 x 4/4 equal W4's solo replies. That overlap ran in it: the server printed
    "prefill/decode overlap: on" and "stream 1 (decode lane) at priority -1", and its 16-at-once ITL p99 was 0.204 s (B's 0.238,
    not A's 0.651). Its `/metrics` snapshot (`intarsia_batch_overlap_steps_total 0`) was taken at startup, before any request, so
    it does not count the steps.
- Verdict: S1 is valid and counts as rep 1.

| S1 (1 rep) | A main5 | B o4 | B / A |
|---|---|---|---|
| real 4: agg / TTFT p50 / ITL p99 | 101.8 / 1.545 / 0.109 | 102.2 / 1.202 / 0.159 | 1.004 / 0.778 / 1.457 |
| real 8: agg / TTFT p50 / ITL p99 | 119.1 / 1.927 / 0.614 | 117.1 / 1.674 / 0.191 | 0.983 / 0.869 / **0.311** |
| real 16: agg / TTFT p50 / ITL p99 | 127.2 / 3.000 / 0.651 | 124.5 / 2.643 / 0.238 | 0.979 / 0.881 / **0.365** |
| filler 1 / 4 / 8 / 16 agg | 61.9 / 113.2 / 142.7 / 154.5 | 61.8 / 113.3 / 140.4 / 153.3 | 0.998 / 1.001 / 0.984 / 0.992 |
| filler 1 / 4 / 8 / 16 TTFT p50 | 0.418 / 1.316 / 2.065 / 3.643 | 0.398 / 1.167 / 1.988 / 3.702 | 0.952 / 0.887 / 0.963 / 1.016 |

`speed_sum.py` fix (21:05): the real-prompt reply files are a dict keyed by prompt, so the TTFT p95 / ITL p90 rows were never
computed; they are now (S1: ITL p90 B / A 1.64 / 1.81 / 1.31 at 4 / 8 / 16, TTFT p95 = max at these N 1.05 / 1.12 / 1.23).

### S2: paired speed rep 2, order BA (fleetq #12353; radiance down 21:21:05 - 21:24:14 = 3:09)

Driver `raw/window_s.sh 2` (sync, the three binaries checked, then `serve_window.sh`); raw `raw/s2/`, card 1 `raw/s2/card1/`.
- Complete: B (o4) then A (main5), real 4 / 8 / 16 (nonce s2) and filler 1 / 4 / 8 / 16, every cell exit 0.
- Bitwise, card 0: B's replies equal A's, 4/4, 8/8, 16/16.
- Card 1: **o5 (merged head) greedy md5 8 / 8 = main** (`--spec off` and the DSpark default, short / 4k, 64 / 256: 04cdb28a,
  9a27c21b, 2252566a, 9675bbb9 each; decode tok/s DSpark 96.8 / 94.8 / 86.3 / 108.6, spec off 83.1 / 83.5 / 79.6 / 80.3, o4's).
  **The solo servers of o4 and o5 did not come up** within the stage's 40 s: started without `--spec off` (W4's solo config), they
  load the DSpark draft and admit its shapes; no reply was sent. Moved to window S4 with W4's flags.

| S2 (1 rep) | A main5 | B o4 | B / A |
|---|---|---|---|
| real 4: agg / TTFT p50 / ITL p99 | 102.3 / 1.574 / 0.114 | 101.7 / 1.281 / 0.169 | 0.994 / 0.814 / 1.489 |
| real 8: agg / TTFT p50 / ITL p99 | 112.7 / 2.056 / 0.588 | 112.0 / 1.707 / 0.193 | 0.994 / 0.830 / **0.328** |
| real 16: agg / TTFT p50 / ITL p99 | 125.1 / 2.762 / 0.626 | 124.7 / 2.673 / 0.256 | 0.997 / 0.968 / **0.409** |
| filler 1 / 4 / 8 / 16 agg | 61.8 / 113.5 / 139.6 / 153.9 | 62.0 / 115.3 / 140.3 / 153.2 | 1.003 / 1.016 / 1.005 / 0.995 |
| filler 1 / 4 / 8 / 16 TTFT p50 | 0.393 / 1.319 / 2.096 / 3.684 | 0.395 / 1.162 / 1.999 / 3.704 | 1.005 / 0.881 / 0.954 / 1.005 |

### S3: paired speed rep 3, order AB (fleetq #12355; radiance down 21:29:35 - 21:32:41 = 3:06)

Driver `raw/window_s.sh 3`; raw `raw/s3/`, card 1 `raw/s3/card1/`.
- Complete: A (main5) then B (o4), every cell exit 0. Bitwise, card 0: B = A, 4/4, 8/8, 16/16.
- Card 1, the pace trade-off (the coordinator's ask), o4 beside card 0's visits (single runs; compare the arms with each other):
  every arm's replies equal W4's solo replies, 8/8 (conc:8:0) and 16/16 (16 at once). (`stage_s3bg.sh` was first revised to run o5
  here, then put back to o4 before the window: o5's admission cache was empty, ~100 s of cases at its first batched start.)

| card 1 arm | 8 real: agg / TTFT p50 / ITL p99 | 16 real: agg / TTFT p50 / ITL p99 | overlap steps (both runs) |
|---|---|---|---|
| main5 (no overlap) | 118.7 / 1.980 / 0.577 | 125.9 / 2.911 / 0.617 | - |
| o4 pace 0 | 113.5 / 1.888 / **0.142** | 123.4 / 3.861 / 0.231 | 177 |
| o4 pace 150 (the measured default) | 116.8 / 1.734 / 0.190 | 125.2 / 3.136 / **0.223** | 103 |
| o4 pace 300 | 118.8 / 1.778 / 0.322 | 125.6 / 3.607 / 0.348 | 52 |

Reading: pace 0 has the lowest ITL p99 at 8 and costs 4 % aggregate at 8 and +33 % TTFT p50 at 16; pace 300 gives back about half
the ITL gain for +1.7 % / +0.3 % aggregate over 150. 150 stays the opt-in's pace. Card 0, S3:

| S3 (1 rep) | A main5 | B o4 | B / A |
|---|---|---|---|
| real 4: agg / TTFT p50 / ITL p99 | 93.1 / 1.697 / 0.248 | 92.2 / 1.358 / 0.180 | 0.990 / 0.800 / 0.726 |
| real 8: agg / TTFT p50 / ITL p99 | 119.7 / 2.424 / 0.588 | 117.9 / 1.598 / 0.368 | 0.985 / 0.660 / **0.627** |
| real 16: agg / TTFT p50 / ITL p99 | 125.1 / 2.812 / 0.633 | 124.4 / 2.660 / 0.238 | 0.994 / 0.946 / **0.376** |
| filler 1 / 4 / 8 / 16 agg | 61.9 / 112.5 / 142.0 / 154.1 | 62.1 / 113.4 / 140.9 / 153.5 | 1.003 / 1.008 / 0.992 / 0.996 |
| filler 1 / 4 / 8 / 16 TTFT p50 | 0.389 / 1.315 / 2.067 / 3.643 | 0.390 / 1.172 / 2.019 / 3.701 | 1.003 / 0.891 / 0.977 / 1.016 |

The 8-stream subset is S1's (REAL_OFF 2 picks the same 8 prompts as 0 at N = 8; nonce s3). B's tail: 19 gaps above 0.3 s, all
in three events of 0.34-0.41 s (7-8 streams each), vs 1 (S1) and 0 (S2); A has 27-34 per window.

### Gate 4 verdict (S1-S3, n = 3)

`speed_sum.py s1 s2 s3` (`raw/speed-s1-s3.txt`): FAIL on `real8.itl99` (upper bound 0.863 > 0.70); the other 15 checks pass
(`real16.itl99` 0.440; aggregate upper bounds >= 1.000; TTFT p50 lower bounds <= 0.997). Per the brief, overlap is opt-in:
`INK_OVERLAP_DEFAULT = false` (commit 5899d3743), `docs/tuning.md` `INTARSIA_OVERLAP` row with these numbers. The opt-in-off
default equals main's batching path (`Model::overlap_default` false: no twin, no decode lane, `opts.overlap` None -> off); S4 runs
the merged head with `INTARSIA_OVERLAP=0`, which takes the same branch.

### Merge of origin/main aee06201c (#535) and o5

`git merge origin/main` (21:01, a merge; commit 20fc7ea25, parents ba3af1417 / aee06201c): auto-merged. #535 (sampled DSpark,
opt-in `INTARSIA_SPEC_SAMPLED_DFLASH`) shares two files with this change, `model.rs` and `model/batch.rs`: its new field
`spec_sampled_dflash` lands in `decode_twin`'s `Model` literal (0 = the env default; a twin never speculates). No kernel or HIP
backend file in common. o5 = the merged tree (hip build fleetq #12346, md5 d20df8b1d67a7bf1182c3e292ac1e938). Its md5 and batched /
solo bitwise checks run on card 1 of S2 / S3. cpu tests of the merged tree on endor (fleetq #12345, `raw/cpu_endor.sh`, log
`raw/cpu-merged-endor.log`): `intarsia-core` lib 274, `tests/scheduler.rs` 54, `batch_recurrent` 15, `inkling2` 17,
`inkling2_batch` 10, `inkling2_dspark_sampled` 2 (+3 ignored), `inkling2_dspark_sampled_gumbel` 1, `inkling2_packed` 6,
`prefix_store` 39, `docs_env_vars` 2, `intarsia-server` lib 71: all pass.


### S4: checks only, the merged head o5 (fleetq #12362; radiance down 21:44:05 - 21:47:16 = 3:11)

The coordinator's extra window. `raw/window_s4.sh` + `raw/stage_s4.sh` (both cards at once under `window.sh`, no speed); raw
`raw/s4/card/`.
- Card 0, o5 with no env (o5 predates the flip, so overlap is on: the `INTARSIA_OVERLAP=1` config of the merged head): its first
  start ran o5's admission (23 kernels, 134 cases, 0 cached; up after 102 s), then **8/8 (conc:8:0) and 16/16 (16 at once) equal W4's
  solo replies**, 113 decode steps beside chunks in flight, stream 1 at priority -1.
- Card 0, o5 with `INTARSIA_OVERLAP=0` (the branch the PR head's unset default now takes): **8/8 and 16/16 equal**, 0 overlap steps,
  ITL p99 0.585 / 0.625 (main's tail).
- Card 1, hip tiny on the PR-head tree (test builds fleetq #12361 after the flip, the cli built with `--features hip` there too):
  `inkling2_batch` **10/10**, `inkling2_packed` **6/6** (46 / 32 decode phases beside chunks in flight at pace 0 / 2 ms).
- Card 1, the solo servers of o4 and o5 (`--spec off`): **again not up** within 100 / 60 s. Their logs stop after "loaded": a solo
  serve loads the P12 (12-bit) weights and admits that plan's shapes, which no earlier run of o4 / o5 had cached (the batched config
  loads bf16). Moved to S5 with a wait that covers the admission.

cpu tests of the PR head (22f76ac23 tree, overlap opt-in) on endor (fleetq #12365, `raw/cpu-final-endor.log`): core lib 274,
scheduler 54, batch_recurrent 15, inkling2 17, inkling2_batch 10, inkling2_dspark_sampled 2 (+3 ignored), inkling2_dspark_sampled_gumbel
1, inkling2_packed 6, prefix_store 39, docs_env_vars 2, server lib 71: all pass.

### S5: checks only, the solo servers (fleetq #12368; radiance down 21:53:36 - 21:56:21 = 2:45)

`raw/window_s5.sh` + `raw/stage_s5.sh`; raw `raw/s5/card/`. Both solo servers at once (`--max-seq 4608 --spec off`, no
`--max-batch`: W4's config), o5 (merged head) on card 0, o4 on card 1: each up after ~115 s (admission 27 kernels, 6 cases run, 106
cached; the P12 cases are the slow ones), then the 16 real prompts one at a time: **o5 16/16 and o4 16/16 equal W4's solo replies**
(wall 34.2 / 33.9 s, W4's b10 / main3 34.8 / 34.0 s). Gate 2's solo-server item is closed.

Radiance for brief 3: O1 4:04, O2 3:32, O3 4:17, O4 4:02, S1 3:58, S2 3:09, S3 3:06, S4 3:11, S5 2:45 = **32:04** (30 min plus the
coordinator's extra 5). The resumed agent's share: S2-S5 = 12:11 of the ~15 left at the resume brief.

## Parked 2026-11-26 20:40 (coordinator: Claude Desktop restart)

(Resumed 2026-11-26 ~21:00 by a new agent: S1 checked valid, S2-S5 run, gate 4 verdict and the opt-in default in the Result
section at the top. The text below is the parked state as it was.)

- **Head**: the commit that adds this section (parent 32bd1e389), branch `ink/hip-overlap`, local only (not pushed; origin has none of it).
  Candidate binary o4 = 8ed77739d (md5 38475e68394d85800f5e970d39006508), baseline main5 = origin/main a4c8f3b19 (md5
  42fdf7a7d871d6694625fba108e086c7); both in tatooine `~/intarsiaml-ink-hip-bpf/.scratch/bin/`, both admission-primed.
- **Passed**: cpu tests (#12319: core lib 274, scheduler 54, inkling2 17, inkling2_batch 10, inkling2_packed 6, docs_env_vars 2,
  server 71); same-ISA check (#12325: 168 kernels identical, diffs only in the 3 ticket kernels); O4 (#12322): default budget
  unchanged (context 3285, 52560 KV both), real prompts 4 x 4/4, 2 x 8/8, 16/16 with overlap default-on, md5 main5 spec off and o4
  DSpark = W3's, hip tiny batch 10/10 and packed 6/6, admission conv_mix / loop_blend / ink_attn_decode / ink_route ADMITTED.
  O1-O3 (#12281, #12293, #12303): overlap bitwise everywhere incl. the poisoned o2 run (16/16, 4 x 4/4).
- **Not verified**: gate 4 (no paired speed rep of o4 vs main5 yet: O3's single runs only); the poisoned run on o4 (o2's predates
  side_staging and the decode-lane hook); md5 of o4 `--spec off` and main5 DSpark; G2 short / long; the solo server on o4.
- **Radiance for brief 3**: O1 4:04, O2 3:32, O3 4:17, O4 4:02 = 15:55, plus S1 (#12331), which fleetq started at 20:38:20 just
  before the park; released at ~20:39, its tatooine window.sh (trap) restored radiance (active at 20:42:08). The Mac driver kept running to the
  window's cap: both S1 visits (A, B) finished their real and filler runs (localeval #12337, #12338); not yet checked or summarised. Left: about 10 min.
- **Logs**: Mac `docs/results/2026-11-26-inkling2-hip-overlap/raw/{o1,o2,o3,o4,s1}/`, `*-fleetq.log`, `isa-check.log`; tatooine
  `~/intarsiaml-ink-hip-bpf/logs/{o1,o2,o3,o4,s1,isa,serve}/`.
- **Next step**: window S1 as specified (`raw/visits-s1.txt`, `raw/stage_s1bg.sh`; then S2 = BA with `stage_s2bg.sh`, S3 = AB with
  `stage_s3bg.sh`), each: `fleetq run --host tatooine-r9700 --takes-down radiance --quiet gpu,cpu --expect 8 -- ... serve_window.sh`
  with `VISITS=raw/visits-sN.txt OUT=raw/sN CAP=250 LE_MIN=20 LE_MARGIN=6 REAL_OFF=N-1 BG=stage_sNbg.sh BG_CAP=240` (sync.sh first);
  then `python3 raw/speed_sum.py s1 s2 s3` against the section-4 rule; default-on stays (INK_OVERLAP_DEFAULT) only if it passes,
  else set it false (opt-in) and record the numbers. The 10 min left fit 2 windows: ask the coordinator for a third or run n = 2.
