# sideeager: does `INTARSIA_SIDE_EAGER=0` help plain base, and is there a second slow-mode trigger? (CUDA, Qwen3.6-35B-A3B NVFP4, default MTP recipe, spark1)

Lane `sideeager`, branch `sideeager` from origin/main `7cfe3d885` (contains #447, postchain2). Measurement only, no engine change. spark1 (DGX Spark GB10) with vLLM up
beside it (it had no request during any run: its counters, polled every second, did not move: `raw/ab*/vllm-*.log`, `raw/ns/ns*/vllm-*.log`); fleetq `--quiet gpu,cpu` for
every timing run, `--load cpu` for the build and the analysis; CARGO_BUILD_JOBS=3 MAX_JOBS=3 FLASHINFER_NVCC_THREADS=1, fleetq `--mem 55`, free-memory watchdog
`scripts/sideeager/wd.sh` (kill below 15 GB available) on every build and serve (never fired; 55-57 GB available at every server start). spark2 not used. Raw output:
`docs/results/2026-11-18-sideeager/raw/`; scripts: `scripts/sideeager/` (lane copies of `scripts/postchain2/`'s `build.sh`, `wd.sh`, `ab.sh`, `nsys` and gates scripts, adapted,
plus `nsys2.sh`, `nsys_req.py`, `nsys_tab.py`, `srv_screen.py`, `prev_eager.py`). Nothing default-on.

**Result.**
1. **Plain base does gain from the knob, but only where requests end on an eager 1-draft step.** Base servers enter the slow graph-replay mode after such a request
   (nsys: 205-374 of 1191 graph launches slow in 9 of 9 base servers, 0-1 in 13 of 13 knob servers), and the request after it runs 1.3 % slower. On a cell where about half the requests end that way
   (code-gen, 300 tokens) the knob is **1.013x** of base (4 of 4 rounds above 1, SE 0.0014). On the four default cells, where only 1-3 of 12-13 requests per server end on an eager step,
   it is neutral: **1.0000 median, 1.0008 mean (SE 0.0018) over 8 rounds**. Greedy ids and served replies identical. The knob is a default-on **candidate** for CUDA base; the flip is the user's call.
2. **The unexplained slow server is not a second trigger of the slow mode.** It reproduced (ab1 round 1: the knob server +0.205 ms per step, from the first 100 steps to the end; traced once: k4, +0.311 ms from its first request), but a repeat of the same two
   servers, same prompts, gave the opposite sign (ab3 round 1: knob -0.105 ms). Fresh servers of the same arm differ persistently by up to ~0.3 ms per step, in both arms; the difference is in the replay's span (kernel time, r = 0.83 over 22 traced servers;
   concentrated in the big weight-reading GEMVs), not in its idle time (68.5 us per replay everywhere, slow-mode launches excepted). Nothing precedes it: it is there from the first request. Why a server starts slow or fast is not found (section 4).

## Method

Binary: one build for every arm, `intarsia` from origin/postchain2 `3d6f7b812` (the #447 commit; its `crates/` tree is identical to origin/main `7cfe3d885`, `git diff` empty) +
the lane's scripts, `INTARSIA_BUILD_STAMP=sideeager-3d6f7b812...-dirty`, built on spark1 in 6 min 26 s (`raw/build-run.out`). Arms differ by env only: `base` (no knob) and `bk`
(`INTARSIA_SIDE_EAGER=0`). Every server: `serve --backend cuda --max-batch 1 --max-seq 8192 --spec mtp --spec-k 2 --mem-budget 40`, `INTARSIA_SPEC_TRACE=1`, 2 warm-up requests (chat-short), then
the cells.

- **ABBA** (`scripts/sideeager/ab.sh`, postchain2's, cells from `scripts/mcopy/prompts.py`): one server per arm and round, arm order reversed on even rounds (base first on odd rounds), 4 rounds, cells in
  the order chat-short, code-gen, reason, agent-edit, 3 reps each (400 / 800 / 500 / about 300 tokens; max_tokens 400 / 800 / 500 / 1200); decode tok/s of every rep from the stream.
  Per round and cell: median of the 3 reps; ratio bk / base; headline = **median of the per-round ratios**, with the mean and SE and the per-round values (`scripts/sideeager/ab_an.py`).
  "ALL" = mean over the four cells of the per-round ratio. Run twice (`ab1`, replicate `ab3`, same prompts) and pooled as 8 rounds. A fifth cell, `code-gen300` (the 300-token code-gen request
  that postchain2 used for PRE; not one of the brief's four), ran as `ab2` (4 rounds x 8 reps) because the four default cells rarely end on an eager step (below).
- **nsys** (`scripts/sideeager/nsys2.sh`): per server, `nsys launch` (CUDA trace, graphs traced per node) and `nsys start` right after it (server start and load included), 2 warm-ups, 3 x code-gen 300 tokens (PRE,
  r2-r4), 2 x code-gen 800 tokens (MID, r5-r6); then `nsys_req.py` on the sqlite. A graph launch (the 1002 / 1000 / 994-node MTP-step variants) is **slow** when its idle time (span minus the union of node intervals) is above 200 us,
  postchain2's rule (0.07 vs 0.32 ms bimodal). Requests are cut where two launches are more than 60 ms apart (7 segments for 7 requests in every server).
- **Eager step** = a `spec: token` line with no graph step after it (the 1-draft MTP step that a request ends with when one token is left). Counted per server from the serve log.
- **Gate** (`scripts/sideeager/gates.sh`): `generate --spec mtp --spec-k 2 --chat --no-think --temperature 0 --show-ids`, knobs unset vs `INTARSIA_SIDE_EAGER=0`, same binary.

## 1. Base vs base + knob, ABBA, four default cells (`raw/ab1/`, `raw/ab3/`, `raw/ab13-pooled.txt`)

Ratio of per-round median decode tok/s, bk / base (`ab1`, then replicate `ab3`; round 1-4 each):

| cell | ab1 r1-r4 | ab3 r1-r4 | median of 8 | mean (SE) | > 1 |
|---|---|---|---|---|---|
| chat-short | 0.9888 0.9978 0.9984 0.9977 | 1.0030 0.9927 1.0012 1.0040 | 0.9981 | 0.9979 (0.0018) | 3/8 |
| code-gen | 0.9872 0.9976 0.9999 0.9976 | 1.0041 0.9979 1.0049 1.0060 | 0.9989 | 0.9994 (0.0021) | 3/8 |
| reason | 0.9908 1.0004 1.0003 0.9970 | 1.0069 1.0130 1.0044 1.0029 | 1.0017 | 1.0020 (0.0023) | 6/8 |
| agent-edit | 1.0000 1.0020 1.0027 0.9986 | 1.0193 0.9954 1.0070 1.0051 | 1.0024 | 1.0038 (0.0026) | 5/8 |
| **ALL** | 0.9917 0.9995 1.0003 0.9977 | 1.0083 0.9997 1.0044 1.0045 | **1.0000** | **1.0008 (0.0018)** | 4/8 |

(ab1 alone: ALL median 0.9986, mean 0.9973; ab3 alone: median 1.0044, mean 1.0042.) **Neutral on these cells: no gain, no loss beyond the noise.** The two runs disagree on sign for the same prompts
because a fresh server's step time varies by ~0.1-0.2 ms (section 4), more than the effect. Reason: these cells almost never end on an eager step. Per server, 1-3 eager 1-draft steps in 12-13 requests (positions 3180, 871, 576, 438: each is the final position of a request that ran to its
`max_tokens`, agent-edit's 1200, code-gen's 800, reason's 500, chat-short's 400; only some such requests end that way, the other end on a 2-draft step), identical in both arms of a round; requests that stop on their own (agent-edit
298 tokens, chat-short 351-394) had none.

Greedy replies (sha of the full reply text) equal across the two arms in **every** round, cell and rep: 16 of 16 round-cells in ab1 and in ab3 (`raw/ab*-an.txt`, 3 reps each; the agent-edit cell
is checked only this way, it has no single-message prompt for `generate`).

## 2. Where the knob helps base: requests that end on an eager step (`raw/ab2/`, `raw/ab2-an.txt`, `raw/ab2-prev-eager.txt`)

Cell `code-gen300` (code-gen, 300 tokens: the request ends on an eager 1-draft step at position 371 in 3-6 of its 8 reps per server, the same in both arms): 4 rounds ABBA, 8 reps per server.

| round | base tok/s | bk tok/s | bk / base | graph step bk - base (ms) | eager end steps per server |
|---|---|---|---|---|---|
| 1 | 153.12 | 155.00 | 1.0122 | -0.144 | 6 |
| 2 | 156.45 | 158.02 | 1.0100 | -0.042 | 4 |
| 3 | 153.50 | 155.67 | 1.0141 | -0.159 | 5 |
| 4 | 154.51 | 157.11 | 1.0168 | -0.069 | 3 |
| | | median **1.0132**, mean 1.0133 (SE 0.0014) | 4/4 > 1 | mean -0.103 | |

Replies equal across arms in every round (8 of 8). Per request (`prev_eager.py`, pooled over the 4 rounds, 16 requests per cell of the table): **base**, request after one that ended on an eager step:
153.82 tok/s (n 16), after one that did not: 155.84 (n 16), **-1.3 %**; **bk**: 156.05 and 156.06, no difference; bk matches base when the previous request ended normally (156.06 vs 155.84).
So the knob removes the post-eager-step penalty, and the size of the gain scales with how many requests follow an eager-ending request (here about half; in section 1, almost none).

## 3. nsys: does base enter the slow mode? (`raw/ns/`, `raw/ns/ns*/*.req.txt`; table: `scripts/sideeager/nsys_tab.py`)

Slow graph launches per request (r0-r1 warm-ups chat-short, r2-r4 code-gen 300, r5-r6 code-gen 800), all launches, side-stream eager kernels in the whole session, and the median replay span over r0-r3
(before any request-ending eager step, so the same state in both arms) and the median idle inside the replay:

| server | arm | slow launches r0 r1 r2 r3 r4 r5 r6 | slow / launches | side-stream eager kernels | span r0-r3 (ms) | idle (us) |
|---|---|---|---|---|---|---|
| b1 | base | 0 0 0 0 19 284 4 | 307 / 1191 | 560 | 17.434 | 68.5 |
| b2 | base | 0 0 0 0 108 195 0 | 303 / 1191 | 560 | 17.389 | 68.5 |
| b3 | base | 0 0 0 0 108 122 0 | 230 / 1191 | 560 | 17.336 | 68.6 |
| b4 | base | 0 0 0 0 108 213 0 | 321 / 1191 | 560 | 17.445 | 68.5 |
| b5 | base | 0 0 0 0 7 280 0 | 287 / 1191 | 560 | 17.520 | 68.9 |
| b6 | base | 0 0 0 0 1 276 0 | 277 / 1191 | 560 | 17.517 | 68.7 |
| b7 | base | 0 0 0 0 108 193 0 | 301 / 1191 | 560 | 17.517 | 68.7 |
| b8 | base | 0 0 0 0 53 284 37 | 374 / 1191 | 560 | 17.489 | 68.3 |
| b9 | base | 0 0 0 0 108 97 0 | 205 / 1191 | 560 | 17.461 | 68.5 |
| k1 | knob | 0 1 0 0 0 0 0 | 1 / 1191 | 0 | 17.465 | 68.5 |
| k2 | knob | 0 0 0 0 0 0 0 | 0 / 1191 | 0 | 17.344 | 68.7 |
| k3 | knob | 0 0 0 0 0 0 0 | 0 / 1191 | 0 | 17.442 | 68.8 |
| k4 | knob | 0 0 0 0 0 0 0 | 0 / 1191 | 0 | 17.571 | 68.5 |
| k5 | knob | 0 0 0 0 0 0 0 | 0 / 1191 | 0 | 17.503 | 68.6 |
| k6 | knob | 0 1 0 0 0 0 0 | 1 / 1191 | 0 | 17.519 | 68.4 |
| k7 | knob | 0 0 0 0 0 0 0 | 0 / 1191 | 0 | 17.360 | 68.7 |
| k8 | knob | 0 0 0 0 0 0 0 | 0 / 1191 | 0 | 17.512 | 68.5 |
| k9 | knob | 0 0 0 0 0 0 0 | 0 / 1191 | 0 | 17.566 | 68.5 |
| k10 | knob | 0 0 0 0 0 0 0 | 0 / 1191 | 0 | 17.523 | 68.7 |
| k11 | knob | 0 0 0 0 0 0 0 | 0 / 1191 | 0 | 17.416 | 68.7 |
| k12 | knob | 0 0 0 0 0 0 0 | 0 / 1191 | 0 | 17.425 | 68.8 |
| k13 | knob | 0 0 0 0 0 0 0 | 0 / 1191 | 0 | 17.468 | 68.8 |

- **Base enters the slow mode, in the request after a 300-token request's eager end step.** In all 9 base servers the end of the second and third PRE request (r3, r4) runs one eager 1-draft step that launches 280 kernels on the side stream
  (the only side-stream eager kernels in the session, 560 per server; knob servers have none, their eager step has 906 kernels on the main stream), and the first slow launch of every one of them is launch 0 of r4. From there on 205-374 launches are slow
  (b1: the first 19 launches of r4 and all 284 of r5, the first MID request, ~5 s; b2 and b3: 108 of 108 of r4 and 195 / 122 of 284 of r5; b8: 53 of r4, 284 of r5, 37 of r6), then it ends by itself, as in postchain2 (r6, after an 800-token request that ends without an eager step, is fast in 7 of 9; b1 and b8 have 4 and 37 slow launches there, the tail of the run).
  Idle per replay in a slow launch: 324 us against 68.5 us.
- **Knob servers: no run of slow launches in any of 13.** Two servers (k1, k6) have one isolated slow launch each, at launch 96 / 98 of r1 (150 launches), no neighbours, no side-stream eager kernel.
- Cost of the slow mode in these sessions: 0.25 ms on 205-374 of 1191 launches; the bk throughput gain in section 2 (1.3 %) is the same size as 0.25 ms of 17.5 ms.

## 4. The unexplained slow server (`raw/ab*-screen.txt`, `raw/ns/`, `raw/tsv/`)

Question: postchain2's `bx` r3 server ran +0.3 ms per step in every request, from its first, with no chain and no side-stream eager pass. Is there another trigger?

**It reproduced without any trigger, in both arms, it was traced once (k4), and it is not the slow mode: the server is slow from its launch and the extra time is kernel time.**
- ab1 round 1: bk is +0.205 ms per step against the base server of the round, in every 100-step window from the first (+0.20 +0.22 +0.22 +0.18 +0.21 ...) to near the end (last window -0.01): the bx r3 picture. Throughput
  0.9917 in that round. Knob on, so no eager pass on the side stream; the steps of the two servers are identical (all 2681 matched by position and drafts).
- The replicate ab3 (same prompts, fresh servers): round 1 is **-0.105 ms** (bk faster in every window), round 3 -0.114, round 4 -0.074, round 2 +0.010. So a fresh server runs 0.1-0.2 ms per step slower or faster than another
  one of the same arm for its whole life, from its first request, on the same steps. Over 12 ABBA rounds (ab1, ab2, ab3) the per-round mean step difference spans -0.159 to +0.205 ms.
- **Traced (nsys from launch, `raw/ns/`): k4**, a knob server, is +0.311 ms per step against the median of the 22 traced servers over its first 515 steps (r0-r3, before any request-ending eager step), high from r0 (the first warm-up),
  and its session has no side-stream eager kernel at all (0 of 0), no slow launch, idle per replay 68.5 us like every other server. What it has is a longer span: r0 17.60, r1 17.68 ms, r2-r3 17.53 / 17.47, then 17.70-17.75 in
  r5-r6 (other servers 17.3-17.5). Nothing precedes the offset: it is there at the first request after the launch (nsys started at server launch). One base server (b6) has the same offset (+0.324 over r0-r3) before its own slow mode;
  its span (17.43-17.62) does not explain it: unexplained.
- **The offset is span (kernel) time.** Across the 22 traced servers (9 base, 13 knob), the step-time offset over r0-r3 against the group median correlates with the median span of r0-r3 at **r = 0.83, slope 1.19 ms of step per ms of span**
  (b6 the main outlier); the host gap between steps is not related (r = -0.01, 0.569-0.615 ms for all). Spans: base 9 servers 17.457 ms (sd 0.063, range 17.336-17.520), knob 13 servers 17.470 ms (sd 0.071, range 17.344-17.571): the same
  spread in both arms, 0.2 ms from lowest to highest, no knob offset (+0.013 ms). Median idle per replay 68.3-68.9 us in every fast request of every server.
- **Which kernels** (`raw/cmp/cmp.txt`, `nsys_cmp.py`: three high servers k4 / k9 / k6 against three low ones k2 / k7 / k11, whole sessions, 555 launches each of the 1006-node graph; null = differences within the low and within the
  high group): span +0.142 ms (nulls +0.011, +0.039); per kernel name, us per launch: `fp8_gemv_rows_full_kernel` (130 nodes, 6104 us) **+91** (nulls +7, +29), `gemv_split_kernel` (140 nodes, 2136 us) **+77** (nulls +39, +21),
  `nvfp4_gemv_split_kernel` +29 (nulls -1, +27), `moe_dec_gate_up_kernel` +22 (nulls -13, -9); the sum over all nodes is +220 us against nulls +133 / +115, so only the first two names clearly exceed their nulls. The 1004- and 998-node graphs
  show the same names. The slow servers are slower in the big weight-reading GEMVs, not in the small kernels; that points at memory (inference: where the weights' pages sit in the unified memory), not at launch latency. Not tested.
- The engine thread is on a Cortex-X925 core in all 8 sampled ABBA servers of ab1 (`raw/tsv/ab1`: all CPU ticks on X925 cores; bk r1, the slow one, on cpu 15 for its whole life, like base r3), so it is not the A725 / X925 placement of
  the cpupin lane. vLLM had no request in any run (counters polled every second in ab1-ab3 and ns1-ns3: no change).
- Consequence for readings: ABBA ratios from one server pair per round carry ~0.1-0.2 ms of per-server noise (0.6-1.1 % of a 17.5 ms step); 4 rounds do not pin an effect below ~0.5 %. Section 2's 1.3 % (4 of 4 rounds, with
  the per-request split beside it) is above that; section 1's neutral is neutral to about +-0.4 %.

Not done: tracing ab1 round 1's server itself (not traced; the knob servers traced from launch are k1-k13, of which k4 shows the same kind of offset). Why a fresh server's weights / kernels run 0.2 ms apart is not established.

## 5. Gates (`raw/gates.run.out`, `raw/gates/`; `scripts/sideeager/gates.sh`)

| gate | result |
|---|---|
| greedy ids, `generate --spec mtp --spec-k 2`, knobs unset vs `INTARSIA_SIDE_EAGER=0`, same binary | **PASS** chat-short 364 ids, reason 500, code-gen 500: identical per cell |
| greedy replies, served, base vs bk | identical sha in every round, cell and rep: ab1 16 of 16 round-cells (3 reps each), ab3 16 of 16, ab2 4 of 4 (8 reps each); agent-edit is covered only by this (multi-message prompt) |
| greedy replies in the nsys servers (same prompts, 7 requests) | identical sha list in every server, base and knob (`raw/ns/ns*/*.jsonl`) |

Not run: the repo's CUDA test suite (no engine change here).

## Verdict

- **Question 1: yes for base, conditionally.** The knob removes a 1.3 % penalty on every request that follows a request that ended on an eager 1-draft step (base: 153.8 vs 155.8 tok/s; knob: no difference), measured as
  1.013x on a cell where half the requests end that way, and it costs nothing measurable where they do not (1.0000 median over 8 rounds, SE 0.0018). Base servers do enter the slow mode (nsys: 205-374 slow launches
  per server in 9 of 9, none in the 13 knob servers). How much a real workload gains depends on its share of requests that run to `max_tokens` and end on a 1-draft step: the default cells here have 1-3 per 12-13 requests (they do not lose);
  in these logs every eager end step is at a request that ran to its cap, none at a request that stopped on its own.
  **Default-on candidate for CUDA `--max-batch 1` MTP serving: yes on this evidence. Not tested: `--max-batch` > 1, tensor parallelism, HIP, Metal, sampled decoding. The flip is the user's call; nothing was changed.**
- **Question 2: no second trigger found.** The +0.2-0.3 ms servers appear in both arms and flip sign between repeated runs of the same pair, with the same steps and the same idle per replay: a per-server step-time offset of
  about +-0.1-0.3 ms (kernel time, mostly in the big weight-reading GEMVs), independent of the knob and of eager passes. Why a server starts that way is not explained; k4 was traced from its launch and shows no eager pass, no slow launch, normal idle, longer span (section 4).

Revert: nothing in the engine changed; delete this doc, `scripts/sideeager/`, `raw/` and the LOG row. To drop the knob itself, revert #447.

## Not verified / limits

- 8 ABBA rounds on the default cells and 4 on `code-gen300`, one server pair per round: per-server noise 0.1-0.2 ms (above). Cells other than the five above, other models and hardware, and real traffic mixes not measured.
- nsys covers 9 base and 13 knob servers over a 7-request session; the slow-mode counts are of that session's design (three 300-token requests, then 800-token requests), not of live traffic.
- The mechanism of the slow mode and of the per-server offset is not visible below the driver (no GPU metrics without admin rights).
- Time: part 2 (the unexplained server) ran from 06:40 to 08:05 by the clock, interleaved with parts 1 and 3 (ABBA, gates); counted as machine time it used about 42 minutes (nsys ns1 + ns2 + ns3), which I took as the 45-minute box.
- The `code-gen300` cell is an addition to the four default cells; the per-cell request mapping of `ab_an.py`'s serve-log part does not print (the log's request boundaries come out 12-13 for 14 requests), only the per-server totals.

## Instructions received

None beyond the brief (lane rules `/opt/code/intarsiaml/.worktrees/scan/lane-rules.md`). Deviations, decided in-lane: #447 was not in main when the lane started (merged 06:22 EST); the lane branched from `origin/postchain2`
(`3d6f7b812`, the same crates as main) to start the build and was rebased onto origin/main `7cfe3d885` before the commit; the `code-gen300` ABBA, the ab1 replicate and the extra traced servers were added.

## Files

`scripts/sideeager/`: `build.sh`, `wd.sh`, `ab.sh` (ABBA; vLLM counter poller and `SAMPLE=1` thread sampler added), `ab_an.py` (ratios, medians, graph step, eager steps; several dirs pool as more rounds), `nsys2.sh` (collection from server
launch), `nsys_req.py` (per request: slow launches, idle, span, eager kernels per stream, first slow launch), `nsys_tab.py`, `nsys_mode.py` / `nsys_an.py` / `nsys_cmp.py` (copies of postchain2's), `srv_screen.py` (per-server step offset),
`prev_eager.py`, `gates.sh`.
Raw: `raw/ab1`, `raw/ab2`, `raw/ab3` (serve logs, jsonl, replies, metrics, vLLM poller), `raw/ab*-an.txt`, `raw/ab*-screen.txt`, `raw/ab13-pooled.txt`, `raw/ab2-prev-eager.txt`, `raw/ab*.run.out`, `raw/ns/` (per server serve log,
jsonl, `*.req.txt`, sampler, nsys stop logs; the `.sqlite` / `.nsys-rep` files are not committed), `raw/tsv/` (thread samples), `raw/cmp/` (k4 / k9 / k6 against k2 / k7 / k11, per kernel name), `raw/gates/`, `raw/gates.run.out`, `raw/build-run.out`, `raw/ns1a.run.out` (the first nsys attempt, failed on a session-name race, fixed in `nsys2.sh`).
spark1's `~/intarsiaml-sideeager` is removed.
