# postchain: what an accepted copy chain leaves behind (CUDA, Qwen3.6-35B-A3B NVFP4, spark1)

Lane `postchain`, branch `postchain` from origin/cpupin (PR #439 head `5b51c9ff5`, which carries #436's opt-in probe hooks). Measured on spark1 (DGX Spark GB10)
with vLLM up beside it (fleetq `--quiet gpu,cpu` for every timing run, `--load cpu` for the build; CARGO_BUILD_JOBS=3 MAX_JOBS=3 FLASHINFER_NVCC_THREADS=1, fleetq `--mem 55`,
free-memory watchdog `scripts/postchain/wd.sh` on every build and serve). Raw output: `docs/results/2026-11-18-postchain/raw/`; scripts: `scripts/postchain/`.
Nothing default-on.

**Result: cause not found; narrowed. No fix.** Two hypothesis rounds, then the stop rule.

- **The trigger is the chain pass itself, accepted or rejected.** With an instrument whose positive control works (the 800-token request right after the event,
  "MID1", against the arm's own PRE requests), a single bogus 8-row chain that is rejected (`L7`) leaves +0.133 ms per graph step (5 of 5 rounds > 0, 2.6 SE),
  a single accepted 7-draft chain (`a7`, new probe) +0.158 (5 of 5 > 0, 1.7 SE), real n20 chains +0.278 (5 of 5, 3.0 SE). cpupin's "a rejected chain does not
  reproduce it" came from its 300-token later-request window, which does not see real chains either (+-0.04 here).
- **Not the head.** A head K / V-only pass of 4 or 8 rows (`k4`, `k8`, no chain; 4 rows = the bf16 GEMV path, 8 = cuBLAS) leaves nothing (-0.008 +- 0.030,
  +0.005 +- 0.021); capping the head's post-chain catch-up at 3 rows (`INTARSIA_SPEC_COPY_CATCHUP=3`, opt-in, new) leaves real chains as slow (+0.185, 4 of 4).
- **Not host time, not different work.** Base and n20 run identical drafts at every matched step; the whole difference is between the graph launch and the
  outputs' download (device time), not in any host segment of the step (all under 0.002 ms) and not in the launch call.
- **It fades.** The offset is in the request right after a request with a chain pass, and gone two requests later (POST1-9: n20 +0.067 +- 0.076, L7 +0.018, a7 +0.059).
  In the workloads that made the 0.984x (800-token code-gen, a chain in most requests) that is the same as permanent.
- The remaining suspect is what the eager `1 + len`-row verify pass of the main model does that a 3-row captured step does not: its 8-row kernel variants, the GDN
  verify writing save rows 2..6 (the whole recurrent state per row; about 60 MB each by my estimate of 30 GDN layers, not measured; rows that only chains touch), and the eager launches on the model's stream. Not separated here.
- Code-gen stays at about 0.984x of MTP base (cpupin, mcdiag; not re-measured); the 0.995x bar is not met; nothing default-on. A full fix of the offset would give
  about 0.9965x (estimated from pin1 / real3, not measured).


## Step 1: what an accepted chain does that a rejected one does not (code reading, no hardware)

`LocalEngine` (`--max-batch 1`) holds ONE `SequenceState` for the whole process (`engine.rs`, `LocalEngine::new`); a new request truncates or restores it, it is
never reallocated. So "the slowdown survives into later requests (fresh KV, fresh head queue)" does not by itself mean device or driver state: anything on that
object (save buffers, snapshot pool, the head's cache and queue buffers) is process-lifetime too.

The path of a chain (`Seq::prepare_with`, `spec_verify` / `spec_accept`, `SequenceState::spec_rollback`, `Model::mtp_catch_up`), accepted (`j` >= 1 drafts) vs rejected (`j` = 0):

| state | rejected chain (and the bogus probe `L<l>`) | accepted chain |
|---|---|---|
| verify pass | eager, `1 + len` rows, row-exact kernels | the same |
| recurrent rollback | from save row 0 (`spec_rows.bufs[0]`, as a plain MTP step) | from save row `j` (rows 2..6 only chains touch), or none when all are accepted |
| position | `+1` | `+j+1` in one pass |
| head catch-up after the chain (`copy_behind` -> `mtp_catch_up`, K / V-only, eager) | `p0` rows, `p0` = rows the previous step queued, 1..3: the sizes the captured step itself runs | `p0 + j` rows: **4 to 10** in the real runs (below), a row count no other decode-time head pass has (the captured step takes `n <= k + 1 = 3`; prefill takes the prompt) |

Row counts of the head's catch-up after every real chain in cpupin's `raw/pin1` (`scripts/postchain/segments.py`): the first chain of each server
5, 4, 5, 5 rows; all 35 chains 3 (accepted 0) to 10 (accepted 7). The head's projections switch kernels by row count (CUDA: bf16 `matmul_bf16_gemv` at M <= 4,
cuBLAS `matmul_bf16` above, `crates/intarsia-backend-cuda/src/kernels.rs`; NVFP4 GEMV NT pick 1 below M = 4, `kernels/cuda/nvfp4.cu`).

## Step 1b: what the existing logs already say (no hardware)

Re-analysis of mcdiag `raw/real3` and cpupin `raw/pin1` (same deterministic chains, 4 rounds each; base vs n20 of the same round):

- **The work is identical.** At every graph step matched by (request, position, catch-up rows `n`), base and n20 have the same drafts and the same accepted
  count, before and after the first chain (`scripts/postchain/drafts_id.py`: 0 differing steps of 8 885 in each of pin1 and real3). So the same work runs slower; the head's state is not
  different in any way that changes a draft.
- **Not the rollback copy.** Split by the step's own outcome (`scripts/postchain/outcome_an.py`), the post-chain offset is as large in steps that accept both drafts
  (no rollback at all: pin1 +0.240, real3 +0.185 ms) as in steps that roll back from save row 0 or 1. It is in the replay (or its download), not in the rollback.
- **The server level drifts.** Base servers drift by about +-0.2 ms per step from request to request (`scripts/postchain/perreq.py`), and the "no chain yet" bins of the
  real-chain runs are about one request per server. Per round, seen minus not-seen (matched steps): pin1 +0.17 +0.29 +0.16 +0.24, real3 +0.09 +0.17 +0.38 +0.23 (8 of 8 positive).
- **Ceiling of a fix.** With the matched post-chain offset removed, n20 / base in graph-step time would be 0.9981 0.9958 0.9992 0.9928 (pin1) and 0.9981 0.9966 0.9988 0.9924 (real3),
  mean 0.9965 (`scripts/postchain/ceiling.py`): a full fix would clear the 0.995 bar on average, not in every round (the r4 server's 13 mostly-missed chains cost 0.7 % by themselves).
- **"A rejected chain slows its own request": not shown.** cpupin's single-event runs (`raw/sw1`, `raw/sw2`) measured only later requests. Inside the event request,
  matched steps after the bogus chain minus before it, arm minus base (`scripts/postchain/midmatch.py`), give L3 -0.21 +0.27 +0.24 +0.10, L5 +0.12 +0.26 +0.31 +0.12,
  L7 +0.05 +0.43 +0.16 +0.10. But the same single-request estimator gives +0.11 / +0.19 for arms with no event at all before position 828 (n20 and cap of round 1
  below, against a7's -0.03 when a7 fired nothing), so a per-request value of this estimator is noise of +-0.2; the null I first compared with (mcdiag's n256)
  averaged 8 requests per round, not one. So mcdiag's gen1 "in-request" repro is not confirmed by these data either way.

## Round 1: the head pass, a capped catch-up, and the later-request instrument (`raw/r1b/`, `raw/r1.log`)

Build: this branch with the probe kinds `a<l>` (accepted chain from a reference text), `k<m>` (head K / V-only pass of `m` rows), `INTARSIA_SPEC_COPY_PROBE_ONCE=<i>`
(fire in the i-th request that reaches `g`) and `INTARSIA_SPEC_COPY_CATCHUP=<rows>` (all opt-in, `docs/tuning.md`), plus a split of the traced graph step
(host before the launch / launch to download done / host after it / acceptance; the launch call alone from round 2 on).
Design (`scripts/postchain/serve.sh`, from cpupin's serve2): per server 2 warm-ups (chat-short), REF (code-gen, 500 tokens, prompt tag `c1-0`), PRE x6 (code-gen, 300 tokens),
MID x2 (code-gen, 800 tokens, tags `c1-0`, `c1-1`: the prompts of pin1 r1, whose replies make real n20 chains: MID0 at position 828 accepted 2, MID1 at 665 accepted 1,
the same chains as pin1 r1 and real3 r1), POST x10 (code-gen, 300 tokens). Probes fire at 450 generated tokens in MID0 (`ONCE=2`; REF is the first request to reach 450).
Arms, 4 rounds ABBA, one server per arm and round: `base`; `n20` (real chains, the positive control); `a7` (accepted chain probe: it fired NOTHING in round 1, a bug: the
reference was replaced by every later, shorter request; fixed for round 2, so in round 1 `a7` is a knob-on, no-event null); `k4`, `k8` (one head pass of 4 / 8 rows,
no chain: 4 rows is the bf16 GEMV path, 8 rows the cuBLAS path; the MTP head is unquantized bf16, the checkpoint's quantization config ignores `mtp*`); `cap`
(n20 with `INTARSIA_SPEC_COPY_CATCHUP=3`). Greedy replies equal across all arms of a round (sha per request in `raw/r1.log`). n20 r2 is void (its PRE requests ran at 31 ms per
step, the outside-load mode cpupin saw in L7 r3).

Arm minus base per request, mean over valid rounds, ms per graph step (matched steps; MID0 over positions 100-440, MID1 over 100-660, i.e. before each request's own chain):

| arm | PRE0-5 | MID0 | MID1 | POST0 | POST1-9 |
|---|---|---|---|---|---|
| n20 | +0.092 +0.026 -0.011 +0.225 -0.122 -0.082 | +0.092 | **+0.189** | **+0.155** | +0.001 .. +0.248 (mean +0.007) |
| cap | +0.075 +0.024 -0.045 +0.189 +0.020 -0.158 | -0.065 | **+0.208** | **+0.260** | mean +0.032 |
| a7 (no event) | -0.019 .. +0.049 | +0.071 | -0.008 | -0.028 | mean -0.028 |
| k4 | | -0.150 | -0.039 | -0.055 | mean -0.039 |
| k8 | | -0.131 | -0.082 | -0.095 | mean -0.076 |

MID1 per round: n20 +0.34 +0.24 -0.02 (r2 void); cap -0.06 +0.47 +0.13 +0.30; a7 -0.01 +0.10 -0.05 -0.08; k4 -0.06 +0.01 -0.07 -0.03; k8 -0.20 +0.14 -0.08 -0.20.

- **The head's catch-up row count is not the trigger.** k4 and k8 run the head pass a chain's catch-up runs (4 rows on the GEMV path, 8 on cuBLAS) and leave nothing in the
  requests after them; cap (real chains, catch-up in passes of at most 3 rows, so never a head pass of 4+ rows) is as high as n20. **Hypothesis round 1: null.**
- **The later-request offset of real chains is short-lived here.** It is in MID1 and in POST0 (the requests right after a request with a chain), not in POST1-9 (carry
  POST minus PRE: n20 -0.04 -0.08 +0.06, cap +0.19 -0.15 +0.05 +0.06). So cpupin's 300-token POST instrument cannot see it, and its "a single rejected chain leaves
  nothing in later requests" says nothing about chains either. MID1 is the one window of this design where the positive control is positive (n20 r1: +0.34 after pin1 r1's
  exact chain; pin1 r1 measured +0.41 for the steps after that chain).
- **No host cause inside the timed step.** Every host segment of the split differs by under 0.002 ms between arms; all of the step-time difference is in launch-to-download
  (`scripts/postchain/split_an.py`). (This holds for every comparison of round 1; it does not locate a real offset by itself, since most of them are at noise level.)

## Round 2: accepted vs rejected, on the MID1 instrument (`raw/r2/`, `raw/r2.log`, `raw/r2-an.txt`)

Primary measure, fixed before the run (`scripts/postchain/r2_an.py`): per arm and round, (arm - base) over MID1 positions 100-640 (before MID1's own n20 chain at 665)
minus (arm - base) over the PRE requests; steps matched by (request, position, n) with the same drafts. Positive: > 0 in at least 4 of 5 rounds and mean > 2 SE.
Arms, 5 rounds ABBA: `base`, `n20`, `a7` (fixed: "probe accept chain of 7 at token 450", accepted 7 of 7 at position 523 in every round), `L7` (bogus chain of 7 at 450,
accepted 0 at position 523 in every round). Greedy replies equal across arms in every round. No void round.

The same measure over round 1's arms is in the table too (`raw/r1b-r2an.txt`; n20 r2 void):

| arm | event in MID0 | MID1 per round | MID1 mean (SE) | POST0 mean | POST1-9 mean |
|---|---|---|---|---|---|
| base | none | | 0 | 0 | 0 |
| a7, round 1 (fired nothing) | none | +0.033 +0.040 -0.028 +0.028 | +0.018 (0.016) | -0.004 | -0.004 |
| k4, round 1 | head pass, 4 rows | +0.010 +0.055 -0.089 -0.006 | -0.008 (0.030) | -0.026 | -0.014 |
| k8, round 1 | head pass, 8 rows | -0.023 +0.051 -0.038 +0.031 | +0.005 (0.021) | -0.005 | +0.014 |
| cap, round 1 | real chains, catch-up <= 3 rows | +0.168 +0.103 +0.117 +0.351 | **+0.185 (0.057)** | +0.243 | +0.015 |
| n20, round 1 | real chains | +0.286 (void) +0.201 +0.019 | +0.169 (0.079) | +0.134 | -0.035 |
| **L7**, round 2 | bogus chain, rejected | +0.037 +0.268 +0.053 +0.062 +0.246 | **+0.133 (0.051)** | +0.057 | +0.018 |
| **a7**, round 2 | accepted chain of 7 | +0.079 +0.536 +0.034 +0.047 +0.092 | +0.158 (0.095) | +0.074 | +0.059 |
| **n20**, round 2 | real chains | +0.343 +0.580 +0.041 +0.146 +0.281 | **+0.278 (0.092)** | **+0.240** | +0.067 |

(Round 2's r2 base server is slow in its later requests: every arm is high in r2. Without r2: L7 +0.10, a7 +0.06, n20 +0.20.)

Reading (the advisor's table, decided before the run): n20 +, a7 +, L7 + -> **any chain pass**; acceptance is not what matters. POST0 is high for n20 only, because
MID1 has its own n20 chain (at 665), so POST0 is again "the request after a chain"; for a7 and L7 POST0 is two requests after the event. **Hypothesis round 2
(accepted-only trigger): null.** Stop rule reached; no third probe round.

What this does not settle: whether a7 differs from L7 (they are within 1 SE of each other); whether the 8-row verify pass or the rollback is the trigger (L7 rolls
back from save row 0, as a plain MTP step does, so a rollback from rows 2+ is not needed); why the offset reaches the next request and not the one after.

## What is now ruled out (with the earlier lanes)

| candidate | evidence |
|---|---|
| graph re-capture / eager fallback, clocks, core placement | mcdiag, cpupin |
| different work after the chain | identical drafts and accepted counts at all 8 885 matched steps of pin1 and of real3 |
| the rollback copy | the offset is as large in steps with no rollback |
| host time inside the step | the split: under 0.002 ms in every host segment, 0 in the launch call |
| the head's large post-chain catch-up (row count, GEMV vs cuBLAS) | k4, k8 null; cap (no head pass above 3 rows) as slow as n20 |
| acceptance (a commit from save row j >= 2, the position jump, the accepted head rows) | a rejected bogus chain (L7) is positive too |

## Next measurement, if any (not run)

All on the MID1 instrument (it costs about 2.4 min per server; 5 rounds of 4 arms is 50 minutes):
1. Split the verify pass: probe `e` (one eager 3-row MTP step: eager launches, no 8-row kernels, save rows 0..1 only), `L2` (a 3-row bogus chain) and `L7`. If `e` is
   null and `L2` is positive, the trigger is the eager verify pass, not the row count; if only `L7`, the 8-row variants or save rows 3..6.
2. Real chains with `INTARSIA_SPEC_COPY=20,3` (4-row passes) vs `20` (8-row): the same split with real text.
3. Only then nsys: kernel times of MID1's ~280 replays in an n20 server vs a base server (many replays of one request, matched by position), to name the kernels.

## Gates (`raw/gates.log`, `scripts/postchain/gates.sh`; main = origin/main `835796344` with #439 merged, built as `intarsia-main`)

| gate | result |
|---|---|
| `plan show --json`, every new knob unset, `--spec mtp --spec-k 2 --mem-budget 40 --max-seq 8192`: main vs lane | **PASS** byte-identical (15570 bytes, sha `c446afdca09ab0ec`; main twice identical) |
| greedy ids, `generate --spec mtp --spec-k 2`, knobs unset, main vs lane | **PASS** copy3k 2902, json-xform 2600, chat-short 364, reason 500, code-gen 500 ids identical (the agent cells have no `.txt` prompt for `generate`; not run) |
| greedy replies with the hooks on: every arm of every round (n20, cap = n20 + `INTARSIA_SPEC_COPY_CATCHUP=3`, a7, L7, k4, k8) vs base | equal sha for all 19 requests per server (`raw/r1.log`, `raw/r2.log`) |
| `cargo test -p intarsia-cli --test docs_env_vars`; `cargo test -p intarsia-core --lib spec::` (incl. the new `probe_ref_drafts_continue_a_matching_prefix`) | PASS (Mac, after the merge of origin/main) |

Not run: the repo's full CUDA test suite; a speed ABBA of the task cells (there is no fix to time); the `--spec off` ids gate (no change on the decode path besides the
`Instant` stamps of the split).

## Verdict and revert

The copy knob stays opt-in on CUDA; nothing default-on. This PR adds measurement hooks and one experiment knob, all off by default: probe kinds `a<l>` and `k<m>`,
`INTARSIA_SPEC_COPY_PROBE_ONCE=<i>`, `INTARSIA_SPEC_COPY_CATCHUP=<rows>` (it does not fix the offset; kept as the record of the `cap` arm, removable), the step split
on the `INTARSIA_SPEC_TRACE` graph-step line, and `scripts/postchain/`. Revert: revert the PR's commits.

## Instructions received

None beyond the brief (lane rules `/opt/code/intarsiaml/.worktrees/scan/lane-rules.md`, including the 21:40 spark1 build caps and free-memory watchdog).

## Files

`scripts/postchain/`: `build.sh`, `wd.sh` (free-memory watchdog, never fired: 54-57 GB available at every server start), `serve.sh` (the REF / PRE / MID / POST design),
`gen.sh` (generate runs, prepared, not used), `gates.sh`; analyzers `r2_an.py` (the primary measure), `an.py`, `midmatch.py`, `split_an.py` (host/device split),
`drafts_id.py`, `outcome_an.py`, `segments.py`, `perreq.py`, `ceiling.py` (the re-analyses of pin1 / real3). Raw: `raw/r1b/` and `raw/r1.log` (round 1; `raw/r0-aborted.log`
is a first start of round 1 stopped after two servers because its MID prompt made no n20 chains), `raw/r2/` and `raw/r2.log` (round 2), `raw/*-an.txt`, `raw/gates/`,
`raw/gates.log`, `raw/build*.out`. Serve logs are gzipped; the analyzers read `.gz`. spark1's `~/intarsiaml-postchain` is removed.

