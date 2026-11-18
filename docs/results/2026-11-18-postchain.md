# postchain: what an accepted copy chain leaves behind (CUDA, Qwen3.6-35B-A3B NVFP4, spark1)

Lane `postchain`, branch `postchain` from origin/cpupin (PR #439 head `5b51c9ff5`, which carries #436's opt-in probe hooks). Measured on spark1 (DGX Spark GB10)
with vLLM up beside it (fleetq `--quiet gpu,cpu` for every timing run, `--load cpu` for the build; CARGO_BUILD_JOBS=3 MAX_JOBS=3 FLASHINFER_NVCC_THREADS=1, fleetq `--mem 55`,
free-memory watchdog `scripts/postchain/wd.sh` on every build and serve). Raw output: `docs/results/2026-11-18-postchain/raw/`; scripts: `scripts/postchain/`.
Nothing default-on.

(Result section: filled in at the end.)

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
  count, before and after the first chain (`scripts/postchain/drafts_id.py`: 0 differing steps of 13 077). So the same work runs slower; the head's state is not
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

## Instructions received

None beyond the brief (lane rules `/opt/code/intarsiaml/.worktrees/scan/lane-rules.md`, including the 21:40 spark1 build caps and free-memory watchdog).
