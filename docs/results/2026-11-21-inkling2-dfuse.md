# Inkling-2-Mini decode controller fusion on Metal (lane ink-dfuse, 2026-11-21)

Status: in progress. Changes 1-5 are committed, default off, and bitwise on the real model. G2 / G3 pass for 1-3 (#10952). Per-switch timing (#10964): best = route + msg + hoist + norm_c is -0.67 ms per step. The tail is removed (ac2371c81). ink/metal 0ee1bec29 is merged (ae617da2f; the hoist is row-table aware). Job E (#11003, snapshot ac2371c81 = .scratch/snap3) runs the final gates and the timing on that code before any default changes. Branch `ink/dfuse` (from `ink/metal` a88db4620), worktree `.worktrees/ink-dfuse`. Not pushed.

## Briefs received

- 2026-11-21 (coordinator, initial brief): cut the GPU time and dispatch count of the per-token controller ops in Metal decode
  (ConvMix, AttnResMix, LoopBlend, router GEMV, MoeRoute). Router fusion approved. New fused ops take discriminants from 76 up,
  message main with number and contract before building. CPU reference composing the existing references, Metal admission cases
  at T = 1, 4, 16. Decode only (T <= 16). Row-local or row-table form for anything absorbing ops 69/70/71/74/75; message main
  before fusing op 74 or any stateful op. Stay out of the backend encoder (ink/hostgap). Rows-exact and the state tests bitwise.
  G1-G3 on Metal before reporting. No default flips for other models; `plan show` for other models unchanged. Original targets:
  <= 350 dispatches per token, controllers <= 1 ms per token.
- 2026-11-21 (coordinator, retarget from Fable's measured review, `.worktrees/ink-fable/decode-profile-review.md` 0.2, 1 item 4,
  2, 3.2):
  - The real decode step is 15.76 ms (the 19.8 ms included the `generate` step profiler, `--profile-every 16`). Controllers are
    3.19 ms per token, 20 % of the step: 46 blended ConvMix x 38 us (4 dispatches each) 1.76 ms; 50 plain ConvMix x 15.6 us
    0.78 ms; 48 MoeRoute x 10.4 us 0.50 ms; 26 AttnResMix 0.25 ms; 48 router GEMV 0.20 ms.
  - Kernel-mode shares inflate small kernels 4-7x: do not quote them; use cb-trace or real-step ms.
  - New target: controllers 2.0-2.3 ms per token (stretch 1.7). Steer by controller ms, not dispatch count (413-461 dispatches
    is realistic; <= 350 is not required).
  - Plan (item 4): (1) ConvMix + router GEMV + MoeRoute in one launch on the 24 plain visits; (2) router + route in one launch on
    the looped visits; (3) fold the message mix into the blend's row kernel; (4) blend from 4 dispatches to 3 (to 2 with the lazy
    norm of 3.2: a contract change, message the coordinator first).
  - Op discriminants 76-79 are this lane's.
  - Measure decode with `--profile-every 0` (the default 16 adds about 4 ms per token of fake gap).
- 2026-11-21 (this lane to the coordinator): contracts of op 76 `InkRoute` (router GEMV + MoeRoute) and op 77 `ConvMixRoute`
  (ConvMix then InkRoute on its single mix plane), 78-79 reserved; messages precomputed per pass (one AttnResMix with Q = 48 at
  the loop restart) instead of folded into the row kernel; blend 4 -> 3 -> 2 inside op 74's Metal kernel with a ticketed
  in-dispatch hand-off (no contract change).
- 2026-11-21 (coordinator, go-ahead): approved: 76 InkRoute and 77 ConvMixRoute as specified, 78-79 reserved, the message
  precompute, and the ticketed blend. Conditions:
  1. Order: messages -> 76 -> blend -> 77 (the blend is the biggest term; 77 touches op 74 on the plain visits where ink/batch is
     adding the row-table form, so last means less merge churn). Commit each step separately with G1-G3 passing, so ink-metal can
     merge partial gains.
  2. Message precompute: show in this doc that nothing writes the bank during a pass (spec section or code). It must be row-local
     for the batched step (rows from different sequences). G1(b) teacher-forced and the tiny state tests stay bitwise.
  3. Ticket hand-off and the 76 last-threadgroup top-k: each call site resets its own counter in-kernel (the last threadgroup), no
     host memset per step; no counter shared between two steps in flight (hostgap may overlap encode with execution); a stress
     test of >= 10k calls across all shapes and row counts, with a hang watchdog, while another GPU job loads the device; report
     any timeout.
  4. The fallback when the metal4 library is missing stays tested: the graph keeps Logits + MoeRoute.
  5. For MTP verify (T = 2-3) and batch (T <= 16), measure 76 / 77 at T = 1, 2, 3, 4, 16, not just T = 1.
  6. Plain ConvMix at 15.6 us in-stream vs 8.5 us in the bench: ink-d1row chases in-stream vs bench gaps too (attention, experts);
     the cause may be shared (cache, residency). Put the finding here when there is one.
- 2026-11-21 (coordinator, queue tip): endor's quiet line is 14+ jobs (about 9 h). Correctness work goes in jobs with the default
  gpu load and a short honest --expect; only timing runs take `--quiet gpu,cpu`; CPU-only work goes to spark1 / spark2. Merge
  ink/metal (a0751f302 fixes the rows_exact pin-list test). Done: merged at a0751f302; cpu tests on spark2.
- 2026-11-21 (coordinator, after the status message):
  1. Skip op 77 for now (about 0.1 ms per token is not worth a shared-header cut across files ink/pack12 and ink/batch are
     editing); revisit after both merge into ink/metal. Keep 76-79 reserved (the pack12 placeholder has them).
  2. The gate of each of the three steps: fused == unfused bit for bit on the real model at 1 row and at 16 rows (ink/batch runs
     16-stream passes through these paths), plus G2 / G3 with everything on. Name the exact comparison in #10880's log lines.
  3. The ticketed blend tail is fine as described. Keep the watchdog stress in #10879 and report its max wait.
  4. Fold the build-only job into #10879. Commit each step separately once its gate passes; defaults stay off until #10882's
     timing shows a gain. Done: all three switches default off (`=1` turns each on).
- 2026-11-21 ~23:40 (this lane to the coordinator): op 74's row kernels are latency chains of cold loads at T <= 16 (one to
  three threadgroups per call): (1) `ink_mix_row` with every operand load issued before the first use (compile-time K <= 4,
  first query batch, norm and msg rows), (2) `ink_blend_norm` one channel per thread, (3) maybe a system-cache prefetch of the
  blend weights from the row dispatch. All bitwise, no contract change. #10952 / #10964 pinned to a snapshot at f4702646a.
- 2026-11-21 ~23:45 (coordinator, go-ahead):
  1. Base: build on ink/metal; leave ink/batch's posb[t] / ring-pointer / xoff setup lines alone so the merge stays local
     (ink/batch will not land in ink/metal soon).
  2. Order: (1) and (2) first, each behind its own default-off switch. (3) only if (1)+(2) leave measurable latency in the
     h GEMV / gate.
  3. Dropping the blend tail after #10964: agreed.
  4. No endor `--load cpu` window (24 jobs behind #10890 until about Sunday morning). Compile checks on bespin:
     `fleetq run --host bespin --load cpu --mem 6 --expect <honest>`; cargo check of the Metal crate, and the shader through
     MTLDevice runtime compile if possible. No GPU timing and no real model on bespin (unanimis services, ~14 GB spare).
  5. Endor: ONE combined job for (1)+(2): build, bitwise tests (tests/inkmix.rs, rows-exact), admission, then the per-switch
     normal-step timing, all under `--quiet gpu,cpu` with an honest --expect.
- 2026-11-21 ~23:55 (coordinator, rule for all Inkling lanes): this session and every worktree run on endor, so no local
  cargo build / check / test or shader compile on endor outside a fleetq slot (they skew the quiet timing jobs). macOS compile
  checks on bespin (`fleetq run --host bespin --load cpu --mem 6 --expect <min>`, a git archive in ~/intarsiaml-<lane>, the
  toolchain in ~/.cargo/bin); CPU tests on spark2; GPU work in endor fleetq jobs. Reading files, git and editing locally are
  fine. This lane: the compile checks of changes (1) and (2) ran on bespin (#10992, #10995, #10997: the shader through
  MTLDevice runtime compile, `cargo check --release -p intarsia-backend-metal --examples --tests`).
- 2026-11-22 ~00:30 (coordinator, after the #10952 gate): the gate of changes 1-3 is accepted. Flag: under load the blend
  tail's slowest batch was 205 ms per 200 calls (~1 ms per call) against 15.4 ms for ink_route. A ticketed in-dispatch
  hand-off could be spinning on threadgroups that are not co-resident, which risks a stall or hang under real contention
  (serve with batching, another process on the GPU).
  1. In #10964 also report the tail's unloaded per-call time and its p99 / max.
  2. Answer in this doc: is the hand-off bounded under contention, and does it need every threadgroup co-resident?
  3. The tail can be default-on only if its worst case is bounded and #10964 shows a real decode gain. Otherwise it stays
     opt-in or is dropped. Route and msg are judged on their own timing.
- 2026-11-22 ~01:25 (coordinator): #10882 (`.scratch/jobC.sh`) left an orphan `intarsia serve --port 8099 --max-batch 1`
  (pids 53944 + child 53955, ~20 GB resident) running from 23:07 outside any slot. ink-batch's #10946 probe hit it. The
  coordinator killed it. (1) Fix the teardown in every job script: `trap 'kill $SRV_PID; wait' EXIT` on the serve parent, a
  free port per job, and a check after readiness that the listener is our own pid. (2) Flag this lane's quiet results from
  23:07 until the kill, say whether any conclusion depends on them, and note that #10964's first minutes overlapped. Done:
  see "The orphan server" under Results; `.scratch/serve_lib.sh` is the fixed teardown, used by jobC.sh and job E.
- 2026-11-22 ~01:35 (coordinator): teardown fix and flags accepted; dropping the tail agreed. Rule for the committed
  switches, as for the other lossless lanes: (1) default-on only if bitwise (shown) AND the real-step timing gain is larger
  than the interleaved rep spread; (2) a microbench gain alone is not enough (#10882: fewer dispatches, no decode gain);
  (3) everything else stays opt-in or is dropped; (4) send the per-switch real-step table and the hash to the coordinator
  before ink-metal merges anything.
- 2026-11-22 ~01:45 (coordinator, go-ahead for job E): best = route + msg + hoist + norm_c is the candidate; dropping the
  tail is right. Rule for E (#11003): best goes default-on, as one commit with per-switch env offs, if E shows (a) best beats
  every off rep by more than the off spread at short AND 5k, and (b) G1-G3 pass with best on and greedy md5 equals off. A
  switch whose leave-one-out costs nothing beyond the spread stays opt-in (minimal default set). Send E's table and the hash;
  ink-metal then merges ink/dfuse. Fusion should stack with P12 (controllers touch no dense weights); the combined gate on
  ink-metal confirms it.
- 2026-11-22 ~02:10 (coordinator): E (#11003) keeps its place in the queue (fleetq has no reorder; --force needs the user).
  The coordinator relays the hoist row-table test (`INTARSIA_INK_MIX_HOIST=1`) to ink-batch and routes the no-metal4
  `ink_in_proj` NoKernel at m 31-63 to ink-prefill. The missed dispatch and controller targets are accepted as stated: the
  evidence beats the target. The ink_route cold-load lever goes in this doc as the next item; it is not built until E is in.
- 2026-11-22 ~02:20 (coordinator, go-ahead): build ink_route candidate 1 now, without waiting for job E.
  1. Keep job E (#11003) exactly as finalized: do not touch the tree or hash it runs (.scratch/snap3, ac2371c81).
  2. Build on a new branch ink/dfuse-route off 05d46d30b, in a fresh git worktree at
     /opt/code/intarsiaml/.worktrees/ink-dfuse/.scratch/route with its own target dir. Use only that path.
  3. Candidate 1 only: load each weight row and x row before the first dot product, with the same reduction order, behind a
     new switch INTARSIA_INK_ROUTE_PRELOAD (default off). It must be bitwise with off at 1 and 16 rows.
  4. Compile check and the Inkling model tests on bespin (--load cpu). Then queue one timing job on endor after E: off vs on,
     short prompt and 5k, interleaved, `--quiet gpu,cpu`, honest --expect.
  5. Candidates 2 and 3 (top-8 tail, cross-layer prefetch) stay unbuilt. Report candidate 1's numbers first.
  Report: the bitwise result, us per ink_route call (real stream vs warm), ms per step, and the branch hash.
- 2026-11-22 ~02:30 (coordinator): hold the flip. Rule (a) is a clean pass: -4.0 % short / -3.8 % 5k per step, all four
  switches needed, localeval +4.1 % / +4.6 %.
  - Decode-mode G2 has gone to ink-metal as a probable regression on the ink/metal tip. ink-metal will gate decode-mode rows
    from now on, bisect f55bfe8b7 / 331238a4e (m-mix) / 45887198b (batch), and compare the two outlier rows against the fp32
    oracle.
  - Job R's (#11111) G2 decode-mode check (off vs best vs best + preload, one binary) is the switch-side proof. If off ==
    best in decode mode there, rule (b) clears for these switches whatever the base turns out to be. Then make the flip
    commit on ac2371c81: one commit, with per-switch env offs. It merges into ink/metal only after ink-metal's decode G2 is
    fixed or explained.
  - Send R's G2 numbers to ink-metal (af43898d63e5b5190) too, as a data point for their bisect.
- 2026-11-22 ~03:03 (coordinator): keep the widened F (#11145) and R (#11111) where they are in line; do not requeue or
  trim them. Requeueing costs about 10 h, and the full-default Metal suite is worth about 25 min more before the flip
  merges. One guard: a job that starts as a backfill gets SIGTERM at its old --expect, so both scripts must write each
  step's result to the log as it finishes. Next time a job is widened, put the new --expect in a fresh job, unless that job
  is deep in a queue like this one. Done 03:04: every step of R and F prints its result as it finishes, and a TERM trap
  names the step that was cut.
- 2026-11-22 ~03:12 (coordinator, FYI):
  - batch's #11104 ran this lane's row-table tests (ac2371c81) on Metal with `INTARSIA_INK_MIX_HOIST=1` and `=0`. Every
    group passed both ways (6 / 6, 3 / 3, 8 / 8), so the batched path is clear for the hoist.
  - The bisect cleared the flip's accuracy question: the decode-mode G2 excess comes from m-mix NE 4 (25ff7864a), not from
    dfuse, and ink-metal is fixing it.
  - The flip 2f952f697 now waits only on that fix and on F (#11145).
- 2026-11-22 ~03:19 (coordinator): ink-metal's NE fix gate #11179 holds every merge, and the coordinator is clearing the
  endor line ahead of it. F (#11145) stays in place, because it gates the flip merge. R is not merge-blocking, and F does not
  need R's output (F's comparison with R's all-off self-goldens is optional and is skipped when the files are missing).
  So #11111 was released at 03:19 and the same script was requeued at the back as **#11190**, with the same flags and the
  same log (`.scratch/route/.scratch/jobR.out`) and an honest `--expect 35`.
- 2026-11-22 ~05:00 (coordinator): #11190 never ran. fleetq hit a PermissionError at 04:25 because jobR.sh was mode 644. The
  client died with the slot held and blocked the endor line for 35 min, until the coordinator released #11190 at ~05:01.
  Requeue R as `sh <path>/jobR.sh` with the same flags, and check that it really runs. Done 05:01: jobR.sh is now
  executable, R is requeued as **#11258** with the command `sh .../jobR.sh` and the same flags, and fleetq estimates a start
  around 11:00.
- 2026-11-22 ~06:00 (coordinator): drop the preload and leave ink/dfuse-route unmerged; skip candidates 2 and 3 (their
  combined ceiling is about 0.18 ms per token, roughly 1.2 % of the 15 ms step, and 3 needs a contract change). Next:
  break the ~3.2 ms per token of per-dispatch controllers down by op (ms per token, dispatches, share) from the existing
  trace without a new job, and propose the single next target with a ceiling of at least 0.3 ms per token, with its bitwise
  plan. Do not build until the coordinator answers. Before any further decode-mode G2 comparison, move onto the ink/metal
  tip that has the NE fix (done 06:05: merged ink/metal 1c1ee202c as 10cf8988d, which includes 7eca1b012; a merge, not
  a rebase, because the branch has merge commits and the brief says merge). bespin #11298 on 10cf8988d: cargo check of
  the Metal crate (examples + tests) and every models test target; cpu Inkling tests 9 / 9 and 6 / 6; cpu ink_route 1 / 1;
  docs_env_vars 2 / 2. No Metal run of 10cf8988d exists yet (flip + NE fix + pack12 + gumbel together); the next endor
  job will be the first.
- 2026-11-22 ~06:15 (coordinator, go-ahead): run step 4's probe on 10cf8988d, measuring only; the merge is fine, no rebase
  needed. Run order inside the slot:
  1. re-trace the controllers on the tip (NE 3 now);
  2. the row kernel alone with the pair stage at S = 1 and S = 3, with and without `--evict`;
  3. the stop rule: if the ceiling over all 96 calls is under 0.3 ms per token, stop and report;
  4. only if it clears 0.3 ms, prototype the split in the evict bench; the go bar is 0.3 ms saved, measured.

  No shipped-kernel or metal4 header changes until the coordinator answers, because the header cut affects op 77's path
  too. Heads-up: ink-batch is chasing a possible B16 batched regression on the merged tree (about 75 ms against 60), with
  the dfuse merges among the suspects. If ink-batch asks which switch disables which controller change at B > 1, answer
  that first.
- 2026-11-22 ~06:45 (coordinator): probe P acknowledged; stopping on the 0.085 ms ceiling was correct. Main removed
  .scratch/snap6 (afaf4e8c9 is on ink/dfuse). **Next item: a measured roofline for Metal decode at the tip.** The step is
  13.3 ms (74.8 tok/s), against a bandwidth floor of about 8.9 ms (5.49 GB at 614 GB/s). Find where the ~4.4 ms goes.
  1. Measure only, with multi-dispatch timing, not per-dispatch command buffers (those overstate small kernels). Break
     the step into classes: attention projections, attention core, MoE experts (gate+up, down), router, unembed,
     controllers (row kernel, blend gate, h GEMV, norm_c, AttnResMix, LoopBlend), everything else. For each class: ms per
     token, bytes read, achieved GB/s, and the gap to 614 GB/s in ms.
  2. Also measure the pure dispatch cost: 543 dispatches per step, at the cost per dispatch inside one command buffer.
  3. Output: the top 3 gaps in ms per token, each with a measured or bounded ceiling for its fix. Propose building only an
     item whose ceiling is at least 0.3 ms per token, the same rule as P. Do not touch shipped kernels.
  4. Run it in a fleetq slot with --quiet gpu,cpu and an honest --expect. Check the box first: if the tip's default step is
     not about 13.3 ms (within 5 %), report BOX SLOW and stop (endor ran about 50 % slow from 06:27 to 06:35, cause
     unknown).
  5. Coordinate with ink-fp8, which owns reduced-precision weights: count bytes at native bf16, and note what fp8 would
     change.
- 2026-11-22 ~07:10 (coordinator, after job L): no dfuse build, agreed. Main removed .scratch/snap7. No m-moe or Metal
  attention lane is active now, so this lane takes the next item:
  1. **NVFP4 experts at 1 row** (gap 1.03 ms, inferred 0.65). Measure first:
     - why 1 row reaches 438 GB/s against 535 at 4 rows: threadgroup count against the core count, occupancy, per-call
       latency;
     - microbench candidate configs at 1 row (more threadgroups by a split over N or K, tile shape) for a measured 1-row
       ceiling.

     Build only if the measured gain is at least 0.3 ms per token. Avoid the persistent gate+up -> down kernel unless
     forward progress can be shown safe on Metal.
  2. If 1 does not clear the bar: **the attention core** (1.07 ms gap, 23-24 us per call, flat in context). Find what the
     fixed per-call latency is made of, then the same rule.
  3. Then **the 0.98 ms of transitions**: check whether the decode encoder uses concurrent dispatch with explicit
     barriers. If it does not, bound what overlap of independent kernels could save.

  Gates for any build: rows-exact or tolerance-justified admission, G2 / G3, greedy md5 at 64 / 256 tokens unchanged, and
  3 alternating reps against the tip. Each build gets its own commit with an env revert, and stays opt-in until the gates
  pass. Box check first, as in L. Jobs use fleetq --quiet gpu,cpu with an honest --expect; the job script runs its body in
  its own process group and kills the tree on TERM.
- 2026-11-22 ~07:55 (coordinator, go-ahead): **build approved**, opt-in and T = 1 only (gate+up with one pair per union
  leader + fused down with 3 chunks ahead), with exactly the gates and decision rule below.
  - **Speed rule (all endor lanes):** BOX SLOW is a label, not a stop. Run the arms ABAB in one slot after a short warm-up,
    take the per-pair median, and quote the warm steady-state gain as primary. N2's slots were short (about 2 min) and
    probably cold, so the gate run is the one that counts against the 0.3 ms bar.
  - Stay out of ink-prefill's NAX code in the shared launcher. If a merge conflict in that function turns out
    unavoidable, tell the coordinator before resolving it.
  - **Decision rule, fixed before any timing:** in-stream ABAB in one slot, per-pair median gain of at least 0.3 ms per
    token in the warm steady state; otherwise the change stays opt-in or is dropped.
  - **Gates:**
    1. greedy md5 at 64 and 256 tokens byte-identical (any difference is a bug: stop);
    2. G2 / G3 self-goldens byte-equal to the tip's;
    3. admission at T = 1, 4, 16;
    4. the Metal tests inkmoe, inkling2_batch and G1 tiny, plus rows-exact checked across the paths (T = 1 on the new
       kernels against T > 1 on the shipped ones);
    5. `plan show` for Qwen3-8B and Llama-3.1-8B unchanged.

    The change gets its own commit, an env revert and stays default off. A default-on proposal would report tok/s through
    localeval.
- 2026-11-22 ~08:10 (coordinator, FYI): ink-batch has the in_proj rows path for B > 1 (T >= 2 decode, below the prefill
  NAX threshold); their B16 profile has it at about 6x its bandwidth floor.
  - The T = 1 in_proj kernel and its dispatch stay with this lane, and ink-batch will not touch them. Their new kernel
    must match the T = 1 output bitwise per row (rows-exact).
  - If either lane needs to change the shared dispatch function, tell the coordinator first, so the merges can be
    sequenced.
  - Lane note: at T = 1 the stream takes ink-pack12's `ink_in_proj_p12_gemv` (crates/intarsia-backend-metal/src/p12.rs,
    P12 default on for in_proj / wo up to 4 rows), whose bf16 twin is `ink_in_proj_bf16_gemv` (inkmix.rs
    `launch_in_proj_gemv` / `in_proj_rows`). Both are bitwise-equal by pack12's contract. That T = 1 output is the
    reference ink-batch's rows kernel must match. This lane has no pending change to either.

## Probes (endor, before any code)

`.scratch/tgbw.swift`, `.scratch/chain.swift` (fleetq #10793, #10813):

- An empty dependent dispatch in one serial encoder costs 0.84 us; a dependent stage of 1 / 8 / 24 small threadgroups costs
  3.70 / 1.84 / 3.53 us as its own dispatch and 1.68 / 1.60 / 3.65 us as a ticketed stage inside one dispatch (0 mismatches).
- One threadgroup streams about 64 GB/s, so a router (590 KB) or blend weights (1.2-3.5 MB) cannot be read by one threadgroup
  per row: the fused kernels keep the GEMVs' grids and hand results across threadgroups instead.

## Changes

### 1. Loop messages per pass (graph only; `INTARSIA_INK_MSG_PRE`)

A looped visit's messages are `attn_res(bank[:1], bank[1], MQ[layer])` (spec a12). The bank is written in exactly two
places, both at a visit boundary (`crates/intarsia-core/src/model/inkling2.rs`, `ink_pass_rows`): the restart visit's copy
(`a12: bank = [the block after the prelude's, the partial]`) and the teacher-forced upload of every visit. So between two
restarts of a free-running pass the bank is constant, and every looped visit's messages can be computed at the restart.
`InkModel::msg_q_all` stacks the 12 core layers' 4 message queries into `[48, D]`; at the restart one `AttnResMix` with Q = 48
writes all 48 f32 planes (`ws.msg` grows to 48 planes of up to 16 rows, 7 MB). The row kernel computes queries in batches of 4
with code that does not depend on Q beyond the batch, so each plane is bitwise the 4-query call's. The next-visit lookahead
(the MLP ConvMix that computes the next visit's attention input) only runs when the next visit is not a restart, so it always
reads this pass's planes. Teacher-forced passes (bank uploaded per visit) and passes of more than 16 rows keep the per-visit
call. Row-local: a row's messages read only that row's bank, so rows of different sequences in a batched step are independent.
Decode: 24 -> 2 message launches per token.

### 2. `InkRoute` (op 76): router GEMV + MoeRoute (`INTARSIA_INK_ROUTE_FUSED`)

Contract in `crates/intarsia-core/src/inkling2/ops.rs`: `MoeRoute(Logits(x, w), bias)` with the logits as an optional output.
CPU: `ink_ref::route` runs the kernel set's own `Logits` then `dsv41_ref`'s `MoeRoute` (bitwise by construction;
`crates/intarsia-backend-cpu/tests/ink_route.rs` checks both kernel sets on every sweep case). Metal `ink_route_bf16`
(`kernels/metal/inkling2_ctrl.metal`, metal4.0 library for the device-scope fences): grid (E / 4, T) threadgroups of 4
simdgroups, each logit exactly as `gemv_f32o_v` / `gemv_f32o_scalar` (the `logits_bf16_gemv` form choice), the last threadgroup
of a row (a device counter) routes it with `ink_moe_route`'s code. Counters come from a per-backend arena (`sync.rs`, 64k u32,
zeroed once); each launch takes fresh ones round the arena and the last threadgroup resets them, so there is no host memset
and two steps in flight never share a counter. Graph: fused on Metal for calls of up to 16 rows when the backend admits it;
without the metal4.0 library the kernel is not registered and the router matmul + `MoeRoute` run. Sweep cases: T 1, 2, 3, 4,
7, 16, 40; D 64 / 60 / 2304; E 16 / 40 / 128; every score function, renorm on / off, temperature, with and without logits.
Decode: 96 -> 48 launches per token.

### 3. The blend tail: h GEMV + gate + norm in one dispatch (`INTARSIA_INK_BLEND_TAIL`, no contract change)

`ink_blend_tail_g*` (`inkling2_ctrl.metal`), launched from `blend_rows` (`inkmix.rs`) for blend chunks of up to 16 rows.
Threadgroups take a ticket at start; tickets `< S * G` are h items (one weight row of `down` for every row, the in-projection
GEMV's per-row code with SG 8, U 2), the rest are gate items (256 channels of a stream: the channel's gain / forget rows are
loaded first, then thread 0 waits until the stream's h counter reaches G; every h item has a lower ticket, so it has started
and finishes without waiting, so there is no deadlock without a forward-progress guarantee). The last gate item of a stream (a
second counter) runs the stream's norm. Every value is computed by the same code as the three dispatches it replaces
(`ink_in_proj_gemv` per row, `ink_blend_gate`, `ink_blend_norm`). Blended ConvMix 4 -> 2 dispatches, LoopBlend 4 -> 2.
Only with the default in-projection family (`INTARSIA_INK_PROJ_CFG` unset).

### 4. Load order in the row kernels (`INTARSIA_INK_MIX_HOIST`, no contract change)

Why: in the stream the ConvMix row kernel costs 16.6 us per call (96 per token, 1.6 ms, half the controller time) against 8.5
us in the warm microbench. At T <= 16 a row is one threadgroup (576 threads at D 2304 since ink/metal's NE 4 change; 768 before), so nothing hides its latency, and its
cost is the chain of dependent cold loads (ink-d1row's cold-cache finding: ~200 MB of weights pass between two visits). The
chain in `ink_mix_row`: the conv loops over a runtime K (4), and each tap's taps row and x / ring row are loaded only after the
previous fma (4 rounds); the queries are loaded after the conv's stores (the compiler cannot move loads above possibly aliasing
stores: +1); the norm rows and the blend's msg rows after the reductions (+1 to 2); the simdgroup partials are summed by a
serial loop of threadgroup loads.

`ink_mix_row_h_*` (template flag HO, `kernels/metal/inkling2_mix.metal`): for K <= 4 every tap's taps row, x row and ring row
are loaded (clamped addresses) before the first fma, and only the ring rows wait on pos; whether a tap reads x or the ring
depends on the row's index in the call, not on pos. The first query batch, its norm rows and the msg rows are loaded at the
kernel start. The partial sums read eight values at a time, with the adds in the same order. Same values and the same fma and
add order: bitwise the plain instance (`tests/inkload.rs`: SconvRes, ConvMix with and without the blend, AttnResMix; widths of
every NE, K 1..5, rows 1..16, the model's ring and a short one, host and device pos, one and two query batches, Q 48). K > 4
keeps the loop. The switch falls back to the plain instance when a pipeline admits fewer threads than the row needs. The
row-table form (ink/batch, merged with ink/metal 0ee1bec29): a row of a row-table call has in-call index 0 and its x row at
`xoff` = t, so the HO conv takes the x / ring choice from the in-call index `dx` (t, or 0 with a row table) and adds `xoff`,
as the plain loop does through `qq - pos + xoff`. The batched-decode tests (`tests/inkling2_batch.rs` on Metal, batch ==
solo bitwise) run with the switch on in job E.

### 5. The blend norm, one channel per thread (`INTARSIA_INK_BLEND_NORM_C`, no contract change)

`ink_blend_norm` runs S x rows threadgroups of 256 threads. At T = 1 that is 3 threadgroups, and each thread loads its 9 y /
norm values one loop iteration at a time (9 serial cold rounds; 9.3 us per call in the stream, 46 per token).
`ink_blend_norm_c` runs S x rows x ceil(D / 256) threadgroups, one channel per thread. Each thread loads y and norm first, then
sums the partials in the same order (eight loads in flight). It is bitwise `ink_blend_norm` (`tests/inkload.rs`: LoopBlend
and the ConvMix blend, D 200..2304, G 8..128, rows 1..40, with and without HO).

### Not done: op 77 (ConvMix + router)

Fusing the router into the attention-side ConvMix needs the ConvMix row kernel's code in the metal4.0 file (a header cut out of
`inkling2_mix.metal`), which ink/pack12 and ink/batch are both editing. Estimated gain about -24 dispatches and 0.1 ms per
token; left for after those lanes land.

## Results

### Correctness (fleetq #10879, raw/correctness_A2.txt)

- Whole Metal crate (`cargo test -p intarsia-backend-metal --no-fail-fast`, stress skipped) passes except this lane's route test,
  whose helper sized output buffers at 16 bytes (fixed in 86c7d8def; the fixed test passes in #10882's fail-fast).
- The blend tail is bitwise the three dispatches (LoopBlend and ConvMix with a blend, D 200 / 256 / 320 / 2304, R 1..16,
  S 1 / 3), also with `INTARSIA_INK_BLEND_CS=32` and `1024`; `tests/inkmix.rs` passes with the tail on.
- Stress, 12000 calls each, every result bitwise its single call: ink_route slowest batch of 200 4.1 ms, blend tail 193 ms (the
  batch time includes the per-call host allocations). The concurrent bandwidth load overlapped only about 1 s, so job B loops
  the stress through the G2 / G3 model runs.
- Admission (`intarsia kernels test --backend metal`): ink_route 124 cases, max_abs 1.98e-7, ADMITTED; conv_mix 532 and
  loop_blend 27 cases with the tail on, ADMITTED; attn_res_mix, moe_route ADMITTED.
- G1 (tiny model on Metal): pass with the switches off and all on; tiny on cpu: pass off and on. With
  `INTARSIA_METAL_NO_METAL4=1` the tiny model does not load on this branch or on ink/metal (InkInProj prefill has no metal3.1
  kernel); `tests/inkctrl_nometal4.rs` covers the fallback of the new kernels.
- `plan show --backend metal` for Qwen3-8B and Llama-3.1-8B-Instruct: byte-identical stdout and stderr against the ink/metal
  f55bfe8b7 binary.

### Timing, all three switches off vs on (fleetq #10882, `--quiet gpu,cpu`, raw/timing_C_summary.txt)

| | off (= ink/metal graph) | all on |
|---|---:|---:|
| dispatches per decode step | 613 | 447 |
| decode step, one command buffer (cb trace, 124 steps, median; 2 reps) | 15.44 / 15.49 ms | 15.54 / 15.56 ms |
| `generate` decode, short prompt (129 tokens; 2 reps) | 64.7 / 64.3 tok/s | 64.3 / 64.2 tok/s |
| `generate` decode, 5051-token prompt (128 tokens; 2 reps) | 60.3 / 60.7 tok/s | 60.8 / 61.0 tok/s |
| localeval speed decode, 136 / 5136 prompt tokens | 65.0 / 61.6 tok/s | invalid: the on server never bound its port, so this run measured the off server (orphan, below) |
| controllers, per-dispatch trace (pipelined, one buffer per dispatch) | 3.70 ms | 3.46 ms |
| controllers, profiled steps (one waited buffer per op call, corrected) | 2.80 ms | 3.27 ms |

(The two controller rows disagree because each mode adds its own per-buffer cost; neither is the in-step time, which a single
command buffer does not expose.) **166 fewer dispatches gave no decode gain.** Dependent dispatches inside one command buffer
cost little next to the kernels' own latency, and two of the fused kernels are no faster than what they replace:

In-stream microbench (`examples/ink_ctrl_bench.rs`, raw/ctrl_bench_C.md; us per call, median):

| T | router pair | ink_route (loop select) | LoopBlend S=3, 4 dispatches | S=3 tail | S=1, 4 dispatches | S=1 tail |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.68 | 9.47 | 25.37 | 32.46 | 19.99 | 20.61 |
| 2 | 9.79 | 9.38 | 28.03 | 44.16 | 21.54 | 28.55 |
| 3 | 10.02 | 9.49 | 29.62 | 55.50 | 23.29 | 37.01 |
| 4 | 10.64 | 9.54 | 30.44 | 66.62 | 24.95 | 45.70 |
| 16 | 23.73 | 12.97 | 76.20 | 209.00 | 47.48 | 148.17 |

- The router's cost is the selection (K dependent shuffle rounds on one simdgroup), not the launch: e0ab9b674 replaces it in
  the fused kernel with a rank selection over the whole last threadgroup (bitwise for finite logits; `INTARSIA_INK_ROUTE_RANK`).
- The ticketed blend tail is slower than the three dispatches: each h item reloads its weight row once per row (the GEMV keeps
  it across the rows), and the gate items spin on a counter while the h items run. Recommendation: leave it off (or drop it).
- Message precompute: per-dispatch trace attn_res_mix 229 -> 102 us per token (26 -> 4 calls).

### Real-model gate of changes 1-3 (fleetq #10952, snapshot f4702646a, raw/gates_B_*.txt)

- Bitwise, every switch alone and all three against all off:
  - 1 row: 64 greedy tokens, text md5: EQUAL for route, msg, tail and all.
  - 16 rows: the G2-short prompts prefilled in 16-row chunks, so every pass runs the 16-row paths. Self-golden tensor bytes
    are EQUAL for all four (`raw/stcmp.py`). The file md5s differ only by the header's build time and run metadata.
- The fused kernels ran (kernel traces): 1-row decode 613 -> 447 dispatches per step, 16-row passes 805 -> 591. ink_route_bf16
  replaces 48 logits_bf16_gemv + 48 moe_route_f32. attn_res_mix 26 -> 4 calls. Blended conv_mix 4 -> 2 dispatches.
- G2 / G3 with all three on: PASS.

| gate | KL21 (scored rows) | limit | top-1 | confident rows, misses |
|---|---:|---:|---:|---|
| G2 short, prefill | 2.380e-3 | 4.0e-3 | 98.95 % | 363, 0 (allowance 1) |
| G2 short, decode (teacher-forced) | 2.589e-3 | 4.0e-3 | 97.66 % | |
| G2 long, prefill | 3.418e-3 | 6.0e-3 | 99.17 % | 1206, 1 (allowance 2) |
| G2 long, decode (teacher-forced) | 2.474e-3 | 6.0e-3 | 97.50 % | |
| G3 short (vLLM golden), prefill | 8.391e-3 | 1.05e-2 | 97.89 % | 367, 0 (allowance 1) |
| G3 long (vLLM golden), prefill | 1.248e-2 | 1.65e-2 | 97.64 % | 1211, 0 (allowance 2) |

- Stress under real GPU load (coordinator condition 3): 12 rounds of the 12000-call watchdog stress ran while the G2 / G3
  model runs loaded the GPU. Every round exited 0 and every call was bitwise. The slowest batch of 200 calls took up to 15.4 ms
  for ink_route (watchdog 270 s) and up to 205 ms for the blend tail (watchdog 660 s). No timeout.

### The blend tail's hand-off: bounded, no co-residency needed (coordinator question, 2026-11-22)

- **Bounded, and no threadgroup has to be co-resident.** A threadgroup's role comes from a ticket it takes from an atomic
  when it starts (`inkling2_ctrl.metal`, `ink_blend_tail`), not from its grid index. Tickets below S x G are h items, which
  never wait. A gate item holds a ticket of at least S x G, and it can only have drawn that ticket after all S x G h items
  drew theirs, so every h item it waits on has already started. The one assumption is that a started threadgroup keeps
  making progress while other threadgroups of the same dispatch run. Metal documents no forward-progress guarantee, but it
  also documents no case where a running threadgroup is parked in favour of its siblings, and a switch to another process
  stops and resumes the whole context. So a gate item spins for at most the remaining time of its stream's h items that are
  already running. It never waits on a threadgroup that has not been scheduled. Threadgroups that have not started
  are not waited on by anyone, so the dispatch needs no forward-progress guarantee and no co-residency. The norm runs in the
  gate item that finishes last (a counter it increments), which waits on nothing. Every counter is reset inside the dispatch
  by its last user, so nothing carries over between calls or between two steps in flight.
- **Cost under contention.** Gate items that are waiting occupy their slots and poll one device atomic. If another process
  holds most of the cores, the h items run slower and the waiting gate items hold their slots longer. That is a slowdown
  bounded by the h items' time, not a stall.
- **The 205 ms per 200 calls is mostly host time, not spinning.** The stress test's batch time is wall time for encoding 200
  calls, each with a freshly allocated and uploaded output buffer (up to 3 x 16 x 2304 bf16), plus the wait. The case mix
  (D 2304 / 200 / 320, rows 1-16, S 1 / 3) is about 7 ms of GPU work per 200 calls at the microbench's per-call times. The
  same stress without load (#10879) had a slowest batch of 193 ms; under load it was 72-205 ms over 12 rounds. ink_route's
  outputs are tiny, which explains its 4-15 ms. #10964 measures the tail's per-call GPU time directly
  (`ink_ctrl_bench --tail-dist 2000`: one command buffer per dispatch, median / p99 / max), unloaded and while a second
  process (`.scratch/gpuload.swift`) keeps every GPU core streaming.

### The orphan server (2026-11-21 23:07 until the coordinator killed it at about 01:20)

**Cause.** `.scratch/jobC.sh` started each server as `run $m $B serve ... &`, where `run` is a shell function. So `$!` was the
function's subshell, and `kill $SP` killed the subshell, not the server supervisor. The off server (port 8099, Inkling
resident, idle) outlived the job. When the on server started it could not bind 8099 (`serve_on.log`: "Address already in
use", exit 78). The readiness probe then answered from the off server.

**Effects on this lane's results:**
- #10882: the localeval row (65.0 / 61.6 vs 65.1 / 61.8) is invalid, because both arms measured the off server; it is marked
  so in the table. The cb traces, `generate` and 5k rows ran before 23:07 and are unaffected. No conclusion of this lane rests
  on the localeval row: "no decode gain" stands on the cb-trace step spans.
- #10952 (00:18-00:22): bitwise equality and the G2 / G3 KL are deterministic outputs, so they are unaffected. The stress
  batch wall times (72-205 ms) ran with the idle server holding ~20 GB. The only conclusion drawn from them is "no hang,
  every call bitwise".
- #10964 (from 01:18): its tests are bitwise checks (unaffected). The microbench (01:19-01:21) overlapped the server's last
  minute or so. Its comparisons are A/B within one process, run back to back, and an idle resident server weighs on both
  arms alike. The per-switch normal-step timing and the per-dispatch trace ran in the job's recovery block from 01:21 on,
  after the kill.

**Fix.** `.scratch/serve_lib.sh` (sourced by jobC.sh and job E):
- `serve_start` runs the server on a free port of its own, through `env` (so `$!` is the supervisor itself).
- After readiness it checks that the port's listener is the supervisor or its child.
- `serve_stop` sends SIGTERM to the supervisor, waits, and kills any listener of ours that is left.
- An EXIT trap stops a server that is still up when the job ends.

### #10964 first pass: the binary lost its Metal feature

`cargo test --release -p intarsia-cli --test docs_env_vars` (step 2) rebuilt `target/release/intarsia` without
`--features metal`. Every model step after it failed at once ("built without the Metal backend"): admission, real-model
bitwise, timing and the per-dispatch trace. A recovery block appended to the running script rebuilt the binary and reran
those steps in the same slot (the first-pass logs are in `.scratch/D/first/`). Job scripts now build the metal binary after
any `cargo test` of intarsia-cli.

### Load-order switches and per-switch attribution (fleetq #10964, snapshot 61e954172 = commits 88b2b089c / f64199417 / fe16c5654, raw/*_D*)

**Correctness:**
- `tests/inkload.rs`: hoist on == off in 1980 cases, norm_c on == off in 147 cases.
- inkctrl and the no-metal4 test pass. `tests/inkmix.rs` rows-exact passes with both switches on.
- Admission with both on: conv_mix 532, sconv_res 64, attn_res_mix 97, loop_blend 27 cases, all ADMITTED.
- The tiny model on Metal with both on (G1) passes, and docs_env_vars passes.
- Real model, off vs hoist / norm_c / both / best (route + msg + hoist + norm_c): EQUAL at 1 row (64 greedy tokens, text
  md5) and at 16 rows (G2-short prompts in 16-row chunks, self-golden tensor bytes).

**Normal-step timing**: cb trace, span median per 256-token decode, 3 interleaved reps, short prompt, `--profile-every 0`.

| config | dispatches / step | reps (ms / step) | median | vs off (median) | vs off's fastest rep (15.514) |
|---|---:|---|---:|---:|---:|
| off (= ink/metal) | 613 | 15.514, 17.185, 15.724 | 15.724 | | |
| route (76) | 565 | 15.501, 15.569, 15.598 | 15.569 | -0.155 | +0.055 |
| msg (per pass) | 591 | 15.389, 15.968, 15.570 | 15.570 | -0.154 | +0.056 |
| tail | 517 | 15.577, 15.890, 15.879 | 15.879 | +0.155 | +0.365 |
| route + msg | 543 | 15.415, 15.592, 15.376 | 15.415 | -0.309 | -0.099 |
| hoist | 613 | 15.469, 15.524, 15.526 | 15.524 | -0.200 | +0.010 |
| norm_c | 613 | 15.388, 15.482, 15.272 | 15.388 | -0.336 | -0.126 |
| hoist + norm_c | 613 | 15.205, 15.435, 15.131 | 15.205 | -0.519 | -0.309 |
| best: route + msg + hoist + norm_c | 543 | 15.058, 15.063, 14.933 | 15.058 | -0.666 | -0.456 |

Off's second rep (17.185) is an outlier; its other two reps differ by 0.21 ms. Only hoist + norm_c and best beat off by more
than that spread in every rep. Every single switch alone stays within about one spread. The tail is slower. Job E (#11003)
repeats this with 6 interleaved reps and leave-one-out configs before any default changes.

**Per-dispatch trace** (one buffer per dispatch, which inflates small kernels; raw/disp_D.txt): controllers go from 3.694 to
2.981 ms per token with best on.

| kernel (in-stream us per call) | off | best |
|---|---:|---:|
| ConvMix row kernel (96 per token) | 16.50 | 13.75 |
| blend norm (46) | 9.29 | 4.38 |
| blend gate (46) | 8.37 | 8.12 |
| blend h GEMV (46) | 5.08 | 5.06 |
| router GEMV + MoeRoute (48) | 4.46 + 10.38 | ink_route 12.17 |
| AttnResMix, us per token | 230 (26 calls) | 90 (4 calls) |

**Microbench** (`ink_ctrl_bench`, 48 rotating operand sets, us per call, T = 1 / 4 / 16; raw/ctrl_bench_D.md):
- Attention-side plain ConvMix: 8.36 / 8.53 / 8.81 off, 5.96 / 6.13 / 6.30 with hoist.
- MLP-side plain ConvMix (Q 3): 11.25 / 11.49 / 12.67 off, 9.99 / 10.22 / 10.80 with hoist.
- Blend S=1 (attention side): 27.81 off, 25.28 hoist, 22.38 norm_c, 20.07 both (T = 1).
- Blend S=3 (MLP side): 35.80 off, 36.54 hoist, 31.37 norm_c, 31.70 both (T = 1). At T = 16: 92.01 off, 74.12 hoist,
  70.92 norm_c.
- Router: ink_route with rank selection 8.60 / 8.82 / 13.06, without it 9.55 / 9.72 / 13.06, the pair 9.65 / 10.65 /
  24.19.

**The blend tail's per-call distribution** (coordinator brief 2026-11-22; `ink_ctrl_bench --tail-dist 2000`, one buffer per
dispatch, us; raw/tail_dist_*_D.md):

| case | unloaded: off median / p99 / max | unloaded: tail median / p99 / max | under gpuload: off median / p99 / max | under gpuload: tail median / p99 / max |
|---|---|---|---|---|
| T=1 S=3 | 31.3 / 118.2 / 124.7 | 43.8 / 58.5 / 71.0 | 156.5 / 608.2 / 920.6 | 107.1 / 543.5 / 766.4 |
| T=1 S=1 | 27.4 / 106.9 / 142.1 | 23.8 / 27.3 / 32.1 | 130.6 / 539.7 / 891.5 | 73.5 / 501.4 / 906.1 |
| T=4 S=3 | 37.3 / 159.9 / 176.8 | 89.3 / 117.5 / 139.8 | 212.9 / 720.7 / 979.9 | 214.6 / 689.8 / 852.4 |
| T=16 S=3 | 92.6 / 359.2 / 399.4 | 234.2 / 340.6 / 384.5 | 257.5 / 983.8 / 1314.5 | 405.3 / 864.3 / 1097.0 |

Under a second process that keeps every core streaming, the tail's maximum stays below the four-dispatch path's. Its worst
case is bounded, which matches the analysis above. But its median is higher at T=1 S=3, T=4 and T=16 (T=1 S=1 is a wash),
and the real step is 0.155 ms slower with it on. Decision (coordinator: agreed): drop the tail.

### After #10964: merge, row-table fix, tail removed, job E

- **Merge of ink/metal 0ee1bec29** (ae617da2f) brings ink/batch's row tables (`conv_mix_bf16_rows`, `MX_ROWTAB`) and the NE 4
  mix geometry: D 2304 now runs NE 4 x 576 threads per row instead of NE 3 x 768.
- **The hoist and row tables.** As committed in 88b2b089c, the HO conv took the x / ring choice and the x address from the
  row index t. That matched the plain loop only for contiguous calls. In a row-table call, row t is a T = 1 call at posb[t]
  with its x row at `xoff` = t, so HO would have read another sequence's x rows. The merge resolution makes HO use the in-call
  index `dx` (t, or 0 with a row table) and add `xoff`, as the plain loop does through `qq - pos + xoff`. This leaves the
  contiguous path unchanged, so the #10964 bits hold for it. Job E runs `tests/inkling2_batch.rs` on Metal (batch == solo
  bitwise, row tables) with best on.
- **The hoisted instances at NE 4** admit 1024 threads (bespin runtime compile, #11099). The model's rows need 576.
- **The tail is removed** (ac2371c81).
- **Checks of the merged tree on bespin:** `cargo check` of the Metal crate (examples + tests) and of the models' inkling2 /
  inkling2_batch tests pass (#11102).
- **No-metal4 tiny failures (#10879) are not this lane's.** With `INTARSIA_METAL_NO_METAL4=1` the three tiny tests fail with
  NoKernel for `ink_in_proj` at prefill shapes (m = 31..63 rows: only the NAX in-projection takes those), not for `ink_route`,
  whose fallback to the pair is shown by `tests/inkctrl_nometal4.rs`. Job E runs the tiny suite without metal4 with the
  switches off and with best on, to show that both fail the same way.
- **Job E** (`.scratch/jobE.sh`, `raw/` after the run):
  1. fail-fast bitwise tests, inkmix rows-exact with best on, G1 (tiny inkling2 on Metal) with best on, batched decode off /
     best, tiny no-metal4 off / best
  2. the metal binary, admission with best on, plan show for Qwen3-8B / Llama against the ink/metal binary
  3. real-model bitwise off vs best (1 and 16 rows)
  4. G2 / G3 with best on
  5. normal-step span of off / best / the four leave-one-outs: short prompt 6 interleaved reps, 5k prompt 4 reps
  6. the per-dispatch trace (controller ms per token), localeval speed 128 / 5120 off vs best (`serve_lib.sh` teardown)

### Next item (named, not built until job E is in): ink_route's cold-load chain

`ink_route_bf16` costs 12.17 us per call in the stream (per-dispatch trace, #10964) against 8.60 us warm in the microbench:
about 3.6 us x 48 calls, roughly 0.17 ms per token above warm, and 0.58 ms per token in all. Its bytes are small (590 KB of
router weight per visit, about 1 us at 560 GB/s), so like the ConvMix row kernel before the hoist, its cost is latency. The
candidates, in the order to try:
1. Issue each simdgroup's whole weight row (9 uint4 per lane at D 2304) and its x row before the first dot, instead of
   chunk by chunk, keeping the fma order (bitwise, as the hoist).
2. Shorten the last threadgroup's routing (sigmoid, top-8 of 128, renorm in one simdgroup after the hand-off fence). It is
   already rank-based (`INTARSIA_INK_ROUTE_RANK`, 8.60 vs 9.55 us warm).
3. A system-cache prefetch of the next visit's router rows from an earlier dispatch. This is a cross-op change, so it needs a
   contract note first.
Measure the same way as the hoist: microbench warm vs rotating sets, the per-dispatch trace, and the normal-step span with
interleaved reps.

### ink_route candidate 1: preload (branch ink/dfuse-route, 120cfe4b9; timed in job R #11258: slower, dropped)

- **Kernel.** `ink_route_p9` / `ink_route_p16` (`INTARSIA_INK_ROUTE_PRELOAD`, default off). Each simdgroup loads its whole
  weight row and the x row before the first dot: 9 uint4 per lane at D 2304, up to 16. The dot8 terms are added in the same
  ascending order as `rt_logit_v`'s block loop and tail (m = 4b + u, then 4 nblk + r), so the logits are bitwise. It covers
  the 16-byte form with D <= 4096; everything else keeps `ink_route`.
- **Checks so far, on bespin:**
  - (#11107) runtime MSL compile: the three pipelines admit 1024 threads.
  - (#11107) `cargo check` of the Metal crate (examples + tests).
  - (#11107) the Inkling model tests on cpu (inkling2, inkling2_batch) with the switches off and on.
  - (#11110) docs_env_vars.
- **Job R** (#11111, queued behind E; `.scratch/route/.scratch/jobR.sh`). off = best, on = best + preload.
  1. The preload bitwise test against the pair: rows 1..16 and 40; D 2304, 64, 3000, 4096, 4104 and the 2-byte form.
  2. The stress test with it on, and admission.
  3. Real-model bitwise at 1 and 16 rows.
  4. us per ink_route call in the stream (per-dispatch trace) and in the microbench.
  5. Normal-step span: short prompt 6 interleaved reps, 5k 4 reps.
- The ink/dfuse docs check of the tail removal passed on spark2 (#11103).
- **120cfe4b9** (on ed14676a6) counts the `ink_route_p*` launches (`route_preload_served()`); the bitwise test asserts the
  count grew, and the first launch prints one stderr line. This guards against the two build hazards (a cloned target dir
  keeping the preload uncompiled, and an empty or failed run compared with itself). Job R now touches every tracked source
  after cloning the target, counts a model-run match only when both runs finished, and reports how many runs the preload
  served (expect on 10 of 10, off 0 of 10). bespin #11163 checked the counter build. Job F got the same touch and run check.
- Queue at 02:51: R (#11111) is 4th behind #11104 on endor; fleetq estimates a start around 12:30 if the jobs ahead end on
  time. F (#11145) follows R. At 03:19 R was released and requeued at the back as #11190 (coordinator, above); F now runs
  first.
- 03:00: R keeps its slot useful if the preload test fails. The failure is recorded, the preload arms are dropped, and the
  off / all-off G2 comparison still runs. Timing is skipped in that case.

### Job E (#11003, 02:01-02:14, snapshot ac2371c81; raw/E_*)

best = route + msg + hoist + norm_c. off = no switch set, which is the ink/metal 0ee1bec29 graph.

**Correctness:**
- Fail-fast passed: inkload, ink_route vs the pair, and the no-metal4 fallback.
- inkmix rows-exact with best on: pass.
- **G1** (tiny inkling2 on Metal, best on): 8 / 8.
- Batched decode on Metal (inkling2_batch: batch == solo, row tables): 6 / 6 off and 6 / 6 with best on, so the row-table
  form also passes with the hoist on.
- Tiny without metal4: the same 3 tests fail off and with best on. All three are NoKernel for `ink_in_proj` (10 prefill
  shapes); routed to ink-prefill.
- Admission with best on: ink_route 124, conv_mix 532, sconv_res 64, attn_res_mix 97, loop_blend 27 cases, all ADMITTED.
- plan show for Qwen3-8B and Llama-3.1-8B-Instruct: byte-identical to the base binary.
- Real model, best vs off: EQUAL at 1 row (64 greedy tokens, text md5) and at 16 rows (self-golden tensor bytes).

**G2 / G3 with best on.** Prefill (the gated mode, as in ink-metal's gate table) passes everywhere. G3 passes. G2's
teacher-forced decode rows, which ink-metal reports but does not gate, are above the 4e-3 / 6e-3 floors, so the tool exits 1:

| gate | prefill KL21 (limit) | decode KL21 (teacher-forced) | top-1 prefill | confident rows, misses |
|---|---:|---:|---:|---|
| G2 short | 2.719e-3 (4e-3) PASS | 1.109e-2 (#10952: 2.589e-3) | 98.77 % | 363, 0 (1) |
| G2 long | 3.544e-3 (6e-3) PASS | 9.205e-3 (#10952: 2.474e-3) | 99.04 % | 1206, 1 (2) |
| G3 short | 8.069e-3 (1.05e-2) PASS | | 97.72 % | 367, 0 (1) |
| G3 long | 1.227e-2 (1.65e-2) PASS | | 97.39 % | 1211, 0 (2) |

The decode KL comes from two rows: chat_reasoning (prompt KL 5.8e-2, one row at 0.43 nats) and chat_system (2.4e-2, one row
at 0.30). The cpu reference has a 0.61-nat chat_reasoning row in the same mode (ink-metal doc). The prefill numbers also moved
between #10952 and E (2.380e-3 -> 2.719e-3 short) while best == off bitwise in prefill. That points at the ink/metal
0ee1bec29 merge (NE 4 row geometry, a different reduction order) rather than the switches. But best == off has not been shown
for the teacher-forced decode rows themselves. Job R (#11111) now first runs G2 short / long in both modes with all switches
off, with best, and with best + preload on one binary, and compares their self-goldens. ink/metal's own merge gate #11048
(45887198b, the same code) covers G2 for the tip. **Rule (b) is open until then.**

**Timing**: cb-trace normal-step span, ms per step, 256-token decode, interleaved reps.

| config | disp / step short (5k) | short: reps | median | vs best | 5k: reps | median | vs best |
|---|---:|---|---:|---:|---|---:|---:|
| off | 613 (621) | 15.678, 15.680, 15.738, 15.774, 15.708, 15.709 | 15.709 | +0.623 | 16.633, 16.653, 16.565, 16.593 | 16.613 | +0.625 |
| **best** | 543 (551) | 15.069, 15.113, 15.176, 15.093, 15.041, 15.078 | **15.085** | | 15.999, 15.980, 15.961, 15.995 | **15.988** | |
| best - route | 591 (599) | 15.142 ... 15.214 | 15.191 | +0.105 | 16.139 ... 16.092 | 16.104 | +0.116 |
| best - msg | 565 (573) | 15.296 ... 15.291 | 15.283 | +0.198 | 16.185 ... 16.177 | 16.181 | +0.193 |
| best - hoist | 543 (551) | 15.184 ... 15.261 | 15.256 | +0.171 | 16.156 ... 16.150 | 16.153 | +0.165 |
| best - norm_c | 543 (551) | 15.361 ... 15.444 | 15.376 | +0.290 | 16.244 ... 16.232 | 16.258 | +0.270 |

- **Rule (a): PASS at both lengths.**
  - Short: min(off) - max(best) = 0.502 ms against an off spread of 0.096 ms.
  - 5k: 0.566 ms against 0.088 ms.
  - Step time: -4.0 % short, -3.8 % 5k.
- **Leave-one-out: every switch costs more than the off spread when removed**, so all four belong in the default set:
  - route +0.105 / +0.116 ms
  - msg +0.198 / +0.193 ms
  - hoist +0.171 / +0.165 ms
  - norm_c +0.290 / +0.270 ms

**Per-dispatch trace** (one buffer per dispatch, inflated): controllers 3.897 -> 3.243 ms per token. ink_route is 12.37 us per
call in the stream. ConvMix row kernel 18.25 -> 15.67 us. AttnResMix 238 -> 98 us per token.

**localeval speed** (decode tok/s, median of 3; through `serve_lib.sh`, no orphan: no intarsia listener after the job):

| prompt | off | best | change |
|---|---:|---:|---:|
| 138 / 133 tokens | 64.3 (64.3-64.4) | 67.0 (65.9-67.0) | +4.1 % |
| 5157 / 5158 tokens | 60.2 (60.1-60.4) | 63.0 (62.9-63.1) | +4.6 % |

Result files: /opt/code/deepeval/results/20261122-021312-adhoc-endor-intarsia-metal-dfuse-e-off,
/opt/code/deepeval/results/20261122-021343-adhoc-endor-intarsia-metal-dfuse-e-best.

### The default-on flip (prepared, held for rule (b))

- **Change**: snapshot 6318887b5 = ac2371c81's code + the flip; the patch is .scratch/flip.patch; not yet a branch commit.
  - `INTARSIA_INK_ROUTE_FUSED` and `INTARSIA_INK_MSG_PRE`: unset / `auto` = on for Metal, off elsewhere. `0` / `off` or
    `1` / `on` force either way.
  - `INTARSIA_INK_MIX_HOIST` and `INTARSIA_INK_BLEND_NORM_C` (Metal kernels only): unset = on; `0` / `off` = the plain
    kernels.
  - The cpu default path is unchanged (auto = the pair / per visit), and so are other models.
- **Checks so far:**
  - bespin #11141: cargo check of the Metal crate (examples + tests), and the Inkling model tests on cpu (inkling2 8 / 8,
    inkling2_batch 6 / 6).
  - spark2 #11143: docs_env_vars.
- **Job F** (#11145, queued after R; `.scratch/jobF.sh`) checks the flip on Metal:
  1. Every env setting against job E: no env, all `=1`, all `=0`, each switch `=0`. It compares greedy md5 and
     dispatches per step; no env must equal E's best.
  2. The 16-row self-golden with the defaults against E's best.
  3. G2 in both modes with the defaults.
- The flip becomes one ink/dfuse commit (tree = 6318887b5's) once job R shows all-off == best in G2's decode mode. It merges
  into ink/metal only after ink-metal's decode-mode G2 is fixed or explained (coordinator).
- **F retargeted (03:00) to the tip 00c81dda7** (the flip + ink/metal dbec19036 merged; snapshot .scratch/snap5; the old
  script is jobF.v1.sh). Since the flip, no Metal test suite has run with the new defaults, and the merge adds
  tests/inkling2_mix_decode_cpu.rs. That test launches the mix ops by name, so the defaults choose the hoisted and norm_c
  kernels. F now runs, with the defaults and again with all four switches off:
  - inkload, inkctrl, inkctrl_nometal4 and inkmix;
  - the models' Inkling tests on Metal: inkling2 (G1 tiny), inkling2_batch and inkling2_mix_decode_cpu.

  After the tests it runs the admission of the five ops, `plan show` for Qwen3-8B and Llama-3.1-8B against the base binary,
  the env matrix, and G2 (both modes) and G3 with confident rows. Each self-golden is byte-compared with E's best.
  No test relied on the old defaults: the only test that sets the switches (inkload) sets both arms explicitly.
- bespin #11167: the merge 00c81dda7 passes cargo check (Metal crate examples + tests, every models test target) and the cpu
  Inkling tests (8 / 8, 6 / 6).

### Rule (b): decode-mode G2 is the ink/metal tip's, bit for bit (02:35)

ink/metal's merge gate #11048 ran on 45887198b, the same code as the merged 0ee1bec29 but without any of this lane's code
paths. Its G2 decode-mode numbers equal E's with best on to every printed digit: short 1.109e-2, long 9.205e-3, chat_system
2.376e-2, chat_reasoning 5.808e-2, max row 0.4334. Prefill also matches: 2.719e-3 / 3.544e-3.

The self-goldens of the two runs are byte-identical for G2 short, G2 long, G3 short and G3 long (`raw/stcmp.py`). That
includes the teacher-forced decode rows (`dec_pos_*`, `dec_topk_ids_*`, ... are in the files). So the decode-mode rows above
the floor come from the ink/metal tip, not the switches: with all four switches on, the output is bitwise the tip's. Rule (b)
clears for the switches.

Job R (#11111) still runs its same-binary off / best / best + preload G2 comparison, as one more data point.

**Cause found (coordinator, ~03:12):** ink-metal's bisect puts the decode-mode G2 excess on m-mix NE 4 (25ff7864a), not on
dfuse, and ink-metal is fixing it.

**The flip is committed:** 2f952f697 on ink/dfuse. Its code is ac2371c81 + the flip, with only docs commits in between.
Job F (#11145) checks the new defaults on Metal. It merges into ink/metal after ink-metal's decode-mode G2 is fixed or
explained.

### Job F (#11145, 03:26-03:29, tip 00c81dda7 = the flip + ink/metal dbec19036; raw/F_*)

The defaults on Metal are E's best, bit for bit. The full test suite passes with the defaults and with all four switches off.

| Check | Defaults | All four `=0` |
|---|---|---|
| Metal crate: inkload, inkctrl, inkctrl_nometal4, inkmix | 8 / 8 | 8 / 8 |
| Models on Metal: inkling2 (G1 tiny), inkling2_batch, inkling2_mix_decode_cpu (+ NE 3 child) | 17 / 17 | 17 / 17 |

- **Admission** with the defaults: ink_route (124 cases), conv_mix (532), attn_res_mix (97), sconv_res (64) and loop_blend
  (27) are all ADMITTED.
- **`plan show`** for Qwen3-8B and Llama-3.1-8B: stdout and stderr byte-identical to the base binary (f55bfe8b7).
- **Env matrix** (64 greedy tokens, 1 row). Every arm gives md5 65cde8f35a1802e3e2fadb44d94d080d, which is also E's best and
  E's off.

  | Arm | Dispatches / step | tok/s (engine, one run) |
  |---|---|---|
  | no env (defaults) | 543 | 66.03 |
  | all `=1` | 543 | 66.05 |
  | all `=0` | 613 | 63.61 |
  | route `=0` | 591 | 65.95 |
  | msg `=0` | 565 | 63.92 |
  | hoist `=0` | 543 | 65.72 |
  | norm_c `=0` | 543 | 65.41 |

- **16-row self-golden** with the defaults: byte-equal to E's best.
- **G1-G3** with the defaults. Each self-golden is byte-equal to E's best, so these are E's numbers.

  | Gate | KL21 (limit) | Confident misses (allowance) | Verdict |
  |---|---|---|---|
  | G1 (tiny on Metal) | | | 8 / 8 |
  | G2 short, prefill | 2.719e-3 (4e-3) | 0 / 363 (1) | pass |
  | G2 long, prefill | 3.544e-3 (6e-3) | 1 / 1206 (2) | pass |
  | G2 short, decode mode | 1.109e-2 (4e-3) | | fail: the m-mix NE 4 rows on the ink/metal tip, not dfuse (bisect) |
  | G2 long, decode mode | 9.205e-3 (6e-3) | | fail: same cause |
  | G3 short | 8.069e-3 (1.05e-2) | 0 / 367 (1) | pass |
  | G3 long | 1.227e-2 (1.65e-2) | 0 / 1211 (2) | pass |

  The accuracy tool exits 1 on G2 because the decode-mode rows are above the limit. The prefill rows, which are the gated
  numbers in ink-metal's table, pass.
- **Merge condition.** The flip 2f952f697 (and ink/dfuse up to 00c81dda7) merges after ink-metal's NE fix (7eca1b012,
  gate #11179). The merged tree then needs a decode-mode G2 rerun with the defaults and with `INTARSIA_INK_MIX_HOIST=0`;
  both must pass and give the same bytes.
  - The fix is host-side: `mix_geom` goes back to the fewest NE that fits, NE 3 at D 2304.
  - The plain and the hoisted (`ink_mix_row_h_e*`) instances take NE from that same `mix_geom` (inkmix.rs, the pipeline
    table indexed by NE with a hoist offset), so the one fix should cover both. The rerun checks that.
  - inkling2_mix_decode_cpu passed here and on the faulty base, so it is no evidence for this point.

### Job R (#11258, 05:46-05:52, ink/dfuse-route 120cfe4b9; raw/R_*): the preload is bitwise but slower, so it is dropped

off = best (route + msg + hoist + norm_c); on = best + `INTARSIA_INK_ROUTE_PRELOAD=1`.

- **Correctness.**
  - The Metal tests pass, and so does the stress run with the preload on (4000 calls, all bitwise). With the preload on,
    ink_route is ADMITTED on all 124 cases.
  - With the preload on vs off, the real model's 1-row greedy md5 is EQUAL (65cde8f35a1802e3e2fadb44d94d080d) and the
    16-row self-golden is EQUAL.
  - The preload kernel really ran: in the bitwise runs it served on 1 and off 0, and in the timing runs on 10 of 10 and
    off 0 of 10.
- **G2 on one binary.** All four switches off, best, and best + preload give byte-equal self-goldens in both modes, short
  and long: prefill 2.719e-3 / 3.544e-3 (pass), decode mode 1.109e-2 / 9.205e-3. So the decode-mode excess comes from the
  base (m-mix NE 4), whatever the switches.
- **us per ink_route call**

  | Where | Rows | off | on (preload) |
  |---|---|---|---|
  | in the stream (per-dispatch trace, 48 calls per token) | 1 | 12.33 | 13.58 |
  | microbench, 48 rotating sets | 1 | 8.62 | 9.97 |
  | microbench, 48 rotating sets | 4 | 8.84 | 9.84 |
  | microbench, 48 rotating sets | 16 | 13.12 | 12.11 |

  Per-dispatch controllers: 3.210 ms per token (off) vs 3.303 (on).
- **Normal-step span**, interleaved reps (median ms/step):
  - short, 6 reps: off 15.054, on 15.090, so +0.036 against an off spread of 0.096.
  - 5k, 4 reps: off 16.317, on 16.526, so +0.209. This one is noisy (off spread 0.935).
- **Verdict.** Issuing the whole row's loads up front makes ink_route slower at 1 and 4 rows: +1.3 us per call in the stream
  and in the microbench. It gains only at 16 rows. Decode runs at 1 row, so candidate 1 is dropped.
  - ink/dfuse-route (ed14676a6, 120cfe4b9) stays unmerged, and the switch is not on ink/dfuse.
  - The cold-load gap stays: 12.33 us per call in the stream against 8.62 warm, about 0.18 ms per token.
  - Candidates 2 (a shorter top-8 tail) and 3 (cross-layer prefetch, which needs a contract note) are unbuilt and wait
    for the coordinator.

### Controller breakdown by op (06:05, from the existing trace, no new job)

Source: job E's per-dispatch trace with best on (.scratch/E/disp_best.jsonl, ac2371c81). That is the same code as job R's
off arm and as the defaults on the tip. R's own trace was removed with the route worktree; only its summary is kept
(raw/R_disp_summary.txt, 3.210 ms per token, which agrees to within 1 %). Splitting by call type:
raw/E_disp_best_calltype.txt (raw/calltype.py). The measurement uses one command buffer per launch. Its per-dispatch GPU busy,
16.01 ms, is 0.92 ms above the normal-step span (15.09 ms), so each dispatch is overstated by about 1.7 us on average.

| Controller dispatch | ms / token | Dispatches / token | Median us | Share |
|---|---|---|---|---|
| ConvMix row kernel, calls with the blend (conv + mix + pair) | 0.885 | 46 | 18.37 | 27 % |
| ConvMix row kernel, calls without the blend | 0.727 | 50 | 13.3 to 15.4 | 22 % |
| ink_route (op 76) | 0.593 | 48 | 12.37 | 18 % |
| blend gate (`ink_blend_gate`, ConvMix #2) | 0.400 | 46 | 8.42 | 12 % |
| blend h GEMV (ConvMix #1) | 0.272 | 46 | 5.23 | 8 % |
| blend norm (`ink_blend_norm_c`, ConvMix #3) | 0.210 | 46 | 4.50 | 6 % |
| AttnResMix (loop messages) | 0.098 | 4 | 24.15 | 3 % |
| LoopBlend (pair, h GEMV, gate, norm) | 0.059 | 8 | 4.5 to 9.9 | 2 % |
| **all controllers** | **3.243** | **294** | | 100 % |

The calls without the blend split as follows:
- 24 per token on the attention side (after wo, before the router), at 14.46 us.
- 23 per token on the MLP side (after the experts, before in_proj), at 13.27 / 15.42 us.
- 3 others.

The row kernel is half the controller time (1.61 ms per token, 96 dispatches). ink_route's remaining lever is capped at
about 0.18 ms. The fused blend tail was tried and was slower (change 3).

### Proposed next target: the ConvMix row kernel's cold cost in the stream (not built; waits for the coordinator)

- **Gap.** The same instance costs far less in `ink_ctrl_bench`, where 48 operand sets rotate and stay in the system
  cache (raw/R_bench.md, T = 1, hoist + norm_c), than in the stream:

  | Call | Bench (us) | Stream (us) | Gap (us) |
  |---|---|---|---|
  | attention side, plain | 5.96 | 14.46 | 8.5 |
  | MLP side, plain, Q 3 | 9.99 | 13.27 | 3.3 |
  | blend call S = 1, whole (4 dispatches) | 20.07 | 36.5 | 16.4 |

  **Ceiling, measured part.** Only the plain calls have the row kernel alone in both columns. Net of the ~1.7 us per
  dispatch that the measurement adds:
  - attention side: (14.46 - 5.96 - 1.7) x 24 = 0.16 ms per token;
  - MLP side: about 0.04 ms.

  That is **0.2 ms per token measured**.
  **Unmeasured part.** The blend calls' 16.4 us gap covers all four dispatches; the split touches only the row kernel
  (#0, 18.37 us in the stream, 46 per token), and the bench has no number for the row kernel alone with the pair stage.
  If that kernel's gap matches the attention-side plain one (6.8 us net), it adds about 0.3 ms, for about 0.5 ms in all.
  **Baseline caveat.** All of these numbers are NE 4 (job E's trace on ac2371c81, R's bench on the route branch). The tip
  10cf8988d has the NE fix: NE 3 at D 2304, 768 threads and 24 simdgroups per row. m-mix chose NE 4 for speed, so the row
  kernel's cost on the tip has probably moved; the probe measures both columns again on 10cf8988d.
- **Cause, as far as the data goes.** In the stream each visit's controller weights are cold: about 200 MB of expert
  weights pass between two visits. Those weights are the conv taps (4 rows), the queries (f32, Q rows) and the norm rows, as
  is the 8-source mix chain. One threadgroup per row runs all of it on one GPU core. A single threadgroup can stream about
  60 GB/s (tgbw probe, #10793), so bandwidth is not the limit: about 80 KB per call is about 1.3 us. The cost is the
  dependent rounds:
  1. the loads, then reduction 1 (the source squares and query dots, partials in simdgroup order, softmax in simdgroup 0);
  2. the weighted sum;
  3. reduction 2 (the norm, or the pair's mix / msg squares), then the stores.
  
  The hoist cut the load rounds before reduction 1 to one; the rest of the chain is untouched.
- **Lever: split the row's phase 1 across threadgroups by source, with a counted hand-off.** Same contract, no new op, no
  backend-encoder change.
  - A row gets G <= NS threadgroups of the current size (TPG threads; 768 at D 2304 after the NE fix). TG g loads source g's
    row (TG 0 also does the conv and writes y / sum / ring as now). Each computes reduction 1's partials for its source
    (its squares, and its dot with each query of the batch) with the current thread layout. The partials go to device
    scratch [row, NV, nsg]. TG 0 also writes an f32 copy of the conv's `last` row to the row's scratch. `last` is not always
    stored: with a residual and no sum output it is the f32 value rv + yv, and only bf16(yv) is written. The last TG reads
    that copy rather than recomputing the conv, which would re-enter the ring / pos logic and the deferred ring write.
  - Each TG then does a device fence and an atomic add on the row's counter, as ink_route already does before its last
    threadgroup routes. The last TG to arrive runs the rest of the current kernel unchanged: it sums the partials in
    simdgroup order (eight at a time in the HO instance), runs the softmax, the weighted sum, reduction 2 and the stores. It
    then resets the counter. The other TGs exit, so nothing spins and no co-residency is assumed. Its source, norm and msg
    rows were just loaded by the other cores (phase 1 also touches the norm and msg rows), so its loads hit the cache.
  - **Why it is bitwise.** Every partial is computed by the same thread index, over the same elements, in the same fma order,
    with simd_sum over the same 32 lanes as today. The last TG adds them in the same order and then runs today's code. So
    the coefficients, the mix, the norm and the outputs keep their bits at every NE, NS, QB, K and row count, and in the
    row-table form: the counter and scratch are per row, so the change stays row-local, as the batch lane requires.
  - **Tests**, before any timing:
    - tests/inkload.rs-style equality with the plain instance: SconvRes, ConvMix with and without the blend, AttnResMix;
      widths of every NE, K 1..5, rows 1..16, Q 1 / 3 / 48, host and device pos, the row-table form;
    - rows-exact;
    - the batched-decode tests on Metal;
    - real-model md5 and self-golden equality with the switch off;
    - admission at T = 1, 4, 16.
  - **Switch.** A new switch, off by default, like the hoist.
  - **Build cost to know up front.** The hand-off needs MSL 3.2's device-scope fences, so the split instance must be compiled
    -std=metal4.0 (METAL4_SOURCES), as ink_route is. That means cutting the row kernel's body out of inkling2_mix.metal (3.1)
    into a header shared with a metal4.0 file: the same step op 77 was waiting on. With pack12 now merged, it is free
    unless ink/batch still edits the mix shader. Without the metal4 library, the current kernel stays: the same fallback as
    InkRoute in tests/inkctrl_nometal4.rs. The per-row counters use the backend's sync arena (sync.rs), as ink_route does;
    the encoder is not touched.
- **Step 1, before building (needs the go-ahead): a probe, on 10cf8988d.** No model run and no shipped kernel.
  1. First, the missing number: the row kernel alone with the pair stage on, at S = 1 and S = 3. Measure it, and the
     plain calls, with eviction off and on. Eviction comes from a new `--evict MB` mode in `ink_ctrl_bench`, which
     streams a buffer of about 256 MB between units the way the expert weights do.
  2. Then the stream per-dispatch trace on the same tree, giving the ceiling on NE 3.
  3. Stop rule: if the measured ceiling is below 0.3 ms per token, stop and report.
  4. Otherwise time a phase-1-only split prototype in the evict-mode bench. Go if the evict bench reproduces the stream
     within about 2 us and the split saves at least 0.3 ms per token over the 96 calls.
- **Alternative held back:** prefetching the next visit's controller rows from the previous row kernel. It is a cross-op
  change (contract note) with the same ceiling, so it is only worth it if the probe shows the split cannot hide the cold
  rounds.

### Probe job P (#11305, 06:37-06:39, snapshot .scratch/snap6 = afaf4e8c9; measurement only)

- **afaf4e8c9** adds `ink_ctrl_bench --probe [--evict MB,..]`, example code only.
  - It covers the four ConvMix call sites with the default switches, timed per dispatch label. Each dispatch runs in its
    own command buffer (`INTARSIA_METAL_COMMIT_EVERY=1`), as in the stream per-dispatch trace, so the row kernel is timed
    apart from the blend stages and both columns carry the same per-dispatch overhead.
  - `--evict` streams one shared weight (96 or 256 MB) through `logits_bf16_gemv` before every unit, so the call's operands
    come cold. That dispatch is reported apart.
  - raw/ceil.py sums, over the four call sites, (stream row-kernel median - cached bench median) x calls per token. It
    prints GO (at least 0.3 ms) or STOP.
  - bespin #11304: the examples cargo-check cleanly.
- **Job steps:**
  1. Build.
  2. Per-dispatch trace on the tip (NE 3), plus 3 normal-step reps.
  3. The probe at T = 1 with evict 0 / 96 / 256 MB.
  4. The ceiling.

  No prototype is in this job; step 4 of the go-ahead (the split) is built only if the ceiling clears 0.3 ms. A watcher
  (.scratch/watchP.sh) reports the start and the build step. fleetq estimates a start around 12:05.
- **For ink-batch (B16 hunt):** which switch disables which controller change at B > 1. Every change is per row or
  rows-exact and applies up to 16 rows:
  - `INTARSIA_INK_ROUTE_FUSED=0`: op 76, the router GEMV + MoeRoute fused, used when t <= 16 rows. B16 decode is t = 16;
    the microbench at T = 16 has the fused call at 13.1 us against the pair's 23.6.
  - `INTARSIA_INK_MSG_PRE=0`: loop messages per pass, for passes of up to 16 rows.
  - `INTARSIA_INK_MIX_HOIST=0`: the hoisted row kernels, including the row-table form. #11104 passed the batch tests with
    it set to 1 and to 0.
  - `INTARSIA_INK_BLEND_NORM_C=0`: the one-channel-per-thread blend norm, at any row count.

  All four at `=0` give the pre-dfuse kernels. Bisect order: all off first, then one at a time.

### Job P results: the row-kernel ceiling is 0.085 ms per token, so STOP (raw/P_*)

- **The tip (afaf4e8c9 = 10cf8988d + the probe; NE 3, pack12, gumbel; the defaults).**
  - Normal step: 13.31 / 13.33 ms, 74.8 / 74.6 tok/s (engine bench, short prompt). A first, cold rep read 14.94 ms.
  - 543 dispatches per step. Greedy md5 d03580fb172c4b088977979647b67712: not E's, since the base changed (NE 3, pack12).
  - Per-dispatch controllers: 2.972 ms per token (E on ac2371c81: 3.243).

  | Controller dispatch on the tip | ms / token | Dispatches / token | Median us |
  |---|---|---|---|
  | ConvMix row kernel, with the blend | 0.779 | 46 | 16.37 |
  | ConvMix row kernel, plain | 0.637 | 50 | 11.8 to 14.1 |
  | ink_route | 0.569 | 48 | 11.81 |
  | blend gate (#2) | 0.376 | 46 | 8.17 |
  | blend h GEMV (#1) | 0.264 | 46 | 5.00 |
  | blend norm_c (#3) | 0.199 | 46 | 4.29 |
  | AttnResMix | 0.092 | 4 | 22.81 |
  | LoopBlend | 0.056 | 8 | 4.5 to 9.6 |
  | **all** | **2.972** | **294** | |

  The row kernel is faster at NE 3 in the stream (14.00 us median against 15.67 at NE 4).
- **The probe** (T = 1, one command buffer per dispatch on both sides). Row kernel alone, us:

  | Call site | Stream | Bench cached | Bench, 96 MB evicted | Bench, 256 MB evicted |
  |---|---|---|---|---|
  | attention side, plain | 12.37 | 10.65 | 48.95 | 57.29 |
  | attention side, blend S = 1 | 15.46 | 13.58 | 42.89 | 56.36 |
  | MLP side, plain, Q 3 | 11.98 | 15.38 | 60.32 | 72.61 |
  | MLP side, blend S = 3 | 17.42 | 21.43 | 68.07 | 84.16 |

  The bench's MLP shapes (4 stored sources, Q 3) cost more than the model's MLP calls, so the stream is under the bench there.
- **Ceiling** (raw/P_ceil.txt): sum of (stream - cached) x calls = 0.041 + 0.043 + 0 + 0 = **0.085 ms per token**,
  under the 0.3 ms bar. Stop rule: stop. No prototype was built.
- **What this shows.**
  - The row kernel's operands are not cold in the stream. With a real eviction (96 or 256 MB) the kernel takes 43-84 us,
    3-5x its stream cost, so in the stream they come from the cache.
  - Its stream cost is its own cached cost plus about 2 us. The cold-load hypothesis behind the split (and behind the
    hoist's stream-vs-warm reasoning) does not hold on the tip; a split for memory-level parallelism has nothing to win.
  - The per-dispatch command buffer costs a small kernel much more than the ~1.5 us average: the attention-side plain row
    kernel is 5.96 us in the multi-dispatch bench (raw/R_bench.md, NE 4) against 10.65 here. So the per-dispatch
    controller totals overstate the real controller time, perhaps by up to 1 ms per token.
- **Not measured.** What is left in the row kernel is its own dependent chain: two full-row reductions per query batch with
  their barriers, the softmax in simdgroup 0, and the stores. Only a restructured kernel could cut it, and that has no
  measured ceiling yet.

### Roofline job L (#11336, 06:57-06:59, snapshot .scratch/snap7 = 91355d5f7, code = afaf4e8c9; measurement only)

- **Method.** Multi-dispatch timing only.
  - Each class is measured by its lane's existing microbench at the real shapes: many launches per command buffer, each
    on its own operand set, with the weights rotating past the system cache.
  - The class times (per call x calls per token from probe P's trace) are summed against the measured step. What is left
    over is the cost the benches do not see: dependencies between different kernels, and the per-dispatch launch cost.
  - The benches: ink_p12_bench (InkInProj S 1 / 3, wo, unembed), ink_attn_bench (attention core, ctx 512 / 4096 / 32768),
    inkmoe_bench (NVFP4 and MXFP8 experts, 1 row), ink_ctrl_bench (router, controllers, LoopBlend), argmax_cost, and
    dispatch_cost (cost per dependent dispatch in one serial command buffer).
- **Box check first.** Rep 0 is dropped; the median span of reps 1-3 must lie in 12.64-13.97 ms (13.3 +- 5 %), otherwise
  the job prints BOX SLOW and stops. One more rep runs at the end.
- **Bytes per token by class** at native widths (raw/roof_bytes.py; bf16 dense, experts as stored):

  | Class | MB / token | ms at 614 GB/s |
  |---|---|---|
  | in_proj (wq, wr, gate, sink, wk, wv) | 1902 | 3.10 |
  | wo | 680 | 1.11 |
  | experts, NVFP4 (44 visits) | 1577 | 2.57 |
  | experts, MXFP8 (4 visits) | 263 | 0.43 |
  | router | 28 | 0.05 |
  | controllers | 113 | 0.18 |
  | small (norms, sconv) | 7 | 0.01 |
  | unembed | 922 | 1.50 |
  | **total** | **5492** | **8.94** |

  The KV read is small at the short prompt: about 4 KB per context token per visit, so about 30 MB per token at a context of
  150.
- **What reduced precision changes** (ink-fp8's lane; from tools/inkling2_floor.py):
  - The tip runs P12, which is lossless, for the dense projections and the unembed: about 12/16 of the bf16 bytes plus
    escapes. The P12 bench reports the exact bytes.
  - fp8 (e4m3) would bring the projections from 2582 MB (bf16) to 1301 MB and the unembed from 922 to 465 MB. Against P12,
    that is about -0.64 GB, or -1.0 ms per token at 614 GB/s, if it hits the same GB/s.

### Job L results: where the 4.4 ms goes (raw/L_*, raw/roofline.py -> raw/L_roofline.txt)

- **Box check: OK.** Default step, reps 1-3: 13.331 / 13.282 / 13.830 ms; median 13.331, inside 12.64-13.97. The rep after
  the benches read 13.330. One rep: GPU busy 13.281 ms per step, idle 0.001, 543 dispatches.
- **Classes.** Multi-dispatch microbench time per call x calls per token. Bytes are those the tip reads: P12 for the dense
  projections and the unembed, experts as stored, and KV at a mean context of 150. The floor is those bytes at 614 GB/s.

  | Class | ms / token | MB / token read | GB/s | Floor (ms) | Gap (ms) |
  |---|---|---|---|---|---|
  | Controllers: ConvMix row kernel and the blend stages, LoopBlend, AttnResMix | 1.728 | 113 | 66 | 0.184 | **1.543** |
  | Attention core: fused decode, 40 SWA + 8 full visits, ctx <= 512 (an upper bound for the short prompt) | 1.118 | 30 | 27 | 0.048 | **1.070** |
  | Experts NVFP4 (44 visits; gate+up, down) | 3.599 | 1577 | 438 | 2.568 | **1.031** |
  | Attention projections: InkInProj S 1 / 3, wo (P12) | 3.735 | 1950 | 522 | 3.175 | 0.561 |
  | Router (ink_route) | 0.409 | 28 | 69 | 0.046 | 0.363 |
  | Experts MXFP8 (4 visits) | 0.504 | 263 | 522 | 0.428 | 0.076 |
  | Unembed (P12) | 1.202 | 692 | 576 | 1.128 | 0.074 |
  | Argmax, norms, embedding, other | 0.031 | 8 | | 0.013 | 0.018 |
  | **Sum of the classes** | **12.327** | **4660** | | **7.590** | **4.737** |
  | **Not seen by the class benches** (step 13.31 - sum) | **0.983** | | | | |

  Notes on the controller rows:
  - The two MLP-side controller rows use the bench's shapes, which are heavier than the model's (probe P: the stream's MLP
    row kernel is under the bench). The controller row is an upper bound by about 0.1-0.2 ms, and the unseen share is
    larger by the same amount.
  - AttnResMix (4 per token) has no multi-dispatch bench; its per-dispatch stream time (0.091 ms) is used as an upper bound.
- **Pure dispatch cost.** A near-empty kernel costs 0.88 us per dependent dispatch in one serial command buffer
  (dispatch_cost). 543 x 0.88 = **0.478 ms per step**. That cost is inside the class times above, not on top of them. In a
  concurrent encoder without barriers it falls to 0.12 us, but that does not apply to a dependent chain.
- **Floors.** 7.59 ms at the bytes the tip reads (4.66 GB), 8.94 ms at native bf16 bytes (5.49 GB). The step is 13.31 ms:
  5.7 ms over the tip-bytes floor, 4.4 ms over the native one.
- **Top 3 gaps, with ceilings for a fix.**
  1. **Controllers, 1.54 ms gap.** Bounded ceiling: at most about 1.28 ms. Even a perfect set would keep its 294
     dependent dispatches (294 x 0.88 = 0.26 ms) and its bytes (0.18 ms).
     - The biggest part is the two blend chains (ConvMix with the blend, 4 dispatches each). They cost 1.175 ms against
       floors of about 5.4 us (S 1) and 9.3 us (S 3) per call (bytes + 4 dispatches), which bounds the ceiling at about
       0.84 ms.
     - No lever with a measured gain exists: the row kernel's cold cost is 0.085 ms (probe P); fusing the blend tail
       measured +0.155 ms, i.e. slower (change 3); ink_route's lever is at most about 0.18 ms.
  2. **Attention core, about 1.07 ms gap.** It is latency-bound: 23-24 us per call at ctx <= 512, almost no bytes, and
     nearly flat in context in the stream (about 30-33 us per dispatch from ctx 20 to 277).
     - Bounded ceiling: about 0.33 ms if the fused kernel cost no more than its longer half (attention alone 16.3 us, SWA)
       with the prologue fully hidden. About 0.58 ms if it cost no more than the prologue alone (10.8 us). Neither is a
       measured kernel.
     - Owner: the attention lane (m-attn / ink-metal).
  3. **Experts NVFP4, 1.03 ms gap** (438 GB/s at 1 row).
     - Inferred from 4-row data, not measured at 1 row: the same kernels stream at 535 GB/s at 4 rows (m-moe results doc).
       At that rate the 1-row call would be 67 us instead of 81.8, which is 0.65 ms per token.
     - Bound from 1-row data: m-moe's 1-row split is gate+up 50 us at 480 GB/s and down 33 us at 355 GB/s. If down ran at
       gate+up's rate it would take about 24 us, which is 9 us x 44 visits = about 0.40 ms per token. That clears 0.3 ms
       with no assumption about more rows.
     - Owner: m-moe. Its doc names the lever: one persistent gate+up -> down kernel, which has a forward-progress caveat.

  Also measured:
  - The unseen remainder, 0.98 ms: kernel-to-kernel transitions in the dependent chain, outside any one class. It has no
    ceiling, and the levers would sit in the backend encoder (hostgap's area).
  - The projections: 0.56 ms gap. The best GEMV rate measured on this box is the unembed's 576-582 GB/s, which bounds the
    gain at about 0.35 ms (pack12's kernels).
- **What reduced precision changes (ink-fp8's lane).** e4m3 for the projections takes their bytes from 1950 MB (P12) to
  1301 MB, and for the unembed from 692 to 465 MB. That is -0.88 GB, about -1.4 ms at the rates the P12 kernels reach now:
  the biggest single lever in the table, and not this lane's.
- **Proposal.** Nothing in this lane has a lever with a measured ceiling of at least 0.3 ms, so I propose building nothing
  here. The best-supported item over the bar is the NVFP4 1-row experts (measured bound 0.65 ms, owner m-moe); after it come
  the attention core (bounded 0.33-0.58 ms) and the projections' GEMV rate (bound 0.35 ms).

### Current G1-G3 with the flip (ink-metal's combined gate #11206, tree 7e5e6010e: NE fix + hostgap / fp8 / prefill / dfuse)

ink-metal's combined gate ran this lane's merge condition after the flip landed (ink-metal results doc, "Combined gate
#11206", PASS):
- decode-mode G2 short: 2.928e-3 with the defaults and 2.928e-3 with `INTARSIA_INK_MIX_HOIST=0` (limit 4e-3);
- decode-mode G2 long: 2.761e-3 / 2.761e-3 (limit 6e-3).

The self-golden tensors are identical; the files differ only in metadata. Greedy md5 on that tree is 65cde8f35..., this
lane's defaults md5 from job F. The flip stands (merge b1a4f3f62, no revert). These supersede job F's decode-mode numbers,
which predate the NE fix.

### Job N (#11362, 07:25-07:28, snapshot .scratch/snap8 = 914738f23; measurement only): NVFP4 experts at 1 row

- **Shape at T = 1.** 8 experts per visit, 35.83 MB: gate+up 23.89 MB, down 11.94 MB. The GPU has 40 cores.
  - Gate+up (`ink_moe_gu_nv_a_x`, union): 576 threadgroups of 128 threads (16 rows each, 8 lanes per row, 9 chunk pairs per
    lane, one chunk in flight per lane).
  - Fused down (`ink_moe_dn_f_nv_a_bf16`): 288 threadgroups of 128 threads (8 rows; each simdgroup serves 2 of the 8 slots
    one after the other, 4 lanes per row, 9 chunks per lane per slot). That is half the threads of the gate+up and twice
    the dependent chain per lane.
- **Steps:**
  1. Box check (job L's window). Only batch.rs changed since L's tree, so the B = 1 decode path is the same.
  2. Streaming floor (raw/moebw.swift): a no-math kernel reading the same regions, with a sweep of TG count, threads and
     loads in flight. It measures gate+up alone, down alone, both as two dependent dispatches, and all in one dispatch, plus
     the 29-expert (4-row) union. This answers how much of 438 against 535 GB/s is dispatch size against kernel design.
  3. Shipped kernels: the gate+up / down split (INKMOE_PROBE) at 1 and 4 rows, and fused-down threads 64 / 256.
  4. 1-row variants of the shipped kernels (raw/moevar_gen.py, raw/moevar.swift; prototypes compiled at run time, nothing
     shipped): lanes per row (gate+up 4 / 8 / 16, down 4 / 8 / 16), chunks ahead 1 / 2, simdgroups 4 / 8, fused-down threads
     128 / 256. Each is timed, and max |out - shipped| is checked over 4 visits. The variants that keep the per-lane chunk
     order (chunks ahead, simdgroups, threads) should be bitwise.
  5. Box re-check.
- The job body runs in its own process group, and TERM kills the tree (tested with dummy sleeps: no descendants left).
- **Item 3, first part (code read, no job):** the decode encoder is `MTLDispatchTypeSerial`
  (crates/intarsia-backend-metal/src/lib.rs:790; lib.rs:13-15 names concurrent dispatch with targeted barriers as a phase B
  option). The only concurrent dispatch is in the ICB path (decode.rs:179). So every dispatch waits for the one before; the
  overlap bound comes later.

### Job N results (raw/N_*)

- **Box: OK.** Median span of reps 1-3 is 13.328 ms; the re-check after the benches read 13.294. Greedy md5 is
  d03580fb... in all 5 reps.
- **Streaming floor of one 1-row visit** (raw/N_moebw.txt; no math, every byte loaded once, GPU time, 5 layer copies
  rotating). An empty dependent dispatch costs 0.81 us.

  | Read | Best config | us / visit | GB/s |
  |---|---|---|---|
  | gate+up regions alone (23.89 MB) | 4 loads in flight, 1024 threads, 2304 TGs | 41.76 | 572 |
  | down regions alone (11.94 MB) | 4 loads in flight, 512 threads, 2304 TGs | 21.57 | 554 |
  | both, two dependent dispatches (35.83 MB) | best of each | **63.73** | 562 |
  | both, one dispatch (no dependency) | 2 loads in flight, 1024 threads, 4608 TGs | 61.76 | 580 |
  | both, two dispatches, at the shipped geometry (1 load, 128 threads, 576 / 288 TGs) | | 71.06 | 504 |
  | 29 experts (the 4-row union), two dispatches | best 8-expert configs | 223.15 | 582 |

- **Shipped kernels, 1 row** (raw/N_moe_split.txt, wall time of synced batches): 81.9-84.2 us per visit, 426-438 GB/s.
  Gate+up is 49.7 us (480 GB/s), down 33.3-34.2 us (349-359 GB/s). Fused-down threads 64 / 128 / 256 change nothing
  (32.6-34.6). At 4 rows the default is 271 us, 487 GB/s (gate+up 520, down 423); m-moe's doc number of 535 does not
  reproduce on this tree.
- **1-row variants of the real kernels** (raw/N_moevar.txt, GPU time; prototypes, nothing shipped):

  | Variant | gate+up us | down us | visit us | vs shipped |
  |---|---|---|---|---|
  | shipped (run twice) | 47.94 / 48.18 | 31.27 / 31.36 | 79.70 / 79.76 | |
  | down 2 chunks ahead (dn_nb2) | 47.96 | **27.10** | **77.46** | -2.24 |
  | down 8 lanes per row, 576 TGs (dn_ld8) | 47.90 | 29.49 | 77.87 | -1.83 |
  | dn_ld8 + 256 threads | 49.50 | 28.72 | 77.69 | -2.01 |
  | down 16 lanes per row (dn_ld16) | 47.91 | 30.38 | 78.64 | -1.06 |
  | gate+up 16 lanes per row, 1152 TGs (gu_lg16) | 48.15 | 31.37 | 81.10 | +1.40 |
  | gate+up 256 threads (gu_sg8) | 48.62 | 31.41 | 80.29 | +0.59 |
  | gate+up 2 chunks ahead (gu_nb2) | 52.00 | 32.57 | 86.65 | +6.95 |
  | gate+up 4 lanes per row (gu_lg4) | 59.60 | 31.39 | 90.93 | +11.23 |

  The "max abs vs shipped" column reads 0 for every variant. With these weights that is weak evidence: every byte was
  filled with 0x22, and the bf16 roundings of g, u and y hide most order changes. The next probe uses random weights.
- **Why 438 GB/s at 1 row.**
  1. Dispatch size is a small part. A perfect stream of the 8 experts reaches 562 GB/s as two dependent dispatches,
     against 582 for the 29-expert union: about 2 us per visit.
  2. The dependency between gate+up and down costs about 2 us per visit (63.73 against 61.76 for one dispatch), i.e.
     about 0.09 ms per token. A persistent gate+up -> down kernel has little to win, so the forward-progress question
     does not need answering.
  3. Gate+up runs at the no-math streaming speed of its own geometry: 47.9 us against 47.7 for a no-math kernel of 576
     TGs x 128 threads with 1 load in flight. Its ceiling is the geometry. But more loads in flight, more threads or more
     TGs in the real kernel do not help (gu_nb2, gu_sg8, gu_lg16), so per-thread work and registers limit it, not the
     memory system.
  4. The fused down runs 7 us over its geometry's no-math time (31.3 against 24.4): its per-lane chain (two slots in
     sequence per simdgroup, 18 chunks with one load in flight, the barrier and the k-order sum) is latency-bound. Two
     chunks ahead takes 4.2 us of it.
- **Ceilings at 44 visits per token.**
  - Bound: shipped minus the perfect two-dispatch stream = (79.70 - 63.73) x 44 = 0.70 ms. A kernel that does the math
    cannot reach it fully.
  - Measured: the best variant (dn_nb2) saves 2.24 us per visit = **0.099 ms per token**, under the 0.3 ms bar. Nothing
    built.
- **One more probe before item 1 is closed (job N2).** The candidates not yet tried are aimed at the gate+up's per-thread
  cost:
  - a T = 1 instance with one pair per leader (INK_XR 1 instead of 4), fewer registers, bitwise for T = 1 calls;
  - down with 3 or 4 chunks ahead;
  - combinations with dn_nb2.

  Random weights for the numerics check, and each variant's maxTotalThreadsPerThreadgroup as an occupancy hint. If the
  best measured combination stays under 0.3 ms per token, item 1 stops and item 2 starts.

### Job N2 (#11382, 07:49-07:51, snap8 914738f23 and job N's binary; measurement only; raw/N2_*)

- **Box:** median span of reps 1-3 is 13.275 ms; the re-check after the variants read 13.264. Greedy md5 is d03580fb... in
  all 5 reps.
- **Weights are random this time** (finite e4m3 scales; all 9216 outputs nonzero, no inf / nan). Every variant reads
  0 diff against the shipped output over 4 visits. For the variants that keep each lane's chunk order (one pair per
  leader, chunks ahead) that holds by construction. For the lane-count variants it is still only probable, because the
  bf16 roundings of g, u and y hide most f32 order changes.
- **Visit us, two passes in opposite orders.** Shipped: 79.72 / 80.16 (pass 1), 80.84 / 80.81 (pass 2).

  | Variant | Pass 1 | Pass 2 | Mean gain vs the shipped runs of the same pass (us) | ms / token (x 44) |
  |---|---|---|---|---|
  | **xr1_dn_nb3**: gate+up one pair per leader (INK_XR 1) + down 3 chunks ahead | 73.19 | 73.23 | **7.17** | **0.316** |
  | xr1_dn_nb2 | 72.73 | 74.93 | 6.55 | 0.288 |
  | xr1_dn_ld8_nb2 | 75.11 | 75.09 | 5.07 | 0.223 |
  | dn_nb3 | 75.89 | 76.91 | 4.04 | 0.178 |
  | dn_nb2 | 76.14 | 76.30 | 4.16 | 0.183 |
  | gu_xr1_lg16 | 78.00 | 77.81 | 2.32 | 0.102 |
  | gu_xr1 | 79.03 | 77.88 | 2.00 | 0.088 |
  | dn_nb4 | 80.09 | 81.25 | -0.25 | |
  | gu_nb2 | 86.07 | 86.56 | -6.0 | |

  Down alone: 26.65 / 26.44 us at 3 chunks ahead against 32.57 / 31.67 shipped. Gate+up alone with one pair per leader:
  45.8-47.4 against 47.9.
- **Reading.** The two levers add up.
  - **Gate+up:** one pair per leader is the only change that helps it, by about 2 us. At T = 1 every pair leads a run of
    one, so the per-pair math is the shipped kernel's; the change only drops the 4-slot arrays and loops (registers and
    instructions).
  - **Down:** more chunks in flight per lane shortens its latency-bound chain; 3 is the best, and 4 is worse again.
  - **Combined:** xr1_dn_nb3 measures 0.316 ms per token in the microbench, at the bar. It is bitwise by construction (no
    change to any pair's chunk order, fma chain or reduction).
- **Proposed build, opt-in, T = 1 only:**
  1. A T = 1 gate+up instance with one pair per leader (`ink_moe_gu_nv_*` templated on the pairs per leader).
  2. The fused down `ink_moe_dn_f_nv_*` with 3 chunks ahead; it serves only T = 1 calls under the Auto policy.
  3. The T > 1 paths (dn_u, union gate+up with 4 pairs per leader, MXFP8) stay unchanged, which keeps the batch lane's
     paths and rows-exact intact: a T = R call still runs the shipped kernels, and its pairs' bits equal the T = 1
     calls'.
  4. One env switch to revert. Gates as in the brief: the model step ABAB, 3 alternating reps, is the real measure of the
     gain.

### Build: INTARSIA_INK_MOE_T1 (fcd05b172, default off) and its gate job G (#11400, queued 08:05)

- **fcd05b172** (own commit). kernels/metal/inkling2_moe.metal:
  - `ink_gu_impl` / `ink_lead` take the pairs per union leader, and `ink_dn_row` / `ink_dn_f_impl` the chunks ahead, as
    template parameters. The defaults are INK_XR / INK_NB, so the shipped instances are the same code; the explicit
    shuffle reductions are unchanged.
  - New NVFP4 instances: `ink_moe_gu1_nv_{a,u}[_x]` (one pair per leader) and `ink_moe_dn_f3_nv_{a,u}_{bf16,f32}` (3 down
    chunks ahead).

  crates/intarsia-backend-metal/src/inkmoe.rs: `t1_enabled()` (`INTARSIA_INK_MOE_T1=1` / `on`; `set_t1` for tests). A
  one-row NVFP4 call takes the new instances. The first call prints one stderr line, and `t1_served()` counts the calls.
  The NAX prefill code is not touched (the switch sits after its early return). docs/tuning.md has the row. bespin
  #11397 / #11399: the Metal crate's lib, tests and examples check, and docs_env_vars passes.
- **tests/inkmoe.rs.** The NVFP4 rows-exact sweep now runs twice. With the switch on, the one-row reference calls take the
  new instances and the T-row calls the shipped ones, so the compare crosses the kernels. The test asserts that the new
  instances served when on and never when off.
- **Job G** (raw/G_jobG.sh; one binary, the switch by env; body in its own process group):
  1. box label;
  2. greedy md5 at 64 / 256 tokens, off vs on (stop on any difference);
  3. inkmoe, plus G1 tiny and inkling2_batch on Metal with the switch on;
  4. admission `--op moe_experts` off and on (the situ sweep has T = 1, 3, 4, 7, 16, 40, 64, 2048);
  5. G2 (both modes) and G3, short and long, off and on, with the self-goldens byte-compared;
  6. `plan show` against the base binary;
  7. speed after a warm-up: ABAB in this slot, 4 pairs at the short prompt and 3 at 5k, per-pair span and tok/s
     differences, with the decision rule printed.
- **Unslotted CPU work on endor:** a 14 s `cargo check` at about 07:56 while ink-metal's quiet baseline #11387 ran, plus a
  few seconds of offline `xcrun metal` / `swiftc -typecheck` earlier. Reported to main; compile checks now go to bespin.


### Job G results (#11400, 08:41-08:47, snapshot .scratch/snap9 = fcd05b172; raw/G_*)

Every correctness gate passes. The speed gain is real but below the 0.3 ms/token bar, so by the rule fixed before timing
`INTARSIA_INK_MOE_T1` **stays opt-in (default off)**. Keeping it opt-in or dropping it is the coordinator's call.

- **Greedy md5, off vs on:** equal at 64 tokens (65cde8f3...) and 256 tokens (61fd59fe...). The on runs served the new
  instances, the off runs never did.
- **Tests:** inkmoe 2/2, including the crossed rows-exact sweep (3840 one-row calls took gu1 / dn_f3 when on, 0 when
  off). G1 tiny + inkling2_batch with the switch on: 16/16.
- **Admission `moe_experts`:** off and on identical (situ 10/10 and rows 7/7 ADMITTED, same error numbers).
- **G2 / G3, self-goldens off vs on byte-equal in all four.** Decode-mode G2: 2.928e-3 short, 2.761e-3 long, the same as
  ink-metal's #11206. G3: 8.420e-3 short, 1.229e-2 long.
- **`plan show` Qwen3-8B and Llama-3.1-8B:** byte-identical to the base binary.
- **Box:** median 13.290 ms/step at the start and 13.243 at the end (543 dispatches, 2 buffers per step, idle 0.01 %).
- **Speed, ABAB in one slot (median span ms/step, off minus on):**

| Prompt | Pairs | Per-pair gain (ms/token) | Median | tok/s on minus off (median) | Bar |
|---|---|---|---|---|---|
| short | 4 | 0.275, 0.314, 0.236, 0.270 | **0.272** | +1.52 (75.1 -> 76.6) | below 0.3 |
| 5k | 3 | 0.293, 0.308, 0.248 | **0.293** | +1.18 (70.1 -> 71.4) | below 0.3 |

  The on side wins all 7 pairs, about 2 %. The microbench predicted 0.29-0.32 ms/token (N2), and the real step lands at
  the bottom of that range. The difference at 1 row is the one-row gate+up (one pair per leader) plus the down kernel
  loading 3 chunks ahead; 44 visits per token.
- **Lane recommendation:** keep it opt-in. It is bitwise, the code is template parameters on the shipped kernels, and
  the gain is consistent. No default flip unless the coordinator changes the bar.

### Coordinator decisions on job G and the next item (2026-11-22 ~08:55, verbatim in substance)

- **T1 stays opt-in and is kept, not dropped.** The pre-registered rule holds: `INTARSIA_INK_MOE_T1` stays default off.
- **Bundling candidate.** If a later T = 1 change lands in the same kernels (`inkling2_moe.metal` NVFP4 gate+up / down),
  gate it together with T1 in one ABAB, so that the two gains are judged as one.
- **Next: item 2, the decode attention core.** Take its ceiling from roofline L (measured ms per token vs floor, short and
  5k context). If the ceiling is >= 0.3 ms per token, probe it as for the experts (a no-math stream of the same geometry,
  then variants) under the same pre-registered rule. If it is under 0.3 ms, report the number and wrap up the lane.
- **Housekeeping.** Merge ink/metal into ink/dfuse (done: 3d85517f7). Remove finished snapshot worktrees with
  `git worktree remove` (snap8 and snap9 removed). The merge's compile check runs on bespin or in a slot.

### Merge check (bespin #11429): green

ink/metal 5ef24b165 merged into ink/dfuse at 3d85517f7 with no conflicts. On bespin, cargo check passes for the Metal crate
(examples and tests) and for the models tests. The Metal `kernels::tests` (including ink-metal's new shader-count test) pass
7/7, the CPU Inkling tests pass 11 + 7, and docs_env_vars passes 2/2.

### Item 2: decode attention core, ceiling from roofline L (no new job)

The attention core is the fused `ink_attn_dec_g3` (H 24 / Hk 8, D 128, d_rel 16, E 1024, K 4, SWA window 1024) plus
`ink_attn_comb_bf16` when there is more than one split (64 keys per split, at most 32). That is 40 SWA and 8 full visits
per token. Times are L's `ink_attn_bench --fused`, rows 1, raw/L_b_attn.txt. The floor counts the KV bytes at 614 GB/s:
4 KiB per key per visit (K and V, 8 kv heads x 128 x bf16); the q / rel-proj bytes are negligible.

| Prompt | Context | Measured (ms / token) | Floor (ms / token) | Gap = ceiling (ms / token) |
|---|---|---|---|---|
| short | <= 512 (L's ctx 512 values, an upper bound) | 1.118 (40 x 23.09 + 8 x 24.34 us) | 0.048 | **1.07** |
| 5k | 5051-5308 (job G's 5k run; mean 5180) | 1.847 (40 x 31.36 + 8 x 74.1 us; full interpolated 4096 -> 32768) | 0.550 (337 MB) | **1.30** (1.20 with full at its ctx-4096 value) |

- **Ceiling >= 0.3 ms at both prompts, so this lane probes it** (coordinator's rule).
- L's tighter bound for the short prompt (the fused kernel costing no more than the attention alone, or the prologue alone)
  is 0.33-0.58 ms. That is also >= 0.3.
- **Cross-check against the real step.** Job G's 5k step is 0.92 ms slower than its short step. The bench's growth is 0.73
  ms (0.64 lower bound), and more than that from the short prompt's real context (about 150, below the ctx 512 used). The
  bench may under-count the stream at 5k by up to about 0.2 ms. One likely cause is the bench's 16 reused operand sets: 67
  MB of SWA KV at ctx 4096 can stay cache-resident. The probe measures in the real step too.

### Item 2: pre-registered rule (written before any timing)

1. **Harness calibration first.** The probe harness's shipped kernel must be within 5 % of L_b_attn's fused times (SWA and
   full, ctx 512 and 4096), and its output bitwise equal to the backend's for one visit at each. If either check fails, the
   harness is fixed before any variant number is read.
2. **Stage 1: no-math kernels of the same geometry, priced per visit.**
   - Latency skeleton: the dec + comb pair with the same grid (8 x splits x 1, 128 threads), threadgroup memory, barrier
     sequence and loads, and the arithmetic dropped.
   - The same skeleton as one dispatch (no comb), to price comb. At 64 keys per split, nearly every decode position has at
     least 2 splits.
   - At 5k only: a byte stream of the same bytes on the same grid.

   Ceiling = sum over visits of (shipped - best no-math), at short (ctx about 150 and 512) and at 5k. If it is under 0.3
   ms/token at both prompts: stop, report, wrap up the lane.
3. **Stage 2: variants, only where stage 1 >= 0.3.**
   - Same-bits restructurings first: key-loop lookahead loads, and issuing the first K/V and rel-proj / tap loads before
     the prologue barriers.
   - Non-bitwise variants (split keys 16 / 32 / 128 through `INTARSIA_INK_SPLIT_KEYS`; more simdgroups) are measured for
     information only.
4. **Build and default rule (as for T1).**
   - A build is proposed only on a microbench gain >= 0.3 ms/token at its target prompt.
   - It goes default-on only if the real-step ABAB median gain is >= 0.3 ms/token, warm, at its target prompt, with no
     regression beyond 0.1 ms/token at the other prompt. Same-bits variants also need equal greedy md5. Otherwise the
     variant stays opt-in or is dropped.
5. **Bits.** A split-policy change reorders the rounding, so md5 will differ. Whether non-bitwise variants are in scope,
   and what replaces md5 for them, is the coordinator's call (asked 2026-11-22 ~09:15).
6. **State.** InkAttnDecode writes the KV cache and the conv ring, and `ink_attn_dec<GP, RT>` is shared with ink-batch's
   row-table form. Before any build: message main, and run `inkling2_batch` and the attention rows-exact tests.

### Coordinator go-ahead for item 2 (2026-11-22 ~09:20, verbatim in substance)

- **The probe plan and the pre-registered rule above are approved as written.**
- **(a) Non-bitwise variants are in scope, but opt-in only (default off).** In place of md5 their gate requires:
  1. G2/G3 within ink-metal's thresholds;
  2. decode-mode G2 no worse than now;
  3. rows-exact and the state tests still bitwise, so the split choice depends only on context and shape, never on T or
     the batch row count, and T = 1 and the rows form split identically.

  Making a non-bitwise variant default-on is the **user's** call, not the coordinator's. If one clears 0.3 ms/token in the
  real-step ABAB, bring the coordinator its G2/G3 numbers and the coordinator will put it to the user. Same-bits variants
  that clear the bar can go default-on under this lane's rule.
- **(b) Any build that touches `ink_attn_dec<GP, RT>` goes to the coordinator before it starts.** ink-batch's rows form
  shares the template, and ink-batch is working on the B > 1 in_proj now. The gate runs `inkling2_batch` and the
  attention rows-exact tests.

### Probe job Q (queued 09:05, estimated start ~11:37 behind #11426 / #11427 / #11428; snapshot .scratch/snap10 = 6368ad219)

The tools are in raw/Q_tools. Nothing is shipped: the backend dump is an example that is not tracked in the snapshot, and the
variants are macro switches on a probe copy of the shader.

- **`ink_attn_dump.rs`**: one backend `ink_attn_decode_bf16` visit for each (SWA / full) x ctx (150, 512, 4096, 5180), at
  rows 1 and ink_attn_bench's shapes. It writes the operands, the output, and the caches and rings after the visit.
- **`attnvar_gen.py`** writes `ship.metal`, which is the shipped file with common.h inlined, and `probe.metal`:
  - `PR_NB` n: key-loop lookahead loads (same bits);
  - `PR_EARLY`: the first n steps' cache loads are issued before the prologue (same bits);
  - `PR_NOMATH`: the key loop keeps its loads and drops the score / softmax / merge arithmetic;
  - `PR_NOPRO`: the prologue keeps its loads and barriers and drops its arithmetic;
  - `pr_bytes`: a byte stream over the split's K / V on the same grid.

  `build_libs.sh` compiles them with build.rs's flags (`-std=metal3.1 -O3 -fno-fast-math`, Xcode.app).
- **`attnprobe.swift`** has three parts:
  - **cal**: the shipped copy (and p0) against the dump, bitwise.
  - **warm16**: L's bench setting, 16 sets x 8 passes, us per visit.
  - **token**: 40 SWA + 8 full visits per token over 6 / 4 / 2 distinct token sets at ctx 150 / 512 / 5180, so that K/V
    arrive cold. Two passes per command buffer, ms per token.

  Variants: ship, p0, nocomb, nopro, nomath, skel_c (both off, with comb), skel (both off, no comb), bytes,
  la1 / la2 / la4, early2 / early4, and split keys 32 / 128. Each number is the median of 7 rounds, with the order rotated.
  Same-bits variants are compared bitwise with ship on a SWA and a full visit.
- **Pre-checks (bespin #11434):** the example passes cargo check, the harness compiles with swiftc, and all 10 macro sets
  compile with the OS Metal compiler (bespin has no Xcode, so the backend-flag compile runs in the endor slot).
- **Real step in the same slot:** a box label, then the split-keys sweep 64 / 32 / 128 at short and 5k, two runs each in
  ABC CBA order. This is for information only; the bits differ.
- (Housekeeping: this job's first draft was written over the original `.scratch/jobA.sh` from 2026-11-21. That script
  belonged to the superseded first correctness pass; the rerun's script is archived as raw/jobA2.sh.)

### Job Q results (#11433, 09:20-09:24; raw/Q_probe.txt, Q_speed.txt, Q_jobQ.out)

**Bottom line:** the stage-1 ceiling is large (0.74 ms/token at short, 1.06 at 5k). No pre-registered stage-2 variant comes
near 0.3. The same-bits lookahead family gains at most 0.02 ms/token and is slower at ctx 512 and 5k. The best
non-bitwise variant, split keys 32, gains about 0.1 ms/token at short and nothing at 5k. **By the rule, no build is
proposed.**

**Calibration.**
- Bitwise: PASS. The shipped copy, and p0 (the probe source with every switch off), equal the backend on out, k / v
  caches and k / v rings at all 8 points (SWA / full x ctx 150 / 512 / 4096 / 5180). So the harness's InkP, split count,
  bias table and flags are the backend's.
- Timing against L (raw/L_b_attn.txt, fused rows 1):

| Point | L (us/visit) | Harness ship (us/visit) | Difference |
|---|---|---|---|
| SWA ctx 512 | 23.09 | 22.53 | -2.4 % |
| full ctx 512 | 24.34 | 22.52 | **-7.5 %** |
| SWA ctx 4096 | 31.36 | 30.59 | -2.5 % |
| full ctx 4096 | 62.32 | 60.90 | -2.3 % |
| empty dependent dispatch | 3.50 | 3.32 | |

  Three of four points are inside the 5 % bar. **Full ctx 512 misses it at -7.5 %.**
  - The harness has full 512 equal to SWA 512 (22.52 / 22.53): same keys, same splits, the dyn conv adds about nothing.
  - L's own full-512 row is its noisiest: the unfused attention p10..p90 is 17.31..18.48 there, against 16.22..16.45 for
    SWA 512.
  - The cause is not established. A geometry error is ruled out, since the bits (which depend on the split geometry)
    match. Independent evidence that the harness tracks the stream: sk32 gains 0.117 ms/token in the harness and 0.103 in
    the real step at short.
  - Stated plainly: the bar was not met at that point. The no-build decision below compares variants with ship inside one
    harness, so it does not depend on that row.
  - Before this harness is the basis for any build, L's ink_attn_bench full-512 row must be rerun in the same slot.

**Stage 1: no-math kernels (token mode, cold K / V, ms per token; ship = 1.074 / 1.168 / 1.864 at ctx 150 / 512 / 5180).**

| Variant | ctx 150 (short) | ctx 512 | ctx 5180 (5k) | What it removes |
|---|---|---|---|---|
| nocomb | -0.153 | -0.186 | -0.279 | the combine dispatch (about one 3.3 us dependent dispatch per visit) |
| nopro | -0.411 | -0.365 | -0.484 | prologue arithmetic plus its rel-proj / tap / ring loads |
| nomath | -0.215 | -0.283 | -0.252 | key-loop score / softmax / merge arithmetic |
| skel_c (nopro + nomath, with comb) | -0.638 | -0.626 | -0.811 | |
| **skel** (no arithmetic, no comb) | **-0.741** | **-0.784** | **-1.062** | stage-1 ceiling |
| bytes (byte stream, same grid) | -0.860 | -0.896 | -1.184 | |

- **Ceiling >= 0.3 at both prompts, so stage 2 ran.** The three parts are roughly additive: comb + prologue + loop =
  0.78 at short, against 0.74 for skel. The per-visit numbers in warm16 tell the same story (raw/Q_probe.txt).

**Stage 2: variants (ms per token, minus = faster).**

| Variant | Bits | ctx 150 | ctx 512 | ctx 5180 |
|---|---|---|---|---|
| la1 / la2 / la4 (key-loop lookahead) | equal | +0.001 / -0.003 / -0.005 | +0.019 / +0.042 / +0.093 | +0.057 / +0.142 / +0.251 |
| early2 / early4 (first K/V loads before the prologue) | equal | -0.020 / -0.020 | +0.011 / +0.063 | +0.181 / +0.288 |
| sk32 (split keys 32) | differs (max 5e-7) | **-0.117** | -0.124 | -0.023 |
| sk128 | differs | +0.226 | +0.333 | +0.240 |

- The key loop is not load-latency-bound. Holding more loads in flight costs more (registers) than it hides.
- **Real step, split keys (information only; md5 differs as expected), mean of 2 runs, ms/step against sk 64:**
  - short: sk32 **-0.103**, sk128 +0.441;
  - 5k: sk32 +0.009, sk128 +0.178.

  This matches the harness: sk32 gains about 0.1 at short and nothing at 5k. Box label 13.245 / 13.246 at the start and
  13.288 at the end: normal.

**Decision (pre-registered rule).** Stage 1 >= 0.3 at both prompts. But no stage-2 variant reaches a microbench gain of
0.3 ms/token at either prompt (best: sk32 -0.12 at short, non-bitwise), so no build is proposed.

**What the attribution leaves open (not pre-registered, for the coordinator).**

| Candidate | Measured bound (ms/token) | Notes |
|---|---|---|
| (i) fold the combine into the last-arriving split threadgroup (device counter + fence) | <= 0.15 short, <= 0.28 at 5k (nocomb) | The partials can be merged in split order with comb's arithmetic, so it can stay bitwise. Under 0.3 alone at both prompts. |
| (ii) prologue | <= 0.41 short, <= 0.48 at 5k (nopro) | A ceiling for deleting the prologue, not for doing it faster. The same-bits overlap of loads already failed for K/V (early). The rest is serial reductions, precise sqrt / divide and the 16-deep r . relproj chain, whose order is fixed by the bits. |

Neither has a measured lever >= 0.3. A pair (i) + (ii) might reach it only if (ii) recovers more than half its bound.

**Two findings outside the attention verdict.**
- **Dispatch cost.**
  - The harness's empty dependent dispatch costs 3.32 us (pr_floor: one read-modify-write of a device word, serial
    encoder, 256 in a chain). L's own bench floor (add_bf16 through the backend) was 3.50 us. Dropping comb saves 3.2
    us per visit (token mode) and 3.9 (warm16).
  - Roofline L priced a dependent dispatch at dispatch_cost's 0.88 us (raw add_bf16, same buffers every time, 2000 per
    command buffer). That gave its 0.48 ms per step and its controller bound of 294 x 0.88 = 0.26 ms.
  - If a transition in the real chain costs about 3.3 us, those bounds are about 4x low: cutting 543 -> 350 dispatches
    would be worth about 0.6 ms per token, not 0.17. That would reopen this lane's original dispatch target.
  - Not proven: what separates 0.88 from 3.3 (a true data dependency, private vs shared storage, the backend's launch
    path) has not been measured.
- **5k residual.** The cold-KV harness grows 0.79 ms/token from ctx 150 to 5180, but the real step grows 0.93 (job Q:
  14.27 vs 13.34; job G: 0.92). About 0.14 ms/token at 5k is not explained by the attention core.

### Coordinator brief: item 2b, the dispatch cost (2026-11-22 ~09:45, verbatim in substance)

Go with (b), measure-only, with one change so that the result applies to the real engine:
1. **The deciding measurement runs in the real decode step, not the harness.** A debug env var inserts k empty dependent
   dispatches into the real decode chain at a realistic point (e.g. after attention), for k = 0, 50, 100, 200. Time
   ms/token at short and at 5k, ABAB after a warm-up, all in one quiet slot. The slope is the real cost per transition.
2. **Keep the harness chains** (empty, write-only, read-modify-write, read-previous-output; private vs shared), but
   configured as the engine is: the same encoder dispatch type, hazard-tracking mode and barrier scope. If these differ
   from what dispatch_cost assumed, write down which setting explains the gap.
3. **The prologue split, the combine fold and L's full-512 rerun ride along**, as long as they don't delay item 1.

**Decision rule:**
- Real-step slope >= 2 us per dispatch: reopen the dispatch-count target and send the coordinator ranked candidates, with
  ms/token = removable dispatches x slope. The coordinator takes the build decision.
- Slope < 1.5 us: do (a), the wrap-up.
- In between: report it, and the coordinator decides.

**Other instructions:**
- The 0.14 ms/token unexplained at 5k is noted (job Q results) and is not chased now.
- The env var is committed as debug-only, off by default, with no effect on plans or bits when unset.

### Job K (#11450, queued ~10:05; snapshot .scratch/snap11 = 88d28e48a): what a dependent dispatch costs in the real step

- **88d28e48a** adds `INTARSIA_INK_DEBUG_DISPATCHES=k` (debug only; unset = 0, no other effect). Each solo one-row decode
  step gets k extra `Add` [1, 32] bf16 dispatches, spread over the 48 visits right after each attention. They form a
  chain: the first reads the attention output, and each next one reads the previous output, ping-ponging between the heads
  of aproj and aout. wo and the attention-side ConvMix overwrite both right after, so the bits do not change.
  Documented in docs/tuning.md. bespin #11449: check, the CPU Inkling tests and docs_env_vars pass.
- **Job K** (raw/K_jobK.sh when archived):
  1. build;
  2. greedy md5 at 64 tokens, unset vs k = 200 (stop on a difference), and plan show of Qwen3-8B / Llama-3.1-8B against
     the base binary;
  3. **the deciding measurement**: warm-up, then k = 0 50 100 200 200 100 50 0 at short and at 5k. The least-squares
     slope of span ms/step on k is the cost per transition;
  4. ride-alongs:
     - the harness chains in the engine's setting and in dispatch_cost's (raw/Q_tools/dispcost.swift: shared vs private,
       tracked vs untracked, serial vs concurrent + barrier);
     - dispatch_cost itself;
     - the prologue split (noclip / noconv / nobias, token mode);
     - ink_attn_bench (L's full-512 row).
- **Engine setting, from the code:** `MetalBackend::alloc` makes buffers `StorageModeShared` with hazard tracking ON
  (`new_mtl_buffer(bytes, false)`). One `MTLDispatchTypeSerial` encoder, no barriers. dispatch_cost's buffers are
  untracked (`new_mtl_buffer(bytes, true)`), and the attention harness used private, tracked buffers. So hazard tracking
  is the first suspect for 0.88 vs 3.3 us.

### Coordinator add-on to job K (~09:35, verbatim in substance)

If job K shows that hazard tracking explains 0.88 vs 3.3 us, rank one more candidate next to the dispatch cuts: decode
buffers untracked (`MTLHazardTrackingModeUntracked`) with explicit barriers. That would cut the cost of all 543
transitions, not only the removed ones, and it would help every model on Metal. Report, measure and list only, no build:
1. from the chain matrix, the slope difference between tracked and untracked buffers in the engine's encoder setup;
2. whether the cost is CPU encoding or GPU time: the encode wall time and the GPU time of a step at k = 0 and k = 200;
3. where the engine creates and uses buffers (file:line), and which hazards it relies on Metal to track today: reads after
   writes across encoders, blits, and the CPU readback of sampled ids.

**Added to job K before it started:**
- dispcost.swift: an untracked conc+barrier column, and the host encode wall per dispatch for every setting.
- stepenc.py: per fb step, GPU busy and the lead (GPU end - last commit).
- Two plain-mode runs (`INTARSIA_TOKEN_FEEDBACK=0`, k = 0 and 200). In plain mode, cbstat `enc` gives host encode us per
  dispatch from the full 128-dispatch buffers. In fb mode the held buffer's commit time includes the hold, so encode is
  not separable there.
- From job Q's traces (k = 0): the host commits each step's last buffer about 13.2 ms before the GPU finishes it (lead
  13.230 ms short, 14.173 at 5k; GPU idle 0.01 %). Host encoding is therefore off the critical path today, and a
  transition's cost is GPU time. Job K checks that this still holds at k = 200.

**Item 3: buffers and the hazards Metal tracks today** (read-only survey of crates/intarsia-backend-metal/src and the
core readback callers; key lines spot-checked):
- **Creation.**
  - Every engine buffer is created at `lib.rs:1119` (`alloc` -> `new_mtl_buffer(bytes, false)`, `lib.rs:670-677`):
    StorageModeShared, tracked. That covers workspaces, KV, state and the `ws.tok` token ring.
  - mmap'd weights: `host.rs:83-87` (`newBufferWithBytesNoCopy`, shared, tracked, read-only).
  - Already untracked: the sync counter arena `sync.rs:32`; the decode.rs bench (`decode.rs:85-86`,
    `IK_METAL_TRACKED=1` measures tracked); `bench.rs:131-132`; the keep-warm scratch `host.rs:358`.
  - No MTLHeap.
- **Encoders.**
  - One `MTLDispatchTypeSerial` compute encoder on the decode path (`lib.rs:788-791`, `with_encoder_opt`). No blit
    encoders: copy and zero are compute dispatches (`fill.rs`), and an overlapping same-buffer copy synchronizes first
    (`lib.rs:1151-1157`).
  - Cuts:
    - `commit_every` (128, `INTARSIA_METAL_COMMIT_EVERY`) and `first_commit` (4);
    - in token feedback, the held cut of 16 dispatches, then the rest on `release` (`lib.rs:806-815, 1237`). cbstat
      shows 2 buffers per decode step in fb mode.
  - No memoryBarrier, MTLFence, MTLEvent or MTLSharedEvent. `useResources` appears only for row-table indirect sets
    (`lib.rs:832-860`).
- **Hazards that rely on tracking today.**
  - (a) Read-after-write across command buffers: the held cut -> rest within a step, and step N -> step N+1 (token
    feedback has two steps in flight). Inside one serial encoder, the dispatch order alone covers RAW.
  - (b) Blits: none on the decode path.
  - (c) CPU readback of the sampled id: `feedback_read` -> `read_after` (`lib.rs:1272-1293`). It waits
    (`waitUntilCompleted`) on the fence's command buffers, then memcpys the `ws.tok` slot. The next step reads the other
    ring slot, so this does not depend on tracking. The plain path's `download` drains first (`lib.rs:1134-1136`).
  - (d) Host writes (`upload`) always drain first (`lib.rs:1123`), so they do not depend on tracking.
  - (e) Row tables reach state by device address, which tracking cannot see, so those already rely on serial order plus
    `useResources`.
- **What an untracked switch would need.** Explicit ordering at each command-buffer boundary: an MTLFence or MTLEvent
  at the held cut and between steps, about 2 per step in fb mode. Inside the serial encoder, nothing extra is needed. The
  single flip point is `lib.rs:1119`. This is the hostgap lane's area (encoder and commit), so it is a listing only.

### Coordinator add-on 2 (~09:50): a concurrent encoder with barriers only at real dependencies

**The coordinator's reasoning.** Untracked buffers probably will not remove a transition's GPU cost, because order inside
the serial encoder comes from the dispatch type. The serial drain and refill will. So the ranking should include a
concurrent-dispatch encoder with `memoryBarrier(scope: .buffers)` only at real dependencies, letting independent neighbours
overlap. Measure and list only.

- **Chain matrix (dispcost.swift, before job K starts):**
  - It already had concurrent + barrier-after-every-dispatch columns (tracked and untracked).
  - Added: a concurrent no-barrier column, the overlap bound for independent neighbours (dependent chains race there, so
    it is timing only). Compiles (bespin #11452).
  - Comparing serial with conc+barrier on the dependent chains gives the per-transition cost of the barrier form.
- **How many of the ~543 transitions have no data dependency on the dispatch before them:** about **6 per step**.
  - Source: a read-only reading of `ink_pass_rows` and the Metal launchers (inkattn, inkmoe, inkmix, p12); the
    AttnResMix pair was spot-checked. Confidence is high on the dependencies and medium on the exact count.
  - Every visit is a strict read-after-write chain: in_proj -> attention dec (-> comb) -> wo -> ConvMix -> InkRoute ->
    experts gate+up -> down -> MLP-side ConvMix -> the next in_proj.
  - The blend stages (h GEMV -> gate -> norm) and LoopBlend's 4 stages are chains too.
  - At each of the 2 loop restarts there are 3 independent transitions:
    - restart copy 1 after the previous MLP ConvMix;
    - copy 2 after copy 1 (disjoint bank planes);
    - `AttnResMix(dst)` after `AttnResMix(msg_all)`: the second reads the stack and partial and writes `ws.dst`; the
      first writes `ws.msg`.
  - All 6 are tiny ops (about 4.6 KB copies, small AttnResMix).
  - The fusions already in place remove the classic independent neighbours. q / k / v / r / gate are one `InkInProj`
    GEMV, and gate / up are one expert kernel.
- **Implication for the ranking.** Overlap of independent neighbours is worth at most about 6 transitions per step. A
  concurrent encoder can only pay off if a barrier-separated transition is cheaper than a serial one. In dispatch_cost
  (untracked) it was not: 1.03 us with a barrier vs 0.88 serial. Job K's matrix measures this in the engine's tracked
  setting.

### Job K results (#11450, 10:06-10:10; raw/K_*)

**Bottom line: one more dependent dispatch costs about 1 us in the real decode step: 1.03 us at short (se 0.20) and
0.85 us at 5k (se 0.19).** That is below 1.5 us, so by the coordinator's rule this lane wraps up (option a). The 3.3 us
seen in the attention harness and L's bench floor does not occur in the real step. Hazard tracking makes no measurable
difference, and a concurrent encoder with barriers costs more per transition than the serial one.

- **Bits and plans.**
  - Greedy md5 at 64 tokens is equal unset vs k = 200 (65cde8f3..., the same as job G).
  - Every timing run's md5 is equal across k (short d03580fb..., 5k d108e0e9...).
  - plan show Qwen3-8B / Llama-3.1-8B is byte-identical to the base binary.
- **Real-step slope (span ms/step, two runs per k, order 0 50 100 200 200 100 50 0; dispatches/step 543 -> 743 as
  intended, 2 command buffers per step throughout):**

| k | short | 5k |
|---|---|---|
| 0 | 13.309, 13.359 | 14.322, 14.264 |
| 50 | 13.351, 13.355 | 14.294, 14.282 |
| 100 | 13.404, 13.366 | 14.333, 14.290 |
| 200 | 13.482, 13.587 | 14.434, 14.474 |
| **slope** | **1.03 us / dispatch** | **0.85 us / dispatch** |

  Box label: 13.954 at the start (the first run, cold), 13.447 at warm-up, 13.280 at the end.
- **CPU or GPU (add-on item 2).** It is GPU time.
  - In token-feedback mode the host commits each step's last buffer 13.2-14.4 ms before the GPU finishes it, at k = 0
    and at k = 200 (raw/K_stepenc.txt). GPU busy per step rises by 0.95 us per added dispatch at short and 0.78 at 5k.
  - Host encode (plain mode, full 128-dispatch buffers) is 1.19 us per real dispatch and about 0.96 averaged at k = 200.
    It is hidden while the host runs ahead.
- **Chain matrix in the engine's setting (add-on item 1; GPU us per dispatch, 1000 per command buffer, raw/K_dispcost.txt):**

| chain | shared tracked serial (engine) | shared untracked serial | private tracked serial | shared tracked conc+barrier | shared untracked conc+barrier | conc, no barrier |
|---|---|---|---|---|---|---|
| write | 0.666 | 0.664 | 0.665 | 1.084 | 1.012 | 0.124 |
| rmw | 0.940 | 0.880 | 0.917 | 1.398 | 1.161 | 0.121 |
| add3 (dispatch_cost's) | 0.820 | 0.846 | 0.829 | 1.303 | 1.292 | 0.127 |
| chain (reads previous output) | 0.944 | 0.946 | 0.951 | 1.357 | 1.497 | 0.129 |
| chain8 (+8 bound buffers) | 0.978 | 0.915 | 0.979 | 1.526 | 1.553 | 0.130 |

  - **Tracked vs untracked: no systematic difference** (-0.06 to +0.03 us). The one outlier is `empty` (1.057 vs 0.756),
    and `empty_b` reverses it. So hazard tracking does not explain the gap.
  - **The barrier form costs +0.4 to +0.6 us per dependent transition** over serial. Only the ~6 independent
    transitions per step could overlap (0.13 us). Net, a concurrent encoder would be about 0.2 ms/token slower.
  - dispatch_cost itself, rerun: add_bf16 0.92 us serial, 1.11 with barriers.
  - **Where 3.3 us came from: not established.** The real step (0.85-1.03 us) and this matrix (0.7-1.0 us) both
    disagree with it. Both 3.3 us figures were taken on a lightly loaded GPU with little warm-up (the harness floor right
    after cal; L's bench floor). The GPU clock state is a plausible cause, but unverified.
- **Ranking at the measured slope (ms/token = removable dispatches x ~1 us; all below the 0.3 bar):**

| Candidate | Dispatches removed per token | ms / token (short / 5k) | Note |
|---|---|---|---|
| reach the 350 target (any mix) | 193 | 0.20 / 0.16 | |
| blend tail h GEMV + gate + norm -> 1 | 92 | 0.09 / 0.08 | change 3 (fused tail) was slower net |
| attention comb folded into dec | 48 | 0.05 / 0.04 | the harness's nocomb (0.15 / 0.28) also counts comb's own work |
| ink_route folded into the ConvMix | 48 | 0.05 / 0.04 | |
| experts gate+up -> down in one launch | 48 | 0.05 / 0.04 | forward-progress caveat |
| decode buffers untracked | 0 | about 0 | matrix: no difference |
| concurrent encoder + barriers at real dependencies | 0 | about -0.2 (slower) | ~6 independent transitions per step |

- **Ride-alongs.**
  - **L's bench rerun:** full ctx 512 fused 23.12 us, against L's 24.34 and the harness's 22.52 (-2.6 %). With that, the
    harness is within 5 % at all four points: SWA 512 22.97, SWA 4096 31.65, full 4096 61.56. L's full-512 row was
    the outlier.
  - **Prologue split (token mode, ms/token vs ship; CAL PASS again):**

| Part removed | ctx 150 | ctx 512 | ctx 5180 |
|---|---|---|---|
| rel bias (r . relproj, 16-deep chain per distance) | **-0.242** | **-0.302** | **-0.258** |
| q / r clip + k clip / v norm | -0.104 | -0.070 | -0.081 |
| conv | +0.035 | +0.003 | -0.163 |
| all (nopro) | -0.413 | -0.361 | -0.486 |

    The rel bias is the largest part, but 0.24-0.30 is a bound for deleting it, not for computing it faster in the same
    order. It reaches 0.3 only at ctx 512. Under the rule this is no build.
- **Decision: wrap up (option a)**, per the coordinator's rule (slope < 1.5 us).

### Wrap-up job W (#11465, queued ~10:20; estimated start ~12:29)

- **Merge:** ink/metal bb2edc527 merged into ink/dfuse at 81bb8c4b8. The only code it brings is a test gate in
  inkling2_batch.rs and a HIP kernel, so Metal decode is unchanged. Job W's Metal test step is the check.
- **Arms:**
  - A = after: the tip 81bb8c4b8 (.scratch/snap12) with its defaults;
  - B = before: the same binary with the lane's four switches at 0. It isolates only the switchable changes. Two smaller
    changes sit in both arms: route rank selection (part of InkRoute, so off whenever route is off) and the in_proj
    rows-exact pin;
  - C = the brief's base a88db4620 (.scratch/snapC), included until the coordinator says otherwise.

  `INTARSIA_INK_MOE_T1` stays default off and so is in no arm; INTARSIA_INK_DEBUG_DISPATCHES is unset everywhere.
- **Steps (raw/W_jobW.sh):**
  1. Metal tests at the tip: the backend-metal Inkling tests (inkmoe, inkling2_attn, inkctrl, inkctrl_nometal4, inkmix,
     inkload) and the models tests inkling2 (G1 tiny), inkling2_batch and inkling2_mix_decode_cpu.
  2. Greedy md5 A vs B at 64 and 256 tokens.
  3. G2 short / long (decode mode) and G3 short / long at the tip, with the self-goldens byte-compared with job G's.
  4. plan show for Qwen3-8B / Llama-3.1-8B.
  5. For each arm, a normal-step cb trace (short, 5k) and job E's per-dispatch trace with disp.py: dispatches and controller
     ms per token.
  6. localeval speed (sizes 128 / 5000, gen 128, 3 reps; serve --max-batch 1 --max-seq 8192) in the order A B C C B A.
- **unanimis:** job K's finding is captured as record 46fa531e (about 1 us per dependent dispatch, tracking makes no
  difference, conc + barrier is dearer).
