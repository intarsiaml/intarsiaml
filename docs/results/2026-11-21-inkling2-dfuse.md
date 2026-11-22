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
  tip that has the NE fix (done 06:05: merged ink/metal 1c1ee202c as 10cf8988d, which includes 7eca1b012).

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

  After taking off the ~1.7 us per dispatch that the measurement adds, the row kernel's stream-over-bench gap is about
  0.5-0.75 ms per token. That is the ceiling: the row kernel running in the stream as fast as it runs with its operands
  cached.
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
    scratch [row, NV, nsg].
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
- **Step 1, before building (needs the go-ahead): a probe.** Add an `--evict MB` mode to `ink_ctrl_bench` that streams a
  buffer of about 256 MB between units, so the bench reproduces the stream's 13-18 us. Then time a phase-1-only split
  prototype in the bench, with no model run and no shipped kernel.
  - Go if the evict-mode bench reproduces the stream within about 2 us, and the split saves at least 3 us per call (at
    least 0.3 ms per token over 96 calls).
  - Otherwise stop and report what the probe shows.
- **Alternative held back:** prefetching the next visit's controller rows from the previous row kernel. It is a cross-op
  change (contract note) with the same ceiling, so it is only worth it if the probe shows the split cannot hide the cold
  rounds.

