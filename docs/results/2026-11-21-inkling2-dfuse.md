# Inkling-2-Mini decode controller fusion on Metal (lane ink-dfuse, 2026-11-21)

Status: in progress. Changes 1-5 are committed, default off, and bitwise on the real model. G2 / G3 pass for 1-3 (#10952). Per-switch timing (#10964): best = route + msg + hoist + norm_c is -0.67 ms per step. The tail is dropped. Job E (#11003) confirms with 6 reps and runs the final gates before any default changes. Branch `ink/dfuse` (from `ink/metal` a88db4620), worktree `.worktrees/ink-dfuse`. Not pushed.

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
us in the warm microbench. At T <= 16 a row is one threadgroup (768 threads at D 2304), so nothing hides its latency, and its
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
keeps the loop. Every NE 3 instance admits 1024 threads (bespin runtime compile), and the switch falls back to the plain
instance when a pipeline admits fewer threads than the row needs. The row-table form of ink/batch only changes the pos / ring
pointer lines above the loop, so the merge stays local.

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

PENDING_ATTRIBUTION
