# Inkling-2-Mini decode controller fusion on Metal (lane ink-dfuse, 2026-11-21)

Status: in progress (correctness and first timing done; per-switch attribution and the real-model bitwise gate queued). Branch `ink/dfuse` (from `ink/metal` a88db4620), worktree `.worktrees/ink-dfuse`. Not pushed.

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
| localeval speed decode, 136 / 5136 prompt tokens | 65.0 / 61.6 tok/s | 65.1 / 61.8 tok/s |
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

PENDING_ATTRIBUTION
