# Inkling-2-Mini prefill / decode overlap on HIP, one R9700 (lane ink-hip-overlap, 2026-11-26)

Branch `ink/hip-overlap` from the `ink/hip-bpf` head 2561d5957 (PR #538: packed prefill, default-on with `--max-batch`), in
`.worktrees/ink-hip-overlap`. User (via the coordinator, 2026-11-26): "build the prefill overlap too". Goal: cut the ITL p99 at 8 and
16 streams (0.59-0.65 s today, about one 2048-row prefill chunk) while every stream stays bitwise equal to its solo run.

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
