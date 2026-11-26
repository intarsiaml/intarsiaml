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

### 1.5 Plan

1. Inkling twin (`InkModel` weights behind `Arc`, `InkModel::twin` allocating its own `InkWs`), `decode_twin` for Inkling; cpu test:
   a twin's batched decode step == the main model's, bitwise.
2. Per-lane tickets (ink_mix, ink_attn, ink_route_fused), lane from the launch stream; per-stream capture flag.
3. Packed in-flight prefill (`prefill_begin_packed`), scheduler: the overlap branch picks the packed set `prefill_packed` would.
4. Default: on for Inkling-2 on HIP with `--max-batch` > 1 only if gates 1-4 pass; opt-in (`INTARSIA_OVERLAP=1`) until then.
5. Gates on tatooine in windows of < 5 min (30 min in total).
