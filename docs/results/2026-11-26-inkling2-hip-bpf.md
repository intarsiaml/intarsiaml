# Inkling-2-Mini packed prefill across streams on HIP, one R9700 (lane ink-hip-bpf, 2026-11-26)

Branch `ink/hip-bpf` from main `80719483a` (#529 merged). Goal (user, verbatim, 2026-11-26): "make prefill batch across streams. the
main goal is to maximize speed while keeping accuracy high to exact". Bar: with `serve --max-batch N` on HIP, several streams' prompt
chunks prefill in one forward pass, and every stream's reply stays bitwise its solo run's. Raw: `docs/results/2026-11-26-inkling2-hip-bpf/raw/`.

## Briefs received

- 2026-11-26 (coordinator, lane brief): build packed prefill for Inkling on HIP; design first (per-sequence state, per-kernel row
  independence, rows-exact rule, chunk boundaries, R3d overlap, expected gain); gates: unit packed == solo bitwise (tiny cpu/hip and
  real shapes on hip, 17..2048 rows per segment, 2..16 segments), serve 16/16 real prompts at 4/8/16 equal to solo (also with prefix
  reuse if resumed prompts pack), solo md5 unchanged (64/256, short/4k), G2 short/long equal to main, tests (`inkling2_batch` hip
  tiny, `inkling2`, `scheduler`, admission for every touched op), speed paired vs main's batching server (3 reps, VRAM peak per N):
  real-prompt sweep 4/8/16, 1k filler 1/4/8/16, pure-prefill rows 16x49, 8x512, 4x1024, 2x2048. Radiance: at most 8 windows,
  each <= 5:00, <= 30:00 total. Overlaps: ink-hip-p12 owns `kernels/hip/ink_proj.hip`, the wo/unembed GEMVs in `matmul.hip`,
  `quant/p12.rs`; ink-hip-verify owns the decode/rows expert forms in `ink_moe.hip` and `ink_route*.hip`, `ink_spec.hip` (the
  prefill forms in `ink_moe.hip`, WMMA and PF2, are this lane's, in separate functions); ink-sampled owns `spec.rs`, `dspark.rs`;
  `kernels.rs` append only; rebase onto `ink/hip-batch-mixmode` if it lands first.
- 2026-11-26 (coordinator, mid-lane; copied verbatim in substance): **reuse the existing 35B HIP packing path and its numbers.**
  - Existing 35B HIP packing: default-on for HIP at `crates/intarsia-backend-hip/src/device.rs:1644`; `engine/batch.rs:1214-1217`
    copies it into `cfg.pack_prefill`, and it turns off when a prefix store is configured. `engine/batch.rs:558` calls
    `Model::forward_packed`: row-wise kernels run once over the concatenated rows; attention, KV append and the recurrence run once
    per prompt (`model.rs:3607`). Needs a single rank (`model.rs:3588 pack_capable`). A packed segment needs at least 16 rows. With
    `pack_half` it packs at most half of the waiting prompts (`scheduler.rs:432`); total rows capped by `pack_rows_max`
    (`model.rs:3582`). Decode rows and prefill chunks never share a forward pass (`scheduler.rs:30-31`).
  - 35B measured (docs/results/2026-11-05-hip-pack-default.md:61-63; A = packing, B = `INTARSIA_PACK_PREFILL=0`, ~1k prompts, 128
    greedy tokens, tp1): 4 streams -0.2 % aggregate, TTFT p50 0.36 vs 0.36 s; 8 streams +4.6 %, TTFT 0.56 vs 0.62 s; 16 streams
    +9.9 %, TTFT 1.05 vs 1.19 s. Enough to ship it default-on.
  - **Gate change** (replaces ">= 10 % at 8 and 16 streams"): default-on with `--max-batch` if all four hold: (1) every stream bitwise
    equal to its solo run (16/16 real prompts); (2) the solo path unchanged; (3) aggregate or TTFT p50 at 16 streams >= 5 % better
    than with packing off (beyond the paired run-to-run spread); (4) no stream count more than 1 % worse on aggregate or TTFT p50.
    Otherwise opt-in, and report the numbers. Why: max speed at exact output; a bitwise change that is never slower should ship.
  - Inkling points: prefill ~3.1k tok/s vs 12k for the 35B, so TTFT is the bigger lever (4.17 s at B16 vs 1.05 s). The real-prompt
    set (49-4253 tokens) has many short prompts, where packing should win most. Report TTFT p50 / p95 next to aggregate tok/s for
    B 1 / 4 / 8 / 16.

- 2026-11-26 ~16:50 EST (coordinator heads-up, in substance): the default-on PF2 prefill form (`INTARSIA_HIP_INK_PF2`, mask 151,
  from #528) fails the HIP tiny test `inkling2_tiny_g1b_teacher_forced_bf16` (a 40-row pass on tiny shapes, chunk 64); its errors
  change from run to run, and with `INTARSIA_HIP_INK_PF2=0` the test passes. Lane ink-hip-p12 bisects and fixes it on
  `ink/hip-pf2-tiny`. Real-model md5 matches main at the lengths tested so far, but packing may create row counts or segment
  shapes solo prefill never runs. **Any non-bitwise or run-to-run-varying packed result: rerun it with `INTARSIA_HIP_INK_PF2=0`
  before debugging this lane's code, and note the result here.** No other change to the brief.

## 0. Prior work (step 0)

`git log --all -i --grep` over pack / packed prefill / pack_min_rows / pack_rows, the `ink/*` and `lane-*` branches, and the results docs:

- **Inkling, Metal (ink-batch brief 47, `11b084a45`, docs/results/2026-11-21-inkling2-batch.md section 47): scoped, no-go.** On Metal a
  1x2048 pass cost 845-893 ms against 902-911 ms for 2x1024 (two passes), so packing two 1k prompts would save at most 2-7 % of their
  prefill; the lever taken instead was busy chunk 256 -> 2048 (+18..24 % predicted at 16 streams). It lists what packing would need:
  `pack_capable` drops the `ink` exclusion; `run_layers` rejects `Seqs::Packed` for Inkling; per-segment `pos0` / slot for the
  stateful ops (SconvRes, ConvMix, InkKvPrologue, RelSinkAttn, ring writes); MoE tile choice by rows per expert must be proven or
  run per segment; bitwise rule: each sequence equals its solo prefill with the same chunk schedule (`tests/packed_rows.rs` as
  template). Nothing was built.
- **Qwen3.6-35B, CUDA (srv16, srv16b, lane pf; #238, #310) and HIP (hg, fh; #308):** `Model::forward_packed` (row-wise ops over the
  concatenated rows; attention, KV append, GatedDeltaNet once per prompt), `BatchModel::prefill_pack`, `pack_rows_max`,
  `pack_min_rows` (HIP 16: the 35B MoE GEMV / tensor-core switch; CUDA 33), `pack_whole_prompts` (CUDA only: cuBLAS tiling by M),
  `INTARSIA_PACK_ROWS` (3072) / `INTARSIA_PACK_HALF`. HIP default-on since fh (2026-11-05: +9.9 % at 16 streams, see the brief).
  This path assumes a recurrent state: the last prompt token runs alone after the packed bodies (`SlotModel::prefill_pack`).
- **Inkling on HIP:** no packed or varlen prefill on any `ink/*` branch. #529 (batched decode) gives the row-table (`RT`) forms of
  `ConvMix` and `InkAttnDecode` (one row per sequence) in their own translation units (`ink_mix_rows.hip`, `ink_attn_rows.hip`), the
  lesson that RT instantiations in the solo TU change the solo kernels' ISA (-0.43 % decode), and the measurement that HIP Inkling
  prefill is **not chunk-invariant** (busy chunk 256 broke batched == solo, 8 / 16).

## 1. Design

### 1.0 Facts the design rests on (code read, nothing run)

- Inkling-2 has no `StateSlot::Recurrent` (rings and attention KV only), so `SequenceState::has_recurrent()` is false: its prefill
  runs plain chunks `C, C, .., rest` with the head on the final chunk's last row (`engine/batch.rs` `run_prefill`), not the
  recurrent "body + lone last token" schedule. `SlotModel::pack_rows_max` returns 0 for it (it wants `has_recurrent`), and
  `Model::pack_capable` excludes `ink`. So nothing packs today, and HIP's `pack_prefill_default() == true` would turn Inkling
  packing on by default the moment it becomes capable: the model must gate it itself (opt-in until the gate says otherwise).
- Per visit (48 visits, 24 layers): `AttnResMix` / `LoopBlend` / `RmsNorm` (the next visit's x planes) -> `InkInProj` ->
  `InkKvPrologue` + `RelSinkAttn` (HIP prefill: the pair; op 75 is admitted only up to 16 rows) -> `Matmul` wo -> `ConvMix`
  (attention side) -> router `Matmul` (f32) / `Logits` -> `MoeRoute` -> `MoeExperts` / `MoeExpertsMxfp8` -> `ConvMix` (MLP side,
  fused with the next visit's mix).
- Chunk cost on HIP (pf2 profile, 2048 rows, ctx 2k, 679 ms): experts 314 ms (WMMA), prologue 80, flash 80, in-proj + wo 52,
  mix 43, blend 46, router 2, idle 60. SIMT experts (hip-moe doc, per call, NVFP4): T 16 0.62 ms, 64 1.62 ms, 256 5.35 ms,
  2048 ~38 ms: about 2.3-3.2 us per (token, expert) pair, i.e. compute-bound per pair; WMMA at 2048 rows 6.3 ms.

### 1.1 (a) What is per-sequence in an Inkling prefill pass

- Conv rings (`RingKind::InkConv`): K/V conv windows (`kvconv_slot`, 2 planes, read and written by `InkKvPrologue`) and the
  attention / MLP sublayer conv windows (`hconv_slot`, plane 0 by the attention-side `ConvMix`, plane 1 by the MLP-side one).
  `K - 1 + 16` rows.
- SWA KV rings (`RingKind::InkKv`, window 1024 + 16) and full-attention KV (`AttnKv`), read and written by `RelSinkAttn`.
- Rel-bias and the sink: computed per row from the row's own q / r (`InkKvPrologue`), consumed by `RelSinkAttn` over the
  sequence's own keys; position-dependent (`pos0`, log-position tau). Per-sequence through `pos0` and the caches.
- The loop / blend state (bank, messages, block stack, partials): workspace planes `[N, T, D]` that live within one pass, row
  local. Not per-sequence state; but their kernels address planes at stride `T * D` of the call's own `T`.
- The head (final norm row -> unembed -> argmax / logits / Gumbel): the last row of each prompt's final chunk.
- Ring snapshot (`st.ring_snapshot`) after a pass that ends a prompt; `prefill_chunk_rows` reset.

### 1.2 (b) Per HIP prefill kernel: do a row's bits depend on the other rows of the call?

(explorer read + my read of the launchers; to be proven by the unit gate)

| Kernel (op) | Row bits independent of the other rows? | Condition / evidence |
|---|---|---|
| `ink_moe_wmma_kernel` (MoeExperts / Mxfp8, ordinary names) | yes, within the WMMA form | each pair its own B column, fixed 64-k loop, no split-K, no output atomics; pair tile 64 under the rule, `wm_zp` / PF2 pieces only repartition the grid; gather sums k in slot order (ink_moe.hip 1079-1150, 728-737) |
| SIMT pair kernels (`_rows` twins at every T; ordinary names below the rule) | yes (rows-exact: a T-row call is bitwise T one-row calls) | header 12-20; `ink_moe_rows_exact` test |
| **form choice WMMA vs SIMT** | **no: per call** | ordinary names take WMMA iff `P = T K > 512` and `P >= 16 E` (`T >= 256` at top-8 of 128) and lean and H, I % 64 (ink_moe.hip 1132-1141). A 17-255-row segment alone is SIMT; inside a >= 256-row packed call it would be WMMA (other f32 order). **So the packed MoE must run per form: one ordinary call over the WMMA segments' rows, one SIMT (`_rows`) call over the others.** |
| `gemm_wmma_kernel` (wo `matmul_bf16`, router `logits_bf16` f32, `ink_in_proj_bf16_matmul`) | yes | tile by `M >= 128`, same k loop, no split-K; GEMV only at M <= 4 (never for >= 17-row segments); `gemm_simple_kernel` only for K % 32 or misaligned x (row offsets of `[T, K]` slices stay 16-byte aligned) |
| `moe_route` | yes | one wave per row |
| Embedding, RmsNorm | yes | one block per row; vec form by alignment (row slices aligned) |
| `ink_row_kernel` (AttnResMix, LoopBlend, ConvMix mix / blend parts) | yes for the row-local parts | fast form for <= 16 rows is bitwise the other; blend chunks of 256 rows only offset t0 (the <= 2-row-chunk ticket path to be checked) |
| ConvMix conv (`ink_row_kernel` conv taps, `ring_after`) | **no** | one `pos0` and one ring per call: taps at `qp >= pos0` read the call's own x rows, so a packed call would read the previous segment's rows as history; `ring_after` writes the last `min(T, Rr)` rows of the call to one ring (ink_mix.hip 188-307, 979-1008) |
| `ink_prologue_k` / `ink_rel_table_k` / `ink_flash_k` / KV + ring writes (InkKvPrologue, RelSinkAttn) | per segment: one `pos0`, one ring / cache per call | they read only their call's rows plus the caches; a call on a row slice `[lo, lo + len)` of the `[T, ..]` buffers is the solo call (no row-0 assumption; 16-byte alignment holds) |
| Plane stride | n/a | `[N, T, D]` planes are addressed at stride `T * D` with `T` = the call's rows (`src->shape[1] == T` is checked): a segment's rows of a packed `[N, Ttot, D]` plane set cannot be passed as a solo call |

### 1.3 (c) The rows-exact rule

`ink_rows_exact(hip, t)` marks passes of 2..=16 rows (decode numerics). A solo chunk of <= 16 rows keeps them. So **`pack_min_rows`
= 17 for Inkling on HIP** (a segment of >= 17 rows alone runs the fast prefill kernels, and so does any packed pass of >= 34 rows).

### 1.4 (d) Chunk boundaries

Not chunk-invariant (measured by #529: busy chunk 256 gave 8 / 16). The scheduler already packs exactly the solo schedule's chunks
(`prefill_plan` per request; HIP's `pack_whole_prompts()` is false, so later chunks and resumed prompts may pack: each segment is
its own chunk at its own `pos0`). Inkling has no lone last token: each segment is the whole chunk, and a final chunk's head is
computed from its own last row of the packed pass.

### 1.5 (e) Overlap (R3d)

Not on HIP Inkling today, and not built in this lane. What it needs:
1. A decode twin (`Model::decode_twin`) for Inkling: its own `InkWs` of 16 rows, own row table / `batch_pos`, own blend / flash / MoE
   scratch. Today `decode_twin` refuses a model with a DSpark head (`self.dflash.is_some()`), and builds only the generic workspace.
2. **Device-global tickets**: `g_gate_ctr` (ink_mix.hip:57, the fused blend norm's last-block election), `ink_dec_tick`
   (ink_attn.hip:659) and `g_route_ctr` (ink_route_fused.hip:17) are one set per device. A decode step on stream 1 beside a prefill
   chunk on stream 0 can run both kernels at once and corrupt each other's counts. They need one set per stream (or per call, in
   the call's scratch).
3. Every Inkling launcher already takes the call's stream; the second (high-priority) HIP stream exists for the 35B twin.
It would not add throughput (the prefill kernels already fill the GPU), only cut ITL p99 (a decode step waits for at most a
kernel, not a 0.7 s chunk). The batch lane's 500-1000 ms ITL p99 is that wait; packing makes each pass longer, so it can raise
ITL p99 (reported, not gated). Overlap is the follow-up that fixes ITL; packing is the one that fixes TTFT and aggregate.

### 1.6 (f) Expected gain

Per-pass cost on HIP is far from linear in rows: TTFT at 1 stream with ~1k prompts is 0.45 s (one ~1k pass, ~2.3k tok/s), while
4k prefill runs at 3084 tok/s (two ~2048-row passes, ~0.66 s each). A line through those two points: t(T) ~= 0.20 s + 0.225 ms x T.
The fixed part is mostly the MoE (WMMA blocks dequantise their expert's rows per 64-pair tile, so a call costs per tile, not per
pair: 1024 rows ~ 1.5 tiles per expert, 2048 ~ 2.5) plus ~1500 launches per pass. Packing amortises it:

| Workload (16 streams) | today | packed, cap 2048 | packed, cap 4096 (opt-in workspace) |
|---|---|---|---|
| 16 x 1k filler: prefill passes x cost | 16 x 0.43 = 6.9 s (TTFT max 7.2 measured) | 8 x 0.66 = 5.3 s (-23 %) | 4 x ~1.12 = 4.5 s (-35 %) |
| real prompts: 21 passes (49..2048 rows) | ~10 s of the 17.7 s wall | ~14 passes, est. -10..20 % prefill | est. -20..30 % prefill |

At 16 real prompts that is +5..12 % aggregate (prefill is ~55 % of the wall) and a larger TTFT median gain; at 4 streams little
(`pack_half`: at most half the waiting prompts per pass). Short (< 256-row) segments gain least: their MoE must stay on the SIMT
pair kernels (their solo form), which cost per pair (2.3-4.8 us), so packing them only saves launches and some weight reuse.

### 1.7 What gets built (decided before code)

- **Stateful ops per segment.** `InkKvPrologue` and `RelSinkAttn` run once per segment on row slices of the `[T, ..]` buffers
  (q, k, v, rel, pq, pk, pv, o are all row-major `[T, ..]`; every slice offset of these widths is 16-byte aligned), each on its own
  sequence's rings / KV at its own `pos0`: that is the solo call, no kernel change.
- **`ConvMix` segment form (new kernel form).** Its planes (`src`, `msg`, `mix` `[N, T, D]`) are addressed at stride `T * D`, so a
  segment cannot be a slice. New: `OpCall::segs` (a segment table: per segment first row, rows, first position, and its state
  address in the batched step's row-table layout) and `conv_mix_bf16_segs`: one launch over all rows, the row kernel's conv reads
  history from its own segment's rows (`own0` = the segment's first position) and older positions from its own segment's ring, then
  one ring-write launch (each segment's last `min(len, Rr)` rows; segments have >= 17 rows, so the solo call defers its ring write
  too). The template gets a `SEG` parameter; the SEG instantiations live in their own translation unit (`ink_mix_segs.hip`), so the
  solo and row-table modules are compiled from unchanged code (checked by an ISA diff, as #529 did). cpu: `_segs` twins that run
  each segment as the solo call (planes gathered and scattered back).
- **MoE by form.** The HIP launcher takes WMMA for an ordinary call iff `P = T K > 512` and `P >= 16 E` (lean, H and I % 64):
  T >= 256 rows here. A new query (`ik_ink_moe_split_rows`, ink_moe.hip, separate function; `Backend::moe_split_rows`) returns that
  T. The packed pass orders segments longest first; all segments >= 256 rows run as one WMMA call (row bits independent of the
  other rows within the form: own B column per pair, fixed k loop, no split-K, gather in slot order), the shorter ones in SIMT calls
  of < 256 rows each (rows-exact, so any grouping is bitwise). No kernel change.
- **Head per final segment**: the solo one-row head (`head_argmax` / `head_logits` / `head_gumbel`) on that segment's last row.
- **`pack_min_rows` = 17** on HIP for Inkling (section 1.3): a segment's solo chunk then never takes the rows-exact decode numerics.
  **`pack_rows_max`** = the Inkling workspace (the prefill chunk, 2048, so no extra VRAM by default); `INTARSIA_INK_PACK_ROWS`
  grows the workspace of a batching model to hold larger packed passes (2 x 2048, 4 x 1024).
- **Scheduler**: Inkling's final chunk has no lone last token, so a final chunk's body is the whole chunk
  (`BatchModel::pack_last_alone`, default true = today's recurrent rule).
- **Knob**: `INTARSIA_INK_PACK` (0 / 1; default decided by gate 5). An item whose solo prefill would not be one pass (peel, long
  chunk), or a drafting / MTP state, prefills alone beside the pack.

## 2. Build log (what exists, how it was checked)

Code (branch `ink/hip-bpf`):
- `OpCall::segs` / `SegTab` (ops.rs), `KernelImpl::seg_table` (registry: selected only with a segment table, admitted through its
  base `<name>` without `_segs`), `inkling2::ops::SEG_TABLE_OPS = [ConvMix]` and the contract ("Segment tables").
- cpu: `<kernel>_segs` twins of ConvMix run each segment as the solo call (`rowtab::run_segs`).
- HIP: `conv_mix_bf16_segs` in its own translation unit `kernels/hip/ink_mix_segs.hip` (the SEG instantiations of
  `ink_row_kernel` and `ink_ring_seg_kernel` live there only), `ik_ink_moe_split_rows` + `Backend::moe_split_rows` (the experts'
  tensor-core threshold: 256 rows at Inkling-2-Mini's shapes, measured in W1).
- Model: `Model::ink_prefill_packed` (row-wise ops over all rows; `InkKvPrologue` + `RelSinkAttn` per segment on row slices; ConvMix
  segment table; the experts once per range of one form, longest segments first; the one-row head per final segment),
  `ink_pack_capable`, `ink_pack_layout`, `pack_min_rows` >= 17 for Inkling, `INTARSIA_INK_PACK` (default off until gate 5) and
  `INTARSIA_INK_PACK_ROWS` (a larger packed workspace; the prefill chunk stays the recipe's).
- Engine / scheduler: `SlotModel::ink_prefill_pack`, `BatchModel::pack_last_alone` (false for Inkling: a final chunk's body is
  the whole chunk), the pack body uses the long-prompt chunk a request would start with.

Checks without a GPU (tatooine cpu slots):

| fleetq | what | result |
|---|---|---|
| #12163 | main 80719483a cli (`.scratch/bin/intarsia-main`, md5 2f41908f) | built |
| #12201, #12208, #12210 | ISA of every kernel of `ink_mix.hip`, `ink_mix_rows.hip`, `ink_moe.hip`, main vs branch (instructions + VGPR / SGPR / LDS / scratch / kernarg) | **31 / 31, 55 / 55, 427 / 427 identical**; `ink_mix_segs.hip`: 56 kernels, 24 SEG instantiations (new) |
| #12210 | `intarsia-core` lib (scheduler, registry incl. the segment-table selection test, ops) | 269 passed |
| #12208 | `intarsia-backend-hip` lib, `inkling2` + `segment` filter (registration of `conv_mix_bf16_segs`) | 2 passed |
| #12210, #12215 | `inkling2_packed` on the cpu reference: packs of 2 / 3 / 4 / 16 segments, 17 .. 2048 rows, resumed segments, 4096-row pass, blend tails; heads, every state byte, `restorable_len`, the decode step after; the batching engine (4 slots, 9 requests) == the solo engine with packed passes counted | 4 / 4 passed |

## 3. GPU windows (radiance downtime)

| Window | fleetq | down (stop -> answers) | what |
|---|---|---|---|
| W1 | #12216 | 16:44:35 -> 16:46:50 = **2:15** | unit gates on hip (b5) + admission prime |
| W2 | #12224 | 16:59:32 -> 17:00:24 = **0:52** | packed != solo diagnosis (b7): `inkling2_packed` at the defaults and with `INTARSIA_HIP_INK_PF2=0`, main's tiny suite |

### W1 (b5): packed != solo on HIP

- `inkling2_packed` hip: the refusal test passes; tiny `2 x 17` (20 rows each after the 4-row alignment of tiny's 972-byte rows)
  **differs at segment 0's head**; the engine test differs at request 0; the real-shape test refused its first pack (message not
  captured: the stage kept only `panicked` lines). `moe_split_rows` = 256 at real shapes, `pack_rows_max` 2737 there.
- `inkling2` hip tiny: 14 / 17: the two screened-head tests (no HIP `HeadScreen`, known) and **`g1b_teacher_forced_bf16`**, which
  the coordinator's heads-up attributes to PF2 mask 151 on tiny shapes (a 40-row pass: the same row count as this tiny pack).
  `inkling2_batch` did not run (cargo stops after the first failing test binary; now `--no-fail-fast`).
- Admission prime: the `--max-batch 16` server with `INTARSIA_INK_PACK=1` came up (101 s with 134 admission cases, then cached),
  logging "inkling2 packed prefill: on, up to 2048 rows per pass, segments of 17+ rows"; the solo server came up in 17 s.

### W2 (b7): the PF2 tiny-shape fault, not packing

Raw: `raw/w2-diag.log`. b7 = b5 + the per-op dump of a packed pass (`Model::ink_dump_packed`) keyed per pass, not per segment.

| Check (card) | PF2 default (mask 151) | `INTARSIA_HIP_INK_PF2=0` |
|---|---|---|
| `inkling2_packed` real shapes (card 0): 17 + 2048, 4 segments around the expert split (255 / 256 / 257 / 17), blend tails (258 / 261 / 513) | **packed == solo** (head, every state byte, `restorable_len`, the decode step after) | **packed == solo** |
| per-op dump, real 17 + 300 and 257 + 600 (one resumed) | (not reached: the tiny case before it failed) | **1636 / 1636 tensors bitwise, each** |
| tiny 2 x 17 .. 16 mixed (card 0) | differs from visit 0 on (`v00.rel` first: the prologue) | **packed == solo** (2 x 17, 17 / 40 / 300, 2048 / 1000 / 17) |
| per-op dump, tiny 2 x 20 and 40 + 300 | 532 of 548 tensors differ, from `v00.rel` | **548 / 548 bitwise, each** |
| packed engine (4 slots, 9 requests) == solo engine (tiny) | differs | **equal** |
| (card 1) main 80719483a's `inkling2` tiny suite | **g1b_teacher_forced_bf16 FAILS on main** (+ the 2 known screened-head tests) | branch: 15 / 17 (g1b passes; the 2 screened-head tests fail on main too) |
| branch `inkling2_batch` tiny (card 1) | 8 / 8 | |

So the packed pass is bitwise its solo passes wherever PF2 is off, and at real shapes with PF2 on; the tiny-shape failures with PF2 on
are main's PF2 fault (the heads-up: run-to-run varying at tiny shapes; lane ink-hip-p12 owns the fix). Two items left from W2, both fixed
in b8: (1) the opt-in 4096-row workspace (`INTARSIA_INK_PACK_ROWS=4096`) refused a 2 x 2048 pass at real shapes ("kernel needs 226625024
scratch bytes but the plan reserved 113368832"): the plan skipped its 4096-row probe because one sequence's KV room (2737 tokens) is
smaller. b8 adds a plan probe of the largest packed pass (`Phase::Pack`: fresh phantom sequences of at most a chunk each, the real
packed layers recorded), so the scratch and admission cover it. (2) A test case summed to 4097 rows after the 4-row alignment of the
tiny GPU runs (test fix).

