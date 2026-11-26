# Inkling-2-Mini packed prefill across streams on HIP, one R9700 (lane ink-hip-bpf, 2026-11-26)

Branch `ink/hip-bpf` (PR #538) from main `80719483a` (#529 merged), rebased onto 5367023d2. Goal (user, verbatim, 2026-11-26): "make prefill batch across streams. the
main goal is to maximize speed while keeping accuracy high to exact". Bar: with `serve --max-batch N` on HIP, several streams' prompt
chunks prefill in one forward pass, and every stream's reply stays bitwise its solo run's. Raw: `docs/results/2026-11-26-inkling2-hip-bpf/raw/`.

## Summary

**Packed prefill for Inkling-2 on HIP works and is bitwise: every stream's reply equals its solo run's.** With
`INTARSIA_INK_PACK=1` and `serve --max-batch N`, the waiting prompts' next chunks (at least 17 rows each) run as one forward pass.
The row-wise kernels run once over all rows. The attention pair runs once per segment, on that segment's own rings, KV and position.
`ConvMix` runs in a new segment-table form (`kernels/hip/ink_mix_segs.hip`). The experts run once per run of segments that take the
same expert form solo. Each final chunk's head runs alone on its last row. It **ships opt-in** (section 4.5). The recommended config
for `--max-batch` servers is `INTARSIA_INK_PACK=1 INTARSIA_INK_PACK_ROWS=4096` (+606 MiB VRAM). With the default 2048-row passes,
two ~1k prompts never fit one pass, so there is almost no gain.

Speed, paired on card 0 (A = main 6517d98f4's batching server, which does not pack; R = this branch with the recommended config;
2 paired reps of R, 3 of A; ratio R / A with its 95 % interval):

| streams | 1k filler: agg tok/s A -> R | R / A | TTFT p50 s A -> R | real prompts at once: agg A -> R | R / A | TTFT p50 A -> R | ITL p50 / p99 s A -> R |
|---|---|---|---|---|---|---|---|
| 1 | 61.9 -> 61.7 | 0.996 | 0.41 -> 0.39 | (not run; one at a time: solo server 16 / 16 equal) | | | |
| 4 | 111.2 -> 117.4 | **1.056** [0.966, 1.147] | 1.32 -> 1.35 | 97.3 -> 99.8 | **1.041** [0.994, 1.088] | 1.66 -> 1.42 | 0.024 / 0.483 -> 0.024 / 0.173 |
| 8 | 140.9 -> 152.3 | **1.081** [1.066, 1.096] | 2.16 -> 2.14 | 114.0 -> 116.2 | **1.032** [1.024, 1.040] | 2.11 -> 2.08 | 0.034 / 0.591 -> 0.034 / 0.587 |
| 16 | 154.4 -> 168.9 | **1.095** [1.076, 1.114] | 3.67 -> 3.37 (**0.916** [0.870, 0.962]) | 122.0 -> 128.0 | **1.050** [1.000, 1.099] | 3.29 -> 2.57 (**0.779** [0.569, 0.990]) | 0.058 / 0.649 -> 0.058 / 0.653 |

Pure prefill, prompt tok/s A -> R: 16 x 49 560 -> 634 (1.14), 8 x 512 1913 -> 2834 (**1.48**), 4 x 1024 2897 -> 3230 (1.12), 2 x 2048
2799 -> 2783 (1.00: `pack_half` sends two waiting prompts one at a time). VRAM peak 28 529 -> 29 135 MiB. Packing at 2048 rows (B):
real 16 at once 1.016, filler 16 streams 0.994, 8 x 512 1.28.

| Gate | Result |
|---|---|
| 1 unit: packed pass == K solo `prefill_chunk` (head, KV, every ring byte, `restorable_len`, the next decode step) | cpu tiny 5 / 5 tests (2 .. 16 segments, 17 .. 2048 rows, resumed, 4096-row pass, blend tails; per-op dump 548 / 548). **hip real shapes 7 / 7 cases** (W8: up to 16 segments, 2 x 2048, 6 segments 4029 rows, resumed), plus 3 cases on b7 (W2) and the per-op dumps 1636 / 1636 x 2. hip tiny: 4 / 4 tests with `INTARSIA_HIP_INK_PF2=0`. At the PF2 default on b10 they fail from the first prologue, which is main's PF2 tiny-shape fault: g1b failed on main too (W2), fixed by #537 (merged after W8, in this branch's base; not re-run here) |
| 2 serve, 16 real prompts vs the solo server | **16 / 16 at 4 at once (4 waves), 16 / 16 at 8 (2 waves), 16 / 16 at 16** (W5); 16 / 16 with the 4096-row workspace; in the speed windows every packed arm's replies equal main's on the same requests, **232 / 232**. Prefix reuse: the HIP batching engine never resumes an Inkling-2 prompt (`prefix_reuse` off: no batched snapshots), so resumed packs do not arise at serve level; they are unit-tested |
| 3 solo unchanged | `generate` greedy md5 **8 / 8 = main** (spec off and the DSpark default, short / 4k, 64 / 256; W3). Solo server (no `--max-batch`) **16 / 16 = main's** (W4). G2 short 4.491e-3 / long 4.083e-3: main's values exactly |
| 4 tests | `inkling2_batch` hip tiny 9 / 9; `inkling2` hip tiny 14 / 17 (the same 3 as main: g1b (PF2), 2 screened-head); `intarsia-core` lib 270 (scheduler, registry, ops); `plan` 10; `docs_env_vars` 2 / 2; admission: `conv_mix_bf16` 532 cases, `moe_experts` 57 cases ADMITTED, serve admission 135 cases with nothing rejected (b10). ISA of `ink_mix` / `ink_mix_rows` / `ink_moe` kernels identical to main (31 / 55 / 427) |
| 5 speed | above and section 4 (B and R: 2 paired reps each, A: 3; the window budget ran out); the default rule's condition (4) fails on the mean (4-stream filler TTFT p50 R / A 1.020: +0.8 % and +3.2 %), so opt-in |
| knob off == main | `INTARSIA_INK_PACK` unset: every new path is behind `ink_pack_on()` (`Model::ink_pack_capable`, the workspace growth at load, the `Phase::Pack` plan probe through `ink_pack_capable`); `pack_rows_max` then falls back to main's (0 for Inkling), so the scheduler never packs; `pack_last_alone` and the long-chunk lookup in `pack_body_rows` matter only where packing runs. Shown by code read and the cpu `core` / `plan` / `inkling2_packed` tests on b11; a knob-off `--max-batch` server was not run on the GPU (the solo server and `generate` were: = main) |

Radiance: 8 windows, **25:42** down in total (the longest 4:38), restored within 15 s each time. Overlap (R3d) was not built
(section 1.5 says what it needs).

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

- 2026-11-26 ~18:10 EST (coordinator, after PR #538; in substance): one more window on the PR head (b11), at most 5:00 down:
  (1) hip tiny `inkling2_packed` at main's default PF2 (no env) plus `inkling2_batch`, must pass; (2) the 4-stream filler row,
  A (main 178d87ece, or the 6517d98f4 binary if no fresh build fits) vs R (b11, recommended config), 3 more paired reps in
  alternating order, TTFT p50 and aggregate per rep and pooled with the earlier 2 reps, 95 % CI; (3) if time allows, one
  `--max-batch 16` server with packing off on b11, 16 real prompts, md5 / text equal to main's (closes the knob-off gap).
  Decision rule: if the pooled 4-stream TTFT p50 is at most 1 % worse, or its CI includes 1 % or better, make packing default-on
  with `--max-batch` (`INK_PACK_DEFAULT`), plus the 4096-row default if the +606 MiB fits the recipe's B16 budget at the default
  kv-fraction (say whether it fits); otherwise opt-in. Either way update the PR body (table, gates, decision and why), push, report;
  do not merge. Lane ink-hip-verify may also queue a short window; fleetq orders them. Prefill overlap is out of scope (follow-up).

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
- Plan (b8): `Phase::Pack`, a probe of the largest packed pass (fresh phantom sequences of at most a chunk each, the real packed
  layers recorded), so the scratch and admission cover passes of more rows than one sequence's KV room (W2's 2 x 2048 refusal).
- Admission (b10): a segment-table kernel is skipped by the sweep and takes the admission of the kernel it twins, as row-table kernels
  do (W3's REJECTED `conv_mix_bf16_segs`); `INTARSIA_HIP_DISABLE` of the base drops the twin too.
- Dumps: `Model::ink_dump_packed` (the oracle-named intermediates of a packed pass, for the per-op diff test).

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
| W3 | #12236 | 17:24:52 -> 17:28:22 = **3:30** | gate 3 `generate` md5 (main3 vs b9, both cards) + admission prime of the batching servers |
| W4 | #12240 | 17:31:56 -> 17:36:34 = **4:38** | gate 3 serve level (b10 solo vs main3 solo), tiny suites, G2 short; the packed visit sent no request (driver bug) |
| W5 | #12241 | 17:37:45 -> 17:40:14 = **2:29** | gate 2 serve level (b10 packed), the opt-in 4096-row workspace, G2 long, admission |
| W6 | #12242 | 17:42:12 -> 17:46:00 = **3:48** | speed pair A (main3) / B (b10 packing, 2048 rows), tag w6; second-turn reuse check |
| W7 | #12246 | 17:48:25 -> 17:52:46 = **4:21** | speed R (b10 packing, 4096-row workspace) / A / B, tag w7 |
| W8 | #12248 | 17:53:34 -> 17:57:23 = **3:49** | real-shape unit test (card 1, PRE), then speed A / R, tag w8 |
| **total** | 8 windows | **25:42** (budget 30:00, each <= 5:00: longest 4:38) | radiance answered within 15 s after every window |

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

### Rebases and binaries

- The branch was rebased twice while the lane ran: onto b7450223e (#531 shared mix switches, #534 hip-verify) and onto **6517d98f4**
  (#533 hip-p12: a model loaded to batch packs no P12 copies). #531 made the two ink_mix test switches one definition shared by the
  solo and row-table TUs (`IK_INK_MIX_ROWS_TU` declares them extern); the segment TU (`IK_INK_MIX_SEGS_TU`) now declares them extern too
  (else a second definition, and the segment launches would not follow the setters).
- ISA after the first rebase (#12227, main b7450223e vs b8): ink_mix 31 / 31, ink_mix_rows 55 / 55, ink_moe 427 / 427 kernels identical;
  #533 touched none of these files.
- **main3** = origin/main 6517d98f4 (`.scratch/bin/intarsia-main3`, md5 2548ea9f), the reference for every gate from W3 on. **b9** =
  branch on 6517d98f4 (md5 09b8cebb). cpu checks on b9 (#12231): `intarsia-core` lib 270 passed, `plan` 10 passed, `inkling2_packed`
  (cpu) 5 / 5.

### W3 (main3 vs b9): gate 3 `generate` md5 equal, and an admission gap found

Raw: `raw/w3-gate.log`. Greedy `generate --ignore-eos`, graphs on, no env knob.

| prompt | tokens | `--spec off` (card 0) main3 / b9 | DSpark default (card 1) main3 / b9 |
|---|---|---|---|
| short-0 | 64 | 04cdb28a / 04cdb28a | 04cdb28a / 04cdb28a |
| short-0 | 256 | 9a27c21b / 9a27c21b | 9a27c21b / 9a27c21b |
| 4k-0 | 64 | 2252566a / 2252566a | 2252566a / 2252566a |
| 4k-0 | 256 | 9675bbb9 / 9675bbb9 | 9675bbb9 / 9675bbb9 |

**8 / 8 equal.** Prefill at 4k: 3085-3096 tok/s both binaries (main3 3084.8 / 3096.1, b9 3072.5 / 3091.9 at 64 tokens).

The prime of the b9 packed server (`INTARSIA_INK_PACK=1 --max-batch 16`, 135 cases, 112 s) **REJECTED `conv_mix_bf16_segs`**: the
plan's new `Phase::Pack` probe records the segment-table ConvMix, the plan queued that kernel as untested, and admission swept it with
its op's ordinary shapes, which have no segment table ("invalid op call"). Under `serve` (admitted kernels only) every packed pass would
then fail. b10: a segment-table kernel is skipped by the sweep and takes the admission of the kernel it twins, exactly as row-table
kernels do (`admission.rs`, `plan.rs`; `INTARSIA_HIP_DISABLE` of the base also drops the twin, `device.rs`). Main's batching server
(main3, 134 cases, 106 s) primed clean.

### W4 (b10): solo server == main's, 16 / 16; the packed visit lost to a driver bug

Raw: `raw/w4/` (driver.log, bg.out, jsons). b10 = b9 + segment-table kernels outside the admission sweep (cpu check #12239: `kernels
test --backend cpu --op conv_mix` on b9 FAILs `cpu_f32_conv_mix_segs` 8 x "invalid op call"; on b10 only `cpu_f32_conv_mix` is swept,
976 cases, ADMITTED). cpu on b10 (#12238): core 270, plan 10, `inkling2_packed` 5 / 5.

- **Gate 3, server level: 16 / 16.** The branch's solo server (b10, card 0, `--max-seq 4608 --spec off`, no `--max-batch`) and main's
  (main3, card 1, same flags): the 16 real prompts one at a time, 128 forced tokens each, every reply equal (text and md5). Wall
  34.82 s (b10) / 33.98 s (main3): different cards, not a speed comparison.
- **The packed visit ran nothing:** the Mac driver's `serve_bitwise.py` looked for the prompts next to `.scratch` (its tatooine path);
  fixed (`_prompts_dir`). Its server's admission (135 cases, 101 s, no REJECTED) is cached for the next windows.
- Tiny suites (card 1, b10 test builds): `inkling2_batch` 9 / 9; `inkling2` 14 / 17 (g1b and the two screened-head tests: the same
  three fail on main, W2); `inkling2_packed` at the PF2 default 1 / 4 (main's PF2 tiny fault, W2), with `INTARSIA_HIP_INK_PF2=0` **4 / 4**:
  2 x 17, 17 / 40 / 300 (resumed), 2048 / 1000 / 17, 4096 rows (two resumed), blend tails, 16 mixed segments: packed == solo; per-op dump
  548 / 548 bitwise twice.
- G2 short (b10, card 1): 21-way KL 4.491e-3, top-1 99.47 %: main's value (the hip-g2 doc's 4.491e-3 after #526). G2 long: out of time.
- The real-shape test on card 1 got a 1121-token KV cap (the solo server's memory not yet freed) and could not run its first case.

### W5 (b10): gate 2, every stream bitwise its solo reply

Raw: `raw/w5/`. One `INTARSIA_INK_PACK=1 --max-batch 16 --max-seq 4608 --kv-fraction 0.9` server (card 0, admission cached: up in
19 s), the 16 real prompts (49 .. 4253 tokens, 128 forced greedy tokens), plain, compared with W4's solo b10 replies (= main3's):

| wave | prompts | prompt tokens | wall s | aggregate tok/s | TTFT median / max s | ITL p50 / p99 s | replies equal to solo |
|---|---|---|---|---|---|---|---|
| 4 at once, offset 0 | 4 | 5 241 | 5.13 | 99.8 | 1.56 / 2.08 | 0.024 / 0.087 | **4 / 4** |
| 4 at once, offset 1 | 4 | 5 406 | 5.19 | 98.6 | 1.68 / 2.17 | 0.024 / 0.487 | **4 / 4** |
| 4 at once, offset 2 | 4 | 6 778 | 5.56 | 92.1 | 1.58 / 2.46 | 0.024 / 0.241 | **4 / 4** |
| 4 at once, offset 3 | 4 | 8 327 | 6.38 | 80.3 | 2.40 / 3.26 | 0.025 / 0.571 | **4 / 4** |
| 8 at once, offset 0 | 8 | 12 019 | 8.80 | 116.3 | 1.89 / 4.51 | 0.034 / 0.614 | **8 / 8** |
| 8 at once, offset 1 | 8 | 13 733 | 9.29 | 110.3 | 2.11 / 5.10 | 0.034 / 0.600 | **8 / 8** |
| 16 at once | 16 | 25 752 | 16.48 | 124.2 | 2.84 / 9.72 | 0.058 / 0.628 | **16 / 16** |
| 16 at once, `INTARSIA_INK_PACK_ROWS=4096` (opt-in) | 16 | 25 752 | 15.98 | 128.1 | 3.23 / 9.26 | 0.058 / 0.652 | **16 / 16** |

- **16 / 16 at 4 (four waves), 16 / 16 at 8 (two waves), 16 / 16 at 16 at once**, and 16 / 16 with the 4096-row workspace.
- Packing ran: `intarsia_batch_pack_passes_total` 0 -> **7 passes carrying 22 chunks** over the visit; the 4096-row server 4 passes,
  13 chunks (16 at once plus the two pure-prefill rows).
- For scale (not paired, batch lane window 10 on main b7): 16 at once 115.7 tok/s, wall 17.31 / 17.70 s; here 124.2, 16.48 s.
  The paired numbers are section 4.
- Prefix reuse (`reuse-conc:6` vs main3's `reuse-solo:6`): 6 / 6 equal, **but 0 cached tokens on both servers**: an extended
  prompt (A's text + more instruction) never resumes on Inkling, whose slot can restore only at its ring snapshot (the end of A's
  prompt), which the extension does not keep as a prefix. A second-turn mode (`reuse2-conc`: first prompt, reply, new message;
  that request starts with the whole first prompt) runs in W6 on both arms.
- Pure prefill with the 4096-row workspace (1 rep): 2 x 2048 2812 tok/s, 4 x 1024 3238 tok/s.
- G2 long (b10, card 1): 21-way KL **4.083e-3**, top-1 98.98 %: **main's defaults exactly** (hip-g2 doc, arm A1: long 4.083e-3 / 98.98 %, short 4.491e-3 / 99.47 %).
- Admission (b10, card 1, `kernels test --backend hip`): `conv_mix_bf16` 532 cases ADMITTED (the segment twin takes its admission);
  `moe_experts` 6 kernels, 57 cases, all ADMITTED.
- The real-shape unit test got a 1121-token KV cap on an empty card 1: since #533 a model loaded unbatched packs P12 copies and
  measures the KV budget with them resident; the test now loads with `LoadOptions::batched` (as the serve loader does).

### W6 - W8: speed, and more bitwise evidence

Raw: `raw/w6/`, `raw/w7/`, `raw/w8/`, `raw/speed-summary.txt` (`speed_sum.py w6 w7 w8`). Section 4 has the tables.

- **W8 PRE, gate 1 at real shapes (card 1, b10 at the PF2 default, models loaded batched): 7 / 7 cases packed == solo** (head, every
  state byte, `restorable_len`, the decode step after): 17 + 2048; 4 segments 255 / 256 / 257 / 17 (the expert split); blend tails
  258 / 261 / 513; **2 x 2048 = 4096 rows** in the 4096-row workspace; **6 segments, 4029 rows** (one resumed at 1024); **16 segments,
  3512 rows** (priors 0 / 49 / 0 / 1500); 2 x 1024. KV cap 2791 per sequence (W4 / W5's 1121 was the unbatched load's P12 copies).
- **Every reply of the speed visits matches across arms.** The tagged real-prompt levels and the pure-prefill rows send the same
  requests to each arm of a window, so each packed arm's replies compare with main3's: B vs A (W6, W7) and R vs A (W7, W8), each window
  4 / 4 + 8 / 8 + 16 / 16 real-prompt replies and 16 / 16 + 8 / 8 + 4 / 4 + 2 / 2 pure-prefill replies: **232 / 232 equal**.
- Second turns (`reuse2-conc:6`, W6): A vs B 6 / 6 equal, but again 0 cached tokens on both: **the batching engine never resumes a
  prompt on HIP Inkling-2** (`engine/batch.rs`: a recurrent model with no batched snapshots, `Model::batch_snapshots() == 0` off the cpu
  backend, gets `prefix_reuse = false`; the prefix store, the other way to resume, turns packing off). So the gate's "with prefix
  reuse on" case does not arise at serve level on HIP; resumed segments are covered by the unit gates (tiny: 3 cases, real: 2 cases,
  per-op dumps: 2 cases).
- Packed passes per visit (`intarsia_batch_pack_passes_total`): B 10 passes / 36 chunks (W6), 7 / 28 (W7); R 21 / 59 (W7), 21 (W8).
- VRAM peak (card 0, the whole visit): A 28 528-28 529 MiB, B 28 530-28 532 MiB, **R 29 135 MiB (+606 MiB)**; context 4608 tokens
  for all three (`--kv-fraction 0.9`).

## 4. Speed (gate 5), paired on card 0

Arms, all `--max-batch 16 --max-seq 4608 --kv-fraction 0.9`, default busy chunk 2048, no `--spec`, the same card, servers in turn:
**A** = main3 (origin/main 6517d98f4, no Inkling packing: packing off), **B** = b10 with `INTARSIA_INK_PACK=1` (2048-row passes),
**R** = b10 with `INTARSIA_INK_PACK=1 INTARSIA_INK_PACK_ROWS=4096` (4096-row passes). Order: W6 A B, W7 R A B, W8 A R. Ratios are
paired within a window (X / A), mean over the windows with a 95 % t interval (n = 2: t = 12.7, so wide). Real-prompt levels: the
batch lane's subsets (every 16 / N-th of the 16 sorted by length, offset = window), a window nonce in every request (no shared
prefix). Filler: localeval `speed --sizes 1024 --gen 128 --reps 1 --concurrency 1,4,8,16` (1039-token prompts). Pure prefill: K
prompts at once, 1 token each, prompt tokens / wall.

### 4.1 Filler sweep (1k prompts, 128 tokens)

| streams | A agg tok/s | B agg | R agg | **R / A agg** | A TTFT p50 s | B | R | **R / A TTFT p50** | TTFT max A / R s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 61.9 (3) | 61.9 (1) | 61.7 (2) | 0.996 [0.945, 1.047] | 0.41 | 0.42 | 0.39 | 0.955 [0.442, 1.467] | 0.41 / 0.39 |
| 4 | 111.2 (3) | 110.6 (1) | 117.4 (2) | **1.056** [0.966, 1.147] | 1.32 | 1.33 | 1.35 | 1.020 [0.870, 1.170] | 1.53 / 1.37 |
| 8 | 140.9 (3) | 139.4 (1) | 152.3 (2) | **1.081** [1.066, 1.096] | 2.16 | 2.08 | 2.14 | 0.975 [0.322, 1.627] | 3.09 / 2.57 |
| 16 | 154.4 (3) | 153.9 (1) | 168.9 (2) | **1.095** [1.076, 1.114] | 3.67 | 3.65 | 3.37 | **0.916** [0.870, 0.962] | 6.36 / 5.06 |

B / A (W6 only): 1.003 / 0.994 / 0.990 / 0.994 aggregate: **with 2048-row passes the filler never packs**, since two 1039-token
prompts are 2078 rows. The design's estimate (section 1.6: 8 x 0.66 s instead of 16 x 0.43 s) assumed two 1k prompts fit one
2048-row pass. They do in a 4096-row pass (three per pass): R.

### 4.2 Real prompts (16 real-text prompts, 49 .. 4253 tokens, 128 tokens each)

| N at once | A agg tok/s | B agg | R agg | B / A agg | **R / A agg** | TTFT p50 / max s: A | B | R | **R / A TTFT p50** | ITL p50 s A / R | ITL p99 s A / B / R |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 97.3 (3) | 99.6 (2) | 99.8 (2) | 0.992 [0.929, 1.056] | **1.041** [0.994, 1.088] | 1.66 / 2.24 | 1.63 / 2.12 | 1.42 / 2.08 | **0.843** [0.455, 1.231] | 0.0239 / 0.0242 | 0.483 / 0.273 / 0.173 |
| 8 | 114.0 (3) | 113.9 (2) | 116.2 (2) | 1.012 [0.809, 1.215] | **1.032** [1.024, 1.040] | 2.11 / 4.84 | 2.03 / 4.81 | 2.08 / 4.64 | 0.962 [-0.235, 2.160] | 0.0337 / 0.0336 | 0.591 / 0.605 / 0.587 |
| 16 | 122.0 (3) | 123.7 (2) | 128.0 (2) | 1.016 [0.948, 1.084] | **1.050** [1.000, 1.099] | 3.29 / 10.09 | 2.94 / 9.87 | 2.57 / 9.27 | **0.779** [0.569, 0.990] | 0.0576 / 0.0576 | 0.649 / 0.633 / 0.653 |

(A / B / R columns are unpaired means over their windows, whose N = 4 / 8 subsets differ; the ratio columns are paired.)

TTFT p95 of the real-prompt levels (per-request TTFTs, linear-interpolated p95; mean over windows, paired ratio per window):

| N at once | A p95 s | B p95 s | R p95 s | B / A | R / A |
|---|---|---|---|---|---|
| 4 | 2.16 | 2.04 | 1.98 | 1.027, 1.002 (mean 1.015) | 0.887, 0.902 (mean **0.894**) |
| 8 | 4.51 | 4.49 | 4.32 | 0.998, 0.947 (mean 0.973) | 0.939, 0.937 (mean **0.938**) |
| 16 | 9.82 | 9.57 | 9.00 | 0.986, 0.960 (mean 0.973) | 0.914, 0.917 (mean **0.915**) |

The filler sweep's localeval result keeps only the median / min / max TTFT per level (no per-request list), so 4.1 shows median and max.

### 4.3 Pure prefill rows (prompt tok/s)

| K x L (prompt tokens each) | A | B | R | B / A | **R / A** |
|---|---|---|---|---|---|
| 16 x 49 (60-90) | 560 (3) | 635 (2) | 634 (2) | 1.141 [0.787, 1.495] | **1.144** [0.720, 1.569] |
| 8 x 512 (570-612) | 1913 (3) | 2443 (2) | 2834 (2) | **1.279** [1.181, 1.377] | **1.484** [1.418, 1.549] |
| 4 x 1024 (1180-1196) | 2897 (3) | 2813 (2) | 3230 (2) | 0.964 [0.864, 1.064] | **1.122** [0.801, 1.443] |
| 2 x 2048 (2348-2364) | 2799 (3) | 2738 (2) | 2783 (2) | 0.977 [0.784, 1.170] | 0.995 [0.958, 1.032] |

The 2 x 2048 row never packs: a packed pass takes at most half (rounded up) of the waiting prompts (`pack_half`, the scheduler's
default, `INTARSIA_PACK_HALF`), so two waiting prompts go one at a time. 4 x 1024 packs only in R (two 1190-token prompts are
2380 rows: one pass of two, then the last two alone). Short segments (16 x 49) gain least: below 256 rows their experts
stay on the SIMT pair kernels, which cost per pair, so packing saves launches, not expert work (section 1.6).

### 4.4 VRAM

Peak over the whole visit (card 0; one server ran every N of a visit, so this is the peak over all N, not per N): A 28 529 MiB, B 28 531 MiB, R **29 135 MiB** (+606 MiB: the 4096-row Inkling workspace and the experts'
scratch for a 4096-row call). Context 4608 tokens in all three.

### 4.5 Verdict against the default rule (coordinator, brief 2)

| Condition | B (2048 rows) | R (4096 rows) |
|---|---|---|
| (1) every stream bitwise its solo run | yes (W5 16 / 16 at 4, 8, 16; W6 / W7 cross-arm 116 / 116) | yes (W5 16 / 16; W7 / W8 cross-arm 116 / 116; unit 2 x 2048, 16 segments) |
| (2) solo path unchanged | yes (md5 8 / 8, solo server 16 / 16, G2 = main) | yes (same binary) |
| (3) 16 streams: aggregate or TTFT p50 >= 5 % better, beyond the spread | **no** (real 1.016, filler 0.994; TTFT p50 0.879 with an interval over 1) | **yes**: filler aggregate 1.095 [1.076, 1.114], filler TTFT p50 0.916 [0.870, 0.962], real TTFT p50 0.779 [0.569, 0.990], real aggregate 1.050 [1.000, 1.099] |
| (4) no stream count > 1 % worse on aggregate or TTFT p50 | (not met) | **fails on the mean**: filler 4-stream TTFT p50 R / A = 1.008 (W7) and 1.032 (W8), mean **1.020**, over the 1 % bar (interval [0.870, 1.170]: within the noise of these 4-request medians, whose 1-stream ratio, where nothing packs, is 0.995 / 0.914). Every other row of aggregate and TTFT p50 is better or within 1 % on the mean |

**So packing ships opt-in** (`INTARSIA_INK_PACK`, default off): B does not reach (3); R reaches (3) but fails (4) on the mean of the
4-stream filler TTFT p50 (+2.0 %, two reps). B and R got 2 paired reps each, not the brief's 3: the radiance budget (8 windows) ran out. **Recommended for `--max-batch` servers: `INTARSIA_INK_PACK=1 INTARSIA_INK_PACK_ROWS=4096`** (+606 MiB),
which makes the measured gains: up to +9.5 % aggregate and -8 % TTFT p50 at 16 filler streams, +5 % aggregate and -22 % TTFT p50 at
16 real prompts, +48 % prompt tok/s on 8 x 512, every reply bitwise. Flipping it default-on (both knobs) is the coordinator's / user's
call; what it would take: one more paired rep of the 4-stream filler row (radiance budget spent: 8 windows, 25:42).

## 5. fleetq jobs and binaries

- cpu-slot jobs (builds, cpu tests, ISA): #12163 (main 80719483a cli), #12199 (b1), #12201 / #12208 / #12210 (b2-b4, ISA), #12215 (b5),
  #12221 (b6 + main tiny tests), #12223 (b7), #12227 (b8: rebased on b7450223e, ISA, main2), #12231 (b9 on 6517d98f4, main3),
  #12238 / #12239 (b10, cpu admission sweep check), #12243 / #12245 (packed test rebuilt with the batched load, `docs_env_vars`),
  #12249 (**b11 = the PR head on 5367023d2**: cli built, core 270, plan 10, `inkling2_packed` cpu 5 / 5, hip lib 2, `docs_env_vars` 2).
- GPU windows (radiance down): W1 #12216, W2 #12224, W3 #12236, W4 #12240, W5 #12241, W6 #12242, W7 #12246, W8 #12248.
- b11 differs from b10 only by the rebase onto #537 (ink-hip-pf2-tiny: `kernels/hip/ink_attn.hip`, which this lane does not touch)
  and docs; no GPU window ran b11 (budget spent). The GPU gates above ran b10 (on 6517d98f4).

## 6. Leftovers

- **Default flip**: one more paired rep of the 4-stream filler row (R vs A) would show or refute condition (4); then
  `INK_PACK_DEFAULT = true` with a 4096-row default workspace is a two-line change (`model/inkling2.rs`).
- **hip tiny packed tests at the PF2 default**: they failed on b10 from main's PF2 tiny-shape fault, which #537 fixes (now in the base);
  re-run `inkling2_packed` hip tiny without `INTARSIA_HIP_INK_PF2=0` on b11 in the next window.
- **Overlap (R3d)**: not built. ITL p99 stays 0.5-0.65 s at 8 / 16 (decode rows wait for a 2048-row pass); R lowers the 4-at-once
  p99 (0.483 -> 0.173 s) only because fewer passes run. Section 1.5 lists what it needs (a second HIP stream, events, per-slot fences).
- **Prefix reuse on HIP batching**: off for Inkling (no batched ring snapshots); with it, resumed prompts would pack (unit-tested).
- `pack_half` sends two waiting prompts one at a time (the 2 x 2048 row): a per-model `pack_half` could help Inkling, whose
  prefill is slow enough that TTFT gains from packing two.

