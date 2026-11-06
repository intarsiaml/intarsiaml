# Metal DFlash2 port (Qwen3.8-27B dense), then tree drafts

Lane dm, branch `lane-dm/metal-dflash2` (from `origin/lane-df/27b-dflash-gap`, PR #322).
Worktree `/opt/code/intarsiaml/.worktrees/metal-dflash2`.

## Decisions

- 2026-11-06, user, verbatim: "Start it now" (the Metal acceptance work: DFlash2 port to Metal, then tree drafts).

- 2026-11-06 ~12:40, coordinator (by message): split LOGITS_ROWS out of this branch into its own PR. (1) After #322 merges, branch
  `lane-dm/metal-logits-rows` off main; cherry-pick the `logits_rows_scalar_m{2..16}` kernel and its bitwise test (own test file if the
  test depends on DFlash2 code). (2) Default-on, `INTARSIA_METAL_LOGITS_ROWS=0` disables (the #322 pattern). (3) Gates: `plan show`
  byte-identical for 8B / 27B / 35B; kernel tests; greedy ids unchanged on 8B and 35B; 27B MTP ids == `--spec off`; ABAB on endor via
  fleetq (`--quiet gpu,cpu`), 27B MTP k2 at short-0 and 4k. (4) Open the PR with raw paths and the ABAB table, message the coordinator,
  who merges on gates. PR body also answers (not built): cost of an aligned copy of lm_head at load (memory for 27B, load time) vs the
  per-token saving on M = 1 and the M-row path. Then return to DFlash2 phase 1 measurement.
- 2026-11-06, coordinator (by message): #322 merged (main 551c56181), MR now default-on; branch `lane-dm/metal-logits-rows` off origin/main
  now and run the gates with MR unset (default).

- 2026-11-06 ~13:45, coordinator (by message): leave `lane-dm/metal-logits-rows` unmerged (no fleet config runs a bf16 27B head; the fp8
  head is the user's default on all backends); keep the branch and its results doc (`docs/results/2026-11-06-metal-logits-rows.md` on that
  branch) as the record. Continue DFlash2 phase 1 against the fp8-head main; the bar is MTP k2 at 59 tok/s at short-0.

- 2026-11-06 ~13:30, coordinator (by message): **phase 3 first**: the bitwise M-row verify GEMV; trees after it (cost per verify row is
  what loses, not acceptance). Constraint, user decision 2026-11-06 "keep bitwise": every verify row stays bit-identical to the M = 1
  decode; moving M = 1 onto the matrix units is out of scope, do not build it. Cover M 2..8 (MTP k2 verifies 3 rows: a faster M = 2..4
  speeds the default path). Pre-register a kernel rule (M = 8 <= 2.0x M = 1, M = 3 <= 1.4x M = 1) and an end-to-end rule (MTP k2
  default >= 1.05x, ids == `--spec off`, plan show unchanged). Ship opt-in first with the usual gates and an ABAB; the coordinator takes
  default-on to the user. Let #8833 run; keep the tree pieces on the branch, no more tree work until the GEMV result is in.

- 2026-11-06 ~13:50, coordinator (by message): lane dh is finished (#323 merged, main 3f46da4f7); land the dry-run ingest fix
  (3c70a7917) on main myself as its own small PR (`lane-dm/dflash-dryrun-ingest`): CPU regression test beside dh's window test if CPU
  can show it (else say so in the PR body); gates: Metal `plan show` 8B / 27B / 35B byte-identical to main under `--spec off` and mtp,
  `serve --spec dflash` on 27B Metal survives a pass with < 7 accepted and greedy ids == `--spec off`, `cargo test --workspace` (skip the
  endor-only `completions_logprobs_do_not_stall_the_async_threads`); no HIP window needed. Open the PR with the error, cause and gate job
  ids, message the coordinator (merges on green CI). Then rebase this branch onto main, rerun the phase 1 ABAB (mtp2 / dfl7 alternating,
  then dfl3), and record that visits 2, 4 and 5 of #8833 are invalid.
- **#8833 visits 2, 4 and 5 (the DFlash2 arms) are invalid**: `serve` failed every request after the first pass that accepted < 7 drafts
  (`no kernel for matmul on metal/applegpu_g17s (m=1 n=5120 k=5120 act=Some(Bf16) w=Some(Bf16)) [admitted kernels only]`): the DFlash dry
  run planned the context ingest at block rows only. Visits 1, 3 (MTP k2) and 6 (TensorFold) are valid.

- 2026-11-06 ~14:45, coordinator (by message): #324 merged as main e45bd5e3c. Rebase `lane-dm/metal-dflash2` onto origin/main and
  rerun the phase 1 ABAB as planned. Continue the table-lookup e2m1 decode for the M-row GEMV. (Sent before my LUT result reached the
  coordinator: the LUT form measured slower and was reverted; see phase 3. Reported back instead of continuing it.)

- 2026-11-06 ~15:00, coordinator (by message): phase 3 option (c) as ONE time-boxed probe (60 min, ALU MR kernel only, bitwise) of the
  bandwidth assumption: load each activation word once per simdgroup / threadgroup and share it (threadgroup staging or `simd_shuffle`)
  instead of each lane re-reading it per P weight rows; k-ascending fma order per output kept. Rule (pre-registered below). Meets it:
  usual gates + MTP k2 ABAB (>= 1.05x), report. Misses: revert, write up, close phase 3 (bandwidth finding + MMA ceiling as the reason).
  Not (b) (MMA M = 8: ~1.5x M = 1 ceiling puts DFlash k7 at ~60 tok/s, parity with MTP k2 at best, nothing for the default path).
  Push 3fc8a028d now; finish the queued no-op gates, ids and phase 1 ABAB rerun. Final lane report: Metal DFlash2 vs MTP by category, a
  recommendation on whether Metal DFlash2 should exist as opt-in, branch head.

## Goal

Close the acceptance half of the Metal gap against TensorFold (90.3 tok/s at natural-v1 4k vs our MTP k2 31.1, 0.34x;
TensorFold keeps 3.2-4.4 tokens/round from a ~15-node DFlash2 tree, our MTP keeps 2.1-2.4).

## Setup

* Draft checkpoint: `z-lab/Qwen3.8-27B-DFlash2` (HF cache snapshot `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` on endor), `model.safetensors`
  sha256 `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`, identical to HIP's `/data/models/tensorfold/Qwen3.8-27B-DFlash2`
  on tatooine (same sha, same config.json md5): no copy needed. Block 8 (k7), 5 layers, taps [5, 19, 33, 47, 61], sliding window 2048,
  conv K 2 / group 16, selector rank 256 top-16.
* Target: `Qwen3.8-27B-Quark-AWQ-MXFP4` (W4A16, bf16 head).

## Phase 1: DFlash2 on Metal, linear block k7

### What changed (Metal backend; shared files minimal)

* `crates/intarsia-models/src/load.rs`: the DFlash2 load gate admits `metal` (one line).
* `kernels/metal/dflash.metal`, `crates/intarsia-backend-metal/src/dflash.rs`: `dyn_conv_bf16` (`Op::DynConv`, the hip kernel's
  semantics, one thread per (t, c)); `attn_decode_bf16_window` (`.ext()`, priority -1): the sliding-window decode as the existing decode
  shaders over the window's keys (K / V bound `s` rows further in, same per-head stride, position `p - s`), split-KV where it applies.
  Registered in `kernels.rs` (`MAIN_KERNEL_LIST` pin updated: two new names, no existing op changes).
* `crates/intarsia-core/src/model/dflash.rs` / `dflash2.rs`: lane dh's bug-0 hunk (`a8353a3ae`, identical text) so `serve` plans the
  windowed calls (the zero-key / no-kernel symptom: `n = 0` in the HIP message was the windowed `attn_decode` missing from the plan).
* `logits_rows_scalar_m{2..16}` (`INTARSIA_METAL_LOGITS_ROWS`): the M-row twin of `gemv_f32o_scalar`. The 27B lm_head sits at byte 8 mod
  16, so every head call is the scalar kernel (5.8 ms) and the M-row verify / draft head was M one-row launches. Split into its own PR
  (coordinator, see Decisions); removed from this branch after the coordinator's 13:45 decision (the record is branch
  `lane-dm/metal-logits-rows`).

### Correctness gates

* `tests/dflash2.rs` (endor GPU): `dyn_conv` vs an f64 reference (T 1 / 3 / 8 / 16 at H 5120 K 2 G 320, and H 48 K 3 G 3, 2-byte offsets);
  windowed decode bitwise the plain decode over a cache holding only the window's keys (27B draft heads 32 / 8 x 128 on split-KV; 12 / 2 x 72
  on the single-pass kernel; windows inside, at and past the context), and bitwise the plain call when the window covers the context; the
  window kernel is never picked by plain calls; logits rows bitwise the 1-row launches (M 2..16, weight at +8 and +2 bytes, K 5120 / 700 /
  1030), planted fault (non-fma accumulation) caught.
* Greedy ids `--spec dflash --spec-k 2..7` equal `--spec off` at short-0 and 4k-1 (3135-token prompt: the window is active), 256 tokens
  (`raw/p1-try`, `raw/p1-try-mr`, `raw/p1-try-lrs`, `raw/p1-k`).

### Pre-registered speed rule, phase 1 (written before the ABAB window)

Same lane binary (merged main: MR default on), `INTARSIA_METAL_LOGITS_ROWS=1` in both arms, a fresh `serve --max-batch 1` per visit,
visits A B A B (A = MTP k2, B = DFlash2 k7), localeval suite cell `speed-spec` (natural-v1 1k / 4k / 16k, 256 tokens); categories once per
arm; TensorFold one visit in the same window. `--spec dflash` stays opt-in whatever the result. **DFlash2 k7 counts as a win over MTP k2
on Metal if B / A of the two-visit medians is >= 1.05 at natural-v1 4k and no natural-v1 cell is below 0.95**; otherwise phase 1 is
"correct, measured, slower" and the numbers go to the tree / verify-cost decision. TensorFold's categories: it may not honour
`min_tokens` / `ignore_eos`; its rows are compared by decode tok/s over whatever length it generated (flagged in the table).

Deviation (noted before reading #8833's results): the window #8833 ran with `ENVS=INTARSIA_METAL_MXFP4_MR=1` (MR is the default on this
main anyway), not `INTARSIA_METAL_LOGITS_ROWS=1` as written above; #8822 showed LOGITS_ROWS has no effect on the fp8-head main (the head is
`matmul_fp8_gemv_rows`), so the arms are the shipped default path.

### First traces (eager, `INTARSIA_SPEC_TRACE=1`, short-0, 256 tokens, fleetq exclusive)

| arm | knobs | tokens a pass | draft ms (head) | verify + rollback ms | pass ms | decode tok/s |
|---|---|---:|---:|---:|---:|---:|
| off | | | | | 33.4 | 29.95 |
| DFlash2 k7 | none | 3.60 | 52.8 (39.5) | 145.6 | 198.4 | 18.2 |
| DFlash2 k7 | MR=1 | 3.60 | 51.8 (39.0) | 125.5 | 177.3 | 20.4 |
| MTP k2 (eager) | MR=1 | 2.40 | 7.6 | 50.0 | 57.6 | 41.5 |
| DFlash2 k7 | MR=1 LR=1 | 3.60 | 19.4 (6.5) | 89.1 | 108.5 | 33.1 |
| MTP k2 (eager) | MR=1 LR=1 | 2.40 | 7.7 | 39.5 | 47.2 | 50.6 |
| DFlash2 k2 / k3 / k4 / k5 | MR=1 LR=1 | 2.36 / 2.80 / 3.07 / 3.32 | 19.1-19.7 | 41.0 / 45.7 / 73.3 / 83.5 | 60.2 / 65.0 / 93.0 / 102.9 | 38.9 / 42.7 / 32.9 / 32.1 |

(4k-1: DFlash2 k7 MR+LR 3.64 tokens a pass, draft 20.3, verify 96.9 ms, 30.9 tok/s; MTP k2 46.7 tok/s.) DFlash2 draft (MR+LR): flush 2.4,
block 8.7, head 6.5, selector projection 0.3, logits download 0.8, host select 0.7 ms.

Reading: acceptance is there (3.6 tokens a pass at k7 vs MTP's 2.4, as on HIP), but an 8-row verify costs 89 ms against 33 ms for a
plain step (2.7x); MTP k2's 3-row verify is 39.5 ms. The jump from 3 to 5 rows (45.7 -> 73.3 ms) is where the MXFP4 M-row GEMV goes from
its P = 4 to its P = 2 / `_m8` shaders (kernel bench: M = 3 0.95-1.42x of M = 1, M = 8 2.74-4.08x). The verify cost per row, not the draft,
decides DFlash on Metal; see phase 3 / the tree design.

## Phase 2 design: tree drafts (written before any tree code)

Reference read (not copied): TensorFold's DFlash2 tree (Apache-2.0, `tensorfold/drafters/dflash_tree.py`, `dflash_proposer.py`,
`kernels/qwen/dense/v1/{lane_tree,lane_multi,stream_gdn,stream_attention}.py`): one draft block pass of 16 rows; a host best-first search
over the per-row top-k candidates (children = top 4 of the next row, scored with the DFlash2 selector's edge term
`succ[child] . (pred[parent] * hproj[d])`, path score = summed log-softmax), 15 nodes for the 27B; rows parent-before-child, position =
len0 + depth; attention by an ancestor-path gather; GDN by a kernel that walks the nodes keeping every node's state in registers; commit
by gathering the accepted rows' KV and replaying the accepted rows' GDN steps from the pre-round state. Its verify rows are MLX `qmm`
(simdgroup matrices) and are not bitwise its M = 1 decode; ours must be (greedy ids == `--spec off`).

### Constraint that drives the design: the verify cost per row on Metal

Every verify row here is bitwise the one-row decode step (rows_exact kernels). The MXFP4 M-row GEMV keeps the M = 1 fp32 fma chain per
row, so its cost grows ~linearly with M (ALU-bound: the 27B is ~26 G weight-row products per row; M = 8 is 2.7-4.1x of M = 1, the 8-row
verify 89 ms against a 33 ms plain step). A 16-row tree verify on today's kernels would cost ~150-170 ms; at TensorFold's 4.2 tokens a
round that is ~22 tok/s, below MTP k2 (50.6 short with LOGITS_ROWS). So:

1. **A tree only pays at a fixed row budget**: at R verify rows, a tree of R nodes accepts at least as many tokens as a chain of R - 1
   drafts (the chain is one of the trees), for the same verify cost (the draft pass is one block pass either way). The tree is the
   better use of R rows; it does not make R rows cheaper.
2. **R is chosen by measured cost**, as TensorFold's allocator does: rows per pass from the table pass_ms(R) measured on this box
   (phase 1: R = 3 45.7, 4 ~55, 5 73.3, 6 83.5, 8 89 ms + draft 19 ms) and the per-depth acceptance EWMA. Today R = 3..4 wins on
   Metal; R = 8..16 only after the verify GEMV's per-row cost drops (phase 3).
3. Phase 3 (verify cost) is therefore the gate for wide trees; the tree machinery is built so it works at any R <= 16.

### Tree construction (host, after the existing draft block pass)

* One DFlash2 block pass as today (rows [anchor, MASK x 7]); keep per row the top-k (k = 4, device top-k later) logits and ids and the
  selector projection `hproj` (already downloaded for the selector).
* Best-first search: node score = sum over its path of log-softmax over siblings of `logit + selector_edge(parent token, hproj[d])`
  (the edge term our `Selector::select` already computes greedily); pop up to R - 1 nodes; children of a popped node at depth d = top-4 of block row d + 1.
  Depth d uses block row d (rows only see the anchor and MASKs, so row d's candidates do not depend on the parent: the parent enters
  only through the selector edge). With R - 1 = 7 and every pop on the best path the result is the existing greedy chain.
* Output: `TreeDraft { tokens[n], parent[n] (-1 = anchor row), depth[n] }`, parents before children, n <= R - 1.

### Verify pass over a tree (rows = [anchor, nodes...], row i at position len0 + depth(i))

Every row's computation must be bitwise the one-row decode of its own path. Per op family:

* **Row-wise ops** (projections, norms, MLP, head, argmax): unchanged; a row's result depends only on its own input row.
* **Row order**: the rows are the tree in depth-first preorder (`TreeDraft::preorder`): parents before children and each subtree
  contiguous.
* **Rope**: per-row positions `len0 + depth(r)` (a position table, as the captured verify pass already passes one).
* **KV append + full attention** (16 of 64 layers; the verify already runs one decode call per row): no batched append. Row r, in
  preorder, appends its own K / V at slot `len0 + depth(r)` and then runs the plain one-row decode at that position. Preorder guarantees
  slots `len0 .. len0 + depth(r) - 1` hold exactly r's path at that moment (a slot is overwritten only after the subtree that used it is
  finished), so every row is bitwise the plain decode of its own path with **existing kernels** (no path / mask kernel). The fused
  prologue (qk-norm + rope + append) is not used in a tree pass (it appends all rows at once); the unfused ops are bitwise the same.
  The pass keeps each attention layer's roped K and V rows (one copy per layer into a side buffer) for the commit.
* **GatedDeltaNet** (48 layers): the verify runs the conv and the delta rule one row at a time and saves the state after each row
  (`SpecRows`). Tree form: row i starts from its parent's saved state instead of the live state. Implemented as: before row i, copy
  snapshot[parent(i)] (or the pre-pass state for the anchor's children... the anchor row itself starts from the live state) into the
  live slot, run the row (same kernels, bitwise the path's sequential step), save snapshot[i]. Cost: one extra state copy per row whose
  parent is not the previous row (~3.1 MB a layer, ~150 MB a row for the 27B, ~0.4 ms); a later kernel can read src / write dst
  directly and drop the copies. The conv state is part of the saved recurrent bytes, so the conv over ancestors comes for free.
* **Acceptance**: walk from the anchor: target token of the current row; the child whose token equals it is the next row; stop when
  none does. Emits the path's target tokens (as the chain does).
* **Commit**: recurrent state = snapshot of the last accepted row (as `spec_rollback` copies saved row m - 1 today, by node index);
  full-attention KV: re-append the accepted path's rows from the side buffer at `len0 + depth` (one-row `KvAppend`s, 16 layers); the DFlash2 context ingest and the tap staging take the accepted rows only (the same gather on the taps' staging rows).
* MTP is unaffected (`--spec dflash` only); `--spec off` / `mtp` never build a tree.

### Gates for phase 2

* Unit: tree builder (host) property tests (parents before children, budget, chain when the tree degenerates, preorder invariant); no new attention kernel
  test; tree GDN: a tree verify's row logits bitwise equal the sequential decode of each row's path (CPU backend and Metal, small model).
* End to end: greedy ids == `--spec off` (short-0, 4k-1, 16k), `plan show` / `kernels test` no-op for the other models and spec modes.
* Speed rule (pre-registered before the ABAB): tree at R rows vs chain at R - 1 drafts, same binary: tokens a pass >= +8 % and tok/s
  >= +5 % at natural-v1 4k; vs MTP k2 reported, not gated.

### Phase 1 on the fp8-head main (merged `551c56181`; fleetq #8829, `raw/noop/`, `raw/p1-fp8/`)

No-op gates (lane `986530693` vs main `551c56181`): `plan show --json` byte-identical for 8B / 27B / 35B, each with the default spec, `--spec off`
and `--spec mtp` (9 / 9 SAME); `kernels list`: only the two new rows (`dyn_conv_bf16`, `attn_decode_bf16_window`); `kernels test`: 49 -> 51
kernels, 980 -> 1031 cases, every existing row identical, the new rows ADMITTED (`dyn_conv` 12 cases max_abs 0, `attn_decode_bf16_window` 39
cases max_abs 1.4e-3 vs the cpu reference, the admission's sliding-window cases).

Traces (eager, short-0 / 4k-1, 256 tokens; ids equal `--spec off` in every arm):

| arm | tokens a pass | draft ms | verify + rollback ms | pass ms | decode tok/s short-0 / 4k-1 |
|---|---:|---:|---:|---:|---:|
| off | | | | 30.0 / 35.9 | 33.3 / 27.8 |
| MTP k2 eager | 2.49 / 2.55 | 5.9 / 7.1 | 36.0 / 47.0 | 42.0 / 54.0 | 59.1 / 47.1 |
| MTP k2 pipelined (default) | 2.49 / 2.55 | | | | 59.8 / 49.1 |
| DFlash2 k3 | 2.91 / 2.87 | 17.7 / 20.9 | 45.3 / 57.0 | 62.9 / 77.9 | 46.1 / 36.8 |
| DFlash2 k5 | 3.51 / 3.54 | 18.1 / 20.4 | 91.9 / 106.6 | 110.1 / 127.0 | 32.0 / 27.9 |
| DFlash2 k7 | 3.83 / 3.70 | 17.6 / 20.2 | 93.2 / 113.2 | 110.8 / 133.3 | 34.6 / 27.7 |

DFlash2 draft (k7, short-0): flush 2.5, block 9.0, head 4.0 (fp8 head rows), selector projection 0.3, logits download 0.9, host select 0.7 ms.

Verify GEMV headroom (for the phase 2 / 3 decision): fp32 FMA peak on this M5 Max measured 7.51 TFMA/s (fp16 9.61; `.dm`-local Swift
microbench, fleetq #8830); the M-row MXFP4 GEMV at M = 8 reaches 2.75-2.94 TFMA/s (lane df's bench: 8 x 89 M fma in 243-259 us), ~2.6x
under peak. A bitwise M <= 8 kernel at ~1.5-2x of M = 1 looks reachable; M = 16 is 416 G fma a pass = >= 55 ms at peak by fp32 fma alone
(more than a plain step), so 16-row trees need M = 1 itself moved onto the matrix units (a change of the plain decode's bits: the user's call).

## Phase 3 (ordered first): bitwise M-row MXFP4 verify GEMV, M = 2..8

### Pre-registered rules (written before any measurement of the new kernel)

* Kernel (27B shapes 17408x5120, 5120x17408, 12288x5120, 5120x6144; the df bench harness, exclusive slot): time at M = 8 <= 2.0x the M = 1
  call and at M = 3 <= 1.4x, for every shape (the shipped MR auto today: M = 3 0.95-1.42x, M = 8 2.74-4.08x). Every row bitwise the M = 1
  call (test over M 2..8, tails, both output dtypes; planted faults caught).
* End to end: the new kernel behind an opt-in knob; same binary ABAB, MTP k2 default path (pipelined), A = knob off, B = on: B / A >= 1.05
  at natural-v1 1k and 4k (localeval speed-spec), no cell below 0.98; greedy ids == `--spec off` (MTP k1..k4, DFlash2 k3 / k7, short-0 and
  4k-1); `plan show` 8B / 27B / 35B unchanged with the knob unset; `kernels test` unchanged.

### Probe: is fp32 `simdgroup_multiply_accumulate` a sequential fma chain? (fleetq #8835, `raw/p3-probe/mma.swift`, `mma.out`)

20000 trials of 4 chained 8x8x8 MMAs from C0 (zero or random), inputs as the MXFP4 GEMV sees them (e2m1 * 2^-14 times bf16 over
2^-20..2^20, every product exact in f32), a quarter of the trials with adversarial cancellation rows (|x| up to 2^40), 1.28 M outputs:

| reference | outputs that differ |
|---|---:|
| k ascending, a rounded add per product (= the one-row kernel's fma chain) | **0** |
| k descending | 511195 |
| pairwise tree per 8, then added | 547863 |
| one rounding per 8-wide dot | 537842 |

The test discriminates (the other orders differ on ~40 % of outputs) and the M5 Max's fp32 MMA is the k-ascending chain on every output.
This is observed behaviour, not a documented guarantee: the MMA kernel is registered for this arch only and keeps a bitwise test.

### First MMA kernel (`mxfp4_gemv_*_mma`, `INTARSIA_METAL_MXFP4_MR=8`; fleetq #8836, `raw/p3-k1/p3-k1.log`)

* Bitwise: `tests/mxfp4_gemv.rs` `m_row_shaders_are_bitwise_the_one_row_call_per_row` passes with the MMA arm added (M 2..8, N 37 / 130 /
  515 / 64 tails, K 32 .. 17408, bf16 and f32 out, act 0 and 1, x offset 0 and 4: every row bitwise the M = 1 call).
* Speed: **11-18x of M = 1** at every M (17408x5120: 668 us at M = 2, 789 us at M = 8, vs 60 us at M = 1; the shipped MR p2 is 67 / 240 us).
  The draft is a serial chain (4 dependent MMAs per group, one 8x8 tile per simdgroup, 2-byte strided activation loads, 16 KB of
  threadgroup slots per 64 threads).
* Ceiling (`raw/p3-probe/mmapeak.swift`, fleetq #8842): fp32 `simdgroup_multiply_accumulate` peaks at **8.07 TFMA/s** (fp16 8.26) vs the plain
  fp32 fma peak of 7.51: on this GPU the simdgroup MMA runs at ALU rate, not on a separate matrix unit. So the MMA form can only remove
  overhead (decode, widening, operand loads), not add FLOPs, and it pads M up to 8: at M = 3 it does 8 rows of work. Its best case at the
  27B shapes is the M = 8 compute bound (17408x5120: 713 M fma = 88 us at 8.07 T, 1.46x M = 1); M = 3 stays with the ALU MR kernel.
* Bound for the ALU form: M = 1 is bandwidth-bound (17408x5120: 44.6 MB in 60 us, ~740 GB/s); M = 3 needs 267 M fma = 36 us at the fma peak,
  so M = 3 ~ 1.0-1.1x is not excluded by arithmetic: the shipped 1.30-1.44x is instruction overhead (e2m1 decode, bf16 widening, loads).

### Dry-run ingest fix: PR #324 (branch `lane-dm/dflash-dryrun-ingest`)

Opened 2026-11-06 for the coordinator (merges on green CI). CPU regression test fails without the fix, passes with it (#8854: dflash2_cpu
12/12); Metal no-op vs main (#8844): plan show 9/9 identical, kernels list / test identical; Metal serve 27B with this lane's binary
(#8848): with the fix 0 errors over 88 passes (612 proposed, 232 accepted) and all 4 greedy texts == `--spec off`, without it 4 / 4 requests
fail; workspace tests 1368 / 0 (#8849).

### Second attempt: table decode of the e2m1 codes (fleetq #8847, `raw/p3-k2/p3-k2.log`; reverted)

Bitwise (the MR test passes with the arms `12` / `14`), but **slower** than the shipped MR at every shape and M (17408x5120 M = 3: 99.9 / 93.6
us vs 80.8 us; M = 8: 244 / 287 vs 240). Reverted.

Two attempts failed; the assumption that was wrong: "the M-row kernel is ALU-issue-bound by the e2m1 decode". What the numbers fit
instead is the **activation re-read bandwidth**: every lane re-loads each activation word for every M row once per P weight rows, so the
kernel reads 4M/P times the weight bytes of activations from L1/L2 (P = 2: M = 3 6x, M = 8 16x). 17408x5120: M = 3 267 MB in 81 us, M = 8
713 MB in 240 us, both ~3.0-3.3 TB/s, and the four shapes sit on the same line. Larger P cuts it but P = 4 already loses to register
pressure. The only form that shares operands across 8 x 8 tiles in registers is the simdgroup MMA (its B fragment is shared by 8 weight
rows, its A by 8 activation rows), whose ceiling at M = 8 is ~1.5x M = 1, but it pads M to 8 (no help for MTP's M = 3), and the first
version was 11-18x slow for structural reasons (serial chain, 2-byte loads). Reported to the coordinator for a decision before more kernel
work.

### Phase 3 probe: shared activations (pre-registered rule, written before the kernel)

Threadgroup-staged activations in the M-row MXFP4 GEMV (each activation word loaded once per threadgroup, not once per lane per P rows),
k-ascending fma chain per output unchanged. Rule at 17408x5120 (the df bench, exclusive or quiet slot): **M = 3 <= 68.7 us and M = 8 <= 204
us (0.85x the shipped MR's 80.8 / 240.2 us)**, every row bitwise the M = 1 call (`tests/mxfp4_gemv.rs`). Time box 60 min.

### Phase 3 probe result: threadgroup-staged activations (fleetq #8863, `raw/p3-k3/p3-k3.log`): MISS, phase 3 closed

The code was not committed (an uncommitted probe, discarded after the measurement). Bitwise: `tests/mxfp4_gemv.rs` passed with the arm
(M 2..8, every row == M = 1). Speed at 17408x5120 (same run): **M = 3 102.1 us vs the shipped MR 81.2 (rule <= 68.7), M = 8 359.1 us vs 274.0
(rule <= 204)**; slower at every shape. Rule missed, so phase 3 is closed per the coordinator's decision.

Reason recorded for closing: (1) the M-row cost is not the e2m1 decode (table decode: slower) and not the L1 re-read of activations
alone (threadgroup staging removes it and is slower still); the measured rate tracks ~3 TB/s of activation reads per lane, but moving
those reads to threadgroup memory did not help, so the true limiter (likely load / issue slots per lane at this register footprint) is not
identified. (2) The only operand-sharing form, simdgroup MMA, is bitwise on this GPU but runs at ALU rate (8.07 TFMA/s), pads M to 8, and
its ceiling (M = 8 ~1.5x M = 1) would leave DFlash k7 near parity with MTP k2 at best and does nothing for MTP's M = 3. The MMA kernel
(`INTARSIA_METAL_MXFP4_MR=8`) stays on the branch as an opt-in, bitwise-tested experiment.

## Phase 1 ABAB rerun on the merged tree (fleetq #8861, `raw/p1-bench2/`; binary at 3fc8a028d, MR default on, no other knobs)

Gates first (fleetq #8858, `raw/noop2/`, `raw/ids2/`): `plan show` 9 / 9 byte-identical vs main (8B / 27B / 35B x default / off
/ mtp); `kernels test` every existing row identical, 2 new rows ADMITTED; greedy ids == `--spec off` for MTP k2, DFlash2 k3 and k7 at short-0
and 4k-1.

Deviation from the phase 1 rule as written: #8861 ran with no knobs (MR is default on; `INTARSIA_METAL_LOGITS_ROWS` is not on this branch
any more and had no effect on the fp8-head main, #8822).

natural-v1 decode tok/s (localeval `speed-spec`, 256 forced tokens, median of 3; visits A B A B then dfl3):

| visit | arm | 128 | 1k | 4k | 16k | acceptance (accepted / proposed) |
|---|---|---:|---:|---:|---:|---:|
| 1 | MTP k2 | 58.6 | 50.9 | 48.5 | 40.5 | 0.714 |
| 2 | DFlash2 k7 | 53.5 | 28.5 | 27.2 | 19.2 | 0.372 |
| 3 | MTP k2 | 63.8 | 55.2 | 53.7 | 45.6 | 0.702 |
| 4 | DFlash2 k7 | 54.2 | 30.4 | 28.3 | 20.1 | 0.372 |
| 5 | DFlash2 k3 | 59.1 | 44.5 | 41.7 | 32.1 | 0.638 |
| #8866 v1 | MTP k2 (same window as TF) | 61.3 | 52.3 | 50.0 | 40.4 | 0.706 |
| #8866 v2 | TensorFold (DFlash2 tree, MLX 4-bit) | 135.0 | 88.1 | 87.3 | 63.3 | |

B / A of the two-visit medians (DFlash2 k7 / MTP k2): 128 0.88, 1k 0.57, 4k **0.54**, 16k 0.46. **Pre-registered rule (>= 1.05 at 4k,
no cell < 0.95): not met.** DFlash2 k3 (one visit): 4k 0.82x of MTP k2. TensorFold / MTP k2 at 4k, same window (#8866): **1.75x** (128 2.20x, 1k 1.68x, 16k 1.57x). The #8833 TF visit (78.3 at 4k) is not used:
that window ran both engines slow (MTP 33.6 / 39.6 at 4k vs 48.5-53.7 elsewhere).

Categories (`localeval quality --judge none`, 512 forced tokens, decode tok/s; MTP / DFlash2 from #8861; TensorFold and its same-window MTP k2
from #8866, so TF / MTP uses #8866's MTP):

| category | MTP k2 (#8861) | DFlash2 k7 | DFlash2 k3 | MTP k2 (#8866) | TensorFold (#8866) | k7 / MTP | k3 / MTP | TF / MTP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ho-chat | 50.3 | 27.4 | 41.9 | 52.1 | 83.6 | 0.54 | 0.83 | 1.60 |
| ho-code | 58.5 | 42.3 | 52.2 | 60.9 | 134.6 | 0.72 | 0.89 | 2.21 |
| ho-prose | 44.2 | 21.8 | 36.2 | 46.2 | 67.8 | 0.49 | 0.82 | 1.47 |
| ho-recall | 52.3 | 33.2 | 45.6 | 55.1 | 99.1 | 0.63 | 0.87 | 1.80 |
| rcp-code | 58.4 | 47.6 | 54.8 | 61.4 | 137.9 | 0.82 | 0.94 | 2.25 |
| rcp-prose | 44.5 | 22.7 | 36.5 | 46.4 | 67.5 | 0.51 | 0.82 | 1.45 |
| rcp-recall | 53.2 | 36.2 | 47.6 | 54.9 | 103.2 | 0.68 | 0.89 | 1.88 |

Verdict, phase 1: correct (bitwise ids, admitted kernels, serve works), slower than MTP k2 everywhere (0.46-0.88x; k3 0.82-0.94x). The
acceptance half is there (3.6-3.8 tokens a pass vs MTP's 2.5), the cost half is not: an 8-row bitwise verify costs ~3-4x a plain step on
Metal (the M-row MXFP4 GEMV is 4x of M = 1 at M = 8), plus an 18-25 ms draft. Phase 2 (trees): design and host tree builder (`dflash_tree.rs`, preorder,
acceptance, unit tests) done; the coordinator deferred tree work until the GEMV result, which was a miss, so the verify plumbing was not
started and phase 2's pre-registered rule was not run. At today's per-row cost a 16-row tree would be slower than the 8-row chain.

## Recommendation

Keep Metal DFlash2 opt-in only as a correctness-complete port (it costs nothing when `--spec dflash` is not passed: plan show identical),
but do not recommend it to users on Metal: MTP k2 is faster in every category and context length measured. It becomes worth revisiting
only if the verify cost per row drops (a non-bitwise M-row path, which the user has ruled out, or new hardware matrix units reachable from
a bitwise path). The TensorFold gap on Metal (same window: 1.75x at natural-v1 4k, 1.45-2.25x by category) is a verify-cost gap under our bitwise constraint,
not an acceptance gap.

## Merge gates (branch merged with main ea5c6dcda, #325; raw: `raw/merge/`)

Merge commit `ceeb504eb` (textually clean, no `LOG.md` conflict). Semantics against #325 (DFlash TP, window split default on): `INTARSIA_DFLASH_TP` is read
in `model.rs` only for `tp.world > 1` (Metal runs tp = 1) and `OpAttrs::win_split` is consumed only by the HIP kernel registered for it, so neither reaches the Metal path
(`grep win_split` finds no Metal reader); every new `INTARSIA_*` name has a row in `docs/tuning.md`. Lane binary at the merged head vs a base binary at
origin/main `ea5c6dcda`, both `--release --features metal`.

| Gate | Result | fleetq |
|---|---|---|
| Metal `plan show --json`, 8B / 27B / 35B x default / `--spec off` / `--spec mtp` (`scripts/dm/noop.sh`) | 9 / 9 byte-identical | #8869 (endor, exclusive) |
| Metal `kernels test` (base 49 kernels / 1010 cases, lane 51 / 1136) | no FAIL; diff is exactly the 2 known rows `dyn_conv_bf16` and `attn_decode_bf16_window` (ADMITTED) | #8869 |
| Greedy ids == `--spec off` (256 tokens, short-0 and 4k-1): MTP k2, DFlash2 k3, DFlash2 k7 | 6 / 6 PASS | #8869 |
| `cargo test --workspace --features metal --no-fail-fast` | all pass except `intarsia-backend-metal --lib` `attention_is_registered_for_both_ops` (counted 3 `attn_decode` kernels; the lane's window kernel makes 4): a stale count in the lane's own test, fixed in this PR (13 / 13 pass after) | #8869 |
| CUDA (spark1, vLLM left up, `scripts/dm/cuda-noop.sh`): `plan show --json` 35B and 8B base vs lane | byte-identical; `kernels test --quick` exit 0, 0 FAIL | #8868 |
| HIP (tatooine) | not run: radiance holds both R9700s and the box has no free memory, and a HIP build needs the dev image and a cpu slot; the HIP-visible change is nil (no HIP file touched; shared `model.rs` / `load.rs` / `dflash_tree.rs` changes are covered by the CUDA plan check) | - |
