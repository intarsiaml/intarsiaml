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
  (coordinator, see Decisions).

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
