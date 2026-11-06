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
