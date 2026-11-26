# Inkling-2-Mini on Metal: verify-path levers with DSpark on (lane ink-verify, 2026-11-25)

Branch `ink/verify` (worktree `.worktrees/ink-verify`), based on `ink/spec-flip-merge` `b2afb4eaa` (#496 work, DSpark on by default
on Metal, plus `0ad399998`). Nothing pushed. Two levers from the Neon list, each bitwise to what it replaces, each measured on
its own against the base:

1. **Neon #9, screened head on greedy verify groups of 2..=4 rows** (`INTARSIA_INK_VERIFY_SCREEN`).
2. **Neon #10, the DSpark draft's block attention as one row-table launch** (`INTARSIA_DSPARK_ATTN_ROWS`).

## Verdict (details below)

- Both levers are **bitwise** (spec-on greedy ids == spec-off ids at 143 / 1k / 4k, 256 tokens; lever 2 also byte-identical to the per-row
  calls in 1500+ real-model layer checks) and **engage** on the real model, in `generate` and in a default `serve`.
- Neither lever reaches the Neon estimates (+8-12 %, +3-6 %). See the A/B sections: the effect is at most about +1-2 %, inside the
  run-to-run noise of this box at 6-12 samples per arm. (JOB3 numbers go here.)

## What changed

### Lever 1: screened head for verify rows 2..=4

- `crates/intarsia-core/src/model.rs`, `Model::screened_argmax`: takes `c` in `1..=SCREEN_ROWS_MAX` (4) rows instead of 1. For `c > 1`
  (a verify group) it runs the fp8 screen matmul over the `c` rows (`matmul_fp8_gemv_rows`), one `HeadScreen` and one `ScreenArgmax`
  over the group, and **one `LogitsGather` per row** through the 1-row head kernel (`gemv_f32o_v` or P12's `p12_gemv_f32o` in index
  mode, T = 1 as before). A row whose screen is past the candidate path (count 0 or above the cap, non-finite) gathers its full row
  on the device in its own launch, with no host round trip (option B's design, kept). The per-row gathers were chosen over a union of
  the per-row screens or a multi-row gather kernel because the gather is the serving 1-row kernel itself: a gathered logit is that
  kernel's bits by construction, and the verify head's `_rows` kernels are bitwise the 1-row kernels (`tests/rows_exact.rs`), so
  the screened pick equals the full verify head's pick without new kernels.
- `Model::verify_head(.., greedy, ..)`: the new `greedy` argument is true only from `forward_verify_run` with `VerifyRead::None`
  (`Model::forward_verify`, the engine's greedy DSpark verify, `engine.rs` `spec_verify`) and from the plan dry run (which records the
  screen calls as `extra`). A sampled verify (`forward_verify_logits` / `_topk`) reads whole logits rows and takes the full head.
  Unchanged callers: batched verify, tensor parallel / vocab-parallel, captured passes (HIP / CUDA), DFlash's own head.
- Kept fallbacks: `vocab_par`, activation dtype, 16-byte alignment, `disabled`, the first-use probe of the 1-row gather
  (`hs.ok`), plus a per-group-size check (`hs.rows_ok`): the screen matmul, `HeadScreen`, `ScreenArgmax` and the verify head's own
  kernel (`logits_bf16_gemv_rows` / `matmul_p12_logits_gemv_rows` / the cpu dot) must be selectable at `exact_rows`, else the full head.
  Groups of more than 4 rows run the full head.
- Counters: `intarsia_head_screen_verify_groups_total` on `/metrics`, and the drop-time stats line of
  `INTARSIA_INK_HEAD_SCREEN_STATS=1` now says how many were verify groups.
- Test hooks: `INTARSIA_INK_HEAD_SCREEN_FORCE_FALLBACK` bit `i` now forces row `i` of a verify group (its `HeadScreen` runs over the
  +inf bounds, the other rows over the real ones: mixed candidate and fallback rows in one group).

### Lever 2: DSpark block attention in one launch

- `crates/intarsia-core/src/model/dspark.rs`, `Model::ds_attn_rows`: the `for rr in 0..t` loop of single-row `AttnDecode` calls (all
  rows of the block attend the same cache at the same position `last`, with the same window) becomes one row-table `AttnDecode`
  (`OpCall::rows`, the batched-decode kernel `attn_decode_bf16_splitkv_rows`, bitwise the one-row launch per row, `tests/row_tables.rs`),
  the same mechanism as `verify_attn.rs`. The table names the draft layer's K / V cache for every row, moved to the window's first key
  (the Metal windowed call is exactly a cache offset plus a position inside it, `launch_attn_window`), and the position vector holds
  `last - start`. One upload per block pass writes all layers' tables (`ds_attn_tab_fill`, `ATAB_LAYER` = 320 bytes per layer).
- Metal only (the cache-offset equivalence is Metal's windowed call); any other backend, a missing kernel, or too little scratch keeps
  the per-row calls (said once on stderr: `dspark attention rows: ...`). A plan dry run records the row-table call as `extra` (so the
  plan admits its kernel and sizes its scratch) and the per-row calls as before.
- Diagnostic `INTARSIA_DSPARK_ATTN_ROWS_CHECK=1`: runs the row-table launch and the per-row calls into the same buffer one after the
  other, compares the bytes after every layer (a sync and a download per layer), and prints `MISMATCH` on any difference.

### Knobs (docs/tuning.md rows)

`INTARSIA_INK_VERIFY_SCREEN`, `INTARSIA_DSPARK_ATTN_ROWS`, `INTARSIA_DSPARK_ATTN_ROWS_CHECK`. Run-time overrides
`screen::set_verify_override` and `model::set_dspark_attn_rows_override` (`#[doc(hidden)]` benchmark hooks) let one process alternate the
levers per request; the generate loop that calls them is a diag patch kept out of the tree.

## Gates

### Gate 1, bitwise (fleetq #11913, exclusive, 20:36-20:49; binary md5 `161d13ec`, tree `266dd813f`)

`generate --temperature 0 --max-tokens 256 --ignore-eos --show-ids` over three prompts in one process (143 tokens = the first 143
ids of the 1k prompt, 1021 and 4108 tokens: `longp.json` of lane ink-spec), spec on (DSpark default, k 3) against `--spec off`:

| arm | ids == spec-off at 143 / 1k / 4k | verify passes / proposed / accepted | engagement |
|---|---|---|---|
| spec off (new binary) | n/a (reference) | 0 / 0 / 0 | |
| base binary `b2afb4eaa` | yes / yes / yes | 219 / 504 / 404 (80.2 %) | |
| new, lever 1 (default) | yes / yes / yes | 219 / 504 / 404 | 219 verify groups screened, 0 full-head fallback passes (364 screened launches = 145 decode + 219 verify) |
| new, both knobs off | yes / yes / yes | 219 / 504 / 404 | 0 verify groups (145 decode launches) |
| new, lever 2 only (`ATTN_ROWS=1 VERIFY_SCREEN=0`) | yes / yes / yes | 219 / 504 / 404 | `dspark attention rows: engaged` |
| new, lever 2 only + `ATTN_ROWS_CHECK=1` | yes / yes / yes | 219 / 504 / 404 | `1500+ checks, 0 mismatches` (every draft layer of every draft step, all three lengths) |
| new, both | yes / yes / yes | 219 / 504 / 404 | 219 verify groups, rows engaged |

The spec-on == spec-off gate cannot see a non-bitwise draft (the verify pass corrects any proposal), so for lever 2 the gate is the
byte check and the identical proposed / accepted counters of the two arms. Identical counters in every arm also show that the draft
proposed the same tokens with and without the row-table launch.

`serve` smoke (default server, admitted kernels only, two greedy 1k-token chat requests, 160 tokens): default server, 70 spec steps,
70 screened verify groups on `/metrics`; with `INTARSIA_DSPARK_ATTN_ROWS=1` the log says `dspark attention rows: engaged` after
the plan and the counters are identical (70 steps, 126 proposed, 108 accepted).

### Tests

Cpu (no slot): `cargo test -p intarsia-core` 267 passed, 4 ignored; `cargo test -p intarsia-models --test inkling2` 18 passed (the new
`inkling2_tiny_screened_verify_head_equals_full_head`: R = 2..4 at four positions, every subset of rows forced through the full-row
fallback, a sampled verify never screens).

Metal, same slot #11913 (`INTARSIA_INKLING2_BACKEND=metal`, release): `inkling2` 18 passed, `inkling2_batch` 8, `inkling2_dspark` 4,
`inkling2_dspark_cut` 1, `inkling2_verify_softcap` 1 (the tiny head is not 16-byte aligned on Metal, so the tiny screen tests
skip there; the real-model runs above are the Metal coverage of lever 1); `inkling2_dspark` again with `ATTN_ROWS=1 ATTN_ROWS_CHECK=1`:
4 passed, 0 mismatches (the tiny draft's shapes have no row-table kernel, so the path does not engage there); Metal backend
`row_tables` 8, `rows_exact` 11 (4 ignored), `screen` 4 passed.

## Gate 2, speed

Prompts as in gate 1 (143 / 1021 / 4108 ids), `generate --temperature 0 --max-tokens 256 --ignore-eos`, spec on (DSpark k 3, tau 0.8),
decode tok/s from the engine's own decode clock, TTFT = the engine's prefill ms. Every arm proposes and accepts the same tokens
(identical `verify passes / proposed / accepted` in every process), so the arms differ in time only.

**The box is not quiet.** Fleet slots are exclusive among fleet jobs, but other lanes run CPU builds and tests on endor at the same time
(load average 19-42 during the runs; `rustc`, `rsync`, `python` and desktop apps in `ps`). Decode is host-launch bound at these sizes,
so the same arm ran at 116 tok/s (143-token prompt) in one slot and 75-83 tok/s in another an hour later. Cross-process runs also carry the
per-process fast / slow mode L8 found (10-14 % at 16k). Both are bigger than either lever. I therefore ran the A/B twice, in two ways.

### A/B 1: cross-process ABBA against the base binary (fleetq #11914, exclusive, 20:49-21:06)

Arms: A = base binary (`b2afb4eaa`), B = new binary, lever 1 on (`INTARSIA_INK_VERIFY_SCREEN=1` at the time of the run, when the
default was on), C = new binary, lever 2 only (`INTARSIA_DSPARK_ATTN_ROWS=1 INTARSIA_INK_VERIFY_SCREEN=0`), D = both. Eight rounds,
order A B C D / D C B A alternating, each process runs the three prompts twice (so 2 samples per length per round, 16 per arm, and
**14 with round 4 dropped: a release build of mine ran on the box during it (20:55-20:57) and r04-B fell to 68 / 50 tok/s**).
Medians over the 14 samples; "ratio" is the median of the per-round, per-repeat ratio to arm A of the same round.

| decode tok/s (median, [min-max]) | 143 | 1k | 4k |
|---|---|---|---|
| A base | 116.3 [89-125] | 70.6 [64-80] | 84.3 [79-99] |
| B lever 1 | 111.9 [98-125] | 67.8 [62-80] | 85.6 [81-96] |
| C lever 2 | 116.2 [101-125] | 71.1 [65-80] | 85.7 [81-98] |
| D both | 116.5 [98-126] | 70.9 [66-80] | 86.1 [81-96] |
| B / A ratio | 1.003 | 1.001 | 1.002 |
| C / A ratio | 1.008 | 1.011 | 1.013 |
| D / A ratio | 1.009 | 1.014 | 1.023 |

Per-sample ratios range from 0.8 to 1.2 for every arm (min-max in the raw summary), the per-process mode shows plainly (round 1: A
96.8 and 97.3 tok/s at 4k, B / C / D 84.8-87.9; round 8: A 98.9 / 82.1 against 92-98 for the rest). TTFT (prefill ms) medians
within +-2 % of A for every arm and length, ratio medians 0.98-1.02, i.e. noise (1k and 4k B means are pulled up by single slow
processes, medians are not).

### A/B 2: within one process (fleetq #11916 and #11929; diag binary, levers flipped per request through run-time overrides)

One process alternates the four arms request by request (A = both levers off, the base code paths; B; C; D), cycle `c` runs each length with the
arms in order A B C D (even `c`) or D C B A (odd `c`), cycle 0 is warm-up, so a sample is compared with the A sample of the same length a few
seconds away. This removes the per-process mode and most of the drift. Nine cycles of 3 lengths x 4 arms per process; processes
p1-p3 (#11916, 21:15-21:44) JOB4POOL.

JOB3TABLE

Reading: no arm is faster than A by more than the noise at any length. Lever 1 (B) is, if anything, 1-3 % slower in the within-process
pairs (median 0.98 / 0.97 / 0.99), lever 2 (C) and both (D) sit between 0.97 and 1.05 depending on the length, with confidence intervals
that include 1.0 everywhere. TTFT cannot be affected by either lever (verify and draft only), and its pairs show the same +-5 % scatter
(one cell, lever 1 at 1k, reaches p = 0.02 in the favourable direction: chance among 24 cells).

### Why the estimates did not hold

- Lever 1: Neon's +8-12 % needs 2-3 ms per verify pass. A pass is about 25-35 ms (2.84 tokens per pass at 85-120 tok/s). The default head
  is P12 (`matmul_p12_logits_gemv`, 1-row 1.2 ms, 1.58 ms for bf16: `docs/results/2026-11-21-inkling2-fp8.md` 1b), and the screen's fp8
  copy is 0.46 GB against P12's roughly 0.7 GB. The screen can therefore save about a third of 1.2 ms (0.4 ms, ~1.2 % of a pass) before
  it pays for the extra launches (one fp8 matmul, a screen, an argmax and one gather per row, against one head launch plus an argmax).
  The measurement agrees: no gain, a hint of a loss.
- Lever 2: it removes `(t - 1) x layers` host launches per draft step (7 x draft layers; the one-row calls were 2 dispatches each). The
  cross-process ratios (+0.8 to +1.3 % alone) point the right way and the within-process ones (-2.5 % to +3 %) do not contradict them, but
  neither excludes zero.

