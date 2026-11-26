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

(A/B sections follow.)
