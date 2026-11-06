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
