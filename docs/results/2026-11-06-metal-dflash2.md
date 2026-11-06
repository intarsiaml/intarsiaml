# Metal DFlash2 port (Qwen3.8-27B dense), then tree drafts

Lane dm, branch `lane-dm/metal-dflash2` (from `origin/lane-df/27b-dflash-gap`, PR #322).
Worktree `/opt/code/intarsiaml/.worktrees/metal-dflash2`.

## Decisions

- 2026-11-06, user, verbatim: "Start it now" (the Metal acceptance work: DFlash2 port to Metal, then tree drafts).

## Goal

Close the acceptance half of the Metal gap against TensorFold (90.3 tok/s at natural-v1 4k vs our MTP k2 31.1, 0.34x;
TensorFold keeps 3.2-4.4 tokens/round from a ~15-node DFlash2 tree, our MTP keeps 2.1-2.4).
