# bm: batched MTP speculation (`--spec mtp` with `--max-batch` > 1)

Lane `lane-bm/batched-mtp`, 2026-11-05. Base: `origin/main` 602a1fcfe.

## Approval

The user approved starting this work on 2026-11-05 at 17:08 EST ("start 2"), reversing their 2026-10-25 deferral (design note
`docs/results/2026-10-25-sp-batched-mtp-design.md` on `lane-sp/batched-mtp-design`, revision 2).

## Coordinator direction (2026-11-05, during the lane)

Copied from the coordinator's message: Fable's engine review (`.worktrees/perf-ideas/fable-engine/report.md`, idea 1) recommends two phases.
Do phase A first, push it as soon as it passes, and report it before phase B.

- **Phase A:** inside the batching engine, run solo MTP whenever only one slot is active (hooks in `engine/batch.rs`, `spec_off_at` exists).
  A lone user on a `--max-batch` server would get solo-MTP speed: Spark 35B 4k 80.8 -> 114, HIP 27B 56 -> 93, Metal 35B 124 -> 160 (plain batched -> solo MTP).
  Fable's estimate 12-20 h, bitwise, cheap kill test: a CPU tiny-hybrid equality check.
- **Phase B:** the multi-row (ragged) verify for 2-4 streams, as in the design note.

## Status

In progress.
