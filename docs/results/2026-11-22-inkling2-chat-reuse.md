# Inkling-2 serve: KV prefix reuse across chat turns (lane ink-spec-land)

Branch `ink/chat-reuse` from main 22303d704 (worktree `.worktrees/ink-chat-reuse`). Not merged during the #491 freeze; report
before merging (it changes output on multi-turn chat).

## Brief (coordinator, 2026-11-22 ~16:30; the coordinator's message names it "L2c: chat prefix reuse")

- Problem: chat turn 2 TTFT is 1.55 s at about 4.6k against 0.085 s when the prefix is reused, about 18x (#11634, `--spec off`
  server: turn 2 4593 prompt rows, 0 cached; the `/v1/completions` continuation 4560 cached).
- mlx-serve keeps a ring of window + 2048 rows per sliding visit (`.worktrees/mlx-serve-src/src/plugins/inkling2/AGENTS.md`, "A
  sliding visit keeps a ring of window + 2048 rows"). Read it for the design; don't copy it.
- Before building: the memory cost per stream and at B16 for margins 256 / 1024 / 2048, and which margin covers the template's
  ~30-row divergence with room to spare. Compare a larger margin with a ring snapshot at each prompt boundary; pick one.
- Build here, with an env that reverts to margin 16.
- Gates: single-turn md5 64/256 and G2/G3 unchanged; batched == solo digests; prefix-store tests pass; chat flow: turn 2 reuses
  >= 4000 rows and TTFT <= 0.2 s at about 4.6k; the turn-2 text equals a full re-prefill's, or the difference is explained as
  prefill-vs-decode numerics with a greedy G2-style KL on turn 2 against a full re-prefill; server memory at max-batch 16 within
  the measured budget.
- Report before merging; nothing goes to main during the #491 freeze.
