# Inkling-2-Mini on HIP: combined speed on final main (lane ink-hip-combined, 2026-11-26)

Branch `ink/hip-combined` (worktree `.worktrees/ink-hip-combined`), base main `178d87ece` (#536 merged, #537 merged). Docs only.

## Briefs received

1. 2026-11-26 ~17:25 EST (coordinator), verbatim in substance: #536 is merged into main as `178d87ece`, #537 is merged too. One
   last brief: a combined speed measurement on the final main. Record this brief in a new results doc first
   (`docs/results/2026-11-26-inkling2-hip-combined.md`, new branch `ink/hip-combined` from origin/main, worktree
   `.worktrees/ink-hip-combined`).
   1. Build on tatooine (fleetq, `--load cpu`): the CLI at origin/main `178d87ece`, and the CLI at `80719483a` (main before today's
      #533 / #534 / #536 / #537). Reuse a binary already built for `80719483a` if it is that exact commit.
   2. One window (at most 5 min radiance down; split into two windows under 5 min each if needed, at most 8 min in total). 4 + 4
      processes ABBA on both cards, same protocol as before, 256 tokens. Arms: new = `178d87ece` defaults, old = `80719483a` defaults.
      Measure spec-default decode at short / 1k / 4k / 16k (new defaults: k5, P12, vrow); plain `--spec off` decode at short / 4k /
      16k; prefill at 4k and 16k. If time runs short, drop plain 16k first and then plain short.
   3. The same window: greedy md5 of new against old at short / 4k, 64 / 256 tokens, `--spec off` and spec default. They must be
      equal; report any difference at once.
   4. Write the results doc. Lead with one table "start (11-25) / before today (`80719483a`) / now (`178d87ece`)", the start column
      from the 11-25 numbers already in docs/results with a source line each. Ratios with 95 % CIs. One docs PR with this doc only,
      no LOG.md row. Do not merge.
   5. Report the table and the md5 result, then list every leftover (worktrees, tatooine folder and volume); the coordinator removes
      them.
   Rules as before: fleetq for every job, work folders only under `.worktrees/`, real checks.

## Binaries

- **old** = `intarsia-main2`, main `80719483a` exactly (built in lane ink-hip-verify's CPU job #12153, md5
  `2f41908f6b79136e7a59d0f5cbe1d92a`; reused).
- **new** = main `178d87ece` (built for this doc, see below).
