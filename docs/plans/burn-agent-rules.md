# Rules for every quota-burn lane agent (2026-10-21 → 10-22 12:00 EDT)

Read this file in full before you start. The plan is `docs/plans/2026-10-21-quota-burn-plan.md`.

## Fleet (mandatory)
- `/opt/code/fleetq/README.md` applies. Every GPU job or benchmark on spark1, spark2, tatooine or endor takes a fleetq slot. localeval takes its own slot.
- Never `--force` or `--no-queue`. Never release another owner's job. Never work around the hook. Never leave a slot held.
- **Taking a role down:** each window is ONE `fleetq run ... --takes-down <role>` command. It runs a script with `trap '<restore role; wait for /v1/models>' EXIT`, so the role comes back even if you die mid-window. Log every window in your results doc's `raw/downtime.log`.
- **Downtime caps per role per day:**
  - 3 h in total;
  - windows of 30 min or less;
  - at least 5 min of serving between windows;
  - from 00:00 to 07:00 the cap is lifted.
- **Roles:**

  | Role | Where |
  |---|---|
  | spark1-vllm | flight/guidance/capcom |
  | spark2-vllm | retro |
  | radiance | tatooine, both R9700s |
  | endor-kev, endor-tensorfold | standing services; don't take them down |

- **Sparks:** agent-harness has priority on the Sparks' vLLM during the day. Take down spark vLLM only in windows agreed with agent-harness, which the coordinator gives you. Non-exclusive work that leaves vLLM up is fine when memory allows.
- **tatooine:** blocked while fleetq #2898 (the user's RAM/memory test) runs. Queue behind it; don't touch it.
- Kill every process or container you start.

## Code and merges
- Work in your own worktree and branch. The base is the latest origin/main.
- **Defaults** must stay bitwise identical to base. Anything that changes numerics is opt-in, ships with a KL21 number, and is left for the user to decide.
- **Every change needs a real check:** `cargo test --workspace`, `cargo check --workspace`, the feature checks for the backends you touched, and the GPU gates for the backend you touched. The GPU gates are:
  - KL21;
  - 256 greedy ids;
  - prefix reuse;
  - goldens;
  - mutants for sync or order logic;
  - admission.
- **Footprint:** run `git diff --name-only origin/main HEAD | grep -vE '^(docs/|crates/intarsia-backend-<be>/|kernels/<be>/|scripts/|tools/)'`.
  - **Empty:** you may push, run `gh pr create --base main`, then `gh pr merge --merge --delete-branch`, and remove the worktree. The user pre-authorised this.
  - **Not empty** (shared files): don't merge. Report the file list and its effect on each backend to the coordinator. The coordinator runs the Metal no-op on endor, and on spark1 for CUDA, before merging.
- PR bodies end with: `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Benchmarks:** use interleaved A/B ×3 with `--at-most` on both sides. Comparisons against other engines use localeval (`/opt/code/deepeval`, the `local-model-bench` skill), never ad-hoc scripts.

## Reporting
- **Results doc:** `docs/results/2026-10-21-<lane>.md` plus `raw/`. Order: headline, numbers, gates, files, downtime, decisions.
- **Unanimis:** record decisions and findings as accepted or captured, with "origin project: intarsiaml".
- **Final report to the coordinator:** under 300 words. Order: headline, numbers, merged or PR link, decisions for the user.
- **If you are stuck:** after two failed attempts at the same fix, stop and name the assumption that might be wrong.
