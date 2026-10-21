# Decisions

This directory holds two kinds of decision record.

| File | What it is |
|---|---|
| `NNNN-*.md` | Architecture Decision Records (ADRs). One per large, long-lived design choice (kernel ABI, tensor parallelism, Metal backend, continuous batching). Numbered, written once, superseded rather than edited. |
| `LOG.md` | The consolidated decision log. One row per decision of any size: a default switched on or off, a gate threshold, a lever parked, an approach reverted, a scope cut. It indexes the ADRs, the "Decisions" sections of `docs/results/*.md`, `docs/research/*`, `docs/plans/*`, project memory and unanimis. Above the table it has four short sections: open decisions waiting on the user, the six answers of 2026-10-20 23:24, the defaults in force on main, and contradictions between records. |

Use an ADR when the choice shapes the architecture and needs its own argument. Use a `LOG.md` row for everything else. An ADR also gets a row in `LOG.md`.

## Rules for lane agents

1. **One row per decision.** A decision is a choice between alternatives that someone made: a flag's default, an accuracy gate, a value adopted, a path abandoned, an item parked, a scope cut. A measurement or a finding is not a decision. The decision it leads to is.
2. **Same PR as the change.** Add the row in the PR that makes the decision real (code, results doc or plan). A row never lands alone afterwards; a PR with a decision but no row is incomplete.
3. **Newest first.** Insert the row at the top of the log table, below the header. Do not reorder old rows.
4. **Columns**, in this order:
   - **When**: `YYYY-MM-DD HH:MM` in EDT. Use the time the decision was made, not the time of the commit. Drop the time only if the source gives none.
   - **Decision**: one sentence, concrete: name the flag, default or value (`INTARSIA_HIP_MOE_BLOCK` is on by default on HIP; `--prefill-chunk 4096` stays a flag).
   - **Who**: `user` only when the user decided (quote or timestamp in the evidence). `ADR-NNNN` when an ADR decided it. `agent (<lane>)` when an agent decided inside its lane. `open` when the user still has to decide.
   - **Why**: one line, with the number that drove it if there is one.
   - **Evidence**: repo-relative doc path (add `#section`), PR number (`#28`), unanimis id (`unanimis 595e50c6`), fleetq job (`fleetq #2745`). The evidence must let a stranger re-check the decision.
   - **Status**: `active`, `superseded: <by what>`, `reverted (<ref>)`, `parked`, `resolved: <how>` (a question the source left open that was later settled), or `open`.
5. **Changing an old decision.** Add a new row for the new decision and edit only the **Status** cell of the old row (`superseded by <new date and short name>`). Never delete a row.
6. **Open decisions.** A decision waiting on the user goes in the "Open decisions waiting on the user" section at the top of `LOG.md`, with the options and the recommendation. When the user answers, move it into the table as a normal row (`who` = `user`) and remove it from the open section in the same PR.
7. **Defaults.** Whenever a row says a feature is on or off by default, the row's evidence must be the doc or PR that sets it, and the PR also updates the "Defaults in force on main" table at the top of `LOG.md`. If two records disagree about a default, add a line to "Contradictions between records" in `LOG.md` instead of picking silently, and say which record the code supports.
8. **Pipes and length.** Escape `|` inside cells as `\|`. Keep a row under about 400 characters; put detail in the results doc and link it.
9. **Unanimis.** Also capture accepted user decisions in unanimis (`origin project: intarsiaml`) and cite the id in the evidence cell. `mem:<name>` in an evidence cell means a project memory note under `~/.claude/projects/-opt-code-intarsiaml/memory/<name>.md`; it is local to the user's machine, so prefer a doc path or unanimis id when one exists.

## Row template

```
| 2026-10-21 14:05 | `INTARSIA_X` is off by default; opt-in only | agent (hip-round10) | -0.6 % decode at 128-2048 tokens, +1.5 % only above 13k | docs/results/2026-10-21-example.md#decisions; #35 | active |
```
