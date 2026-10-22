# fx7: ARCHITECTURE.md status markers (review finding 7)

Lane fx7, 2026-10-22, docs only, no fleet box. Base `origin/main` cd617de2.

## What changed

- `docs/reviews/2026-10-22-fable-adversarial.md`: the Fable 5.1 review (run 2026-10-21 23:18 EDT) with a Response table
  mapping findings 1-12 to lanes (finding 9 is deferred to the user).
- `docs/ARCHITECTURE.md`: a "Status vs plan" table at the top and a claim / status / evidence table after every
  subsection of sections 1 to 13 (sections 3 to 5 are covered claim by claim). Plan text is unchanged.
- `docs/assets/architecture.svg`: HIP and Metal backends added, dashed PLANNED strip, legend. No `lane-ob` branch or PR
  existed when this was done (`git branch -r | grep lane-ob`, `gh pr list`), so the SVG was edited; if ob lands a dark
  variant later, it needs the same strip.
- `docs/architecture-diagram.md`: legend paragraph.

## Method

Each claim was checked against the code with `grep` and `ls` in the lane worktree created from origin/main. Links are
relative and were checked to resolve (script: every `](../...)` target exists). Status rules: **implemented** needs a file
that does the job; **partial** names what is missing; **planned** means no code.

## Tally (rows in the per-section tables)

| Status | Rows |
|---|---|
| implemented | 70 |
| partial | 50 |
| planned | 37 |

(The top table is a summary and is not counted.) Biggest gaps: paged KV, radix prefix cache, host KV tier, autotuner and
tuning DB, kernel packs, `intarsia-quant` crate, GGUF and AWQ/GPTQ, MLA / sliding window / Mamba2 / LayerNorm, RCCL, fused
all-reduce, `/v1/responses`, OTLP, run manifest and replay.

## Not done

- Rows were verified by presence or absence in the code, not by running anything. "Partial" rows that say a behaviour is
  decided in model code rather than by the plan compiler rest on reading `model.rs` call sites only briefly.
- Sections 0 (goals, competitor gates), 14 (risks) and the research links are claims about targets or process, not code,
  and carry no marker; section 13 has a one-paragraph status.
