# `--spec auto` at tp=1, Qwen3.8-27B on one R9700 (HIP): re-measure, retune, default decision (lane au1, 2026-11-06)

Lane `lane-au1/spec-auto-tp1`, on `origin/main` `71e504101` (after lane dt1, PR #328). tatooine, `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`, `serve --backend hip --max-batch 1` on card 0 (tp=1), `localeval quality --judge none`, 512 forced tokens, the seven category files of `scripts/dfh/cats/` (36 rows), driver `scripts/au1/cat-window.sh` (copy of `scripts/dt1/` plus the auto arms). Raw: `docs/results/2026-11-06-spec-auto-tp1/raw/`.

Background: `--spec auto` was tuned at tp=2 on 10-28 (`2026-10-28-asp-spec-auto.md`, `2026-10-28-asa-auto-priors.md`) when a DFlash2 pass cost 38 ms against 25.5 ms for an MTP k2 pass. Since then the DFlash2 pass at tp=1 fell to about 54.5 ms (from 88.5), and DFlash2 k7 beats MTP k2 on code and recall but not on chat and prose (`2026-11-06-27b-dflash-hip-tp1.md`: DFlash2 / MTP k2 = 0.95 ho-chat, 1.19 ho-code, 0.84 ho-prose, 1.11 ho-recall, 1.31 rcp-code, 0.85 rcp-prose, 1.15 rcp-recall). A per-pass choice between the two could collect the code and recall wins; this lane measures whether `--spec auto` does so at tp=1, retunes it, and decides by the rule below.

## 0. Pre-registered rule (written and committed before any measurement)

Arms: `auto` (`--spec auto --spec-draft <DFlash2 draft> --spec-k 7`, the params under test), `mtp2` (`--spec mtp --spec-k 2`), `dfl7` (`--spec dflash --spec-k 7`). Per category: decode tok/s pooled as tokens / sum of decode times (`tools/dfh/cat_table.py`), mean of the rounds of the window.

1. Parameters are chosen from window 1 only (replay of its per-pass logs, `tools/asa/replay.py`, plus its tok/s) and frozen before window 2. Window 2 is the confirmation window; the decision uses window 2 only, with 2 rounds, arms alternated, the same rows, on the binary that would ship: any retuned constant (including the priors) is a code default (gated, see 4) in a rebuilt binary and the window-2 `auto` arm runs with no `INTARSIA_SPEC_AUTO` params beyond `pass,log`.
2. `--spec auto` (with the retuned params, k7 for DFlash) becomes the default for the 27B on HIP at tp=1 **only if**, in window 2, in **every one of the 7 categories**:
   * auto >= 0.99 x MTP k2 (within noise), AND
   * auto >= 0.97 x max(MTP k2, DFlash2 k7).
3. Also required: greedy ids of `--spec auto` (shipped params) and of a forced-switch arm (`INTARSIA_SPEC_AUTO=alt:1`) equal to `--spec off` on short-0 (256 ids) and 4k-1, in both windows.
4. Otherwise `--spec auto` stays opt-in; only the retuned built-in params are merged, and only if they improve auto over the current ones: the comparison is auto / max(MTP k2, DFlash2 k7) per category, window 2 (retuned) against window 1 (current built-in params, `auto` arm, same ratios to the same-window mtp2 and dfl7); "improves" = the mean of the 7 ratios rises and the worst ratio does not fall by more than 0.01. And only where the data justifies them: params that are global (`AutoParams::default`) stay as they are unless the tp=2 data of 10-28 still supports the new values, otherwise they are gated to tp=1 or not changed. No tp=2 behaviour, no other backend or model default changes.
5. No cherry-picking: every category and round of each window is reported, including rounds with problems; a category with a problem row count (not 36 rows of 512 tokens in every arm) counts as a failure of that category.

Window 1 also measures the structural floor of the machinery at tp=1: `--spec mtp` takes the captured graph step (`Model::spec_step`, ids traces of 11-06 show `mtp graph step`) and the pipe step, `--spec auto` takes neither (`self.auto.is_none()` in `engine.rs`), so auto's MTP passes are eager. Arms `autotrmtp` (auto pinned to the MTP head) and `autotrdfl` (pinned to DFlash) against `mtp2` and `dfl7` give that floor per category. If the floor alone is below 0.99 of MTP k2 in a category that auto must spend on MTP (chat, prose), the rule fails for that code reason regardless of the params; it is then reported as such, and enabling the graph / pipe step for auto's MTP passes is named as the next step, not folded into this lane.

Window 1 shape (45 min cap): ids pre-stage (short-0 and 4k-1: off, auto, alt:1), then one round of `mtp2`, `autotrmtp`, `dfl7`, `autotrdfl`, `auto` (current built-in params, `pass,log`); one round only because five arm visits already take about 30 min. The replay uses the request parity (even / odd rows) as its held-out split.

Wall: window 1 and window 2 each at most 45 min of radiance downtime (user approved).
