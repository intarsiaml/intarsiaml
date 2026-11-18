# copymetal: Metal copy drafts re-measured at len 15 + VERIFY16

## Brief

## Question
Should copy drafts (`INTARSIA_SPEC_COPY=20`) become default-on for Metal serving at `--max-batch 1` with `--spec mtp`? Lane mcopym measured n20 at len 7 (`docs/results/2026-11-17-mcopym.md`: copy3k 1.75x, json 1.54x, agent-edit 1.20x, worst cells ~0.98x = noise per its diag). Since then main changed: fused GDN verify default-on (#445), and PR #456 (merging now / just merged — wait until `gh pr view 456 --json state -q .state` prints MERGED before measuring, polling no more often than every 4 min; if not merged in 60 min, report and stop) makes bare `INTARSIA_SPEC_COPY=20` mean len 15 on Metal with `INTARSIA_METAL_VERIFY16` on. #456's one-shot sanity showed json-xform 0.95x vs len 7, while an earlier 4-round ABBA said 1.006x — this lane settles it.

## Method
Reuse the mcopym / mmoe16 / mlen15 harness (`scripts/mcopy*`, `scripts/mmoe16*`, `scripts/mlen15/`; results docs `docs/results/2026-11-17-mcopym.md`, `2026-11-18-mmoe16.md`, `2026-11-18-mlen15.md` for exact commands, model `$HOME/models/Qwen3.6-35B-A3B-NVFP4`, cells). Served, 4 ABBA rounds, current origin/main (after #456) defaults. Arms: base (copy unset); `INTARSIA_SPEC_COPY=20` (= len 15 now); `INTARSIA_SPEC_COPY=20,7`. All cells: chat-short, code-gen, reason, agent-edit, agent-batch, copy3k, json-xform (whatever the harness defines). Bar (lane mcopy2): no cell below 0.995x vs base, agent cells ≥ 1.03x. State pass/fail per cell plainly, and which len is better per cell. Ids: greedy ids equal vs `--spec off` on copy3k and json-xform for both copy arms.

## Fleet rules
Every build/run on endor takes a fleetq slot: read /opt/code/fleetq/README.md and /opt/code/fleetq/deploy/LOCAL.md; `fleetq status` first; `--host endor`, always `--expect MIN` (guess high), `--quiet gpu,cpu` for timing; label "copymetal ...". FLEETQ_OWNER=claude-0b386a17-copymetal. endor runs standing services (kev, tensorfold) — leave them alone. Lane `herdflip` also needs endor briefly; fleetq orders you. Never leave a slot held; never release another owner's job; kill only by PID (no broad pkill).

## Work folder & hygiene
Only `/opt/code/intarsiaml/.worktrees/copymetal/` (create with `git worktree add` from origin/main after #456 merges, branch `copymetal`). Per-worktree CARGO_TARGET_DIR, and a separate one for tests vs the `--features metal` binary. PR body in `.worktrees/copymetal/pr-body-copymetal.md`. Add a LOG row at the TOP of the dated table in `docs/decisions/LOG.md` ("measurement, no default changed"). Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. PR title `copymetal: Metal copy drafts re-measured at len 15 + VERIFY16`. Do not merge. No single tool call or wait over 4 min (Bash timeout ≤ 240000): long work with nohup, poll at most every 4 min.

Report: PR URL, per-cell medians (both copy arms / base) with round spread, pass/fail vs bar, best len per cell, ids result, recommendation in one line.

## Summary

Date 2026-11-18, endor (Apple M5 Max, Metal), `Qwen3.6-35B-A3B-NVFP4`, `serve --max-batch 1 --max-seq 8192 --spec mtp --spec-k 2`. One binary: origin/main `70e39e19d` (after #456, with fused GDN verify and `INTARSIA_METAL_VERIFY16` default-on), the arms differ by environment only. No code change, no default change. Every run in a fleetq slot (`--quiet gpu,cpu`). Scripts `scripts/copymetal/` (the cells, `cell.py` and `tables.py` are mcopy's); raw in `docs/results/2026-11-18-copymetal/raw/`.

- **Answer: yes on the numbers.** Bare `INTARSIA_SPEC_COPY=20` (len 15 + VERIFY16) passes the mcopy2 bar on every cell and is the better or equal length on every cell. json-xform vs len 7 is a tie (1.003x), as the earlier ABBA said; the served ABBA does not reproduce #456's one-shot 0.95x (that was a single `generate` run per arm; no `generate` A/B was run here).
- **Bar** (no cell below 0.995x of base, agent cells at least 1.03x): **`=20` (len 15) passes all 7 cells; `=20,7` fails code-gen (0.990x)**, and only because of one slow server in round 1 (below).
- **Ids: PASS**, greedy `generate` ids of copy3k (2902) and json-xform (2600) are identical to `--spec off` for both copy arms (4 of 4). Served replies of both copy arms have the same sha as base's in every rep of every round (`tables.py` prints no `IDS DIFFER`).

## Speed: 4 ABBA rounds x 3 reps, served (`raw/ab/`, `raw/ab-table.txt`)

Cell value = median decode tok/s over all reps of all rounds [min-max]; ratio = median over rounds of the round's median ratio to base; the four per-round ratios (rounds 1 to 4) in brackets. Arm order: base, c15, c7 on odd rounds, reversed on even rounds; one fresh server per arm and round, cooled to thermal level 0 before each start (levels 0 to 2 during runs).

| cell | base | c15 = `INTARSIA_SPEC_COPY=20` (len 15) | c7 = `INTARSIA_SPEC_COPY=20,7` |
|---|---|---|---|
| copy3k | 211.9 [200-213] | 444.6 [429-447] **2.102x** (2.119, 2.100, 2.105, 2.084) | 382.6 [301-397] **1.716x** (1.560, 1.587, 1.850, 1.846) |
| json-xform | 204.0 [189-210] | 334.3 [327-350] **1.639x** (1.754, 1.646, 1.614, 1.631) | 330.7 [261-348] **1.625x** (1.477, 1.654, 1.597, 1.653) |
| agent-edit | 213.3 [201-217] | 259.2 [214-264] **1.219x** (1.263, 1.217, 1.218, 1.220) | 220.8 [178-257] **1.122x** (0.955, 1.055, 1.189, 1.192) |
| agent-batch | 221.2 [210-223] | 236.0 [235-237] **1.075x** (1.109, 1.092, 1.058, 1.055) | 231.9 [167-235] **1.052x** (0.879, 1.056, 1.051, 1.052) |
| chat-short | 174.7 [165-181] | 180.9 [177-190] **1.037x** (1.105, 1.049, 1.000, 1.024) | 176.8 [147-186] **1.002x** (0.890, 1.055, 0.985, 1.020) |
| reason | 210.4 [201-218] | 216.2 [214-221] **1.032x** (1.068, 1.036, 1.010, 1.028) | 213.7 [162-221] **1.006x** (0.984, 1.040, 0.983, 1.027) |
| code-gen | 204.3 [194-214] | 207.6 [206-213] **1.024x** (1.062, 1.027, 0.983, 1.021) | 205.0 [173-214] **0.990x** (0.902, 1.034, 0.966, 1.014) |

Lowest cell ratio: c15 1.024 (code-gen), c7 0.990 (code-gen).

### Pass / fail against the bar (no cell below 0.995x, agent cells at least 1.03x)

| cell | c15 (len 15) | c7 (len 7) | better length |
|---|---|---|---|
| copy3k | PASS 2.102x | PASS 1.716x | **15** (c15/c7 1.230x; every round) |
| json-xform | PASS 1.639x | PASS 1.625x | **tie** (c15/c7 1.003x; rounds 1.188, 0.995, 1.011, 0.987) |
| agent-edit (agent, >= 1.03) | PASS 1.219x | PASS 1.122x | **15** (1.089x) |
| agent-batch (agent, >= 1.03) | PASS 1.075x | PASS 1.052x | **15, small** (1.020x; rounds 1.262, 1.034, 1.007, 1.003) |
| chat-short | PASS 1.037x | PASS 1.002x | 15 by 1.0 %, within noise |
| reason | PASS 1.032x | PASS 1.006x | 15 by 1.4 %, within noise |
| code-gen | PASS 1.024x | **FAIL 0.990x** | 15 by 1.2 %, within noise (c7's fail is round 1, below) |

Caveat on the c15 bar: the cell medians pass, but one single round of one cell is under 0.995x (code-gen round 3, 0.983x); chat-short round 3 is exactly 1.000x. Per-round c15 ratios of the three control cells are 0.983 to 1.105.

### Notes

- **json-xform, the open question: settled as a tie at about 1.63x base for both lengths.** Per round c15 / c7 is 1.188, 0.995, 1.011, 0.987 (round medians c15 333 to 335 tok/s every round; c7 280, 336, 332, 338). #456's `generate` one-shot (0.95x vs len 7) is not reproduced in the served runs; it was one run per arm. In the ids run most chains run the full length (full 15 in 138 of 183 len-15 passes, full 7 in 265 of 323 len-7 passes; `raw/ids.log`) and the rest stop early; why the longer pass gains nothing here is not analysed.
- **Round 1 is a slow-server round for c7, again** (the same pattern as mcopym round 1): every c7 cell is 0.88x to 1.0x of base in round 1 (copy3k 327 tok/s against 336, 392, 392 in rounds 2 to 4; chat-short 149 against 177 to 185), and every c15/c7 ratio in round 1 is 1.09 to 1.36x. Without round 1, c7 on code-gen is 1.014 / 0.966 / 1.034 (median 1.014) and its other cells are all at or above 0.98x. c15's round medians are stable across rounds (copy3k 443 to 446 tok/s in all four). The tables above keep round 1; the per-round numbers are in `raw/c15-vs-c7.txt`.
- **Every knob-on arm sits about 2 % above base on the control cells, for both lengths.** In rounds 2 to 4 (`raw/c15-vs-c7.txt`) c7 is 1.020 / 1.027 / 1.014 and c15 1.024 / 1.028 / 1.021 of base on chat-short / reason / code-gen, so c15/c7 there is 0.993 to 1.028: no len-15 effect. The larger c15 numbers in the table come from round 1, where c7 and base were both slow servers (base json-xform 190 against 203 to 208, code-gen 195 against 203 to 211), which inflates every round-1 ratio. About 1089 of the 1114 copy passes per server are copy3k and json-xform reps, so chains barely fire on the controls; the likely reading is a roughly 2 % offset between knob-on arms and base (cause not investigated), so every copy ratio here may carry it. It does not change the verdict: agent-batch 1.075x still clears 1.03x with 2 % removed.
- Copy3k at len 15: 2.10x against base, 1.23x against len 7 (mmoe16, with the fused GDN verify opt-in on, measured 1.138x).
- Base metrics (all cells, one server): 8058 to 8420 MTP steps, 95.7 % of proposed draft tokens accepted; c15 and c7: 2676 to 3047 MTP steps plus 1114 / 2091 copy passes.

## ids gate (`raw/ids.log`, `raw/ids/`)

`generate --chat --no-think --temperature 0 --show-ids`, `--spec off` against `INTARSIA_SPEC_COPY=20` and `=20,7`, (`--spec mtp --spec-k 2`, `INTARSIA_SPEC_TRACE=1`):

| cell | `--spec off` ids vs `=20` (len 15) | vs `=20,7` |
|---|---|---|
| copy3k (2902 ids) | PASS identical (180 chain passes, 178 accepted 15 of 15) | PASS identical (359 chain passes, 357 accepted 7 of 7) |
| json-xform (2600 ids) | PASS identical (183 chain passes) | PASS identical (323 chain passes) |

## Method and commands

- Build (`scripts/copymetal/build.sh`, fleetq `--load cpu`): `cargo build --release -p intarsia-cli --features metal`, `CARGO_TARGET_DIR=target-copymetal`, from origin/main `70e39e19d` (merge of #456). `raw/build.log`.
- ids: `fleetq run --host endor --quiet gpu,cpu --mem 40 --expect 20 -- env ARMS="off copy=20 copy=20,7" scripts/copymetal/ids.sh copy3k json-xform` (fleetq #10246).
- Speed: `scripts/copymetal/run-ab.sh` = four `fleetq run --host endor --quiet gpu,cpu --mem 40 --expect 30 -- scripts/copymetal/round.sh <round>` (fleetq #10249 and the next three), `ab.sh` as mmoe16's (`serve --max-batch 1`, warm-up request, 3 reps per cell, greedy, no-think, per-round nonce, decode tok/s = (completion tokens - 1) / time after the first token). Seven cells of `scripts/mcopy/prompts.py`. Tables `python3 scripts/mcopy/tables.py raw/ab base`.
- Tests: docs only, no code changed; `cargo test --release -p intarsia-cli --test docs_env_vars` PASS (2 passed, fleetq #10263, `raw/docs-test.log`).
- Lane `herdflip` ran its own fleetq jobs on endor between my slots (never during a timing run); the standing services (kev, tensorfold) were up the whole time.

## Not done / unverified

- Sampled requests (temperature > 0), thinking mode, `--max-batch` above 1 (copy drafts are refused there), tp > 1, 16k-context cells, real agent transcripts: not measured. The agent cells are short (167 and 139 tokens), so their ratios rest on a few chains.
- Gate (b) (`plan show` unchanged with the knob unset) was #456's gate and not repeated; this lane changes no code.
- No `--spec off` served arm; the `--spec off` comparison is the `generate` ids gate and the served sha equality between base and copy arms.
- Commits are signed `Co-Authored-By: Claude Sonnet 5.5` (the model that did the work), not the `Opus 5.5` line of the brief.
- The roughly 2 % knob-on offset on the control cells is unexplained (above).

## Recommendation

Make `INTARSIA_SPEC_COPY=20` (len 15 on Metal) the default for `serve --max-batch 1 --spec mtp` on Metal: it passes the bar on all 7 cells (copy3k 2.10x, json-xform 1.64x, agent-edit 1.22x, agent-batch 1.08x, controls 1.02 to 1.04x, of which about 2 % is a knob-on offset seen at both lengths) and is never worse than len 7. Before flipping, the open items above (sampled requests, thinking mode, real agent transcripts, `--max-batch` above 1) need a cell or an explicit scope limit. Nothing is changed by this PR.
