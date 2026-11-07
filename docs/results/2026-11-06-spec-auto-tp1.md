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

## 1. Window 1 (exploration; fleetq #8880, radiance down 21:01:13 to 21:33:21 = 32 min 8 s, answering 10 s after the guard started it)

Binary: `origin/main` code tree `c227c1d909c0` (`71e504101`), built on tatooine as `au1-intarsia` (tree `~/intarsiaml-au1`, volume `intarsia-cargo-au1`, fleetq build #8877). One round, order `mtp2, autotrmtp, dfl7, autotrdfl, auto` (five visits fill about 30 min; the arms `autotrmtp` / `autotrdfl` are `--spec auto` pinned to the MTP head / the DFlash draft, they exist to give the replay its pure per-pass traces and the machinery floor). 35 localeval runs, 0 problems (36 rows of 512 tokens in every arm). `auto` here = the **current built-in params** (tp=2 values of 10-28: theta 2.5, alpha_mtp 0.06, dwell_dflash 16, hyst 1.1, priors 25.5 / 38.0 ms), `INTARSIA_SPEC_AUTO=pass,log`.

| category | mtp2 | dfl7 | auto (current params) | auto / mtp2 | auto / best | rule | DFlash passes of auto |
|---|---:|---:|---:|---:|---:|---|---:|
| ho-chat | 64.1 | 60.7 | **63.4** | 0.9894 | 0.9894 | FAIL | 38% |
| ho-code | 79.8 | 95.7 | **96.6** | 1.2106 | 1.0102 | pass | 77% |
| ho-prose | 59.0 | 49.6 | **58.7** | 0.9949 | 0.9949 | pass | 0% |
| ho-recall | 69.7 | 77.6 | **74.8** | 1.0731 | 0.9638 | FAIL | 56% |
| rcp-code | 78.0 | 102.6 | **100.6** | 1.2887 | 0.9804 | pass | 77% |
| rcp-prose | 59.8 | 50.7 | **59.2** | 0.9891 | 0.9891 | FAIL | 4% |
| rcp-recall | 70.1 | 80.8 | **78.4** | 1.1176 | 0.9698 | FAIL | 63% |

Rule on window 1 (not binding, the decision is window 2): **not met** with the current params: ho-chat 0.989 of MTP k2, ho-recall 0.964 of the best, rcp-prose 0.989, rcp-recall 0.970 (borderline).

Machinery floor, auto pinned to a drafter against the plain arm of that drafter (tok/s, x of the plain arm):

| category | autotrmtp | x mtp2 | autotrdfl | x dfl7 |
|---|---:|---:|---:|---:|
| ho-chat | 63.8 | 0.995 | 62.4 | 1.027 |
| ho-code | 79.4 | 0.995 | 96.2 | 1.006 |
| ho-prose | 58.7 | 0.994 | 49.9 | 1.006 |
| ho-recall | 69.4 | 0.995 | 78.0 | 1.005 |
| rcp-code | 77.4 | 0.992 | 102.7 | 1.002 |
| rcp-prose | 59.5 | 0.995 | 51.2 | 1.010 |
| rcp-recall | 69.4 | 0.990 | 81.2 | 1.004 |

Reading: `--spec mtp` at tp=1 takes the captured graph step (`Model::spec_step`) and `--spec auto` does not (`self.auto.is_none()` in `engine.rs`: auto's MTP passes are eager), but the price is small: pinned to MTP, auto runs at 0.990-0.995 of `mtp2` in every category (mean 0.994). That is the floor; on a category where the best policy is pure MTP (prose, chat) auto cannot exceed about 0.995 of MTP k2, so the bar of 0.99 leaves about 0.5 percent for DFlash excursions and noise. Pinned to DFlash, auto is at 1.002-1.010 of `dfl7` (chat 1.027, not understood, one round). The pass costs in the logs are constant: an MTP pass 35.4 ms (2.1-2.8 tokens), a DFlash pass 50.2 ms (2.5-5.3), a fresh pass after a switch +1.3 ms to MTP and about +0 ms to DFlash; DFlash needs 1.42x the tokens per pass of MTP to tie (tp=2: 1.49x).

Greedy ids (stage `ids`, `raw/w1/ids/`, card 0 short-0 and card 1 4k-1, 256 ids): `--spec auto` (current params) and `INTARSIA_SPEC_AUTO=alt:1` (forced switch every pass) equal `--spec off`: **4 of 4 PASS** (alt:1 on short-0: 39 MTP and 38 DFlash passes, 76 switches).

### Replay and the choice of params (`tools/asa/replay.py --au1`, `tools/au1/pass_stats.py`)

The replay (CPU, per-pass logs of `auto`, `autotrmtp`, `autotrdfl`, one round = 36 requests; `REPLAY_ARMS=auto,autotrmtp,autotrdfl PRIOR_M=2.2,35.4 PRIOR_D=3.6,50.2`) searched `Policy2` (the mirror of `AutoPolicy`) over theta {2.2..2.5}, alpha_mtp {0.04, 0.06, 0.1}, dwell_mtp {2, 4, 8}, dwell_dflash {8, 16, 24}, hyst {1.0, 1.1, 1.2}, probe_mtp {6, 8, 12, 16}, alpha_dflash {0.1, 0.15} (3240 sets, `raw/w1/replay-grid.txt`), score per fold = the worst category of min(0.994 x auto / MTP / 0.99, auto / best / 0.97) (1.0 = the rule holds in the sim; 0.994 = the measured floor), folds = even / odd requests (tune on one, test on the other). Findings:

* The sim is optimistic: for the current params it gives recall and code +3.9 / +0.6 / +2.0 percent over the best pure mode where the real run reached -4.1 / -2.2 / -3.2 (per-pass log of the real run), i.e. a systematic 4-5 points, and single parameters move categories by 1-3 points in both directions (2 requests in ho-code and ho-recall). It cannot rank parameter sets closer than about 1 percent; the grid is used as a plateau and robustness check, not an optimiser, and window 2 is the test.
* Plateau: theta 2.5 is best on the worst-category score (lower theta gains recall but loses prose: theta 2.4 costs rcp-prose 2.5 percent in the sim); a slower MTP EWMA (0.04) protects prose (fewer DFlash excursions: rcp-prose 4 percent of passes on DFlash in the real run, all of them losing); a shorter DFlash dwell (8) and a wider hysteresis (1.2) cut the time spent on DFlash after its rate fell. Chosen (frozen before window 2): **alpha_mtp 0.04 (was 0.06), dwell_dflash 8 (16), hyst 1.2 (1.1), priors (2.2 tok, 35.4 ms) / (3.6 tok, 50.2 ms) (were 25.5 / 38.0 ms); theta 2.5, alpha_dflash 0.1, dwell_mtp 4, probe_mtp 8, probe weight 0.3, seed 2 unchanged.**
* In the sim, vs the current params, % vs the best pure mode (ho-chat .. rcp-recall): current -0.8 +0.9 0.0 +0.3 +0.7 -1.2 +0.7; chosen +0.5 +3.3 0.0 +1.2 +0.7 -0.2 +2.0. Worst category vs MTP: -0.8 to -0.2.
* tp=2 check: the same chosen params replayed on the 10-28 tp=2 logs (`docs/results/2026-10-28-asa-auto-priors/raw/w1`, 2 rounds, tp=2 priors): -0.6 +0.9 0.0 +2.6 -0.7 0.0 +1.1 against the current -0.4 +2.1 0.0 +3.1 -0.2 -0.2 +0.4: no gain, not worse beyond the sim's resolution. The tp=2 data does not support the new values on its own, so they are **gated to HIP with one rank** (`AutoParams::for_target("hip", 1)`, set by the loader; `INTARSIA_SPEC_AUTO` still wins); tp=2 and every other backend keep the 10-28 values.

## 2. Window 2 (confirmation, binary with the retuned params; fleetq #8883, radiance down 21:54:25 to 22:31:27 = 37 min 2 s, guard confirmed it answering 10 s after the start)

Binary: branch head `0481ba7` (code tree `541c1062786b` = main plus the tp=1 `AutoParams`, built as `au1-intarsia`, build #8882, checked for the new load line before radiance went down). `git diff 0481ba7 HEAD -- crates/` is empty: the merged code is the code window 2 measured. `auto` runs with `INTARSIA_SPEC_AUTO=pass,log` only: the retuned values are the code defaults (the serve log carries `--spec auto: policy values for hip with one rank (alpha_mtp 0.04, dwell_dflash 8, hyst 1.2, priors [(2.2, 35.4), (3.6, 50.2)])`). Two rounds, order `auto, mtp2, dfl7` then `dfl7, mtp2, auto`; 42 localeval runs, 0 problems (36 rows of 512 tokens in every arm and round). Decode tok/s pooled per category (`tools/dfh/cat_table.py`, `tools/au1/win_table.py`), mean of the two rounds:

| category | mtp2 | dfl7 | auto (retuned) | auto / mtp2 | auto / best | rule | DFlash passes of auto |
|---|---:|---:|---:|---:|---:|---|---:|
| ho-chat | 64.1 | 61.1 | **63.4** | 0.9897 | 0.9897 | FAIL | 20% |
| ho-code | 80.0 | 95.4 | **92.4** | 1.1553 | 0.9686 | FAIL | 79% |
| ho-prose | 59.0 | 49.3 | **58.8** | 0.9961 | 0.9961 | pass | 0% |
| ho-recall | 69.5 | 77.4 | **77.3** | 1.1112 | 0.9985 | pass | 68% |
| rcp-code | 77.9 | 101.8 | **99.9** | 1.2824 | 0.9809 | pass | 77% |
| rcp-prose | 59.9 | 50.8 | **59.5** | 0.9937 | 0.9937 | pass | 1% |
| rcp-recall | 70.1 | 80.7 | **79.5** | 1.1342 | 0.9848 | pass | 72% |


Per round (`tools/au1/rounds.py`; the policy is deterministic given the text, so rounds differ only by timing noise of 0.1-0.5 tok/s):

| category | auto r1 | auto r2 | dfl7 r1 | dfl7 r2 | mtp2 r1 | mtp2 r2 | auto / mtp2 per round | auto / best per round |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ho-chat | 63.42 | 63.45 | 60.98 | 61.13 | 64.15 | 64.04 | 0.989 / 0.991 | 0.989 / 0.991 |
| ho-code | 92.41 | 92.38 | 95.68 | 95.11 | 79.79 | 80.16 | 1.158 / 1.152 | 0.966 / 0.971 |
| ho-prose | 58.85 | 58.73 | 49.43 | 49.23 | 59.01 | 59.03 | 0.997 / 0.995 | 0.997 / 0.995 |
| ho-recall | 77.29 | 77.26 | 77.21 | 77.57 | 69.68 | 69.40 | 1.109 / 1.113 | 1.001 / 0.996 |
| rcp-code | 99.90 | 99.86 | 101.77 | 101.88 | 77.86 | 77.90 | 1.283 / 1.282 | 0.982 / 0.980 |
| rcp-prose | 59.45 | 59.59 | 50.79 | 50.83 | 59.86 | 59.94 | 0.993 / 0.994 | 0.993 / 0.994 |
| rcp-recall | 79.55 | 79.38 | 80.77 | 80.61 | 70.02 | 70.11 | 1.136 / 1.132 | 0.985 / 0.985 |

**Verdict of the pre-registered rule: NOT MET.** Two categories miss, both narrowly: ho-chat auto / mtp2 = 0.9897 (rounds 0.989 and 0.991; the bar is 0.99) and ho-code auto / best = 0.9686 (rounds 0.966 and 0.971; the bar is 0.97). The other five pass (ho-prose 0.996, ho-recall 0.999 of the best, rcp-code 0.981, rcp-prose 0.994, rcp-recall 0.985). `--spec auto` stays opt-in.

Why the two miss (`tools/au1/pass_stats.py`, per-pass logs in `raw/w2/pass-auto-r*.log`):

* ho-chat: auto's MTP passes are eager, the captured graph step of `--spec mtp` is not used, so a policy that never left MTP would sit at the floor of 0.995 of mtp2 (section 1, `autotrmtp`) and pass the 0.99 bar with 0.5 percent of headroom. Auto spent 20 percent of the chat passes on DFlash (166 of 843, 3.15 tokens against the 3.19 needed to tie at MTP's 2.25): each of those passes is about neutral, but the excursions (the entry after the EWMA crosses theta, the dwell, the probes back) used the 0.5 percent up and ended at 0.9897. A higher theta would protect chat and give back the recall gains (theta 2.6 cost recall 4-6 points in the replay).
* ho-code (two requests of about 512 tokens): 79 percent DFlash passes, but the 50 MTP passes (seeds, dwell, probes, the MTP probes every 8 DFlash passes) cost 3 percent against pure DFlash; DFlash passes average 4.77 tokens (pure DFlash 4.82), so the policy did not find lucky passes to skip as it happened to do in window 1 (5.07 tokens on 174 D passes, 1.010 of best). With 2 requests a category moves by several points on which passes the policy happens to skip; rcp-code (8 requests) is steady at 0.980-0.981 in both windows.

Greedy ids (window-2 binary, stage `ids`, `raw/w2/ids/`): `--spec auto` (retuned defaults) and `INTARSIA_SPEC_AUTO=alt:1` equal `--spec off` on short-0 and 4k-1 (256 ids): **4 of 4 PASS** (alt:1: 76 and 79 switches).

## 3. Window 1 (current params) against window 2 (retuned): the rule-4 comparison

auto / max(MTP k2, DFlash2 k7) per category, each window against its own mtp2 and dfl7:

| category | window 1, current params | window 2, retuned |
|---|---:|---:|
| ho-chat | 0.9894 | 0.9897 |
| ho-code | 1.0101 | 0.9686 |
| ho-prose | 0.9949 | 0.9961 |
| ho-recall | 0.9638 | 0.9985 |
| rcp-code | 0.9804 | 0.9809 |
| rcp-prose | 0.9891 | 0.9937 |
| rcp-recall | 0.9698 | 0.9848 |
| mean of the 7 | 0.9854 | **0.9875** |
| worst | 0.9638 | **0.9686** |

The mean rises and the worst does not fall: by the pre-registered criterion the retuned params improve auto, so they are merged, **gated to HIP with one rank** (`AutoParams::for_target`). The gain is small and uneven: recall +3.5 and +1.5 points, prose +0.1 and +0.5, chat and rcp-code unchanged, ho-code -4.2 points (the two-request category above). Auto / MTP k2 is 0.990-1.282 in window 2 against 0.989-1.289 in window 1.

## 4. Decision

* `--spec auto` is **not** the 27B default on HIP at tp=1: the rule is missed in window 2 by 0.0003 (ho-chat vs MTP k2) and 0.0014 (ho-code vs the best). The 27B tp=1 default stays `--spec mtp --spec-k 2`. No recipe change, no new recipe key: the recipe still has no way to name the DFlash2 draft checkpoint, which a default would need.
* Merged: the tp=1 policy values for `--spec auto` (`AutoParams::hip_tp1`, applied by the loader only on HIP with one rank, `INTARSIA_SPEC_AUTO` on top); tp=2, every other backend and every other model default are untouched (`AutoParams::default` is the 10-28 value set, the tp=2 replay does not support changing it: section 1).
* What would clear the two misses, not done (outside this lane's scope and window budget): (1) give auto's MTP passes the captured graph step / pipe step of `--spec mtp` (`engine.rs`: `self.auto.is_none()` guards `spec_step`, `spec_pipe_step`, the deferred verify): that lifts the 0.995 floor on chat and prose to about 1.0 and removes the reason ho-chat misses; it must keep the lazy DFlash stash and the MTP queue valid across graph steps. (2) A cheaper return to DFlash on code: the MTP probes every 8 DFlash passes cost about 1.5 percent of a code request (probe_mtp 12 gave ho-code +4.0 against +3.3 in the replay but lost rcp-recall); a per-request prior for the first pass choice (the 6-9 passes of seeds and dwell cost ho-code 1 percent).

## 5. Gates

| gate | result |
|---|---|
| ids, `--spec auto` and `INTARSIA_SPEC_AUTO=alt:1` vs `--spec off`, tp=1 HIP, short-0 and 4k-1, 256 ids | window 1 (10-28 params) 4 / 4, window 2 (retuned) 4 / 4 (`raw/w1/ids/`, `raw/w2/ids/`) |
| `cargo test -p intarsia-core` (Mac) | 196 passed, 0 failed, 2 ignored (incl. the new `auto_params_follow_the_target_and_the_env_wins`) |
| CPU forced-switch test `tests/dflash2_cpu.rs` (`spec_auto_greedy_ids_equal_spec_off_with_forced_switches_and_the_policy` and 11 more) | 12 passed |
| `cargo test -p intarsia-cli --test docs_env_vars` | 2 passed (`docs/tuning.md` row `INTARSIA_SPEC_AUTO` updated; no new `INTARSIA_*` name) |
| cross-backend no-op (`plan show` Metal / CUDA, HIP 35B / 8B) | not run: no default changed and no plan, memplan, recipe or kernel file touched. The only behaviour change is `AutoParams` values selected in `load.rs` inside the `--spec dflash` / `--spec auto` branch (`spec_auto` = `--spec auto` given), only for `backend == "hip" && tp == 1`; no `--spec auto` is ever the implicit default, so a run without `--spec auto` cannot reach the changed code |

## 6. Fleet and tooling

* fleetq (owner `claude-e7360a-au1`): builds #8877 (cpu, 20:58-21:00) and #8882 (cpu, 21:52-21:54), ids dry run #8879, window 1 #8880 (radiance down 32 min 8 s), window 2 #8883 (37 min 2 s, no wait in line behind its own build), cleanup #8885. Radiance answered at the end of each window (`raw/downtime.log`: 21:33:21 and 22:31:27); a following job of another lane (#8884) took it down again right after window 2.
* Cleanup: `~/intarsiaml-au1` and the volume `intarsia-cargo-au1` removed on tatooine (`df -h /`: 87G used with the tree and volume, 84G after; 80G used before the first build, the rest is other lanes' growth, not measured separately).
* Tools: `scripts/au1/` (copy of `scripts/dt1/` with the arms `auto`, `autoa`, `autob`, `autotrmtp`, `autotrdfl`, a pass-log fetch, `NEED_MARKER`, an ids stage for auto / alt:N / pinned arms), `tools/au1/win_table.py` (window table and rule verdict), `tools/au1/rounds.py` (per round), `tools/au1/pass_stats.py`, `tools/au1/calibrated_grid.py` and `tools/au1/cmp_params.py` (replay variants), `tools/asa/replay.py` (arm names and priors configurable, `--au1` grid with a parity split), `tools/dfh/cat_table.py` (no `off` arm needed, `PREC`).

## 7. Limits

One round per arm in window 1 (five arms in the 45 min), two in window 2; 36 rows over 7 categories, two requests in ho-code and ho-recall (those two categories move by several points on a few passes; the verdict on ho-code rests on two requests); one tp=1 card (card 0), the 27B only; the replay is a counterfactual with a 4-5 point optimism on code and recall (section 1), so the choice of params is a plateau argument and window 2 is the only measurement of them. The machinery floor (auto pinned to MTP, 0.990-0.995 of mtp2) leaves only about 0.5 percent of headroom above the 0.99 bar on chat and prose, which any DFlash excursion on those categories spends; ho-chat missed it by 0.0003.
