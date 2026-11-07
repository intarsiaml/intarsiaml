# 2026-11-07 (lane ag1): `--spec auto` MTP passes on the captured step, and the default-flip rule re-run for the 27B on one R9700

Lane ag1, branch `lane-ag1/spec-auto-graph` from `origin/main` `406ac001a`. Follow-up to lane au1 (`2026-11-06-spec-auto-tp1.md`, PR #329): at HIP tp=1 `--spec auto` (opt-in) met the pre-registered default-flip rule in 5 of 7 categories and missed it only on ho-chat (auto / MTP k2 0.9897, bar 0.99) and ho-code (auto / best 0.9686, bar 0.97). au1 found the cause of the first miss: auto's MTP passes ran eagerly, because the captured graph step that `--spec mtp` uses (`Model::spec_step`) was closed to auto by the `self.auto.is_none()` guard in `Seq::prepare_with`; auto pinned to the MTP head (`INTARSIA_SPEC_AUTO=mtp`) ran at 0.990-0.995x of `--spec mtp --spec-k 2`.

## 0. Pre-registered rule (written and committed before any measurement)

**Change under test.** The captured step takes the passes the MTP head drafts inside `--spec auto`, under the conditions `Model::spec_step_ok` states for plain `--spec mtp` (HIP, one rank, graphs usable, the head's queue holds `1 <= n <= k + 1` rows, ...). The DFlash draft's passes stay eager. What the step does for auto: the replayed verify rows' DFlash taps go to the draft's context as after an eager verify pass (`Model::spec_step` calls `dflash_after_forward`, a no-op without the draft); the auto policy records the replayed pass (`AutoPolicy::record`) as it records an eager one. A pass that follows DFlash passes with more queued head rows than the step takes (`n > k + 1`) drafts eagerly (which empties the queue), then the next MTP pass replays. Plain `--spec mtp`, `--spec dflash` and `--spec off` take exactly the code they took. Out of scope, unchanged: the pipelined step (default on Metal; `spec_pipe_ok` requires no DFlash state, so auto never takes it), tp=2 (`spec_step_ok` needs one rank, so auto stays eager there as `--spec mtp` does), CUDA and Metal (the step needs HIP's per-row verify attention).

**Window.** ONE window on tatooine, card 0 (tp=1, `--max-batch 1`, `--max-seq 20480`, `--mem-budget 29`), radiance down at most 45 min (user approved), the binary built from the committed code tree of this branch. The 27B (`Qwen3.8-27B-Quark-AWQ-MXFP4`), DFlash2 draft `/models/tensorfold/Qwen3.8-27B-DFlash2`, the 7 categories and 36 rows of `scripts/dfh/cats/` of the au1 windows (512 tokens forced, `localeval quality --judge none`, one call per file), decode tok/s pooled per category and round as tokens / sum of decode times (`tools/ag1/win_table.py`, a successor of `tools/au1/win_table.py`), the mean of the rounds of the window.

Arms: `mtp2` (`--spec mtp --spec-k 2`), `dfl7` (`--spec dflash --spec-k 7`), `auto` (`--spec auto --spec-draft <draft> --spec-k 7`, with au1's HIP tp=1 parameters, which are the code defaults of `AutoParams::for_target` there; `INTARSIA_SPEC_AUTO=pass,log` only logs), `autotrmtp` (the auto machinery pinned to the MTP head, `INTARSIA_SPEC_AUTO=mtp,pass,log`). Rounds: 2 for `mtp2`, `dfl7`, `auto`, arms alternated: round 1 `dfl7, auto, mtp2, autotrmtp`, round 2 `mtp2, auto, dfl7`. `autotrmtp` runs in round 1 only: four arms times two rounds (about 50 min at about 6.3 min per visit) do not fit 45 min; its place between the two `mtp2` visits is why it is compared with the mean of both.

1. **Default flip.** `--spec auto` (with the draft, k7) becomes the default for the 27B on HIP at tp=1 **only if**, in the window, in **every one of the 7 categories**: (a) auto >= 0.99 x MTP k2, and (b) auto >= 0.97 x the faster of MTP k2 and DFlash2 k7 (the means of the 2 rounds of each arm). One category below either bar means no flip. No re-tuning of any constant: the parameters are au1's, frozen.
2. **Mechanism check** (a precondition of any flip): `autotrmtp` pooled over the 7 categories (sum of tokens / sum of decode seconds, round 1) >= 0.995 x `mtp2` pooled over the same categories and the mean of its two rounds. If it is below 0.995, the graph step is not engaged (or not as fast), and the lane reports and stops before any flip, whatever the table says. The per-category ratios are reported, not gated.
3. **Engagement evidence** before the speed arms (in the same window, the ids stage): a traced run of auto pinned to MTP and one of the forced-switch arm (`alt:1`) on short-0 each print `graph step` lines of the trace and no `capture failed`; otherwise the window stops after the ids stage.
4. **Ids**: greedy ids of `--spec auto` (shipped params), `INTARSIA_SPEC_AUTO=alt:1` (a switch every pass), `INTARSIA_SPEC_AUTO=mtp` and `--spec mtp --spec-k 2` equal `--spec off` on short-0 and 4k-1 (256 ids). Any difference anywhere stops the lane without a merge.
5. **If the rule is met and the mechanism check passes:** the default for the 27B at HIP tp=1 becomes `--spec auto` with the DFlash2 draft, and falls back cleanly to `--spec mtp --spec-k 2` when the draft is not present (both tested). **Otherwise** `--spec auto` stays opt-in and only the graph-step change is merged (it is a speed fix for opt-in auto), after the gates.

Limitations stated in advance: auto's priors and parameters were calibrated by au1 on eager MTP passes; replayed passes are faster, which the policy measures itself and may move its choices (not re-tuned here, by the pre-registered rule above); after a failed capture the same pass plans twice (`a.pick()` runs twice for it; harmless to ids, at most twice per server: two failures turn captured steps off). Wall: at most 45 min of radiance downtime.

## 1. The change

* `crates/intarsia-core/src/engine.rs` (`Seq::prepare_with`): the captured-step arm takes `self.auto.is_none() || self.drafter == Drafter::Mtp` (it was `self.auto.is_none()`); a replayed pass of auto is recorded in the policy (`AutoPolicy::record(Mtp, emits, ms)`) as an eager one is, with a trace line containing `graph step; mtp, dflash tok/ms`. `LocalEngine::spec_graph_counts` (doc-hidden, tests).
* `crates/intarsia-core/src/model/spec_graph.rs` (`Model::spec_step`): after the verify rows are committed, `dflash_after_forward(st, q, false)` as `forward_verify` does (the recording holds the DFlash tap launches, so the staging buffers hold the rows' features; the draft's lazy store stashes them). No-op without the draft, so plain `--spec mtp` is unchanged.
* The state that switches between a replayed MTP pass and an eager DFlash pass: GDN state and KV roll back through the same `SpecRows` / `spec_cut` (the replay saves the per-row copies at the same addresses the eager pass uses), the MTP queue holds `q` rows after a replayed pass and `n + 8` after a DFlash pass (then `n > k + 1`, `spec_step_ok` is false and the eager draft empties it), the DFlash stash takes every pass's rows.
* Unchanged on purpose: pipe step (Metal), batched passes, tp=2, CUDA / Metal (no per-row verify attention), `--spec mtp` / `dflash` / `off` code paths.
* Tests: `crates/intarsia-models/tests/dflash2_cpu.rs::spec_auto_with_the_captured_mtp_step_keeps_greedy_ids_equal_spec_off`, on a cpu backend that records graphs like a GPU (`RecCpu`, as `tests/mtp.rs`): ids equal `--spec off` for auto pinned to MTP, pinned to DFlash, `alt:1`, `alt:2`, `alt:5` and the scored policy, with the head's own drafts and scripted ones that hit every acceptance pattern, `max_tokens` 1 / 3 / 8 and follow-up requests; the step replays in every mode but DFlash-only; and the replayed run drafts and accepts exactly as the all-eager run (steps / proposed / accepted equal in the forced modes). Mutation check: with the `dflash_after_forward` call removed the test fails (the draft's context is out of sync, which shows as lost prefix reuse: `cached_tokens` 12 against 0).

## 2. The window (fleetq #8898, radiance down 23:57:34 to 00:42:14 = 44 min 40 s, answering 15 s after the guard started it; `raw/downtime.log`)

Binary: branch head `ea23710` (code tree `89e19a38e789`, built as `ag1-intarsia` by fleetq #8890, the build checked for the `graph step; mtp, dflash tok/ms` string before radiance went down; base `406ac001a` as `ag1-intarsia-base`). Arms and order as registered: round 1 `dfl7, auto, mtp2, autotrmtp`, round 2 `mtp2, auto, dfl7`; 49 localeval runs, 0 row or token-count problems. Raw: `raw/w1/`; table: `python3 -I tools/ag1/win_table.py docs/results/2026-11-07-spec-auto-graph/raw/w1`.

### Ids and engagement (the stage before the arms)

| gate | result |
|---|---|
| greedy ids vs `--spec off`, tp=1 HIP, short-0 and 4k-1, 256 ids: `--spec auto`, `INTARSIA_SPEC_AUTO=alt:1`, `INTARSIA_SPEC_AUTO=mtp`, `--spec mtp --spec-k 2` | **8 of 8 PASS** (`raw/w1/ids/`) |
| engagement, traced short-0, auto pinned to MTP | 204 `graph step` lines, 0 capture failures, 102 MTP passes of 102 |
| engagement, traced short-0, `alt:1` | 16 `graph step` lines, 38 DFlash passes, 0 capture failures (a forced switch every pass leaves an MTP pass behind eight queued rows, which runs eagerly by design: few replays) |
| `capture failed` in the serve logs of all arms | 0 |

### Window table (decision, rounds 1 and 2 together; the rule's columns are the means of the rounds)

rounds [1, 2] together (the mean of each arm's rounds: the decision)

| category | mtp2 | dfl7 | auto | auto / mtp2 | auto / best | rule | DFlash passes of auto |
|---|---:|---:|---:|---:|---:|---|---:|
| ho-chat | 64.1 | 61.0 | **63.5** | 0.9899 | 0.9899 | FAIL | 20% |
| ho-code | 79.6 | 95.7 | **92.4** | 1.1604 | 0.9651 | FAIL | 79% |
| ho-prose | 59.1 | 49.3 | **59.0** | 0.9983 | 0.9983 | pass | 0% |
| ho-recall | 69.6 | 77.6 | **77.3** | 1.1109 | 0.9966 | pass | 68% |
| rcp-code | 77.9 | 101.8 | **99.9** | 1.2818 | 0.9809 | pass | 77% |
| rcp-prose | 59.9 | 50.8 | **59.7** | 0.9961 | 0.9961 | pass | 1% |
| rcp-recall | 70.1 | 80.6 | **79.6** | 1.1347 | 0.9868 | pass | 72% |

rule on rounds [1, 2] together (the mean of each arm's rounds: the decision): **NOT MET**

round 1 alone

| category | mtp2 | dfl7 | auto | auto / mtp2 | auto / best | rule | DFlash passes of auto |
|---|---:|---:|---:|---:|---:|---|---:|
| ho-chat | 64.1 | 61.0 | **63.5** | 0.9902 | 0.9902 | pass | 20% |
| ho-code | 79.5 | 95.8 | **92.4** | 1.1622 | 0.9636 | FAIL | 79% |
| ho-prose | 59.1 | 49.3 | **59.0** | 0.9979 | 0.9979 | pass | 0% |
| ho-recall | 69.4 | 77.6 | **77.1** | 1.1118 | 0.9938 | pass | 68% |
| rcp-code | 77.9 | 101.8 | **99.9** | 1.2832 | 0.9819 | pass | 77% |
| rcp-prose | 59.9 | 50.9 | **59.6** | 0.9953 | 0.9953 | pass | 1% |
| rcp-recall | 70.1 | 80.6 | **79.5** | 1.1337 | 0.9864 | pass | 72% |

rule on round 1 alone: **NOT MET**

round 2 alone

| category | mtp2 | dfl7 | auto | auto / mtp2 | auto / best | rule | DFlash passes of auto |
|---|---:|---:|---:|---:|---:|---|---:|
| ho-chat | 64.1 | 61.0 | **63.4** | 0.9897 | 0.9897 | FAIL | 20% |
| ho-code | 79.8 | 95.6 | **92.4** | 1.1585 | 0.9667 | FAIL | 79% |
| ho-prose | 59.0 | 49.3 | **58.9** | 0.9988 | 0.9988 | pass | 0% |
| ho-recall | 69.8 | 77.5 | **77.4** | 1.1100 | 0.9993 | pass | 68% |
| rcp-code | 78.0 | 101.9 | **99.9** | 1.2803 | 0.9799 | pass | 77% |
| rcp-prose | 59.9 | 50.8 | **59.7** | 0.9968 | 0.9968 | pass | 1% |
| rcp-recall | 70.1 | 80.6 | **79.6** | 1.1358 | 0.9872 | pass | 72% |

rule on round 2 alone: **NOT MET**

Mechanism check (auto pinned to the MTP head, round(s) [1], pooled over the categories present):

| comparison | autotrmtp tok/s | mtp2 tok/s | ratio |
|---|---:|---:|---:|
| vs mean of mtp2's rounds | 67.28 | 67.38 | 0.9986 |
| vs mtp2 of the same round | 67.28 | 67.37 | 0.9987 |

mechanism check (pooled >= 0.995 x the mean of mtp2's rounds): **PASS** (0.9986)

| category | autotrmtp tok/s | x mtp2 (mean of rounds) | x mtp2 (same round) |
|---|---:|---:|---:|
| ho-chat | 64.0 | 0.9989 | 0.9989 |
| ho-code | 79.6 | 1.0002 | 1.0022 |
| ho-prose | 59.0 | 0.9984 | 0.9973 |
| ho-recall | 69.7 | 1.0017 | 1.0044 |
| rcp-code | 77.7 | 0.9970 | 0.9978 |
| rcp-prose | 59.8 | 0.9983 | 0.9981 |
| rcp-recall | 70.0 | 0.9990 | 0.9987 |

## 3. Verdict

**Rule: NOT MET.** Two categories miss: ho-chat auto / MTP k2 = 0.9899 (rounds 0.9902 and 0.9897; bar 0.99, short by 0.0001) and ho-code auto / best = 0.9651 (rounds 0.9636 and 0.9667; bar 0.97). The other five pass (ho-prose 0.998, ho-recall 0.997 of the best, rcp-code 0.981, rcp-prose 0.996, rcp-recall 0.987). **Mechanism check: PASS**: auto pinned to the MTP head runs at 0.9986 of MTP k2 pooled over the 7 categories (0.997-1.002 per category; au1 had 0.990-0.995), so the replayed step is engaged and as fast as `--spec mtp`'s. No flip.

Against lane au1's window 2 (same rule, same params, eager MTP passes): auto / MTP k2 ho-chat 0.9897 -> 0.9899, ho-prose 0.9961 -> 0.9983; auto / best ho-code 0.9686 -> 0.9651 (two requests), ho-recall 0.9985 -> 0.9966, rcp-code 0.9809 -> 0.9809. So the graph step closed the pinned-MTP gap (0.990-0.995 -> 0.9986) but not the two misses: auto's remaining shortfall on ho-chat (20% DFlash passes, about 27 switches and 12 probes per 512-token request in the per-pass logs) is the policy's own cost of switching and probing, not the MTP pass speed; the MTP pass right after a DFlash pass also stays eager (queue of `n + 8` rows), which this window did not measure separately. Nothing was re-tuned (pre-registered), and the policy priors still come from eager passes.

## 4. Decision and the premise to check

* `--spec auto` stays **opt-in**. Only the graph-step change is merged: it makes opt-in auto's MTP passes as fast as `--spec mtp`'s, with greedy ids unchanged.
* **The 27B on HIP at tp=1 has no `--spec mtp --spec-k 2` default.** `recipes/qwen3_5_vl.toml` `[tuning.hip]` sets only `mtp_default_tp_k = 2` (tp > 1, user decision 2026-10-26 11:07, "tp1 stays unchanged"); there is no `mtp_default_k` for HIP, and `docs/decisions/LOG.md` (the speculative-decoding row) names the HIP tp > 1 default only. At tp=1 the 27B runs `--spec off` by default; MTP k2 is the comparison bar of the rule, not the shipped default. The earlier sentence "the 27B tp=1 default stays `--spec mtp --spec-k 2`" (au1 doc and LOG row) is therefore inaccurate. A flip would also have replaced plain decode by auto (a draft checkpoint of several GB inside the 29 GB budget, and the recipe has no key naming the DFlash2 draft); it was not reached because the rule is not met, and it needs its own decision.

## 5. Gates

| gate | result | fleetq |
|---|---|---|
| `cargo test -p intarsia-core` (Mac) | 196 passed, 0 failed, 2 ignored | local |
| `cargo test -p intarsia-models --test dflash2_cpu --test mtp` | 13 passed (incl. the new test and the 12 earlier); 32 passed, 1 ignored | local |
| `cargo test -p intarsia-cli --test docs_env_vars` | 2 passed (`docs/tuning.md` `INTARSIA_SPEC_GRAPH` row updated, no new `INTARSIA_*` name) | local |
| `cargo build --workspace` | ok | local |
| Metal, endor, exclusive (`scripts/dm/noop.sh`, base `406ac001a` vs lane): `plan show --json` 8B / 27B / 35B x default / off / mtp | **9 / 9 identical**; `kernels test` 51 kernels, 1136 cases, no FAIL on either, same output (`raw/metal/noop/`) | #8894 |
| Metal 27B ids vs `--spec off`, `--spec mtp --spec-k 2`, short-0 and 4k-1, 256 ids | **PASS, PASS** (`raw/metal/gen/`) | #8894 |
| CUDA, spark1, vLLM left up (`scripts/ag1/cuda-noop.sh`, MAX_JOBS=4): `plan show --json` 35B and 8B base vs lane | byte-identical (15570 B, 8973 B); `kernels test --quick` exit 0, 0 FAIL | #8892 |
| HIP `plan show --json` base vs lane (`scripts/ag1/hip-noop.sh`, radiance down, `raw/hip-noop/`): 8B tp1 default / off, 35B tp1 and tp2 default / off, 35B tp1 mtp2, 27B tp1 and tp2 default / off / mtp2 | **13 / 13 byte-identical**; 35B tp2 mtp2 refuses identically on both (MoE head under tp). First attempt with one card (`raw/hip-noop-1/`): tp1 plans identical, tp2 runs needed both cards | #8925, #8927 |
| HIP ids (above) | 8 / 8 PASS | #8898 |

## 6. Operations

* fleetq (owner `claude-e7360a-ag1`): builds #8890 (tatooine, cpu), window #8898 (radiance down 44 min 40 s), hip no-op #8925 and #8927 (radiance down 1 min 35 s and 2 min 2 s, answering after 20 s each), Metal builds #8891 and gates #8894, CUDA no-op #8892, dry run #8895 (its `plan show` ran with radiance up and ran out of GPU memory: the plan allocates, so the HIP plan gate needs a free card).
* Radiance downtime in all: 44 min 40 s (window) + 3 min 37 s (two plan-show slots).
* Tools: `scripts/ag1/` (copy of `scripts/au1/` with the engagement stage in `stages-ids.sh`, `MINLEFT`, `IDSONLY`), `tools/ag1/win_table.py` (per-round tables and the mechanism check), `scripts/ag1/hip-noop.sh`, `scripts/ag1/cuda-noop.sh`.

## 7. Limits

One tp=1 card, the 27B only, 36 rows over 7 categories (two requests in ho-code and ho-recall: those categories move by a few points on a few passes); `autotrmtp` ran in round 1 only, compared with the mean of the two `mtp2` visits around it and with the same-round `mtp2` (0.9986 / 0.9987); the replayed MTP pass after a DFlash pass is eager, so a switch-heavy run replays less than a pinned one; ids are 256-token gates on two prompts (the CPU test covers the acceptance patterns).
