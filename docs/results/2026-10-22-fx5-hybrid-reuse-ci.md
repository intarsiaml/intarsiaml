# FX5: hybrid prefix reuse in CI, the skipped-case hole in the gate, and a measurement of the 35B reuse deviation (2026-10-22)

Lane `fx5` (branch `lane-fx5/hybrid-reuse-ci`, base origin/main f9087322). Requirement 3 of the Fable review of 2026-10-22, finding 5.
No default changed; the new flags are opt-in and `qualify` gains one run. fp8 paths and the F2 lane's `--reuse-sweep` are untouched.
**The 35B recipe tolerance is NOT in this PR** (split out at the coordinator's request into draft `lane-fx5/reuse-tol`, "needs user decision", `2026-10-22-fx5-reuse-tol.md`). With the built-in tolerance the 35B's new reuse run FAILS at chunk 1024 (section 2); that FAIL is the honest reading.

## 1. CI test for hybrid prefix reuse

**Correction to the finding.** `crates/intarsia-models/tests/hybrid.rs` (since Phase 1a, 4963b434) already ran two
engine-level tests on the tiny hybrid fixture (6 layers, 4 GatedDeltaNet + 2 attention, fp32, cpu backend):
`engine_reuses_prefix_through_snapshots_and_matches_a_fresh_engine` and `engine_continues_a_conversation_without_recomputing_the_history`.
They go through `LocalEngine::generate`, hence `Seq::new` and `restorable_len`, but compare greedy **token ids** only. The hole was
therefore logits, not the path: a restore that is slightly wrong but keeps the argmax passes.

**Added:** `hybrid_engine_reuse_matches_cold_logits_for_every_restore_kind` in `hybrid.rs`. It records the logits row of every generated
token (`GenRequest::trace_logits`) and requires warm == cold within 1e-4 (cpu, f32)
for: identical prompt (restore at the end-of-prompt snapshot, cached 21); diverging prompts at splits 5, 8, 12, 16, 18, 21 (expected cached
0, 8, 8, 16, 16, 21: the largest snapshot at or below the split); and a continued conversation (prompt + reply + new tokens, cached =
whole history). Each split starts from a flushed engine, so the snapshots are 8, 16, 21 every time. It asserts the reused length, the tokens and the logits.

**Mutants** (applied to a copy of the tree on spark1 by `scripts/fx5/test.sh`, never to the branch; each run fails, the reverted tree passes 11/11;
logs in `2026-10-22-fx5-hybrid-reuse-ci/raw/test-hybrid-*.log`):

| mutant of `SequenceState` (state.rs) | `recurrent_snapshots_restore_exactly` | `engine_reuses_prefix_...` | new logit test |
|---|---|---|---|
| `restorable_len`: `l <= want` -> `l <= want + 3` (restores a snapshot past the shared prefix: stale state) | FAIL | FAIL | FAIL (split 5: reused 8, want 0) |
| `restorable_len`: `unwrap_or(0)` -> `unwrap_or(want)` | FAIL | FAIL | FAIL |
| `restorable_len`: largest snapshot, clamped to `want` | FAIL | FAIL | FAIL |
| `truncate`: restore only the first recurrent slot | FAIL | FAIL | FAIL |
| `truncate(0)`: skip zeroing the recurrent state | FAIL (+ `failed_forward_resets_...`) | FAIL | FAIL |

So the older tests already killed these five; the new test kills all five too and is the only one of the three that compares logits through the
engine for every restore kind. Passing: endor (aarch64 macOS, fleetq #3636) and spark1 (aarch64 Linux, fleetq #3660/#3664), 11 passed.
`cargo test -p intarsia-cli` on the first spark1 runs: 141 passed, 2 failed (`refnoise::rescore_*` read `golden/ref_noise/*.json`, which the rsync to spark1 then excluded; they pass once it is synced, section 4).

## 2. `qualify`: the reuse gate cannot be skipped away

`accuracy` got two opt-in flags (`crates/intarsia-cli/src/cmd/accuracy.rs`):

* `--reuse-require-all`: a prompt that cannot cross a chunk boundary (n <= chunk + 2) FAILS the gate instead of being reported as skipped. Every run
  now also prints `prefix reuse: J prompt(s) judged, S skipped`.
* `--reuse-measure`: prints `REUSE-MEASURE <prompt> <case> split S shared_rows N nu_max X kl_max Y [diverged_at D]` for every `continue` and
  unaligned `restore` case, pass or fail (section 3).

`qualify` (`crates/intarsia-cli/src/cmd/qualify.rs`): `acc-long` (or `acc-4k` where the 4k golden hosts the gate) now also runs
`accuracy --mode reuse --prefill-chunk min(plan chunk, 1024) --reuse-require-all` (`acc-long-reuse` / `acc-4k-reuse`), merges its verdict into the stage and
puts `reuse = {prefill_chunk, plan_prefill_chunk, judged, skipped, require_all, verdict}` into the stage metrics; a pass that printed no skipped count is a FAIL. **`--reuse-require-all` is passed only when every prompt of the golden is longer than chunk + 2**
(read from the golden's `prompts` metadata): at a chunk of 1024 or less the only prompts the gate can skip are ones too short for any chunk, and the standard `golden_long`
(eight 18-90 token prompts plus 300 and 700) would otherwise fail every model that has it. For such a golden the skipped count is still recorded and the stage note says that the skips are reported, not failed.
The existing default-chunk reuse inside `--mode all` is unchanged. Unit tests: `require_all_applies_only_to_goldens_...`, `prompt_lengths_are_read_from_the_golden_header`, `a_skipped_prompt_fails_the_reuse_gate_only_when_all_are_required`,
`the_reuse_gate_counts_and_skip_failures_are_read_back`.

**35B on spark1 CUDA (fleetq #3763; `raw/qualify/`, `raw/reuse-c1024-run*.log`):** `qualify -m Qwen3.6-35B-A3B-NVFP4 --backend cuda --golden-long
qwen36-35b-a3b-nvfp4-vllm-long-top20 ` with only `acc-long`: plan chunk 2048, reuse run at chunk 1024, **5 judged, 0 skipped**
(`full.json`: `reuse.judged 5, skipped 0`). At the CUDA default chunk of 2048 the 4k prompts still cross a boundary; on HIP (4096) they were the skipped ones (M3 2.8) and now run at 1024.

**Final #87 head, built-in tolerance (fleetq #3939, `raw/qualify-final/`):** `acc-long` is a **FAIL** (not ERROR): its `--mode all` run at the CUDA default chunk 2048 PASSES, including its own reuse gate, and the new `acc-long-reuse` run at chunk 1024 reports
`reuse = {prefill_chunk 1024, plan_prefill_chunk 2048, judged 5, skipped 0, require_all true, verdict FAIL}`. One case of 15 fails: `long_docs_4k continue` (split 2000, diverged at token 12, shared rows differ by nu 0.877 / KL 3.1e-2 against the built-in 0.25 nats / 1e-2). This is the failure M3 2.8 predicted for the 35B at chunk 1024
(there on HIP); it was hidden at the default chunk because the gate only rescues a diverging pair by tolerance. Whether to loosen the tolerance for the 35B is a user decision: draft PR `lane-fx5/reuse-tol`.

The earlier `acc-long` ERROR on this golden (`long_docs_12k: sequence of 12064 tokens exceeds the planned context of 8192`) is gone: PR #86 sizes `--max-seq` from the golden's longest prompt, and this branch keeps both changes (the capped reuse run goes through the same `accuracy()` helper, so the 12k prompts are no longer cut to 8152 tokens: `turn2` 12080 in `raw/qualify-final/qualify/logs/acc-long-reuse.out`).

## 3. Measured 35B reuse deviation (evidence for the proposal, not a change)

`accuracy --mode reuse --reuse-measure --reuse-require-all`, 35B NVFP4, CUDA GB10 (spark1, vLLM up), the 35B long golden (5 prompts: two 4k, `chat_reuse_4k`, two 12k), bf16 KV, `--max-seq 8192` (the two 12k prompts cut to 8152 tokens in these runs), fleetq #3763 (`scripts/fx5/run.sh`, `raw/reuse-c*.log`).
`nu` = largest |dlog-prob| over either row's top-5 tokens, `kl` = KL(cold || warm), over the rows that share a context (to the first divergence), as the judge uses them. Every `restore` case (aligned or not) was bit-identical to the cold run (nu 0, kl 0). Only `continue` (reply tokens written by the decode kernels) moves:

| prompt (continue) | chunk 1024 nu / kl (runs 1 and 2 identical) | chunk 2048 nu / kl |
|---|---|---|
| long_docs_4k | 0.877 / 3.10e-2 (diverged at token 12) | 2.138 / 9.42e-2 |
| long_code_4k | 2.420 / 3.54e-2 | 4.056 / 4.56e-2 |
| chat_reuse_4k | 1.183 / 4.40e-2 | 1.075 / 5.66e-2 |
| long_docs_12k | 1.949 / 8.49e-2 | 2.399 / 5.61e-2 |
| long_code_12k | 1.878 / 1.50e-2 (diverged at 26) | 1.977 / 4.00e-2 |

The built-in 0.25 nats / 1e-2 are exceeded by the numbers of every `continue` case, but the tolerance is consulted only when the streams diverge: 14 of 15 cases at chunk 1024 pass without it, the gate FAILS on `long_docs_4k continue` alone, and at chunk 2048 the gate passes. The proposal to set the recipe to 5.1 / 0.12 (max x 1.25) and its old-vs-new verdicts are in the draft PR; none of it is committed here.

## 4. Checks on the final #87 head

`cargo test -p intarsia-cli -p intarsia-core -p intarsia-models --test hybrid --test recipes_doc` and `cargo check --workspace` on spark1, cpu (fleetq #3938): cli 155 passed (including `docs_env_vars`; this PR adds no `INTARSIA_*` name), core 97, hybrid 11, recipes_doc 2; check clean. Mutants: section 1.

## 5. Commands and jobs

* fleetq: #3636 (endor, hybrid test), #3660 and #3664 (spark1, tests + mutants), #3763 (spark1, build + measurement + qualify), #3869 (spark1, recipe verification: draft PR), #3870 / #3901 / #3937 (spark1, tests), #3916 (qualify with the recipe value: draft PR), final #87 head: #3938 (tests + check) and #3939 (qualify). #3908 / #3914 / #3915 were my own submissions that ran from the wrong cwd or were killed; no slot is held.
* Scripts: `scripts/fx5/test.sh`, `scripts/fx5/run.sh`, `scripts/fx5/qualify.sh`. Raw logs: `2026-10-22-fx5-hybrid-reuse-ci/raw/`.

## 6. For the coordinator

* Expect a textual conflict with PR #66 (F2) in `accuracy.rs` (it adds `--reuse-sweep` in the same file); not a semantic one.
* `qualify` on the 35B now reads FAIL at the reuse run until the tolerance decision or the decode-vs-prefill deviation is resolved.

## 7. Follow-up (lane fx5b)

The teacher-forced rerun of the `continue` check was tried in diagnostic form: the 35B deviation survives it (KL max 8.5e-2 on the cold context), and the judge already scores only shared-context rows, so the check was not changed. See `2026-10-22-fx5b-continue-teacher-forced.md`.

## 8. Update (lane fx5c, 2026-10-22): the 35B `continue` check is now oracle-relative

On the user's decision (09:10 EDT) `continue` is judged against a committed CPU oracle where one exists (mean KL(oracle, warm) <= 1.5 x mean KL(oracle, cold) + 3e-3), instead of the absolute 0.25 / 1e-2 that
section 3 shows the 35B cannot meet. The 35B reuse gate on CUDA now PASSES at chunks 1024 and 2048; `restore` / `aligned` unchanged; Qwen3-8B unchanged. See `2026-10-22-fx5c-oracle-gate.md`. The tolerance proposal in draft PR
#89 (`lane-fx5/reuse-tol`) is superseded.
