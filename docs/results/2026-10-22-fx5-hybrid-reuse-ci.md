# FX5: hybrid prefix reuse in CI, the skipped-case hole in the gate, and a measured 35B reuse tolerance (2026-10-22)

Lane `fx5` (branch `lane-fx5/hybrid-reuse-ci`, base origin/main f9087322). Requirement 3 of the Fable review of 2026-10-22, finding 5.
No default changed; the new flags are opt-in and `qualify` gains one run. fp8 paths and the F2 lane's `--reuse-sweep` are untouched.

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

**Final tree, with the recipe tolerance (fleetq #3916, `raw/final/`):** the same `qualify` stage reads `reuse = {prefill_chunk 1024, plan_prefill_chunk 2048, judged 5, skipped 0, require_all true, verdict PASS}`.
The stage verdict is still ERROR, for the pre-existing reason below, not for reuse.

Not fixed, found on the way: the `acc-long` main run (`--mode all`) of that stage ERRORs on this golden, `long_docs_12k: sequence of 12064 tokens exceeds the planned context
of 8192`: `qualify` passes `--max-seq 8192` for every long golden (`acc_max_seq`), and the 35B long golden has 12k prompts. That predates this lane; M3 ran `accuracy` by hand with a larger `--max-seq`.

## 3. Measured 35B reuse deviation and the recipe tolerance

`accuracy --mode reuse --reuse-measure --reuse-require-all`, 35B NVFP4, CUDA GB10 (spark1, vLLM up), the 35B long golden (5 prompts: two 4k, `chat_reuse_4k`, two 12k), bf16 KV,
`--max-seq 8192`. `nu` = largest |dlog-prob| over either row's top-5 tokens, `kl` = KL(cold || warm), both over the rows that share a context
(up to the first divergence), as the judge uses them. Both 12k prompts are cut to 8152 tokens by `--max-seq 8192` (`turn2` column of the log), as `qualify` does; their numbers are for that truncated context. Every `restore` case (unaligned and aligned) was bit-identical to the cold run (nu 0, kl 0): the restore is exact. Only `continue`
(reply tokens written by the decode kernels) moves the rows:

| prompt (continue) | chunk 1024 nu / kl (runs 1 and 2, identical) | chunk 2048 nu / kl |
|---|---|---|
| long_docs_4k | 0.877 / 3.10e-2 (diverged at token 12) | 2.138 / 9.42e-2 |
| long_code_4k | 2.420 / 3.54e-2 | **4.056** / 4.56e-2 |
| chat_reuse_4k | 1.183 / 4.40e-2 | 1.075 / 5.66e-2 |
| long_docs_12k | 1.949 / **8.49e-2** | 2.399 / 5.61e-2 |
| long_code_12k | 1.878 / 1.50e-2 (diverged at 26) | 1.977 / 4.00e-2 |

Runs 1 and 2 at chunk 1024 agree to every digit (the measurement repeats), so n = 3 runs / 15 case measurements, 10 of them distinct. **Max over all: nu 4.056 nats, KL 9.42e-2.**
With a 25 % margin: `reuse_logit_tol = 5.1` (4.056 x 1.25 = 5.07) and `reuse_kl_tol = 0.12` (9.42e-2 x 1.25 = 0.118), committed in `recipes/qwen3_5_moe.toml`. The built-in 0.25 / 1e-2 FAILED
this build at chunk 1024 (`long_docs_4k` continue: nu 0.877 > 0.25, KL 3.1e-2 > 1e-2; the same note M3 2.8 shows on HIP) and passed at 2048 only through the leading-token rule.

What these numbers do and do not guard: a value this wide only rules out gross divergence (a restore of garbage state moves KL by whole nats). The detector for a wrong restore is the bit-exact check on every chunk-aligned
restore (`--reuse-exact`, default on) and the CI test of section 1. Verification of the committed values: section 4.

## 4. Verification with the recipe tolerance

fleetq #3869 (rebuilt with the recipe values; `raw/reuse-recipe-c1024.log`, `raw/reuse-recipe-c2048.log`): `accuracy --mode reuse --reuse-require-all` on the same golden prints
`stream tolerance (fixed, recipe qwen3_5_moe): 5.1 nats, KL 1e-1` (0.12) and, at both chunks, `5 prompt(s) judged, 0 skipped` and `prefix-reuse gate: PASS` (exit 0). At chunk 1024 the same
run with the built-in constants FAILED (section 3). `cargo test -p intarsia-cli -p intarsia-core -p intarsia-models --test hybrid --test recipes_doc` and `cargo check --workspace` on the final tree (spark1, #3937): all pass (cli 153, core 97, hybrid 11, recipes_doc 2), incl. `the_reuse_tolerance_is_a_fixed_constant_...`, which now pins the recipe value.

## 5. Commands and jobs

* fleetq: #3636 (endor, hybrid test), #3660 and #3664 (spark1, tests + mutants), #3763 (spark1, build + measurement + qualify), #3869 (spark1, recipe-tolerance verification), #3870 and #3901 (spark1, tests), #3916 (spark1, qualify on the final tree), #3937 (spark1, check + tests on the final tree). #3908/#3914/#3915 were my own submissions that ran from the wrong cwd or were killed; no slot is held.
* Scripts: `scripts/fx5/test.sh`, `scripts/fx5/run.sh`, `scripts/fx5/verify.sh`. Raw logs: `2026-10-22-fx5-hybrid-reuse-ci/raw/`.
* Build on spark1: head 188101a, code-tree feee85f368e7 dirty (the tree under test).

## 6. For the coordinator

* The recipe value applies to every backend that loads `qwen3_5_moe` (the `[accuracy]` table is not per backend): it is measured on CUDA GB10 only. HIP and Metal inherit it unmeasured; re-measure with `--reuse-measure` and move to a per-backend source if they differ.
* Lane F2 (PR #66) diagnoses the fp8 `chat_reuse_4k` failure against the old 0.25 / 1e-2; on the 35B those are now 5.1 / 0.12, so the same divergence reads as a rounding tie. F2 keeps its baseline with `--reuse-logit-tol 0.25 --reuse-kl-tol 1e-2`. Its doc lists setting the recipe tolerance as a user decision; this lane was told to commit it, and it is one revertable file edit.
* Merge note: `crates/intarsia-cli/src/cmd/accuracy.rs` gains two args after `reuse_exact` and a measure/skip block in the reuse gate; F2 adds `--reuse-sweep` in the same file, expect a textual conflict, not a semantic one.
