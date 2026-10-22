# FX5c: the oracle-relative `continue` gate (2026-10-22)

Gate change on PR #87 (`lane-fx5/hybrid-reuse-ci`), **approved by the user on 2026-10-22 09:10 EDT** (as relayed in the coordinator's message; I
did not see the chat). Evidence: PR #96 / `2026-10-22-fx5c-reuse-vs-oracle.md` (warm is not less accurate than cold: against a CPU f64 oracle, mean KL
0.83x (CUDA), 1.09x (Metal), 1.13x (CUDA chunk 2048), 1.10x (CUDA, code prompt) of cold's).

## 1. The rule

For the `continue` case of `accuracy --mode reuse`, when a committed oracle file matches the golden's model, prompt, split and prompt token ids:

```
mean_rows KL(oracle || warm)  <=  1.5 x mean_rows KL(oracle || cold)  +  3e-3
```

* Rows: the 32 teacher-forced rows of a frozen token list (the golden prompt with a frozen 16-token reply inserted at the split, then 32 frozen generated tokens), run
  through `Engine::forward_script` for the cold path (full chunked prefill) and the warm path (turn 1 prefilled, the 15 fed reply tokens decoded one at a time, resume, as the
  engine does after a turn). KL(oracle || engine) is over the oracle's top-64 ids plus a tail bucket, the quantity `accuracy` gates elsewhere.
* **abs_floor = 3e-3.** Derivation: the floor is the oracle's own noise. The f64 CPU oracle is deterministic per machine but not portable (bf16 activations): the oracle made on
  spark1 and the committed one made on endor (FX1) differ on the 2000 prompt rows they share by KL 3.1e-3 mean (max 2.4e-1) (`2026-10-22-fx5c-reuse-vs-oracle/raw/oracle-xcheck.txt`, PR #96).
  A warm-minus-cold difference below that cannot be told from which machine made the oracle. It is small against the means it adds to (1.4e-2 to 1.8e-2): at those means the limit is
  1.66x to 1.71x cold, and it keeps two near-exact paths (KL ~1e-5, a bit-identical resume) from failing on a ratio of tiny numbers. A fixed constant of the gate, not derived from the run.
* Applies to `continue` only. `restore` and `aligned` keep the absolute stream tolerance and the bit-exact check, unchanged. Without a matching file `continue` keeps the absolute
  0.25 nats / 1e-2 rule (also with `--no-reuse-oracle`). `cached >= split` and the cold-run check stay.
* The gate prints both means, the ratio, the limit and the verdict (`continue judged against the committed CPU oracle <file>: mean KL(oracle||cold) ..., mean KL(oracle||warm) ..., warm/cold ..x; limit ...: PASS`).

Code: `crates/intarsia-cli/src/cmd/accuracy.rs` (`judge_oracle_relative`, `kl_oracle_row`, `find_continue_oracle`, `continue_judge`, `judge_continue_oracle`; flags `--reuse-oracle-dir`, `--no-reuse-oracle`,
`$INTARSIA_REUSE_ORACLE_DIR`; test-only `$INTARSIA_REUSE_TEST_BUG=scale:F`, both documented in docs/tuning.md); `docs/accuracy.md`.

## 2. Committed data (`golden/oracle/`, beside FX1's files)

| file | split / chunk | bytes | sha256 |
|---|---|---|---|
| `qwen36-35b-a3b-nvfp4-continue-long_docs_4k-s2000.continue-oracle.safetensors` | 2000 (chunks up to 1999, the 35B default 1024) | 17608 | `826ab3bdc80bd9c224160e92c1d844468b53e4598b7932a7f1fd208aa44a4ea1` |
| `qwen36-35b-a3b-nvfp4-continue-long_docs_4k-s3072.continue-oracle.safetensors` | 3072 (chunk 2048: the split moves to 1.5 chunks) | 17616 | `fef987548f3fd4ae0a2d4bbd0eb4b1001b6eb4263823c2c15533c9842ec6b645` |

Each has a `.json` sidecar (inputs, provenance, sha256 of the file) and the file itself holds the frozen reply and stream tokens, the model, prompt, split, a fingerprint of the golden prompt's token ids, and the
CPU oracle's top-64 rows for the 32 stream positions. **Built from spark1's CPU oracle, not endor's**: endor's fleetq queue was gated all morning (a stale cpu job, then the #4051 reservation); spark1 ran
the f64 CPU oracle (`accuracy --backend cpu`, FX1's recipe: chunk 256, top-64) in 15.5 min each (fleetq #4089, #4141). The sequences were frozen from the CUDA GB10 cold engine's greedy stream. Rebuild:
`scripts/fx5c/seq-chain.sh` then `scripts/fx5c/make_continue_oracle.py`. FX1's oracle finder takes only `-cpu-oracle` files, so these are not mistaken for it.

## 3. Validation

**Unit tests** (spark1, cpu, fleetq #4149; `raw/test-cli-core.log`, `raw/test-hybrid.log`): `cargo test -p intarsia-cli -p intarsia-core`: cli 159 passed (155 + 4 new), core 97; `docs_env_vars` 2 passed; hybrid 11, recipes_doc 2; `cargo check --workspace` clean.
New: `oracle_relative_warm_1_4x_passes_and_1_6x_fails` (also the floor, a mean not a max, non-finite fails, the measured CUDA and Metal means pass), `continue_oracle_found_only_for_its_model_prompt_split_and_ids`
(missing oracle falls back to the absolute rule; wrong split, model, prompt or token ids are not used, with the reason; `restore` and `aligned` stay absolute even with an oracle present),
`kl_oracle_row_is_top_k_plus_tail` (against a hand-computed value), `oracle_relative_mutation_scaled_warm_logits_fail` (synthetic rows: x1.5 and x0.6 fail, the clean pair passes).

**CUDA GB10, 35B NVFP4, long golden (5 prompts, 15 cases), `accuracy --mode reuse --reuse-require-all`** (spark2, fleetq #4170, non-exclusive, numerics only; `raw/cuda/`). This is the gate `qualify` acc-long runs
(`acc-long-reuse` at chunk 1024, and inside `--mode all` at the CUDA plan chunk 2048); I ran the gate directly, not the 45-min qualify around it (its prefill/decode KL gates are untouched by this change).

| chunk | verdict | `long_docs_4k continue` |
|---|---|---|
| 1024 | **PASS** (FAIL on #87's head) | s2000: cold 1.83e-2, warm 1.52e-2, 0.83x, limit 3.04e-2: PASS; the other 14 cases as before |
| 2048 | **PASS** | s3072: cold 1.40e-2, warm 1.59e-2, 1.13x, limit 2.40e-2: PASS; the other 14 cases as before |

The numbers equal the offline scoring of PR #96 to the digit (same rows, same oracle).

**Mutation** (CUDA, test-only `INTARSIA_REUSE_TEST_BUG=scale:F`: the warm rows' logits x F before judging; chunk 1024, `long_docs_4k`; `raw/cuda/mut-*.log`):

| F | mean KL(oracle, warm) | warm/cold | limit 3.04e-2 | gate |
|---|---|---|---|---|
| 1.01 | 1.53e-2 | 0.84x | below | PASS (not detected) |
| 1.10 | 2.75e-2 | 1.50x | below | PASS (not detected) |
| 1.20 | 5.87e-2 | 3.22x | above | **FAIL** |

**A x1.01 logit scale does not fail, and cannot with this rule.** Scaling the logits by 1.01 adds ~1e-4 nats of KL, 30x below the floor and 100x below the engine's own distance to the oracle (1.5e-2 to 1.8e-2);
the same scan on Metal's rows (PR #96, offline) fails at x1.1 (2.82e-2 > 2.48e-2) and CUDA's at x1.2. The rule resolves a warm-path error of about 10 to 20 % on the logits. Where a 1 % error matters the
same-backend drift gates catch it (FX1: a x1.01 scale mutant reads 9.2e-3 against a self-golden budget of 1e-4), and the exact `restore` / `aligned` checks (bit-identical logits, `cached == s`) still guard the cache
logic with no tolerance. So the requirement "x1.01 must FAIL" is not met: I report the thresholds instead of tuning the floor to hit it.

**Fallback** (`--no-reuse-oracle`, same run): `long_docs_4k continue` FAIL by the absolute rule, as before (`raw/cuda/noorc.log`).

**Metal M5 Max (endor), `long_docs_4k` only** (fleetq #4178, #4195, non-exclusive, numerics only; `raw/metal/`; the other 14 cases run the unchanged absolute path, so only the case the rule judges was run):

| chunk | verdict | `long_docs_4k continue` |
|---|---|---|
| 1024 | **PASS** | s2000: cold 1.46e-2, warm 1.59e-2, 1.09x, limit 2.48e-2 |
| 2048 | **PASS** | s3072: cold 1.66e-2, warm 1.78e-2, 1.07x, limit 2.79e-2; `restore` / `aligned` ok, bit-identical |

**The first Metal run found a flaw in my first version, now fixed** (`raw/metal-v1/`): the version that substituted the frozen turn 2 for the engine's own in the real `generate` runs made Metal's `cached` check read
"reuse covered 3071 < the 3072 tokens of turn 1", because Metal's own 16-token reply differs from the frozen CUDA one and the cache then matches only the turn-1 prompt. The engine runs (reuse, cold, streams) now keep the
engine's own reply, and only the scored `forward_script` rows use the frozen sequence, so `cached >= split` keeps meaning "the engine reuses what it holds". CUDA, where the reply is the frozen one, was unaffected and was
rerun on the fixed build (`raw/cuda/`: chunk 1024 PASS 0.83x, chunk 2048 PASS 1.13x, x1.2 FAIL, x1.01 PASS, `--no-reuse-oracle` FAIL; `mut-x1.1.log` is from the first build, same numbers expected, not rerun).

**Qwen3-8B reuse is unchanged, measured** (spark2, CUDA, fleetq #4203; `raw/extra/`): its 4k fp32 golden (which also has a prompt called `long_docs_4k`) with the 35B continue files on the lookup path: gate PASS with no oracle line, identical
to the run with `--no-reuse-oracle`; the finder passes the files over ("made for model ..., the golden is ...", the model metadata differs).

**`qualify` acc-long was not run.** I ran the gate it runs (`accuracy --mode reuse --reuse-require-all`, chunks 1024 and 2048, above), not `qualify` itself: the qualify run is ~45 min of prefill/decode KL gates around it, and the
spark boxes were taken (exclusive jobs, then #4174 from 11:05) so it could not fit before the 11:30 stop. What that leaves untested: qualify -> accuracy subprocess -> `oracle/` beside the golden on a box -> qualify's parsing of a stage that
prints the new line (it reads only the verdict and the `prefix reuse: J judged, S skipped` line, unchanged).

## 4. Files, jobs

* `scripts/fx5c/`: `validate-cuda.sh`, `metal-validate.sh`, `test.sh`, `seq-chain.sh`, `make_continue_oracle.py`, plus the measurement scripts that PR #96 documents.
* Fleet jobs (owner claude-f31ccf91-fx5c): #4149 (spark1, tests), #4150 (spark1, CUDA build), #4141 (spark1, split-3072 sequence + oracle), #4170 (spark2, build + first CUDA validation), #4178 (endor, Metal, first version),
  #4195 (endor, Metal rerun), #4203 (spark2, 8B + CUDA rerun after the fix). Cancelled before running: #4148 and #4179 (full Metal run and qualify: did not fit the queue), #4167 (spark1 copy of the CUDA validation).
  No slot is held.
* Two `cargo check -p intarsia-cli` runs on endor ran outside a fleetq slot (cpu only, before the test job; no fleet role affected); every build, test and run after that took a slot.
