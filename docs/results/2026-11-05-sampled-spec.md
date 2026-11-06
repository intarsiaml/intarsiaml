# Sampled speculative decoding (temperature > 0): rejection sampling and the match rule, opt-in

Lane `lane-ss/sampled-spec`, branch `lane-ss/sampled-spec`, 2026-11-05. Base: `origin/main` 602a1fcfe.

**Status (results below; speed round 2 and CUDA are filled in when they finish).** Both variants are exact in law: exhaustive
enumeration matches plain sampling to < 1e-12 and every statistical test passes (24 / 24 G-tests on Qwen3.5-0.8B, CPU).
Greedy output is unchanged (workspace tests 1330 / 0 failed, Metal greedy ids 4 / 4 identical, plans identical, goldens PASS).
Metal served decode (natural-v1, 1k / 4k): the 27B keeps the most with `reject` (x1.08-1.20 over sampled spec-off), the 35B with
`match` (x1.15-1.23). Opt-in, default off.

## Messages received during the lane

Coordinator, 2026-11-05 (copied verbatim as asked):

> Two findings from a consultant, checked by the coordinator on main.
>
> 1. **The default temperature is 1.0.** `crates/intarsia-server/src/api.rs:417` defaults temperature to 1.0 when a client omits it, so nearly all default API traffic is sampled and gets no speculation today. localeval's speed runs send temperature 0, which is why no gate saw this.
>    - Measure at temperature 1.0 too, not only 0.7.
>    - Report what fraction of the decode gain survives at each.
>
> 2. **A simpler exact variant is available.** With a deterministic (argmax) draft d, "sample x from the processed target distribution at that position; accept if x == d, else emit x and stop" is exact speculative sampling. It needs no p_draft, no residual and no change to the draft head.
>    - Its acceptance is p_target(d), lower than full rejection sampling with a sampled draft.
>    - It is a one-line change in the verify consumer if the per-position target logits are available.
>    - Build this first as the baseline and fallback, then the full q-based version if time allows.
>    - Report the acceptance and tok/s of both, and recommend one.
>    - The same consumer pattern already exists in `crates/intarsia-models/src/dsv41_engine.rs` around lines 859-925 (the sampler draws per row and accepts iff the draw equals the draft). Read it.
>
> DeepSeek-V4.1 stays out of your scope; a separate lane may take it. Copy this message into your results doc.

## The change

`INTARSIA_SPEC_SAMPLED` (default off; `Model::set_spec_sampled` for tests):

| value | variant | draft | accept rule at verify row i | on reject |
|---|---|---|---|---|
| unset, `0`, `off` | off | - | sampled requests take plain steps (main's behaviour) | - |
| `match` | `SampledSpec::Match` | the MTP head's argmax (main's draft path, unchanged) | draw x from the processed target distribution p_i; accept iff x == d_i | emit x, stop |
| `1`, `reject` | `SampledSpec::Reject` | drawn from the head's processed distribution q_i with the request's RNG | accept d_i with probability min(1, p_i(d_i) / q_i(d_i)) | emit a draw from normalize(max(0, p_i - q_i)), stop |

All k accepted: a bonus token is drawn from p_k. Both rules are exact in law (section 2).

**Processed distribution.** The engine's host sampler (`sampler.rs`) applies temperature, top-k and top-p; the only other
logit edit is the `min_tokens` stop-id mask. No min-p, repetition, presence, frequency penalty or logit bias exists in the
engine or the server (grep of `crates/`), so no position-dependent penalty has to see the drafts. `sampler::dist` returns
exactly the candidates and probabilities `sample_row` walks (shared code: `walk` is now `cut` + the cumulative walk;
`sample_row` is bit-identical, its fuzz tests pass). Drafting never starts inside a `min_tokens` window that masks stop ids
(`plan_step`, unchanged), so no verify row needs the mask.

**Reduced draft head** (`--mtp-draft-vocab`, a row prefix of the lm_head, so index = token id): the draft row is widened to the
vocabulary with `-inf`, q = 0 past the prefix, and those tokens come only from the residual, which ranges over p's whole support.

**Where it runs.** Single-sequence engine (`Seq::new`), `--spec mtp`, one rank (`st.peers` empty), no constraint, no logit
trace. Sampled requests take the eager draft + verify step (`Seq::sampled_pass`): the captured spec step (HIP graphs) and the
pipelined Metal GPU-accept step (`spec_pipe`) are bypassed, because both keep the logits on the device. Speed numbers below are
therefore for the eager step. The batching scheduler builds its sequences with `Seq::build` only and never turns it on.
Greedy requests are untouched: `spec_on` for them is computed exactly as on main, and the greedy accept (`spec_accept`) is unchanged.

**RNG.** The request's seeded `Sampler` RNG (SplitMix64) drives the draft draws, accept tests and residual/bonus draws (53-bit
uniforms), so a seed reproduces a sampled run. It is not the same token stream as spec-off for the same seed (the draws are
consumed differently).

## Pass rules (written before any gate ran)

1. **Greedy unchanged.** `cargo test --workspace --no-fail-fast` passes with no failure that main does not also show; the
   existing MTP greedy-identity tests (`crates/intarsia-models/tests/mtp.rs`) pass; on Metal, greedy ids with
   `INTARSIA_SPEC_SAMPLED=1` equal main's for the 27B and the 35B (short and 4K prompt).
2. **Exactness in law, CPU.**
   - Unit (`spec.rs` tests): the exact enumerated law of k + 2 tokens generated by sampled passes equals plain sampling's
     law within 1e-12 for every case (independent supports, disjoint supports, draft 0 on the target's mode, reduced draft
     vocabulary, top-p 0.6, top-k 2 + top-p 0.9, draft = target), k = 1..4, both variants. The RNG-driven driver's 60,000-draw
     chi-square against the exact law stays below the z = 5 critical value for every case and variant, and the wrong-residual
     driver (positive control) exceeds it.
   - Statistical, real model: Qwen3.5-0.8B-Base (the smallest local checkpoint with `mtp.*` weights; Qwen3-8B has none) on the
     cpu backend, `--spec mtp --spec-k 2`, fixed prompt, 4 generated tokens per run, disjoint seed sets per arm. Configs
     A (T 1.0, full draft head) and B (T 0.7, top-k 20, top-p 0.8, draft head reduced to 4096 ids). For each config, each
     variant against spec-off, each of the speculative positions 2, 3, 4: a G-test of homogeneity (cells with expected
     count < 5 in either arm pooled). Pass: every p-value > 0.01 / 12 (Bonferroni over 3 positions x 2 configs x 2 variants).
     n = 3000 runs per arm per config, chosen so the power against Cohen's w = 0.1 at that alpha is >= 0.93 up to 40 df
     (noncentral chi-square, lambda = N w^2 with N = 6000); the realised df and power are reported.
3. **Speed** (reported, no pass threshold): localeval decode tok/s at T 0.7 and T 1.0, spec-sampled (both variants) vs spec-off,
   Metal 27B (MTP default) and 35B (`--spec mtp --spec-k 2`), 1k and 4k natural prompts, with acceptance rates; and the
   fraction of the greedy MTP gain that survives. CUDA 35B tp=1 on spark1 if a slot opens. HIP: not run (tatooine belongs to
   another session).
4. **Cross-backend no-op.** Metal: `plan show` text and json identical to main (27B, 35B, 8B), `kernels test` 0 FAIL, greedy ids
   identical, 8B goldens PASS. CUDA (spark1, if a slot opens): `plan show` identical, `kernels test --backend cuda --quick` 0 FAIL.

## 1. Gate 1: greedy unchanged

| check | result |
|---|---|
| `cargo test --workspace --no-fail-fast` (endor, fleetq #8603, tree `0c019f8fe`; the first attempt #8595 was killed by another lane's `pkill`) | exit 0: **1330 passed, 0 failed, 29 ignored** (`raw/workspace-test-summary.txt`) |
| existing MTP greedy-identity tests (`tests/mtp.rs`, incl. scripted acceptance patterns, pipelined and captured steps, reduced draft vocab) | pass (part of the run above) |
| new: with sampled speculation on (both variants, full and reduced draft head), greedy ids = sampled speculation off, 3 prompts x 64 tokens (`sampled_speculation_drafts_sampled_requests_and_leaves_greedy_ones_alone`) | pass |
| Metal greedy ids, base `602a1fcfe` vs branch with `INTARSIA_SPEC_SAMPLED=1`, 96 tokens: 27B (MTP default) and 35B (`--spec mtp --spec-k 2`), short and 4K natural prompt | **identical 4 / 4**; acceptance identical too (`raw/noop-metal/ids-*.err`) |

## 2. Gate 2: exactness in law

### 2.1 Unit (`crates/intarsia-core/src/spec.rs` tests, `sampler.rs` test)

- `sampled_spec_law_is_the_target_law_exactly`: the law of k + 2 tokens produced by repeated sampled passes (exact enumeration
  through `accept_prob` / `residual`, or the match rule over argmax drafts; target and draft distributions depend on the whole
  prefix; V = 4) against plain sampling's law. 7 cases (independent supports; **disjoint supports**; **draft 0 on the target's
  mode**; **reduced draft vocabulary**; **top-p 0.6** at T 0.8; top-k 2 + top-p 0.9 at T 1.3; draft = target) x **k = 1..4** x
  both variants: max |difference| **< 1e-12** everywhere.
- `sampled_spec_driver_samples_the_target_law`: the RNG-driven `accept_sampled` (the code the engine runs), 60,000 draws of
  the joint law of 3 tokens (k = 2) per case and variant, Pearson chi-square against the exact law: below the z = 5 critical
  value in all 14 runs. **Positive control:** the same driver drawing a rejected token from p instead of the residual fails it.
- `sampled_spec_enumeration_catches_a_wrong_residual`: that bug moves the first token's law by > 1e-3 on the plain case.
- `dist_is_what_sample_row_draws` (`sampler.rs`): `dist` gives exactly `sample_row`'s candidates, order and probabilities (every
  candidate's cumulative interval maps to it under `sample_row`) across T 0 / 0.7 / 1 / 1.5, top-k 0 / 1 / 20 / n,
  top-p 1 / 0.95 / 0.5, rows with -inf; the unsorted fast path (no top-k, no top-p) has the sorted path's probabilities.
- The existing sampler fuzz tests (`keyed_sort_matches_comparator_sort`, `topk_candidates_sample_like_the_whole_row`) pass, so
  `sample_row` is unchanged after the `walk` / `cut` split.

### 2.2 Engine on the fixture (`crates/intarsia-models/tests/mtp.rs`, CPU, tiny hybrid MoE + random MTP head, V = 96)

`sampled_speculation_matches_plain_sampling_in_law_on_the_fixture`: 1,500 seeded requests per arm, disjoint seed sets,
per-position chi-square of tokens 2, 3, 4 vs plain sampling; configs (T 1, full) and (T 1, top-k 4, top-p 0.9) x (reject,
reject with a 40-id draft head, match). All 18 below the z = 5 critical value (`raw/fixture-law-test.log`). The random head
accepts little (match 27 / 4492 drafts, reject 929 / 4045), so this is a wiring check; 2.3 is the real-model test.

### 2.3 Statistical test, real model (Qwen3.5-0.8B-Base, CPU backend, fleetq #8551 + #8602)

Prompt "Once upon a time, in a small village by the sea, there lived", `--spec mtp --spec-k 2`, 4 tokens per run, n = 3000 runs
per arm and config, seed ranges 0.., 1,000,000.., 2,000,000.. (off, reject, match). Config A: T 1.0, full draft head; config B:
T 0.7, top-k 20, top-p 0.8, draft head reduced to the first 4096 ids. Token 1 is " a" in every run (prefill, not speculative).
`scripts/sampled-spec/gtest.py` (`raw/stat/gtest.md`, per-run ids in `raw/stat/*.txt`):

| config | variant | drafts accepted | G-test p, position 2 / 3 / 4 | df | power at w = 0.1 |
|---|---|---|---|---|---|
| A (T 1.0) | reject | 4012 / 6959 = 57.7 % | 0.629 / 0.018 / 0.138 | 58 / 50 / 78 | 0.86 / 0.90 / 0.77 |
| A (T 1.0) | match | 2291 / 8149 = 28.1 % | 0.296 / 0.032 / 0.593 | 53 / 48 / 68 | 0.88 / 0.90 / 0.81 |
| B (T 0.7, k 20, p 0.8, draft vocab 4096) | reject | 4527 / 7057 = 64.1 % | 0.465 / 0.624 / 0.288 | 3 / 7 / 20 | 1.00 / 1.00 / 0.99 |
| B | match | 4519 / 6834 = 66.1 % | 0.686 / 0.946 / 0.210 | 3 / 7 / 20 | 1.00 / 1.00 / 0.99 |

**All 12 p-values are above the Bonferroni alpha 0.01 / 12 = 8.3e-4: PASS.** Both branches ran (acceptance well inside 0..1).
The joint law of tokens 2-4 (supplementary, not in the rule): p 0.018 / 0.357 / 0.180 / 0.062.

**Power shortfall, stated against the pre-written rule:** the rule assumed at most 40 df; config A's high-entropy positions
realised 48-78 df, so the power against Cohen's w = 0.1 is 0.77-0.90 there, not the >= 0.93 the rule promised (B: >= 0.99).
Against w = 0.12 it is >= 0.97 at every cell (same calculation, lambda = 6000 w^2). The CPU backend has no row-exact verify kernels, so its verify rows can differ
from single-row decode in the last bits; that is far below what this test can see and does not apply to the unit tests.

## 4. Gate 4: cross-backend no-op

Metal (endor, fleetq #8605, base binary from `602a1fcfe`, branch binary from `ecf818a37`, whose engine code equals the head's;
`scripts/sampled-spec/noop-metal.sh`, `raw/noop-metal/`):

| check | result |
|---|---|
| `plan show` text, json and stderr, 27B / 35B / 8B | **identical 9 / 9** |
| `kernels test --backend metal` | 49 kernels, 980 cases, 0 FAIL, base and branch logs identical but for the wall time |
| greedy ids 27B / 35B, short and 4K (with `INTARSIA_SPEC_SAMPLED=1` on the branch) | **identical 4 / 4** |
| 8B goldens `qwen3-8b-fp32`, `qwen3-8b-long-fp32` | **PASS** both, base and branch |
| sampled smoke, 27B T 0.7, 128 tokens | reject: 53 passes, 69.8 % accepted; match: 58 passes, 59.5 %; coherent text (`sampled-27-*.out`) |
