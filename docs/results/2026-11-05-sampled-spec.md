# Sampled speculative decoding (temperature > 0): rejection sampling and the match rule, opt-in

Lane `lane-ss/sampled-spec`, branch `lane-ss/sampled-spec`, 2026-11-05. Base: `origin/main` 602a1fcfe.

**Result.** Both variants are exact in law: exhaustive enumeration matches plain sampling to < 1e-12, and all 12 pre-registered
G-tests on Qwen3.5-0.8B (CPU) pass (power 0.77-0.90 against w = 0.1 on the high-entropy config, below the >= 0.93 the rule
assumed; >= 0.99 on the other). Greedy output is unchanged (workspace 1330 passed / 0 failed; greedy ids identical on Metal 4 / 4
and CUDA 2 / 2; plans identical; goldens PASS). Served decode on natural-v1 1k / 4k, sampled spec over sampled spec-off:

| backend, model | T 0 greedy MTP (today) | T 0.7 match | T 0.7 reject | T 1.0 match | T 1.0 reject |
|---|---:|---:|---:|---:|---:|
| Metal 27B (2 rounds) | x1.115 | x1.076 (66 %) | **x1.117 (102 %)** | x1.029 (25 %) | **x1.090 (78 %)** |
| Metal 35B (2 rounds) | x1.288 | **x1.187 (65 %)** | x1.025 (9 %) | **x1.102 (35 %)** | x1.086 (30 %) |
| CUDA 35B tp=1 (1 round) | x1.335 | **x1.222 (66 %)** | x1.161 (48 %) | **x1.150 (45 %)** | x1.106 (32 %) |

(%) = share of the greedy MTP gain that survives. `reject` wins where a decode step is slow (27B); `match` wins where it is fast
(35B), because `reject` pays k draft-row downloads and k full-vocabulary softmaxes on the host per pass. Default on with `match` on CUDA and Metal since the user's decision below; `0` turns it off.

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

Coordinator, 2026-11-05, after the first report (copied as asked):

> 1. `INTARSIA_SPEC_SAMPLED=1` means `match`. `=reject` and `=match` select a variant explicitly. Update the tuning row, the LOG row, the tests and the doc.
> 2. Rebase or merge onto `origin/main` (now `d40d93896`). It includes #309, batched MTP inside `--max-batch`: `INTARSIA_BATCHED_MTP`, with solo MTP at low occupancy. #310 changes the packed-prefill minimum on CUDA.
>    - Resolve conflicts.
>    - Make sure sampled requests on the batched phase-A path either use your sampled accept correctly or fall back to plain decode. Say which in the doc.
>    - Make sure greedy stays bitwise.
>    - Then run `cargo test --workspace --no-fail-fast` through fleetq with `--load cpu`.
> 3. Remove `~/intarsiaml-ss` on spark1 inside a fleetq slot (`--load none`).
>
> The default stays off until the user decides.

Default-on source (to be confirmed by the coordinator): no relayed message with this decision survives in this lane's
transcript, which was compacted after the message above (which still said the default stays off). The coordinator's own session
notes (`memory/perf-push-2026-11-05.md`) record: "~20:10 EST (user): 'sampled default on': Qwen sampled speculation default on,
variant `match` (INTARSIA_SPEC_SAMPLED unset = match; =1 = match; =reject explicit)". The lane implemented that note as: unset
`INTARSIA_SPEC_SAMPLED` = `match` on the CUDA and Metal backends (wherever `--spec mtp` runs, explicitly or by the recipe default:
CUDA 35B, Metal 35B, Metal 27B); off on HIP (its captured verify replay was not checked with the logits read-back, and its MTP
default is tp > 1, which sampled speculation does not cover) and on the cpu reference; `0` / `off` turns it off, `reject` selects
the other variant. Startup prints `sampled requests (temperature > 0): speculation Match (default; single sequence, one rank)`.

## The change

`INTARSIA_SPEC_SAMPLED` (default `match` on CUDA and Metal, off elsewhere; `Model::set_spec_sampled` for tests):

| value | variant | draft | accept rule at verify row i | on reject |
|---|---|---|---|---|
| `0`, `off` (unset on HIP / cpu) | off | - | sampled requests take plain steps (main's behaviour) | - |
| `1`, `match` (unset on CUDA / Metal) | `SampledSpec::Match` | the MTP head's argmax (main's draft path, unchanged) | draw x from the processed target distribution p_i; accept iff x == d_i | emit x, stop |
| `reject` | `SampledSpec::Reject` | drawn from the head's processed distribution q_i with the request's RNG | accept d_i with probability min(1, p_i(d_i) / q_i(d_i)) | emit a draw from normalize(max(0, p_i - q_i)), stop |

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

**Batched engine after the merge with `d40d93896` (#309 batched MTP inside `--max-batch`, `INTARSIA_BATCHED_MTP`).** Sampled
requests there **fall back to plain decode**, as on main. The batching engine builds its sequences with `Seq::build` only, whose
`spec_on` still requires a greedy request; the sampled switch (`spec_sampled`) is set only by the single-sequence `Seq::new`. So
on the batched phase-A path (solo MTP passes at low occupancy, and the deferred batched verify) a sampled request never drafts,
never reaches the sampled accept, and greedy requests keep main's acceptance unchanged. In `Seq::prepare` the sampled arm comes
first and matches only when `spec_sampled` is not `Off`, so the deferred-verify arm of #309 is untouched for every batched sequence.

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
accepts little (match 27 / 4492 drafts, reject 929 / 4045), so this is a wiring check; 2.3 is the real-model test. It takes about
4 minutes in release, so it is `#[ignore]` (run with `--ignored`); the other new fixture test runs by default.

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
The joint law of tokens 2-4 (supplementary, not in the rule): p 0.018 / 0.357 / 0.180 / 0.062. The two lowest per-position
p-values (config A position 3: 0.018 reject, 0.032 match) are compared against the same spec-off sample, so they are correlated,
not two independent signals; Bonferroni holds under that dependence.

**Power shortfall, stated against the pre-written rule:** the rule assumed at most 40 df; config A's high-entropy positions
realised 48-78 df, so the power against Cohen's w = 0.1 is 0.77-0.90 there, not the >= 0.93 the rule promised (B: >= 0.99).
Against w = 0.12 it is >= 0.97 at every cell (same calculation, lambda = 6000 w^2). The CPU backend has no row-exact verify kernels, so its verify rows can differ
from single-row decode in the last bits; that is far below what this test can see and does not apply to the unit tests.

## 3. Gate 3: speed (reported, no threshold)

Method: one `intarsia serve --max-batch 1` per arm (`scripts/sampled-spec/speed.sh`, `cuda.sh`), `localeval suite --tier adhoc
--cells speed-spec` on the natural-v1 prompts, 256 forced tokens, 3 reps per size, chat template, thinking off. localeval has no
temperature option: the targets file `.scratch/suite/targets.toml` sets it in `extra_body`, which the client applies after its
own `temperature: 0` (checked in the saved request record: `extra_body {'temperature': 0.7}`; localeval's "Method" text still
says temperature 0). top-p 1, top-k 0 (server defaults); localeval pins seed 0. The stock `speed-spec` cell runs only on a spec
lane and at 128 / 1k / 4k / 16k, so a suite copy with sizes [1024, 4096] is used, and for the spec-off arms one with the cell's
lane set to plain; prompts and their sha256 checks are the stock ones. Acceptance: the server's `/metrics` counters over the whole
arm (probe and warmup requests included). Off arms run `--spec off`; T 0 arms are the greedy baseline (today's path: on Metal the
pipelined GPU-accept MTP step); the sampled arms take the eager step. Metal binary: `.scratch/intarsia.br` built from `ecf818a37`
(engine code identical to the head; later commits touch tests, scripts and docs only).

**Metal, endor** (fleetq #8606 round 1, #8674 round 2 in reverse arm order; Qwen3.8-27B Quark MXFP4 with the recipe's MTP k2 and
65536-row draft head; Qwen3.6-35B-A3B NVFP4 with `--spec mtp --spec-k 2`). Round 2 ran 10-20 % slower in absolute tok/s on every
arm, off arms included (box state, not the change), so the tables give each round's own ratios and their mean rather than
pooling medians. Per round: `raw/speed-metal/table.md`, `raw/speed-metal-r2/table.md`; ratios: `raw/speed-metal-ratios.md`.

| model | arm | x off (round 1: 1k, 4k; round 2: 1k, 4k) | mean | greedy gain kept | acceptance r1 / r2 |
|---|---|---|---:|---:|---|
| 27B | T0 MTP (greedy) | 1.200, 1.131, 1.090, 1.038 | 1.115 | 100 % | 67.7 / 69.3 % |
| 27B | T0.7 match | 1.047, 1.124, 1.071, 1.062 | 1.076 | 66 % | 65.8 / 64.2 % |
| 27B | T0.7 reject | 1.083, 1.175, 1.134, 1.074 | 1.117 | 102 % | 70.3 / 69.8 % |
| 27B | T1.0 match | 1.033, 1.053, 1.009, 1.022 | 1.029 | 25 % | 60.3 / 55.8 % |
| 27B | T1.0 reject | 1.154, 1.197, 0.991, 1.018 | 1.090 | 78 % | 63.6 / 65.8 % |
| 35B | T0 MTP (greedy) | 1.366, 1.302, 1.227, 1.257 | 1.288 | 100 % | 70.5 / 69.6 % |
| 35B | T0.7 match | 1.229, 1.210, 1.149, 1.160 | 1.187 | 65 % | 66.6 / 65.9 % |
| 35B | T0.7 reject | 1.125, 1.113, 0.904, 0.958 | 1.025 | 9 % | 70.5 / 71.2 % |
| 35B | T1.0 match | 1.177, 1.147, 0.916, 1.167 | 1.102 | 35 % | 62.3 / 62.1 % |
| 35B | T1.0 reject | 1.067, 1.047, 1.109, 1.122 | 1.086 | 30 % | 67.9 / 67.5 % |

Absolute decode tok/s, round 1 (1k / 4k): 27B off T0 28.0 / 27.5, T0.7 27.8 / 27.4, T1.0 27.2 / 26.4; 35B off T0 123.5 / 120.3,
T0.7 105.4 / 103.5, T1.0 106.3 / 104.1 (host sampling alone costs the 35B ~1.4 ms per token). Spread: m27 T1.0 match round 1 was
rejected by localeval's spread gate (sampled text varies its acceptance per request); its numbers are kept and marked here.
localeval runs: `/opt/code/deepeval/results/20261105-19*-ss-*` (round 1) and `20261105-21*`/`22*-ss-*` (round 2), ids in the tables.

**CUDA 35B tp=1, spark1** (fleetq #8652, one round, `--quiet gpu,cpu` with spark1-vllm up and serving its roles: its live traffic
is an unmeasured noise source for these numbers; `--mem-budget 40`):

| arm | decode 1k / 4k | x off 1k / 4k | greedy gain kept | acceptance |
|---|---|---|---:|---:|
| T0 off | 85.5 / 83.0 | | | |
| T0 MTP | 115.8 / 109.2 | 1.354 / 1.316 | 100 % | 69.0 % |
| T0.7 off | 73.1 / 71.3 | | | |
| T0.7 match | 90.1 / 86.4 | 1.233 / 1.212 | 66 % | 68.0 % |
| T0.7 reject | 85.9 / 81.8 | 1.175 / 1.147 | 48 % | 71.9 % |
| T1.0 off | 73.3 / 71.6 | | | |
| T1.0 match | 85.6 / 81.1 | 1.168 / 1.133 | 45 % | 61.1 % |
| T1.0 reject | 79.1 / 81.1 | 1.079 / 1.133 | 32 % | 66.8 % |

**Reading.** `reject` always accepts more (+4 to +8 points: it drafts from q and keeps min(1, p/q) of the mass, where `match`
keeps p(argmax)). It pays for it on the host: per pass k draft rows (248k floats each) come back and go through a full-vocabulary
softmax, plus the residual on a reject. Where a decode step is ~35 ms (27B Metal) that is small and `reject` keeps the greedy gain
(T0.7) or most of it (T1.0); where a step is ~8-12 ms (35B) it eats most of the extra acceptance and `match` is ahead. Both lose more
at T1.0 than T0.7 (lower acceptance). The sampled step is the eager one; the greedy T0 arm on Metal uses the pipelined GPU-accept
step, so part of the gap to the greedy gain is the step, not the sampling.

HIP: **not run** (tatooine belongs to another session).

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

CUDA (spark1, fleetq #8652, base `602a1fcfe` and branch `c28ad2d3f` built there; `scripts/sampled-spec/cuda.sh`, `raw/cuda/`):
`plan show` 35B text and json **identical**; `kernels test --backend cuda --quick` 106 kernels, 918 cases, exit 0; greedy ids 35B
`--spec mtp --spec-k 2`, short and 4K, base vs branch with `INTARSIA_SPEC_SAMPLED=1`: **identical 2 / 2**.

## 4b. Gates after the merge with `d40d93896` and the default-on change (head `986f6951a`)

`INTARSIA_SPEC_SAMPLED` unset on both binaries (main `d40d93896` vs branch `986f6951a`):

| check | result |
|---|---|
| `cargo test --workspace --no-fail-fast` (endor, fleetq #8716, `--load cpu`) | exit 0: **1336 passed, 0 failed, 32 ignored** (`raw/workspace-test-summary-merged.txt`) |
| Metal `plan show` text + json, 27B / 35B / 8B (fleetq #8722, `scripts/sampled-spec/default-metal.sh`) | **identical 6 / 6** |
| Metal greedy ids 27B (MTP default) / 35B (`--spec mtp --spec-k 2`), short + 4K | **identical 4 / 4** |
| Metal default active, T 0.7, 96 tokens | main: 0 verify passes; branch: startup line `speculation Match (default ...)`, 27B 42 passes 63.1 % accepted, 35B 41 passes 65.9 % |
| CUDA 35B tp=1 `plan show` json (spark1, fleetq #8720, `scripts/sampled-spec/default-cuda.sh`) | **identical** |
| CUDA greedy ids 35B `--spec mtp --spec-k 2`, short + 4K | **identical 2 / 2** |
| CUDA default active, T 0.7 | main: 0 verify passes; branch: `speculation Match (default ...)`, 42 passes, 61.9 % accepted |
| `~/intarsiaml-ss` on spark1 | removed at the end of #8720 |

Reach: `serve` defaults to `--max-batch 1` (the single-sequence engine), so the default reaches default served traffic. A deployment
that sets `--max-batch` > 1 (or the batched-MTP default of the bm-default lane, if it moves serving onto the batching engine) gets
no sampled speculation: there sampled requests decode plainly.

Not rerun after the default change: `kernels test` (no kernel or plan code changed: the change is the env default and one log line;
the earlier Metal and CUDA runs stand), the speed tables (they already measured `match` explicitly), the statistical gate (the
accept code is unchanged).

## 5. Recommendation

- **Single recommendation: `match`, opt-in for now.** Per the coordinator's decision below, `INTARSIA_SPEC_SAMPLED=1` now
  selects `match` (the gates above ran with `1` = `reject`; every speed and statistical arm named its variant explicitly, so
  only the no-op greedy-id runs used `1`, and greedy requests never reach either variant).
- **Superseded by the user's decision (default on, `match`):** the original recommendation was not default-on yet. Exactness is shown, and greedy is untouched, but the sampled output for a seed differs from spec-off
  (same law, other tokens), the best variant depends on the model and backend, and the Metal numbers drift 10-20 % between rounds.
- If one variant is to become the default for sampled traffic (the server's default temperature is 1.0), **`match` on the 35B
  (Metal and CUDA: x1.10-1.22) and `reject` on the 27B (Metal: x1.09-1.12)**; a recipe key per model would express that. `match`
  alone is the simpler and safer single choice: no draft-side change, no host softmax over draft rows, never slower than off in
  the means above (27B T1.0 is the weakest, x1.03).
- Cheap follow-ups for `reject`: keep the draft probabilities on the device (softmax + the drafted token's probability, not the
  whole row), and use device top-k when the request sets top-k (`HeadMode::LastTopK`, as batched decode does).

## 6. What porting would take (out of scope here)

- **DeepSeek-V4.1** (`crates/intarsia-models/src/dsv41_engine.rs:836`, `spec = self.spec.filter(|_| req.temperature <= 0.0 ...)`):
  its verify consumer already draws per row and accepts iff the draw equals the draft (lines ~859-925), which is `match`. Porting
  `match` = lift the temperature filter and make the verify rows of a sampled request come back as logits rows, not device
  argmax (`Row::Logits` instead of `Row::Tok`). `reject` would also need the drafter's per-position distribution.
- **DFlash2** (block draft): `match` needs only the verify logits (`Model::forward_verify_logits`) after `dflash_draft`; `reject`
  needs the block's per-position draft logits (the block predicts all positions in one pass, so q_i does not depend on the earlier
  sampled drafts; sampling each position from its own q_i is still a valid proposal as long as the accept test uses that q_i).
  `--spec auto` would then pick per pass between two sampled drafters with the same acceptance rule.

## 7. Not run, or uncertain

- HIP (tatooine): not run, by the brief. The HIP verify graph replay (`verify_replay`) was not exercised with the logits download;
  sampled speculation runs only at tp=1, so the HIP tp>1 MTP default never reaches it.
- CUDA `plan show` was run for the 35B only (the reference no-op also did the 8B).
- Batched decode (`--max-batch` > 1), constrained output, `trace_logits`, tensor parallelism: sampled requests take plain steps there
  (unchanged); the prefix-store exclusion of drafting requests is unchanged.
- The captured (HIP graph) and pipelined (Metal GPU-accept) spec steps are bypassed for sampled requests; no device-side version.
- Statistical power on config A is 0.77-0.90 at w = 0.1 (section 2.3), below the pre-written >= 0.93.
- Speed: Metal 2 rounds, CUDA 1 round, 3 reps per cell; between-round drift on Metal 10-20 %. Top-p / top-k were not varied in the
  speed runs (server defaults top-p 1, top-k 0, so `dist` takes its unsorted full-vocabulary path).
- The spec-trace host timing of a sampled pass (`INTARSIA_SPEC_TRACE=1` prints draft / verify / accept ms) was not collected in
  the served runs, so the host-cost explanation of `reject` on the 35B is inferred from the ratios, not measured.

## Remote work folder

`cuda.sh` created `~/intarsiaml-ss` on spark1 (source tree, cargo target, the two CLI binaries: 2.7 GB; spark1's disk is 91 % full).
Removed on the coordinator's instruction (fleetq, `--load none`).

## Files

`crates/intarsia-core/src/{sampler.rs,spec.rs,engine.rs,model.rs}`, `crates/intarsia-core/src/model/{mtp.rs,batch.rs,spec_pipe.rs}`
(comment only), `crates/intarsia-models/tests/{mtp.rs,sampled_spec_real.rs}`, `docs/tuning.md` (`INTARSIA_SPEC_SAMPLED` and the
harness variables), `docs/decisions/LOG.md`, `scripts/sampled-spec/{stat-run.sh,gtest.py,speed.sh,speed-table.py,speed-ratio.py,
noop-metal.sh,cuda.sh}`, this doc, `2026-11-05-sampled-spec/raw/`.
