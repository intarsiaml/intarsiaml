# Metal continuous batching: no gate probing for requests that do not pay while the batch is busy

> **Update (2026-10-21, docs-accuracy pass, see `docs/decisions/LOG.md`):** Merged to main as PR #3 (`c00af215`; the branch merge `d793300e`, Merge metal-spec-busy-probe 05c2755, is inside it); "not pushed" / "nothing merged" wording in this page describes the state when written.

Date 2026-10-16. Branch `metal-spec-busy-probe` (local, not pushed) off `spike/metal` 5cd03a5. Machine endor (Apple M5 Max), Qwen3-8B BF16, `intarsia serve --backend metal --max-seq 8192 --max-batch 8 --profile-every 0`, `--spec ngram` (k = 4, `--spec-off-at` 4) unless an arm says otherwise. Binaries (private copies): **new** = engine commit 0c7ef42 (sha256 prefix `8f700c591876fd12`), **base** = 5cd03a5 (`9bb825d494889da6`), both `cargo build --release -p intarsia-cli --features metal`, Xcode toolchain.

Fixes the opt-in regression of the M3 gates doc (`2026-10-16-metal-cb-m3-gates.md` section 5): with `--spec ngram`, open-ended text ran at 0.82-0.89x at B=2 and B=3, because every back-off probe of the n-gram gate is a solo verify pass, a whole extra weight pass the batched step cannot absorb.

## Verdict per target

| target | result | verdict |
|---|---|---|
| open-ended B=2, spec on / off, median >= 0.98 | **1.016** (rounds 1.028 / 0.994 / 1.020 / 1.016 / 0.973), base 0.919 | **PASS** |
| open-ended B=3, spec on / off, median >= 0.98 | **1.001** (1.031 / 0.990 / 1.028 / 1.001 / 0.963), base 0.866 | **PASS** |
| grounded B=2 and B=3 stay >= 1.2x | B=2 **1.891x**, B=3 **1.326x** (min over rounds 1.729 and 1.203; see round 5 note) | **PASS** |
| grounded, change against the 5cd03a5 spec-on arm | B=2 129.3 vs 128.1 tok/s (1.009x), B=3 128.8 vs 127.6 (1.009x); on/off ratio 1.891 vs 1.875 (B=2), 1.326 vs 1.300 (B=3) | no loss |
| gate (a), B=1 against G1 | grounded **3.735x** (base 3.735x), open-ended **1.017x** (base 1.024x). B=1 spec trace byte-identical to 5cd03a5 (section 3) | **PASS** (unchanged) |
| mixed load (1 grounded + 3 open) and B=4 do not regress | mixed-once 1.005, mixed-loop 0.995; open-n4 0.995; grounded-n4 0.976 in the 5-round slot, **0.996** in an 8-round repeat (base 0.995) | **PASS** (see B=4 note) |
| identity, batched vs solo, spec on, reuse on, B=2/3/8 | 268 + 268 + 265 comparable, **0 mismatches** | **PASS** |
| `tools/cbidentity.py` B=8, spec off, reuse off and on | 203/203 and 203/203, 0 mismatches, FIFO check OK | **PASS** |
| cargo gates | all pass (section 5) | **PASS** |

## 1. The rule

Where the cost came from. The gate (`crates/intarsia-core/src/spec.rs`, `SpecGate`) backs off after 6 verify passes in a row that accepted nothing, then skips 2, 4, ... 32 plain steps and probes once, doubling on every failed probe. That is cheap inside one sequence (B=1 open-ended: 1.009x) and costly beside a batched step. With two or three open-ended streams the old gate spent 31-61 solo passes per 256 steps (5cd03a5 arm, `/metrics` `intarsia_spec_steps_total`).

Per-pass traces of the two m3 prompts (`trace-base/server-on-1.log.gz`, solo, 5cd03a5) decided the rule. The grounded prompt has runs of two failed passes (passes 10 and 11 of 64, in all five requests), so any "stop after k consecutive misses" rule with k <= 2 would switch speculation off for the whole busy period; mean accepted tokens per pass stays above 2.5 from the second pass on (2.95 overall). Open-ended text accepts 0.14-0.23 tokens per pass (8 of 190 proposed). The two are far apart on the mean accepted per pass, so the rule uses that.

**Rule (`SpecGate::allow(busy)`, `GateParams` `busy_warmup` = 1, `busy_min_accept` = 2).** `busy` is true when at least two sequences decode in the scheduler iteration (`b > 1`; passed to `BatchSeq::prepare` as a new argument next to `spec_allowed`, and set per step with `Seq::set_spec_allowed(allowed, busy)`).

* Not busy: exactly the old gate. The new code is not reached.
* Busy: after the first verify pass, a request whose passes so far accepted fewer than 2 draft tokens each on average (`accepted < passes * 2`) takes plain steps. Its back-off state does not advance (no countdown, no probe), so nothing changes for it until it is alone.
* Alone again (`b == 1`): the old gate resumes from the state it was in; any accepted pass raises the mean again, so a request that starts paying is allowed to draft beside others once the mean is back over the threshold.
* A request that accepts well (grounded text) never meets the condition and drafts as before. The `fails == 0` gate-off switch and `busy_min_accept == 0` turn the busy check off.

Cost left: one warm-up verify pass per request that starts while the batch is busy (2 passes at B=2, 3 at B=3 in the open-ended scenarios, from `intarsia_spec_steps_total`, against 31-61 before).

`INTARSIA_SPEC_GATE` now takes 4 values (as before) or 6 (`..., busy_warmup, busy_min_accept`) and prints what it parsed to stderr; before, a value with the wrong number of fields was silently ignored.

Exactness is untouched: verify is exact (Q1), the rule only decides when to verify. `spec_allowed` (the `spec_off_at` switch) and `TraceEvent::Decode` are unchanged.

### Unit and mock tests (`cargo test -p intarsia-core`: 71 lib tests incl. 5 new gate tests, 21 scheduler tests incl. 1 new)

* `gate_not_busy_ignores_the_busy_check`: B=1, the schedule is the same for any busy parameters, and the old back-off vector `[0, 1, 6, 15, 32, 49, 66]` is unchanged. All pre-existing gate tests now call `allow(false)` with unchanged expected values.
* `gate_busy_stops_probing_a_poor_request_and_keeps_a_good_one`: busy batch, poor request: one warm-up pass, then no draft for 200 steps and the back-off does not advance; good request: drafts on every step.
* `gate_busy_threshold_is_the_mean_accepted_per_pass`, `gate_busy_warmup_and_off_switches`.
* `gate_resumes_when_alone_again`: blocked while busy, back-off pending and frozen, alone it counts down and probes, then busy drafting is back after accepted passes.
* `busy_flag_follows_decoding_rows` (mock scheduler, `tests/scheduler.rs`): `busy` is false while a sequence decodes alone and true exactly in the iterations two decode.

## 2. Throughput, paired and interleaved

Harness: `tools/cbm3.py` through `tools/cbm3_slot.sh client` (prompts `tools/prompts/m3_*.txt`, 256 tokens, aggregate tok/s), one fresh server per arm, arm order reversed on even rounds, 20 s cool-down before each server start, macmon logged. Arms: **off** (new binary, `--spec off`; same code path as base with spec off), **baseon** (5cd03a5, `--spec ngram`), **on** (0c7ef42, `--spec ngram`). Medians over 5 rounds, per-round ratios in brackets. Raw: `docs/results/raw/2026-10-16-metal-spec-busy-probe/final-a/` (fleetq #2239), `final-b/` (#2240), `final-b4/` (#2248).

| scenario | off | base spec on | new spec on | base on/off | new on/off | new/base |
|---|---|---|---|---|---|---|
| solo-grounded (decode tok/s) | 36.8 | 137.5 | 137.6 | 3.735 | **3.735** | 1.001 |
| solo-open | 36.4 | 37.3 | 37.4 | 1.024 | **1.017** | 1.003 |
| grounded B=2 (aggregate) | 68.7 | 128.1 | 129.3 | 1.875 | **1.891** (1.919 / 1.876 / 1.891 / 1.904 / 1.729) | 1.009 |
| grounded B=3 | 97.1 | 127.6 | 128.8 | 1.300 | **1.326** (1.365 / 1.296 / 1.326 / 1.338 / 1.203) | 1.009 |
| open-ended B=2 | 70.0 | 63.9 | 71.0 | 0.919 | **1.016** (1.028 / 0.994 / 1.020 / 1.016 / 0.973) | 1.111 |
| open-ended B=3 | 101.8 | 87.3 | 102.7 | 0.866 | **1.001** (1.031 / 0.990 / 1.028 / 1.001 / 0.963) | 1.176 |
| mixed-once (1 grounded + 3 open, B=4) | 126.3 | 126.6 | 128.0 | 1.006 | **1.005** (0.999 / 0.998 / 1.019 / 1.046 / 1.005) | 1.011 |
| mixed-loop | 122.3 | 121.0 | 121.6 | 0.991 | **0.995** (0.994 / 0.993 / 0.995 / 1.036 / 0.998) | 1.005 |
| open-ended B=4 | 129.3 | 129.8 | 129.6 | 1.004 | **0.995** (0.988 / 1.007 / 0.987 / 0.995 / 1.009) | 0.998 |
| grounded B=4, 5-round slot | 122.4 | 120.5 | 119.5 | 1.010 | **0.976** (0.975 / 0.982 / 0.969 / 0.990 / 0.976) | 0.992 |
| grounded B=4, 8-round repeat (`final-b4`) | 123.9 | 123.1 | 123.5 | 0.995 | **0.996** (0.984 / 1.010 / 0.979 / 0.998 / 1.000 / 0.988 / 1.029 / 0.994) | 1.003 |

Notes.

* **Round 5 of the `on` arm** has the lowest grounded numbers (118.9 and 118.1 tok/s against 128-131 in rounds 1-4; open-ended 69.0 and 98.1, also the round minimum). No cause was looked for; the medians do not depend on it. Without it the grounded B=3 minimum is 1.296.
* **B=4.** At `b >= 4` speculation is off in both binaries, so B=4 differs from base only in the ramp-up (a few iterations with 1-3 rows): grounded-n4 had 10 verify passes with the new gate against 11 with the base gate. The 5-round slot gave 0.976 for grounded B=4 against 1.010 for base, with base per-round values scattering 0.946-1.032; an 8-round repeat of that scenario alone gave 0.996 (base 0.995). I read the first as slot noise (the B=4 pairs scattered 3-5 % in the M3 doc too), not a regression, but the two slots disagree and I did not find a cause.
* Earlier exploratory runs gave lower open-ended numbers for the same rule and the same number of probes: sweep1 (`sweep1/`, #2236, 3 rounds, new 0c7ef42) 0.978 (B=2) and 0.976 (B=3), sweep2 (`sweep2/`, #2238, 4 rounds) 0.999 and 1.010. All three runs of this rule are within a +-2 % spread between slots; the final 5-round slot is the one reported above, the others are listed so the spread is visible.

### Parameter sweep (exploratory, `sweep1/`, #2236)

Arms `wWtT` set `INTARSIA_SPEC_GATE=6,2,32,2,W,T`; the default is W=1, T=2. 3 rounds, aggregate tok/s ratios to off (median):

| arm | grounded B=2 | grounded B=3 | open B=2 | open B=3 | open-ended verify passes (B=2 / B=3) |
|---|---|---|---|---|---|
| 5cd03a5 (baseon) | 1.851 | 1.270 | 0.885 | 0.824 | 31-39 / 53-61 |
| W=1 T=2 (default) | 1.821 | 1.259 | 0.978 | 0.976 | 2-3 / 3 |
| W=1 T=1 | 1.843 | 1.266 | 0.979 | 0.976 | 2-3 / 3-4 |
| W=2 T=2 | 1.832 | 1.275 | 0.972 | 0.969 | 4-5 / 6-7 |

W=1 T=1 and W=1 T=2 are the same within noise on these prompts; T=2 is the larger margin against text with moderate acceptance. A variant that also required a 3-token suffix match for the warm-up pass (a build that is not in any commit, `sweep2/`, #2238) left the probe counts unchanged (the open-ended prompts have 3-token matches too) and was dropped.

## 3. B=1 schedule unchanged (`b1sched/`, fleetq #2241)

`slot_b1sched.sh`: the same four fixed requests (grounded, open-ended with `ignore_eos`, each twice, 512 tokens, greedy, no nonce), one at a time, on base and new with `INTARSIA_SPEC_TRACE=1`, for `--max-batch 1` and `--max-batch 8`. The `spec:` trace (token index, position, drafted, accepted per verify pass) is **byte-identical** between base and new in both configurations (260 lines each), and so are the token ids.

## 4. Identity

* **Spec on, reuse on, batched vs solo at B=2, 3, 8** (`partB/`, fleetq #2249; `slot_partB.sh` is the soak3 Part B script and `tools/cbreuse.py` the soak3 tool and method: solo reference `--max-batch 1 --spec ngram --prefill-chunk 256`; batched `--max-batch 8 --spec ngram --prefill-chunk 256`, reuse on, 120 chains, 268 requests per B, 15 % of follow-ups after a random pause):

| run | sent | comparable | pass | mismatch | errors | not comparable | started from reused KV | of which decode-written |
|---|---|---|---|---|---|---|---|---|
| B=2 | 268 | 268 | 268 | **0** | 0 | 0 | 148 (55 %) | 148 (17098 tokens) |
| B=3 | 268 | 268 | 268 | **0** | 0 | 0 | 148 (55 %) | 148 (17098 tokens) |
| B=8 | 268 | 265 | 265 | **0** | 0 | 3 | 145 (55 %) | 145 (16704 tokens) |
| total | 804 | 801 | 801 | 0 | 0 | 3 | | |

  The 3 not-comparable requests at B=8 (`not_comparable_cached_differs`) started with another cached length than their reference because an idle slot had been reassigned, the same effect as in soak3 (5 of 268). They are counted, not dropped. The new path was exercised: spectrace shows verify passes at B=2 (2047 of 10658 two-row phases) and B=3 (1860 of 6877), and the new gate declined the rest (B=2 and B=3 drafted 2294 and 2047 passes, against 3901 for the solo reference in soak3; with the new rule they are not equal any more, only tokens must be). B=8: 2904 spec-off phases, 0 verify passes in them, the `b < 4` switch value never violated. No negative control was re-run (soak3 had one for the same comparator).
* **`tools/cbidentity.py`, B=8, stream mixed, spec off**, reuse off and on, 203 cases, refs `--max-batch 1` at chunk 2048 and 256 (`cbid/`, fleetq #2243, `slot_identity.sh` as in the M2 closeout): reuse off 203/203 pass, 0 mismatches; reuse on 203/203, 0 mismatches (25 cached-length differences, none with a mismatch). FIFO replay check OK on both traces.

## 5. Build and test gates

| command | result |
|---|---|
| `cargo test -p intarsia-core` | pass (71 lib, 2, 21 scheduler, 0 + 1 ignored) |
| `cargo test -p intarsia-cli` | pass (80 + 1) |
| `cargo check --workspace --all-targets` | pass |
| `cargo check --workspace --all-targets --features intarsia-cli/metal` | pass |
| `cargo build --release -p intarsia-cli --features metal` | pass |
| `cargo test -p intarsia-backend-metal --release` on the GPU, fleetq #2244 | pass, all test binaries ok (`slot-out/gputest.out`) |

## fleetq jobs (all `--host endor --exclusive --mem 24`, mine)

#2234 per-pass trace (base 5cd03a5), #2236 parameter sweep (3 rounds), #2238 sweep2 (the first launch, #2237, failed at once on a variable clash in `cbm3_slot.sh` and measured nothing), #2239 final A (solo, B=2, B=3; 5 rounds x 3 arms), #2240 final B (mixed, B=4; 5 rounds), #2241 B=1 schedule check, #2242 Part B first launch (failed at once: `$W` unset in the copied script; no result), #2249 Part B identity (rerun), #2243 cbidentity, #2244 Metal tests on the GPU, #2248 B=4 grounded repeat (8 rounds). No server or runner of mine is left (`no intarsia serve left` in each slot log).

## Not run / limits

* No unanimis capture was made.
* GPU clock and thermal state were logged (`macmon.jsonl.gz` in each slot) but not analysed per arm.
* Per-pair spread is 3-5 % in places (section 2), so the 0.98 target is judged on medians, as in the M3 doc; one round is under 0.98 at B=2 (0.973) and one at B=3 (0.963), both round 5.
* No negative control for the Part B comparator was re-run.
* Open-ended prompts that begin with a long copied span (so the warm-up pass accepts) and then turn open-ended are only partly covered: such a request pays one or two extra passes before its mean drops under 2, and the sweep did not include one.
* Weights are only the one model, Qwen3-8B BF16; nothing else was tried.

## Reproduce

```
cargo build --release -p intarsia-cli --features metal   # new; build the same at 5cd03a5 for the base copy
BIN=<new> BIN_BASE=<base> ARMS="off baseon on" SCEN=solo-grounded,solo-open,grounded-n2,grounded-n3,open-n2,open-n3 COOLDOWN=20 MAXMIN=150 \
  fleetq run --host endor --exclusive --mem 24 --label "..." -- tools/cbm3_slot.sh client OUTDIR 5
```
The identity, B=1 and cbidentity slot scripts are in `docs/results/raw/2026-10-16-metal-spec-busy-probe/` (`slot_partB.sh`, `slot_b1sched.sh`, `slot_identity.sh`, `env.sh` with the private binary paths of this session).
