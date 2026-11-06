# DSv4.1 TP=2: speculation for sampled and constrained requests (`INTARSIA_DSV41_SPEC_SAMPLED`)

Lane `dsv41/sampled-spec`, branch `dsv41/sampled-spec`, 2026-11-05. Base: `origin/main` 602a1fcfe at the start, rebased onto 0391ab6b3 and later. The window binary was built from `c8213be9e` (602a1fcfe + the first commit of this branch; the rebase brought no engine change that matters here). No go-ahead or course correction arrived except the coordinator's note that fleetq job #8570 (workspace tests) and window #8579 were killed at about 18:02 by another lane's `pkill -f 'fleetq run'`; both were requeued.

**Result.** With `INTARSIA_DSV41_SPEC_SAMPLED=1`, a request with temperature > 0 or a constraint speculates like a greedy one. Decode by text kind at temperature 0.7 and 1.0, 32 requests x 256 tokens, same fixed seeds in both arms, ABAB:

| T | off (main) | on | gain |
|---|---:|---:|---:|
| 0.7 | 36.2 / 36.2 tok/s | 51.9 / 52.3 | **+43 % / +44 %** |
| 1.0 | 36.5 / 36.4 | 51.7 / 51.6 | **+42 % / +42 %** |

The flag is opt-in; making it the default is a user decision, because sampled text is no longer draw-for-draw the text main gives for a seed (section 2). Greedy ids are unchanged (golden 640/640 and 1120/1120 with the flag off and on).

## 1. The change (`crates/intarsia-models/src/dsv41_engine.rs`)

- **Filter.** `generate` speculated only for `temperature <= 0 && constraint.is_none() && trace_logits.is_none()`. With the flag, temperature > 0 and constraints pass; `trace_logits` still never speculates.
- **Logits rows.** For those requests `next_rows` is called with `logits = true`, so `rank0_verify` takes the `forward_rows` path (`Row::Logits`). Without it a sampled request would silently take the device argmax of each verify row (`Row::Tok`): the one bug that gives a wrong answer with no error. Rank 1 is not told (its verify pass computes every row regardless), so the flag is rank 0 only and does not enter the handshake.
- **Accept rule (read, then proven by test).** The consumer loop already did the right thing, for each row in order: min_tokens mask, then `cons.compute_mask()` (after the commits of the rows before it), then `sampler.sample(&logits)` (or argmax for temperature 0), accept iff the draw equals the draft; on a mismatch emit the draw, truncate the sequence to the accepted rows and drop the rows after it; the last row (`expect = None`) is the bonus draw. The Markov loop's draft is a deterministic function of the context (a point mass), so "draw x from p, accept iff x = d, else emit x" is exact speculative sampling: accept probability p(d), and on rejection x is distributed as p restricted to the other tokens, which is the residual distribution. The RNG is drawn once per token consumed, in order.
- **chooseK retention.** `choose_k(.., retention)` multiplies each position's survival by `INTARSIA_DSV41_SPEC_RETENTION` for sampled / constrained requests (greedy requests pass 1.0 and the code path is skipped: choices bit-identical to main). Default **1.0** (section 4: measured retention 1.0; 0.8 was not faster).
- **Trace.** `INTARSIA_DSV41_SPEC_TRACE=1` adds `pd=` to the round line: the probability the target gave each verified draft (softmax at the request's temperature on the masked logits). `roundcost.py` reads the new field.
- Docs: `docs/tuning.md` rows for `INTARSIA_DSV41_SPEC_SAMPLED` and `INTARSIA_DSV41_SPEC_RETENTION`, `docs/decisions/LOG.md` row.

## 2. Exactness

**Greedy unchanged.** Golden + golden_long + decode prompts (13 prompts, 1120 continuation tokens, of which the first 10 prompts x 64 = **640** are the golden set), chunk 2048, TP=2, against the stored ids of main (`2026-11-05-dsv41-prefill3/raw/ab/gate-tok-main-c2048.st`): **identical 1120/1120 with the flag off and with it on** (`raw/gate-compare-{off,on}.txt`, window #8579, before it was killed).

**Scripted-engine tests** (`failpath_tests`, the real `generate`, `Worker`, `next_rows`, `verify`, `truncate`, rank-1 `Recorder`; rank 0's passes scripted by a context-dependent, multi-peaked target, a draft that is the target's argmax or its second best where the context hashes to 0 mod 3):

| test | what it shows |
|---|---|
| `sampled_speculation_emits_the_ids_plain_sampling_does` | same seed: speculating and plain sampling emit **identical ids, finish reason and sequence length**, 60 seeds x T 0.7 / 1.0, both draft qualities, and top-k / top-p cuts. More than 10 rounds accepted to the bonus row and more than 100 with a rejection (asserted, so the test is not vacuous) |
| `constrained_speculation_emits_the_ids_plain_decoding_does` | the same under a stateful mask (token t allowed at step n unless (t+n) % 3 == 0), sampled and greedy: a mask read before an earlier row's commit, or a draft the mask forbids, would show |
| `sampled_speculation_ends_where_plain_sampling_does` | min_tokens masks, EOS drawn inside a round, stop ids, max_tokens ending mid-round (1, 2, 3, 5, 6, 7, 9, 16), with and without the mask |
| `without_the_flag_only_greedy_requests_speculate` | flag off: a sampled or a constrained request sends the other rank no verify (main's behaviour); greedy still does; flag on: a sampled one does |
| `sampled_speculation_has_plain_samplings_token_frequencies` | G-test of token frequencies at positions 1..5, speculating against plain, **disjoint seeds**, N = 3000 per arm, unconstrained and constrained: p = 0.17 to 0.99 at every position (smallest 0.123, alpha 0.002 per position, 0.01 for the family). Negative control, T 0.7 against T 1.0 at position 1: p = 2e-37 and 6e-32 |
| `choose_k_tests::retention_scales_each_positions_survival` | retention 1.0 is the greedy choice; a smaller one never raises k |

Power of the scripted G-test (noncentral chi-square, alpha 0.002, 6000 draws): detects Cohen's w = 0.063 (df 6) to 0.071 (df 14) at power 0.8.

**Mutation check.** With the accept rule broken so that a rejected row emits the draft anyway (`tok = expect.unwrap_or(tok)` for sampled requests), all three of the id-identity, end-condition and G-test tests fail; restored afterwards.

**G-test on the real model** (TP=2, `gtest.py`: prompt "My favorite color is", 6 tokens at T 1.0, top-p 1, `ignore_eos`; speculating arm `INTARSIA_DSV41_SPEC_CHOOSE=0 INTARSIA_DSV41_SPEC_K=4`, so position 1 comes from the prefill row (a null control) and positions 2..5 from verify rows; seeds disjoint). Cells = tokens with at least 5 expected counts in both arms, the rest pooled. Alpha 0.01 over 6 positions (0.0017 each).

| run | N per arm | p by position 1..6 | smallest p | detectable w (power 0.8) |
|---|---:|---|---:|---:|
| window #8609 | 1500 | 0.203, 0.675, 0.804, **0.0043**, 0.556, 0.355 | 0.0043 | 0.11 to 0.13 |
| replication #8636, fresh seeds | 1500 | 0.837, 0.727, 0.435, 0.209, 0.134, 0.223 | 0.134 | 0.12 to 0.13 |
| both pooled (not planned) | 3000 | 0.601, 0.965, 0.117, 0.134, **0.0044**, 0.202 | 0.0044 | 0.09 to 0.10 |

No position rejects at the planned alpha. Two cells are below 0.005 (position 4 of the first run, position 5 of the pooled sets) and are not the same position in the replication; with 6 positions x 3 tests, one or two such p-values are within chance (the smallest, 0.0043, is 0.026 after Bonferroni over its 6 positions), but this test cannot rule out a difference smaller than w of about 0.1. Verify rows run multi-row kernels, so a small numeric difference from the decode row is expected and is the same kind of difference as greedy speculation's near-ties.

**What is not exact: draws for a given seed.** The scripted test shows identical ids for identical seeds because its verify rows are bit-identical to its decode rows. On the real model the constrained requests did give **identical text with the flag on and off for 32 of 32** (16 requests at T 0.7 with seeds, 16 greedy; `raw-w3/cons-*`) and 8 of 8 token counts in window 2, but the real verify rows are not bit-identical to 1-row passes, so on other prompts a draw near a boundary can move. Sampled output follows the same distribution, not always the same text for a seed, hence opt-in.

Greedy requests under a constraint now speculate with the flag on; their ids can differ from main's by verify-row numerics at near-ties, as unconstrained greedy speculation already can.

## 3. Speed (windows #8579 (killed), #8609, #8636; spark1 + spark2, vLLM down and restored by `run-full.sh`, memory watcher at 5 GiB: lowest 6.5 GiB available, no kill)

`sampcat.py` sends the 32 requests of `catspeed.py`'s set (localeval's categories-rcp.jsonl: 8 recall, 8 code, 8 prose, plus chat-rcp.jsonl: 8 multi-turn chats) as streamed chat requests with the stated temperature, thinking off, 256 tokens, seed 1000 + index (same in both arms): localeval sends temperature 0 and has no temperature flag, which is why no gate saw the sampled path. Decode tok/s = total tokens over total decode time (median per request in the raw files). Binary `c8213be9e`, CAP 66000, chunk 2048, retention 1.0.

**T = 0.7** (window #8609; jobs: off1, on1, off2, on2 in that order, each started fresh):

| kind | off1 | on1 | off2 | on2 | tokens a round (on1) | drafts accepted (on1) |
|---|---:|---:|---:|---:|---:|---:|
| chat | 36.0 | 47.9 | 36.4 | 48.0 | 2.13 | 56.0 % |
| code | 36.4 | 68.9 | 36.2 | 68.8 | 3.58 | 74.7 % |
| prose | 36.2 | 43.1 | 36.1 | 43.6 | 1.84 | 52.1 % |
| recall | 36.3 | 54.0 | 36.2 | 54.8 | 2.54 | 63.9 % |
| **all** | **36.2** | **51.9** | **36.2** | **52.3** | **2.369** | **61.9 %** |

**T = 1.0** (same window and arms): all 36.5 / **51.7** / 36.4 / **51.6**; chat 36.4 / 48.0 / 36.4 / 47.9, code 36.6 / 67.2 / 36.5 / 67.0, prose 36.2 / 42.9 / 36.1 / 42.6, recall 36.6 / 54.7 / 36.4 / 54.8. Tokens a round 2.334, drafts accepted 60.5 %.

**Acceptance per draft position** (accepted of checked; a position is checked only when the ones before it were accepted, 8192 tokens):

| position | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| greedy (T 0, flag on; #8579) | 67.4 % (2303/3415) | 72.7 % | 80.7 % | 84.0 % |
| T 0.7 (#8609 on1) | 67.7 % (2328/3440) | 70.3 % | 80.1 % | 86.0 % |
| T 1.0 (#8609 on1) | 66.9 % (2330/3481) | 69.7 % | 76.4 % | 85.6 % |

Seeded runs repeat exactly (on1 and on2 have identical acceptance and token counts), so the two ABAB passes agree to within the host-timing noise (0.4 tok/s).

For scale: greedy speculation on the same set is 55.6 tok/s (#8579, 2.378 tokens a round); sampled speculation at T 0.7 is within 7 % of it. Per kind that is 74.2 vs 68.9 (code), 45.7 vs 43.1 (prose), 51.2 vs 47.9 (chat), 58.7 vs 54.0 (recall). The part of the gap that is not acceptance is the host side of a sampled round: about 1 ms a row to draw from 129k logits (the traced round lines show the cost outside the pass 6.9 / 8.0 / 9.3 / 10.6 ms for 2..5 rows against 5.1 / 5.2 / 5.2 / 5.2 for greedy, which includes the `pd` softmax of the trace).

**Constrained requests** (`conscheck.py`: chat, json_schema `response_format` with a nested object and a list, thinking off, 300 tokens max; every reply must finish with `stop` and match the schema; window #8636): flag off and on, 16 requests at T 0.7 (seeded) and 16 greedy: **32/32 valid in both arms and identical texts**; wall time for the 16 requests incl. prefill 28.9 s -> 23.8 s (T 0.7, -18 %) and 27.5 s -> 21.7 s (greedy, -21 %); 2.17 tokens a round, 68.7 / 64.5 / 76.3 / 67.8 % by position. (Window #8609 had the first, flawed version of the prompt in its first arm: replies hit the 300-token limit and counted invalid; the second to fourth arms ran the corrected prompt, 8/8 valid each, texts and token counts identical off/on: 488 tokens in 14.8 s off, 12.0 s on.)

## 4. The kill test and chooseK

The consultant's kill test (log p_T(draft) in a greedy window) was folded into the first window: the traced arm ran the sampled path itself with `INTARSIA_DSV41_SPEC_TRACE=1` and then the greedy path on the same 32 requests, and retention = sampled over greedy acceptance per position (`specpos.py ratio`): **1.01, 1.05, 1.09, 1.13, weighted 1.035; from tokens a round 0.90** (T 0.7, 2.246 against 2.378, with the 0.8 default of that binary pruning k). Threshold 0.5: passed by a wide margin. The traced `p_T(draft)` of the 6162 verified drafts at T 0.7 (`raw/serve-tr.log`): mean 0.73, 67 % above 0.9, 26 % below 0.5. The drafts that get verified are the confident ones (chooseK picks k from the head's confidence), and the target is peaked on them, so the sampled draw lands on the draft about as often as the argmax does.

Because the retention came out at 1.0, the window arms ran with `INTARSIA_DSV41_SPEC_RETENTION=1.0` and window #8636 measured 0.8 (arm r08, flag on, T 0.7): **51.5 tok/s against 51.9 and 52.3 at 1.0**, so the shipped default is **1.0** (the knob stays, documented). A temperature far above 1 or a flatter target would lower it.

## 5. Gates

| gate | result |
|---|---|
| `cargo test --workspace --no-fail-fast` (endor, #8610, head `825821c6b`) | exit 0: **1330 passed, 0 failed, 29 ignored** |
| after the retention default change (1.0): `cargo test -p intarsia-models --lib dsv41_engine`, `docs_env_vars` | see section 7 |
| footprint check | prints only `crates/intarsia-models/src/dsv41_engine.rs`; Metal no-op: see section 7 |
| CUDA on spark1 (#8583): `plan show` 35B / 8B, text and `--json`; `kernels test --backend cuda --quick` | **identical to main's stored outputs 4 of 4**; 106 kernels, 918 cases, 0 FAIL, log identical apart from wall time (`raw/gates/cuda/`, reference `2026-11-05-dsv41-prefill3/raw/gates/cuda`) |
| golden 640/640 (1120/1120) at TP=2, flag off and on | identical (section 2) |

## 6. Not run, or uncertain

- Draw-for-draw equality with main for a seed on the real model (only the scripted target, and 32/32 constrained texts, show it).
- A real-model G-test with the production `CHOOSE=1` (the G-test arm used `CHOOSE=0 K=4` so every position 2..5 comes from a verify row). A test at temperatures other than 1.0 and of top-k / top-p on the real model (the scripted test covers top-k / top-p).
- 4K+ prompts and cached prefixes at T > 0 on the real model; long runs; tool-call (lazy) constraints; cancel mid-round at T > 0 (the cancel path is the greedy one).
- Device top-k (`HeadMode::LastTopK`) sampling for the verify rows; the host draws cost about 1 ms a row, the largest remaining part of the gap to greedy speculation (next lever: 7 % at T 0.7).
- The window binary was built before the rebase onto the tp35 and later merges; the rebase changed no file this change touches.
- `cargo test` runs on endor from this session before the workspace job were not in a fleetq slot (`cargo test -p intarsia-models --lib sampled` etc., a few seconds of CPU each).
- Window #8579 was killed at 18:02 by another lane's `pkill`; its gate (both arms), traced arm and the off1 T 0.7 arm completed; the rest was rerun as #8609 (`SKIP_GATE=1 SKIP_TRACE=1`). vLLM was restored by `run-full.sh` on the signal path and confirmed up on both Sparks.

## 7. Metal no-op and final checks

TBD

## Files

`crates/intarsia-models/src/dsv41_engine.rs`, `docs/tuning.md`, `docs/decisions/LOG.md`, `scripts/dsv41-tp2/{sampcat.py,gtest.py,conscheck.py,specpos.py,sampled-hook.sh,sampled-cons-hook.sh}` (`roundcost.py` reads the new `pd=` field), this doc, `2026-11-05-dsv41-sampled-spec/raw/` (window #8579: gate, traced arm), `raw-w2/` (#8609: ABAB, constrained, G-test), `raw-w3/` (#8636: constrained, retention 0.8, G-test replication, pooled test), `raw/gates/` (CUDA gate, scripts).
