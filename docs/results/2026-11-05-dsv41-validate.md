# DSv4.1 TP=2 on hardware: the default replay prefill and the encoder-only dead chunks, validated (dsv41/validate)

Branch `dsv41/validate` from origin/main `1faac7bf5` (binary `1faac7bf5 dirty=no`, built as fleetq #8519 into `~/intarsia-dsv41-validate` on both Sparks). Measurement and test only; no engine code is changed. Scripts: `scripts/dsv41-tp2/validate-test.sh` (the fleetq job), `validate-client.py` (clients and analysis). Raw logs: `2026-11-05-dsv41-validate/raw/`. **Status: done. Pass rules were written and pushed (commit `1d3d9282a`) before the first window; results are in the last section. Two windows: fleetq #8522 (15:24-16:05 EST, sessions S1-S5) and #8527 (16:12-16:31, sessions F1-F5), about 60 minutes of Spark time in all. No defect found.**

## Setup (all arms)

Chunk 2048 (the default), `INTARSIA_DSV41_CAP=66000`, rank 0 spark1 `serve --tp-remote`, rank 1 spark2 `tp-worker`, as `replay-test.sh`. Arms: **default** (both variables unset), **off** (`INTARSIA_DSV41_REPLAY=0`: the old full schedule), **skip0** (`INTARSIA_DSV41_REPLAY_SKIP=0`: replay on, every chunk's decoder half on its last 128 rows). Every start strips an inherited `INTARSIA_DSV41_REPLAY*` (`env -u`). Five sessions (fresh ranks each) in ONE exclusive fleetq job over both Sparks, vLLM taken down and restored by an EXIT trap, memory watcher with the kill line at 5 GiB MemAvailable:

| session | arm | runs |
|---|---|---|
| S1 | default | c1, speed A, c2, c3, c4, catspeed |
| S2 | off | c1, c2, c3, c4 |
| S3 | skip0 | c1, speed B |
| S4 | default | speed A |
| S5 | skip0 | speed B (speed order: A B A B) |

Quality runs of check 2 go through localeval (`quality --dataset ... --judge none`). Checks 1, 3 and 4 use `validate-client.py` instead, because localeval has no temperature or seed flag (check 4), records no token ids (check 3) and sends text, not token ids (check 1: offsets and chunk boundaries must be exact).

## Pass rules (fixed before the run)

**Check 1, prefix-cache hits after skipped chunks.** Prompt A = 6000 tokens (BOS + the first 5999 tokens of `tools/prompts/corpus_docs.txt`), chunks 2048 / 2048 / 1904, whole-prompt floor 5872. All requests are raw token ids to `/v1/completions`, greedy, 64 tokens, `ignore_eos`. The server holds ONE cached sequence, so a separator request (64 ids that start with no BOS) precedes every cold B; the very first request of each fresh server is cold B1 and is compared with a later cold B1 after a separator (the separator method must reproduce the fresh-server result bit for bit). Hit runs: separator, A with `max_tokens` 1 (so the state is exactly A), then B.

The engine reuses a prefix only if the ring margin (16 rows) allows the roll-back (`restorable_len`) and, with replay on, `reuse_ok(keep, floor, replay_from)` holds. Each chunk is a forward call with floor = its end - 128; a dead chunk (encoder-only) leaves `replay_from` = its end, any other chunk its floor. From these rules (ported in `validate-client.py`, `predict`) the expected `cached_tokens` per B is (full table in `predict.md`):

| B | tokens | default | REPLAY=0 | REPLAY_SKIP=0 |
|---|---|---|---|---|
| B1 = A + 300 (the task's first B) | 6300 | 6000 | 6000 | 6000 |
| B1m = A[:-8] + 50 (in-margin variant, added: the case that depends on cache history) | 6042 | 5992 | 5992 | 5992 |
| B2 = A[:4100] + 50 (the task's second B) | 4150 | 0 | 0 | 0 |
| B3 = A[:2040] + 50, after a cancel (the task's third B) | 2090 | 0 | 0 | 0 |
| B3' = A[:4088] + 50 after a cancel that left 4096 rows (added; the review's cold-versus-resume case at the boundary a cancel can reach) | 4138 | 0 | 4088 | 4088 |

B2 and B3 are cold in every arm by the ring margin (the state is 1900 or more rows past them), so they are cold-versus-cold controls, not hits; the hits that exist are B1, B1m (and B3' where predicted). Why B3' and not 2048: the host queues two chunks (4096 rows, `PREFILL_SYNC_ROWS`) before it waits for the device, so a client close lands at the cancel check of chunk 3 (state 4096), or not at all; a delay sweep (0.8, 1.2, 0.5, 1.6, 2.4 s) finds where it lands, confirmed by the probe `A[:4096] + 200` whose `cached_tokens` is the state length (4096; 2048 would be the review's case) and by the server's metrics counters. If no delay lands, the cancel cases are reported as not reachable and nothing is worked around.

PASS when all hold, per arm: (a) no HTTP error, hang or garbage (a reply of 64 tokens with at least max(6, 8) distinct token ids and at most half of its 4-grams repeated); (b) measured `cached_tokens` within 1 of the prediction for every B (a miss is a finding and is reported, the predictions are the engine rules as read); (c) where the predicted keep is 0 the reply ids equal cold B's exactly; (d) the fresh and the after-separator cold B1 are identical. For hits with a predicted keep above 0 the reply need not equal cold B (the hit prefills a different chunk schedule even with replay off, and in the default arm the rows between the new floor and the cached length were computed under the earlier floor, so a reply may depend on cache history); the equal-prefix length against cold B is recorded for default and off side by side and is not a failure. The default hit is a defect only if it is garbage or, on equal footing, its equal-prefix is shorter than off's by more than half of the tokens in two of the three hit cases (a tripwire, not a bit rule).

**Check 2, question first, 32K.** 12 prompts of the frozen chunk32k-2 set (`final` seed, shard sha256s re-verified equal to `items.sha256.json`), two per kind (hop2-L3, hop3-L3, agg_sum-L3, agg_sum-L4, agg_max-L3, repo_sum-L4), chosen from the items that BOTH arms of the earlier step 6 (question last) answered right with the exact FINAL parse, so a floor effect does not hide a change; the question and the instruction move before `<document>` and nothing follows `</document>` (the question ends 100-ish characters into a 90K-character prompt). localeval quality, thinking off, 400 tokens, chunk 2048. PASS when no reply is empty or errored, at least 11 of 12 are scored in both arms, and default is not worse than off by more than 2 items (exact FINAL parse). The replies that differ are listed.

**Check 3, thinking on.** 8 items at 4K (recent-window set, first 8 rows of `recent.4k.jsonl`, scored by the last code in the answer) and 8 at 32K (the first 8 of the 12 above, question last, last `FINAL:` line), chat endpoint with `enable_thinking`, greedy, `max_tokens` 1536 (the window budget cannot afford localeval's default 8192; a reply cut at the cap is counted as unfinished and reported). Reasoning and answer are separated by the server; only the answer is scored. PASS when thinking really is on (reasoning text or reasoning tokens in at least one reply of the default arm), no reply is degenerate, and default is not worse than off by more than 2 items per length. Token-identity rate (reply ids equal) and the mean equal-prefix fraction are reported.

**Check 4, sampled decoding.** 16 requests (the first 16 recent-window 4K prompts, each with a unique first line, so no prefix hit: asserted from `cached_tokens`), temperature 0.7, seed 1000 + i the same in both arms, `max_tokens` 60, plus 8 free-form prompts with 256 tokens (seeds 2000 + i) to see repetition. Nothing degenerates: no empty answer, no free-form reply with fewer than max(6, n/8) distinct ids or over half of its 4-grams repeated, no run of one token longer than 40. Same distribution (as far as 24 requests can show): answer-correct counts within 2, mean free-form length ratio default/off in 0.75-1.33, mean repeated-4-gram fraction within 0.10. Identical replies per seed are reported (not expected to be all identical).

**Check 5, speed.** `localeval speed --sizes 1024,65536 --gen 8 --reps 2` in the order default, skip0, default, skip0 (ABAB, fresh ranks each). The longest length is 65536 under CAP 66000. CAP is not raised: the 11-04 chunk32k-2 run at CAP 66000 with 64K prompts measured a lowest MemAvailable of 5.6 GiB (spark1) / 6.3 GiB (spark2) against the 5 GiB kill line; there is no room for more. PASS when no error occurs, default prefill time is at most 1.05 x skip0's at 1K (one chunk: the skip cannot fire, so about 1.0 is expected; a regression check), and at 64K default is at most 1.00 x skip0's (the skip must not be slower). Decode by kind: `catspeed.py` over `categories-rcp` + `chat-rcp`, 256 tokens, default arm; reported against the 11-05 prefill3 figures (all 55.3 tok/s; chat 50.8, code 73.8, prose 45.6, recall 58.4); PASS when each kind is within 10 % of those. The request count is whatever those two files hold (reported).

## Results

Raw: `2026-11-05-dsv41-validate/raw-w1/` (job #8522) and `raw-w2/` (job #8527): steps log, serve and worker logs of every session, the memory logs, every client output (`c1-*.json`, `c3-*.json`, `c4-*.json`, `cancel-*.json`), the localeval result folders, and the analyses (`analysis-c*.md`). Item files are gitignored (`.scratch/`); regenerate with `gen_items2.py` (frozen shard sha256s match `items.sha256.json`), `replay-recent-gen.py` (sha256s match) and `validate-client.py select` (see the script header). Memory: lowest MemAvailable over both windows 6.1 GiB (spark1) and 6.3 GiB (spark2) in #8522, 6.6 / 6.5 GiB in #8527 (65.7K prefill); the 5 GiB kill line never fired.

**Window #8522: what went wrong and was redone.** The speed sessions asked localeval for 65536 tokens; its filler is 3.2 % larger than its estimate (67.6K real tokens) and the server refused it over CAP 66000 (HTTP 400). The 1K rows of that window are valid and listed; the long rows were redone in #8527 with size 63700 (65.7K real tokens, 8 generated). Sessions S4 and S5 of #8522 were therefore only a restart each. #8527 also ran what #8522's runs showed to be missing (below).

### Check 1: prefix-cache hits after skipped chunks. PASS (all three arms, 0 prediction misses). #8522 (S1, S2, S3), #8527 (F1, F2)

Predicted `cached_tokens` (rules read from `restorable_len`, `reuse_ok`, `replay_keep`, `decoder_dead`; table in `predict.md`) against measured, every case, every arm:

| B (prompt tokens) | default pred / measured | REPLAY=0 | REPLAY_SKIP=0 | reply ids equal to cold B (default / off / skip0) | equal prefix of 64 (default / off / skip0) |
|---|---|---|---|---|---|
| B1 = A + 300 (6300) | 6000 / 6000 | 6000 / 6000 | 6000 / 6000 | yes / yes / yes | 64 / 64 / 64 |
| B1m = A[:-8] + 50 (6042; added) | 5992 / 5992 | 5992 / 5992 | 5992 / 5992 | no / no / no | 11 / 27 / 11 |
| B2 = A[:4100] + 50 (4150) | 0 / 0 | 0 / 0 | 0 / 0 | yes / yes / yes | 64 / 64 / 64 |
| after a cancel that left 2048 rows: B3 = A[:2040] + 50 (2090) | 0 / 0 (#8527 F1, twice) | 2040 / 2040 (#8522) | 2040 / 2040 (#8527 F2, twice) | yes / no / no | 64 / 45 / 45 |
| after a cancel that left 2048 rows: B3' = A[:4088] + 50 (4138) | not run | 2048 / 2048 (#8522) | 2048 / 2048 (#8522) | - / yes / yes | - / 64 / 64 |
| after a cancel that left 4096 rows (default, #8522): B3 / B3' | 0 / 0 and 0 / 0 | - | - | yes, yes | 64, 64 |

- The fresh-server cold B1 and the cold B1 after a separator request are bit-identical in all three arms (the separator method is sound); no reply is garbage; B2, B3 (default) and every predicted-0 case equal cold B exactly.
- **B2 and B3 as literally specified are cold-versus-cold controls** in every arm: the ring margin (16 rows) refuses the roll-back from 6000 or 4096 rows to 4100 or 2040. B1 is a real hit (the engine reuses all 6000 rows) and is bit-identical to cold B1 in the three arms (a 300-row resume against a cold chunk schedule gave the same 64 tokens). B1m is the hit that depends on cache history: the 5992 kept rows were computed under A's schedule, and 11 (default, skip0) or 27 (off) tokens match a cold run, then the greedy path moves; the same difference shows with replay off, so it is the chunk schedule of a resumed prefill, not the replay.
- **The review's case, on hardware.** A client close lands the cancel at a chunk boundary: at 0.02 s the state is 2048 rows in the default arm (probe `A[:2048] + 200` cached 2048, server counter `requests_finished{reason="cancelled"}` +1); at 0.8 s it is 2048 in off and skip0 and 4096 in default (presumably the host has queued both encoder-only chunks by then; it waits for the device every 4096 rows). With 2048 rows cached: **default prefills B3 cold (cached 0) where REPLAY=0 and REPLAY_SKIP=0 resume at 2040**, as the review predicted; the default reply equals cold B3, the resumed replies match cold for 45 tokens and then differ. Reproduced twice in #8527 (identical both times). A lost prefix hit, not a wrong answer.
- Cancel confirmed between chunks by the probe `cached_tokens` (2048 or 4096, never 0 or 6000) and the server's cancelled counter, in every cancel; the engine logs nothing about it.

### Check 2: question first, 32K. PASS. #8522 S1 (default), S2 (off)

12 items chosen as both arms right with the question last; question and instruction before `<document>`; localeval, thinking off, 400 tokens, chunk 2048. **Exact FINAL: default 12 of 12, off 12 of 12**; 10 of 12 replies identical text, 0 empty. The two that differ (`agg_max-L3-32k-000`, `repo_sum-L4-32k-000`) differ in layout of the working and both end on the right answer (`analysis-c2.md`). Limit: the item selection saturates the metric (both arms at 100 %), so this shows no loss at the 12-item level, not that there is none.

### Check 3: thinking mode on. PASS (with one deviation from the written rule). #8522 S1, S2

Chat endpoint, greedy, `max_tokens` 1536, reasoning and answer separated by the server (reasoning present in 16 of 16 replies per arm), only the answer scored.

| set | n | finished before the cap, default / off | correct, default / off | reply ids identical | mean equal-prefix fraction |
|---|---|---|---|---|---|
| 4K (recent-window) | 8 | 8 / 8 | 8 / 8 | 7 of 8 | 0.88 |
| 32K (chunk32k-2, question last) | 8 | 6 / 5 | 6 / 5 | 2 of 8 | 0.37 |

The 32K replies that did not finish were still enumerating document lines at the cap (default 2: `agg_sum-L4`, `agg_max`; off 3: those and `agg_sum-L3`); default is never worse than off. **Deviation:** the pre-registered degeneration proxy (fewer than max(6, n/8) distinct ids, or over half of the 4-grams repeated) flagged two long reasoning replies (`agg_sum-L3-32k-000` and `agg_max-L3-32k-000`, both in the off arm); reading their tails shows coherent enumeration of quoted document lines, and a loop test (any 16-gram six times or more) finds none in any reply (largest count 5). By the letter of the rule this check fails on the proxy; I judge the proxy wrong for thinking replies and report PASS on the loop test (`analysis-c3.md`).

### Check 4: sampled decoding. PASS. #8522 S1, S2 and #8527 F1, F5

Temperature 0.7, seed fixed per request, same in both arms, no prefix-cache hit (asserted from `cached_tokens`).
- 16 answer requests after 4K prompts (60 tokens) and 8 free-form (256 tokens, short prompts of under 128 tokens, so the replay cannot cut): default and off give 24 of 24 identical replies; 16 of 16 correct in both arms. This run did not exercise the replay on the free-form side, so #8527 added:
- **16 long-prompt free-form requests** (3.7K-4.3K token prompts, summary of the log, 256 tokens, seeds 3000 + i): replies differ from off from token 3-160 on (0 of 16 identical, as expected with different numerics and the same seed), mean length 237.1 (default) against 232.6 (off), ratio 1.02; repeated-4-gram fraction 0.005 against 0.007; longest single-token run 1; no empty or degenerate reply; 7 of 16 default replies and 3 of 16 off replies ran into the 256 cap. Nothing degenerates; the lengths and repetition agree (`raw-w2/analysis-c4l.md`).

### Check 5: speed. PASS. #8522 S1-S5 (1K), #8527 F1-F4 (65.7K)

`localeval speed --gen 8 --reps 2`, ABAB, fresh ranks each, chunk 2048, CAP 66000. Prefill time to first token (TTFT, s):

| prompt | default (A1, A2) | REPLAY_SKIP=0 (B1, B2) | default / skip0 | pass rule |
|---|---|---|---|---|
| 1.03K, #8522 | 0.717, 0.713 | 0.716, 0.729 | 0.99 | at most 1.05: met |
| 1.03K, #8527 | 0.719, 0.715 | 0.742, 0.706 | 1.00 | at most 1.05: met |
| 65.7K, #8527 | 33.27, 33.89 | 39.85, 37.68 | 33.58 / 38.76 = 0.87 | at most 1.00: met |

Each cell is the median of two requests. At 65.7K both default cells are faster than both skip0 cells (the skip takes 13 % off the long TTFT: about 1960 tok/s against 1700 tok/s prefill). The 1K row is a regression check: one chunk, the skip cannot fire, and the arms agree within 0.03 s. **CAP not raised:** with CAP 66000 the lowest MemAvailable was 6.1 / 6.3 GiB (32K prompts, #8522) and 6.6 / 6.5 GiB (65.7K, #8527) against the 5 GiB kill line; the 11-04 chunk32k-2 run measured 5.6 GiB at 64K prompts. The state costs 0.203 GiB per sequence at 66000 tokens (engine log), so a CAP above 66000 costs little in KV, but the headroom over the kill line is 1.5 GiB and the effect of a longer prefill on it was not measured; I did not raise it. The longest prompt that fits is 65.9K (CAP minus the 8 generated tokens); 65.7K was run.

Decode by kind (`catspeed.py`, `categories-rcp` + `chat-rcp`, 32 requests, 256 tokens, spec on, default arm, #8522 S1): chat 50.5, code 73.8, prose 45.2, recall 58.3, **all 55.1 tok/s** (medians 51.3 / 73.1 / 45.1 / 57.7 / 54.9) against the 11-05 prefill3 figures 50.8 / 73.8 / 45.6 / 58.4 / 55.3: every kind within 1 %. PASS.

## Defects, not run, uncertain

- **No defect found.** The one behavioural difference from the old schedule is the one the review named (a cancel that leaves 2048 rows makes the next shorter-prefix request prefill cold in the default arm); it costs a prefix hit, the reply equals the cold reply.
- Tooling, not engine: `localeval speed --sizes 65536` renders 67.6K tokens and is refused by CAP 66000 (use 63700). Redone.
- The 4096-row cancel case is the only one reachable in the default arm with a 0.8 s close; the 2048-row case needs a close within about 0.02 s and was found by sweeping (0.02 s on the first try).
- Not run: sampled decoding with thinking on; thinking at lengths other than 4K and 32K; question-first at 4K or 65K; the cancel case at chunk 512; hits with a decode of more than 16 tokens before B (A was served with one token, so the state is exactly A); the resumed-prefill case with a chat-template turn; B3' (A[:4088] + 50) after a 2048-row cancel in the default arm was not run (it was run after a 4096-row cancel, and in off and skip0 after 2048).
- Check 2 has 12 items chosen to be right in both arms, so it cannot show an improvement and has little power to show a small loss. Check 3's 32K accuracy (6 against 5) depends on the 1536-token cap; two items finish in one arm only.
- localeval ran only check 2, speed and catspeed. Checks 1, 3 and 4 use the direct client for the reasons in "Setup".
- Leftovers: `~/intarsia-dsv41-validate` on both Sparks (binaries, source copy and build tree, about the size of a build) is kept for a rerun; the worktree `.worktrees/dsv41-validate` holds `.scratch/` (items, build log); neither was deleted.
