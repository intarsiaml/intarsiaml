# DSv4.1 TP=2 on hardware: the default replay prefill and the encoder-only dead chunks, validated (dsv41/validate)

Branch `dsv41/validate` from origin/main `1faac7bf5` (binary `1faac7bf5 dirty=no`, built as fleetq #8519 into `~/intarsia-dsv41-validate` on both Sparks). Measurement and test only; no engine code is changed. Scripts: `scripts/dsv41-tp2/validate-test.sh` (the fleetq job), `validate-client.py` (clients and analysis). Raw logs: `2026-11-05-dsv41-validate/raw/`. **Status: pass rules written before the run (this section); results below are filled in after it.**

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

(filled in after the window)
