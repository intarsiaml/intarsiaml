# Concurrent-identity harness for continuous batching (gate of milestone M1)

Date 2026-10-15. Branch `metal-cb-identity` off `spike/metal` 56ec2d5. Machine endor (Apple M5 Max), Qwen3-8B BF16. Gate text: `docs/research/continuous-batching-design.md` section 7 M1 "Hard identity gate" and section 8.1 (exact-only batching, 256-token prefill chunk while others decode, so the solo reference must use the same chunk schedule). Added: `tools/` files, this doc and its raw output directory `docs/results/raw/2026-10-15-metal-cb-identity-harness/`; `crates/` is untouched.

**Result: the harness works end to end against a real `intarsia serve` and detects deliberate differences.** On today's serial server (max_batch 1), a 49-case subset replayed at B=2, 4, 8 (streaming, non-streaming, mixed) is byte-identical to the solo reference in all four runs (0 mismatches). A run with `max_tokens` capped at 12 is flagged on exactly the 39 predicted cases. The full 203-case set and B in {3, 6} were not run on the GPU (section 6).

## 1. Files

| file | what |
|---|---|
| `tools/cbidentity.py` | the harness: `gen-cases`, `run`, `compare`, `gate`, `selftest`, `mock`. Standard library only; PEP 723 header, so `uv run tools/cbidentity.py ...` does not sync the torch environment of `tools/pyproject.toml` and `uv.lock` is unchanged. `gen-cases` alone needs `--with tokenizers`. |
| `tools/prompts/cb_identity_cases.jsonl` | the default case set, 203 requests (65 KB: prompts are corpus slices stored as file + char offset + length, like `golden_4k.jsonl`) |
| `tools/cbidentity_e2e.sh` | the GPU check: one exclusive fleetq slot, starts `intarsia serve`, smoke, reference, 4 concurrent runs, perturbation self-test, stops the server, releases by trap. `DRY_URL=<mock url>` runs the same script against `cbidentity.py mock` with no slot. |
| `docs/results/raw/2026-10-15-metal-cb-identity-harness/` | raw output of the final GPU run: `e2e.log`, `server.log`, `ref.jsonl`, `conc-b*.jsonl`, `perturbed-cap12.jsonl`, `report.json`, `perturbed-*report.json`, `smoke-*`, `metrics-end.txt` |

## 2. Usage

```
# the whole gate: reference alone, then one replay per B, then compare (exit 1 on any mismatch)
uv run tools/cbidentity.py gate --ref-url http://HOST:8090 --url http://HOST:8091 --b 2,3,4,6,8 --streams mixed,on,off --outdir OUT
# or the phases separately (reference and test may be different servers, with different flags)
uv run tools/cbidentity.py run --mode solo       --url U1 --out ref.jsonl
uv run tools/cbidentity.py run --mode concurrent --url U2 --b 8 --stream mixed --out b8.jsonl
uv run tools/cbidentity.py compare --ref ref.jsonl --test b8.jsonl b4.jsonl --report report.json
uv run tools/cbidentity.py selftest                       # offline, no GPU
uv run --with tokenizers tools/cbidentity.py gen-cases    # regenerate the case set (seed 20261015, Qwen3-8B tokenizer)
```

* `--cases FILE` (default the committed set), `--sample N` (about N cases, stratified by family with at least 2 per family), `--ids a,b`, `--arrival-scale X` (multiplies every `arrival_ms`; 0 = all at once, still bounded by B), `--timeout S`.
* The reference run does not depend on B: run it once and compare every B run against it.
* **Reference flags are the caller's job.** The reference server is whatever `--ref-url` / `--url` of the solo run points at, so the design's rule (solo run with the same `--prefill-chunk` as the batched server, 256) is met by starting the reference server with that flag. `compare` reads `/v1/intarsia/plan` of both runs, warns when they differ, and prints `PREFILL CHUNK DIFFERS` when `prefill_chunk` differs.
* `--stream on|off|mixed`: streaming for all requests, none, or a deterministic half (hash of the case id). Cancel cases are always streamed. The solo reference always streams (it records the delta list) unless `--stream off`.
* `--flush` (default on for solo): before each reference case a 1-token request with an unrelated prompt (token id 151643, `--flush-id`) drops the KV prefix, so every reference prefill starts at position 0.
* `--cap-max-tokens N` plus `compare --expect-cap N`: perturbation self-test (section 5).

Exit codes: `compare` / `gate` 1 on any mismatch, reference error or missing reference record; `run --mode solo` 1 on request errors.

## 3. What is compared

The server exposes no token ids: `crates/intarsia-server/src/openai.rs` has no logprobs and no ids extension (`GenEvent::Token` carries an id internally, the API drops it), and `crates/` was off limits. **Token ids are therefore not compared.** Compared per request, exactly:

* the response text, byte for byte;
* for streamed runs against the streamed reference, the whole list of SSE deltas. Each non-empty delta is one token's released text, so this is the closest stand-in for an id sequence (it also catches different token boundaries that concatenate to the same text);
* `finish_reason`, `completion_tokens` and `prompt_tokens` (streams send `stream_options.include_usage`);
* `cached_tokens` is recorded for both runs and shown next to every mismatch and as a count, but it is not itself a failure: a different KV prefix reuse shifts the prefill chunk boundaries, so a mismatch with `cached_tokens` differing is explainable, not necessarily a bug. A non-stream test record is compared with the streamed reference on text, finish and counts only.

Cancelled requests (stream closed by the client after `cancel_after_chunks` deltas or `cancel_after_ms` ms): the reference for that case is the full run without the cancel. The emitted text must be a prefix of the reference text, and the emitted deltas a prefix of the reference deltas. A cancel case that finishes before the cancel fires is compared in full. The report counts how many cancelled requests emitted more than 0 bytes, because a cancel that fires before the first token passes without testing anything.

Sampled cases carry a `seed`; greedy cases carry `temperature 0` explicitly (the server defaults to 1.0; the loader rejects a case without `temperature`, or a sampled one without `seed`). Chat cases send `chat_template_kwargs {"enable_thinking": false}`.

## 4. The default case set

203 requests, written by `gen-cases` (seed 20261015): 200 generated cases `cb000`-`cb199`, then the three `golden_4k.jsonl` prompts `cb200`-`cb202`. Family counts of the 200: raw greedy 62, raw sampled 30, chat greedy 16, chat sampled 8, EOS (short natural answer, finish by EOS) 16, `min_tokens` 16, stop strings 24 (raw and chat; 80% strings that fire within a few dozen tokens, 20% rare ones that never fire), cancellations 20 (75% by delta count 1, 2, 5, 12, 30; 25% by 30 ms to 4 s), `ignore_eos` 8. The golden family (3) is the two ~4000-token raw prompts (`long_docs_4k`, `long_code_4k`, 64 new tokens, greedy, with a unique nonce prepended) and `chat_reuse_4k` (the four-message system / user / assistant / user chat with the 15000-char document, 64 new tokens; its `reuse_after_message` prefix-reuse feature is ignored), the only multi-turn chat in the set. Target prompt sizes 128, 256, 512, 1024, 2048, 3072, 4000 tokens (weights 14, 14, 17, 17, 15, 12, 11 percent), cut from `corpus_docs.txt` / `corpus_code.txt` at exact token counts with the Qwen3 tokenizer. `max_tokens` 16 to 160 (cancel cases 128). Arrivals: 30% at the same offset as the previous request, otherwise an exponential gap with mean 500 ms.

Every prompt starts with a unique nonce word chosen so that its first token is unique across the set (no two raw cases share a KV prefix). Chat cases still share the template header and the identical system message (17 tokens), so adjacent chat requests reuse a 17-token prefix on the current server; that is what the `cached_tokens` columns below show. The set deliberately has no shared-prefix cases: prefix reuse across slots is design open question 5, not M1.

Measured prompt sizes of the 49-case subset used in the GPU run (reference `prompt_tokens`): 127 to 4025; histogram `<=192` 7, `193-384` 7, `385-768` 10, `769-1.5k` 7, `1.5k-2.5k` 9, `2.5k-3.5k` 4, `>3.5k` 5. Reference finishes: length 33, EOS 12, stop-string-armed cases that stopped 4 (a stop string and EOS both report `stop`, so they cannot be told apart from the API; EOS is counted only for cases with no stop string).

## 5. Checks

**Offline `selftest`** (no GPU, 5 s): 28 checks pass. Synthetic records: identical passes; a text diff, a delta-only diff, a finish diff, a `completion_tokens` diff, a test-side error, a cancelled prefix violation (text and deltas) are caught; a reference error and a missing reference are flagged; valid and empty cancelled prefixes pass; a `cached_tokens` difference alone is reported but is not a mismatch. Mock server (deterministic fake model with the real response shapes, streaming and non-streaming, stop strings, `min_tokens`, EOS, cancels): a clean replay at B=4 in all three stream modes is identical with overlapped requests; a mock with an injected concurrency bug (token altered when 2 or more requests are in flight) is detected, with no more mismatches than times the bug fired; a `max_tokens` cap produces exactly the predicted mismatch set; a case without explicit `temperature` is rejected.

**End to end on a real server.** The golden family was appended after this run: the GPU run used the 200-case file of commit 7873db9f (`cases_sha256` f5f2947e5c78ba41 in the `ref.jsonl` header; `cb000`-`cb199` are byte-identical in the 203-case file, checked by diff), so `cb200`-`cb202` were checked only against the mock server (they resolve and send), not on the GPU, and `--sample 48` on the 203-case file picks a slightly different subset (51 cases); reproducing the table below needs the file as of 7873db9f. (`tools/cbidentity_e2e.sh docs/results/raw/2026-10-15-metal-cb-identity-harness 48`, fleetq slot #2055, 15:12:32 to 15:22:03). Binary `/opt/code/intarsiaml-spec/target/release/intarsia` (built 14:03:32, after the last change to `crates/` at 14:02:48; its `crates/` tree is identical to 56ec2d5), `intarsia serve --model ~/models/Qwen3-8B --backend metal --max-seq 8192 --profile-every 0 --prefill-chunk 256 --no-supervisor --no-ui --port 8099`. Reference and replays use the same server, so `prefill_chunk` is 256 for both. 49 cases (the `--sample 48` stratified subset), `--arrival-scale 0.25`.

| run | cases | pass | mismatch | stream / non-stream | cancelled (emitted >0 bytes) | `cached_tokens` differs (all pass) | wall |
|---|---|---|---|---|---|---|---|
| reference, alone, sequential, flush before each | 49 | | | all stream | | 0 | 97.8 s |
| B=2 mixed | 49 | 49 | 0 | 29 / 20 | 5 of 5 (3) | 3 | 82.7 s |
| B=8 mixed | 49 | 49 | 0 | 29 / 20 | 5 of 5 (3) | 4 | 82.7 s |
| B=4 stream on | 49 | 49 | 0 | 49 / 0 | 5 of 5 (3) | 4 | 82.6 s |
| B=4 stream off | 49 | 49 | 0 | 5 / 44 | 5 of 5 (3) | 4 | 83.0 s |

`GATE COMPARE rc=0`, "IDENTICAL, 0 mismatches". Peak client-side in-flight requests equalled B in every run (2, 8, 4, 4); the server queued them, since `intarsia serve` runs one request at a time. The 3 to 4 `cached_tokens` differences are chat requests that reused the 17-token template header of the previous request in the replay (reference 0, replay 17); their output was identical although the prefill chunk boundaries moved by 17 tokens.

**Perturbation self-test.** Same sample replayed at B=4 mixed with every `max_tokens` capped at 12 (`--cap-max-tokens 12`, 30 s). Plain `compare`: 40 mismatches, exit 1, "FAIL". `compare --expect-cap 12`: the expected set is every non-cancel case whose reference `completion_tokens` exceeds 12; it has 39 members and the mismatching non-cancel set is exactly those 39 (nothing missing, nothing extra), exit 0. The 40th mismatch is a cancel case that completed under the cap before its cancel fired and so was compared in full (as designed, excluded from the predicted set). Each mismatch line shows the first differing char, the delta index, the finish and token-count differences and both `cached_tokens`.

An earlier GPU run on the first case set (40 cases, slot #2051, B=2, 8 mixed, 4 on, 4 off) was also 0 mismatches with the same perturbation result, but its 4 stop-string cases all ran to `length`: no stop string fired, so that path was not exercised. The stop lists were changed to strings that fire, chat stop cases were added, and `--sample` became stratified (the evenly spaced sample had hit one cancel case); the table above is the rerun. Its raw files were not kept. The first attempt (slot #2047) aborted at the smoke step on a shell quoting bug in `cbidentity_e2e.sh` (fixed, `DRY_URL` added so the script wiring can be checked against the mock without a slot).

## 6. Limits and gaps

* **No token ids.** Text, deltas, finish and counts are compared. For the M1 gate the server needs an ids extension (for example an `x_token_ids` field on each chunk and on the final response) to compare `ids` as the gate text says; the harness would add it to the compared fields.
* **Only the serial server was tested.** Server-side concurrency was 1 in every run, so a pass here shows the harness and the client-side concurrency (B outstanding requests, mixed stream modes, cancels) work, not that batched output is identical. The injected-bug mock shows detection works when requests do overlap.
* **Not run on the GPU:** the full 203 cases (including the three `golden_4k` prompts), B=3 and B=6, and the stream modes at B=2 and B=8 separately. Estimated cost of the full gate on the serial server from the measured 83 s per 49 cases: about 6 min per replay, about 7 min for the reference, so about 40 min for `--b 2,3,4,6,8 --streams mixed`; a batched server will be faster.
* **Cancel effectiveness.** Only 3 of the 5 cancelled requests emitted output; the others were cancelled while queued or at the first token. With the serial queue a millisecond cancel mostly fires before the request starts, so most cancels are by delta count; the report prints the "emitted >0" count so a gate run can require enough effective cancels. The harness does not check that the server released the cancelled request's slot or that later requests are unaffected beyond their own identity.
* **KV prefix reuse is not controlled in the replay.** The reference flushes the prefix, the replay cannot. The unique nonce removes shared prefixes except the chat template header; mismatches with `cached_tokens` differing are marked in the table but not excused. If M1 adds prefix reuse across slots, the reference will need to reproduce the same cached length per case.
* **Stop string versus EOS** cannot be told apart from the response (both `stop`); coverage counts EOS only for cases with no stop string.
* **Plan comparison** covers the whole `/v1/intarsia/plan` dict (any differing key is a warning) and singles out `prefill_chunk`; it does not know about a future `max_batch` field.
* Sampled cases rely on the server's seeded sampler being deterministic across batch composition; 10 sampled cases matched in all runs here.
* The `tools/pyproject.toml` environment is not used (stdlib only); the script is a PEP 723 script run by `uv run`, so `pyproject.toml` and `uv.lock` are unchanged.

## 7. fleetq

All GPU use ran in exclusive endor slots (`--mem 24`), taken by `cbidentity_e2e.sh` with `fleetq acquire --host endor --exclusive --wait`, checked RUNNING before the server started, released by trap.

| slot | what | running | released |
|---|---|---|---|
| #2047 | first attempt, aborted at smoke (script bug) | 14:40:31 | 14:41:16 |
| #2051 | e2e on the first case set (40 cases) | 14:48:18 | 14:57:00 |
| #2055 | e2e on the final case set (49 cases), the table above | 15:12:32 | 15:22:03 |

No `intarsia serve` process of this session was left (`leftover serve on port 8099: 0`).
