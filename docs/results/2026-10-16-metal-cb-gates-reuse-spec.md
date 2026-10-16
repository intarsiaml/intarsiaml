# Metal continuous batching: two missing hardware gates (prefix reuse, speculation)

Date 2026-10-16. Machine endor (Apple M5 Max), Qwen3-8B BF16, `--max-seq 8192`. Branch `metal-cb-gates` from `70b15eb` (`spike/metal` with `serve --max-batch`). Binary: `intarsia-gates2`, sha256 prefix `2f3d4855efa24581`, built `cargo build --release -p intarsia-cli --features metal` at `75ecb17`. Identity contract: `docs/research/continuous-batching-design.md` 4.5 (a batched request equals its solo run at the same prefill chunk and the same cached length, KV written the same way).

| Gate | Result | Numbers |
|---|---|---|
| A: concurrent identity, prefix reuse ON, incl. reuse of decode-written KV, B=8 | **PASS** | 322 comparable requests (328 sent), 0 mismatches; 172 (53%) started from reused decode-written KV |
| B: `--spec ngram` with `--max-batch 8`, B=2/3 identity, B>=4 switch-off | **PASS** | 5 runs x 48 requests, 0 mismatches; drafting off in all 2160 decode phases with 4+ decoding, back on in 36 of 37 transitions |

fleetq jobs: pilot #2135, gate A #2140, gate B #2141 (all `--host endor --exclusive --mem 24`, each ran its own servers and left none). Raw outputs: `docs/results/raw/2026-10-16-metal-cb-gates-reuse-spec/` (scripts `gateA.sh`, `gateB.sh`, `pilot.sh`, `serve_lib.sh`; results under `out/{A,B,pilot}`; server logs gzipped).

## 1. How the prefix cache treats decode-written KV (checked before designing gate A)

* `SequenceState::commit` extends `tokens` on every forward pass, decode included (`crates/intarsia-core/src/state.rs:243`). The held sequence therefore equals prompt + generated tokens, minus the last one (never fed). With speculation, `Seq::finish` truncates back to `plen + generated - 1` (`engine.rs:1204-1220`).
* Solo (`LocalEngine`): `common_prefix(&st.tokens, &req.prompt).min(plen - 1)` (`engine.rs:864`). Batched: `pick_slot` takes the free slot with the longest `min(common_prefix(slot_tokens, prompt), plen - 1)` (`scheduler.rs:562`). Same rule, so the contract can be met for decode-written KV; a solo reference is comparable.
* Measured: solo chain `c000`, P prompt 278 tokens, 80 generated, follow-up cached **357 = 278 + 80 - 1** (`out/pilot/A/gen.log`).

## 2. Gate A: concurrent identity with prefix reuse on

Tool: `tools/cbreuse.py` (new). Reference = reproduced predecessor:

1. `gen` on a solo server (`--max-batch 1 --prefill-chunk 256`, prefix cache on): flush, then each chain in order. Chain = P (cold, random 64-2400-token corpus prompt behind a unique nonce token) then F (prompt = P prompt + P output ids + new tokens) then, for 20% of chains, F2 (same again on F). Chain kinds: `ext` (output + 8-120 new tokens), `ext2` (depth 3), `branch` (keep a random head of the output, then diverge: reuse with truncation), `exact` (prompt = previous prompt + output, cached = plen - 1). Prompts are token-id arrays, so there is no retokenisation drift. 30% sampled (seeded), 50% streamed. Solo records include `cached_tokens` and the prefill chunk.
2. `run` on `serve --max-batch 8 --prefill-chunk 256` with reuse ON (no `INTARSIA_CB_NO_REUSE`), `INTARSIA_CB_TRACE=1`, 8 concurrent clients, each running a chain at a time with the same prompts; 15% of follow-ups wait 50-500 ms first so another client can take the predecessor's slot.
3. `compare`: token ids, text, deltas, finish_reason, usage against the solo record. A request counts only if it and every earlier request of its chain had the reference's cached length and prefill chunk (the KV it started from was then written the way the solo run wrote it). Anything else is reported as not comparable, never dropped.

150 chains, 328 requests (150 P, 150 F, 28 F2), prompts 69-2409 tokens.

| Run | Server | Sent | Comparable | Pass | Mismatch | Errors | Started from reused KV | From decode-written KV | Not comparable |
|---|---|---|---|---|---|---|---|---|---|
| A1 (the gate) | `--max-batch 8 --prefill-chunk 256`, reuse on, B=8 | 328 | **322** | **322** | **0** | 0 | 172 (53% of comparable) | 172 (12305 decode-written tokens reused) | 6 (follow-up's slot was taken: test cached 0, reference cached 422-1446) |
| A2 (default chunking) | `--max-batch 8` (2048 alone, 256 while others decode), reuse on, B=8 | 328 | 321 | 321 | 0 | 0 | 172 (54%) | 172 (12501 tokens) | 7 (5 slot taken, 1 first request ran alone at chunk 2048, 1 chain follow-on of that) |

Confirmation of reuse: the sum of `cached_tokens` over A1's usage records is 104355, equal to the server's `intarsia_batch_cached_tokens_total` (A2: 104594, equal); 173 of 329 trace `Admitted` events in each run have `cached > 0` (the 329th admit is the harness probe). Server decode passes in A1: 1985 with 8 rows, 680 with 7, 181 with 6 (peak client in-flight 8). By chain kind (A1, pass): ext 119, ext2 80, branch 85, exact 38. By step: P 150, F 147, F2 25 compared and passed.

* Almost every request in A2 ran at chunk 256 (327 of 328; the server picks 256 whenever others decode), so A2 repeats A1 under the default configuration; the 2048 chunk with reuse is not exercised at B=8.
* Negative control: three token ids flipped in a copy of A1's test file, `compare` reports exactly those 3 mismatches and exits 1 (`out/A/negative-control.txt`).
* Informational: for the 6 not comparable requests of A1 (5 with test cached 0, 1 with a shorter reuse), the token ids still differ from the reuse reference in 4 (different cached length, so a different KV history). That is what 4.5 items 2 and 4 predict, and why the reference must reproduce the predecessor instead of re-prefilling cold. Cause not isolated.
* Pilot (#2135, 8 chains, 18 requests): 18/18 pass in A1, 16/16 comparable in A2.

## 3. Gate B: `--spec ngram` with `--max-batch 8`

Server: `serve --max-batch 8 --spec ngram` (default `--spec-k 4`) with `INTARSIA_CB_NO_REUSE=1` (reuse is gate A), `INTARSIA_CB_TRACE=1`, `INTARSIA_SPEC_TRACE=1`. References: `--max-batch 1 --spec ngram` at `--prefill-chunk 2048` and `256`; each batched request is compared with the reference of the chunk it reports (`cbidentity.py compare --ref-alt`). Cases (`tools/cbspec_cases.py`, `tools/prompts/cb_spec_cases.jsonl`): 48 input-grounded copy tasks (20 raw text copy, 13 chat copy, 15 code copy; 121-365 prompt tokens), 48-288 tokens, arriving in 9 bursts of 6,3,8,5,2,7,4,6,7 every 14 s so the number of decoding sequences rises and falls repeatedly. Mostly greedy; 5 sampled, 4 stop-string, 4 `min_tokens` requests share the batch. Client bound B in {2,3,4,6,8}, streams mixed (27 streamed, 21 not).

### 3.1 Identity and drafting per run

Spec counters are `/metrics` deltas over the run. Solo references (`--max-batch 1 --spec ngram`, each chunk): 914 verify passes, 3656 proposed, 2899 accepted.

| B | Requests | Pass | Mismatch | Verify passes | Proposed | Accepted (rate) | Chunks 256/2048 |
|---|---|---|---|---|---|---|---|
| 2 | 48 | 48 | **0** | 914 | 3656 | 2899 (79%) | 47/1 |
| 3 | 48 | 48 | **0** | 914 | 3656 | 2899 (79%) | 47/1 |
| 4 | 48 | 48 | **0** | 406 | 1624 | 1281 (79%) | 40/8 |
| 6 | 48 | 48 | **0** | 430 | 1720 | 1289 (75%) | 39/9 |
| 8 | 48 | 48 | **0** | 457 | 1828 | 1346 (74%) | 39/9 |

At B=2 and B=3 (always fewer than 4 decoding) the batched server proposes and accepts exactly what the solo server does, to the token count. 240 of 240 requests identical, token ids compared in all.

### 3.2 Switch-off and return (`tools/cbspectrace.py` on the server log)

The scheduler trace hook prints a marker per decode phase (`iter`, `b` = decoding sequences, `spec_allowed`) before the sequences are prepared, so each `spec:` line is attributed to its phase.

| Run | Decode phases | Phases with `spec_allowed=false` | Verify passes in those phases | Gate value != (b < 4) | on to off | off to on | Drafting resumed after off to on |
|---|---|---|---|---|---|---|---|
| B=2 | 3192 | 0 | n/a | 0 | 0 | 0 | n/a |
| B=3 | 2182 | 0 | n/a | 0 | 0 | 0 | n/a |
| B=4 | 2333 | 898 (all b=4) | **0** | 0 | 23 | 23 | 22 |
| B=6 | 2129 | 662 (b=4..6) | **0** | 0 | 7 | 7 | 7 |
| B=8 | 2137 | 600 (b=4..8) | **0** | 0 | 7 | 7 | 7 |
| all | 11973 | 2160 | **0** | 0 | 37 | 37 | 36 |

Across the whole log 12484 draft tokens were proposed, all in phases with b < 4 (drafted while off: 0), in 3121 verify passes (equals the 914+914+406+430+457 from `/metrics`). The one off to on transition without drafting afterwards (B=4 run) was not examined further. Per b: drafting passes occurred at b=1 (383), 2 (1198), 3 (1540) and never at b=4..8 (2160 phases). Identity held in every run that crossed these transitions (B=4: 23 cycles, B=6/8: 7 each); sequences that were mid-generation when drafting switched off or on are inside these runs and all 144 requests of B=4/6/8 are identical.

## 4. Findings (reported, not fixed)

1. **`INTARSIA_CB_TRACE=1` printed nothing.** `SchedulerConfig::trace` only pushes events into `Scheduler::trace`; nothing drains it outside the tests (`take_trace`), contrary to the doc comment "Log scheduler events to stderr", and the vector grows without bound while the variable is set. Needed for gate B, so a hook was added (section 5).
2. **No flag for the B >= 4 speculation threshold.** `SchedulerConfig::spec_off_at` is 4 by default (`scheduler.rs:185`) and nothing in `intarsia serve` or `BatchedOptions` sets it; `serve --help` has `--spec`, `--spec-k`, `--max-batch` only. Gate B ran the hard-coded 4.
3. **`serve --help` is stale for `--max-batch`**: it says "Only 1 is implemented so far; larger values are rejected".
4. No bug found in the engine: no identity mismatch in 3248 earlier + 328 + 240 comparisons here.

## 5. Code change (test hook, `crates/` touched)

`crates/intarsia-core/src/scheduler.rs`, commits `0bd5db7` and `75ecb17`, +9 lines: when `SchedulerConfig::trace` is set (`INTARSIA_CB_TRACE=1`) each trace event is also printed to stderr (`cb-trace t=<epoch ms> <event>`), and a `DecodePhase { iter, b, spec_allowed }` line is printed before the decode rows are prepared. Env-gated, off by default, no change to behaviour or the `TraceEvent` enum. It is in the backend-generic scheduler, not Metal-only, but the scheduler is only reachable through `--max-batch > 1`, which is Metal-only. Check: `cargo test -p intarsia-core` 62 + 2 + 16 passed, 0 failed (`out/cargo-test-intarsia-core.txt`); the binary was then run for the gates. `engine/batch.rs` (the parallel `metal-cb-fix` branch) is untouched.

## 6. Not covered

* The default 2048 chunk with reuse at low concurrency (idle server) has no reference: chains were generated at chunk 256 only. A2 shows the default configuration at B=8, where it almost always uses 256.
* In gate A, 6 of 328 (A1) and 7 of 328 (A2) requests were not comparable (11 because the follow-up's slot had been taken by another client, 2 from one run-start chunk effect); they are listed in `out/A/report-*.json` (`not_comparable`) and are not claimed as passes.
* Gate B used `--spec-k 4` and one prompt family (copy tasks); sampled requests never speculate (server rule) and were only batch companions. `INTARSIA_SPEC_GATE` (per-request adaptive back-off) was left at its default.
* Chunk-size invariance (256 vs 2048 prefill) remains unproven, as before; both gates compare at the chunk the server reports.

## 7. Reproduce

```
cargo build --release -p intarsia-cli --features metal    # copy target/release/intarsia to the scratch bin dir, see serve_lib.sh
fleetq run --host endor --exclusive --mem 24 --label "cb gate A" -- docs/results/raw/2026-10-16-metal-cb-gates-reuse-spec/gateA.sh OUTDIR   # CHAINS=150
fleetq run --host endor --exclusive --mem 24 --label "cb gate B" -- docs/results/raw/2026-10-16-metal-cb-gates-reuse-spec/gateB.sh OUTDIR
python3 tools/cbspectrace.py <(gunzip -c OUTDIR/server-batched.log.gz) --window b4 START_MS END_MS   # windows in OUTDIR/windows.txt
```

Tools added: `tools/cbreuse.py` (gen, run, compare), `tools/cbspec_cases.py`, `tools/cbspectrace.py`. Raw: `out/A/{chains/ref.jsonl,chains/chains.jsonl,conc-*.jsonl,compare-*.log,report-*.json,metrics-*.txt,server-*.log.gz}`, `out/B/{ref*.jsonl,conc-b*.jsonl,compare.log,report.json,spectrace.{txt,json},metrics-*.txt,windows.txt,server-*.log*}`, `out/pilot/`.
