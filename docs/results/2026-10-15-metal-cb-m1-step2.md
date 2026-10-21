# Metal continuous batching, M1 step 2: scheduler wired to the model (`intarsia serve --max-batch 2..8`)

> **Update (2026-10-21, docs-accuracy pass, see `docs/decisions/LOG.md`):** C16: the "chunk-size identity is unproven" limitation is closed by `2026-10-15-metal-chunk-golden.md` (bitwise identical at 256 and 2048 on 33 prompts); an absent `--max-batch` means 1; a scheduler queue-full refusal is 503, the server's own admission 429.

Date 2026-10-15. Branch `metal-cb-wire` (= `metal-cb-m1` b2818bd + `spike/metal` d4e6c32, then `spike/metal` 28e5c5a and 32a42d8 merged,
so chunk-no1, forward-no1 and test-gpu-gate are in). Machine endor (Apple M5 Max), Qwen3-8B BF16, `--max-seq 8192`.
Plan: `docs/research/continuous-batching-design.md` section 2, 7 (M1), 8.1. Scheduler core: `2026-10-15-metal-cb-scheduler.md`.
Model side: `2026-10-15-metal-cb-m1-step1.md`.

**Verdict.** `intarsia serve --max-batch N` (2..8, Metal only) now serves concurrent requests in one batched decode, and every
request is byte-identical (token ids, text, deltas, finish reason, counts) to its solo run: **0 mismatches in 16 runs x 203 = 3248
request comparisons** (server `--max-batch 8` and `4`, client in-flight B = 2/4/8, streaming and non-streaming, dense and
sparse arrivals, prefix reuse off and on), 20/20 cancellations really cut short in every run. `--max-batch 1` (default) is the
unchanged `LocalEngine` path: the M0 solo identity subset is identical to the 8254311 binary (33/33, spec off and `--spec ngram`).
Throughput gate at the requested `--gen 256`: aggregate **1.88x / 3.20x / 4.37x** at N = 2 / 4 / 8 against 1.6 / 3.0 / 4.5:
**N = 2 and 4 met, the N = 8 aggregate gate is MISSED (4.37x, 3% short)**; decode-window aggregate at N = 8 is 187 tok/s (gate >= 175, met).
Supplementary, with the design doc's own `--gen 512` wording: 1.92x / 3.39x / 4.77x (all three met).

## 1. What was built

| Commit | Content |
|---|---|
| `825b46f` | The wiring (below), CPU tiny-model identity tests `crates/intarsia-models/tests/batch_engine.rs`. |
| `efca589` | Metal-only check before the weights load. |
| `1bdcbca` | `avoid_single_row_prefill` in the adapter (after the chunk-no1 merge `a0f66b0`); `1a38892` fixes the stale `SERVE_MAX_BATCH == 1` test assertion. |
| `9d635a6` | `tools/cbidentity.py --ref-alt` (reference per reported chunk). |
| `f1307cf`, `7283398` | `cancel_after_ms` fix + selftest cases; the "cut short" column (section 5). |
| `1aa7c97`, `2d36908`, `f7eda13`, `7f7ae1d`, `0d9f64a` | Gate scripts (zsh word-splitting fix, REFDIR, GEN, interleaved gate 4 with macmon / ps). |
| `64df17f`, `9fcd29f` | Raw gate outputs; this document. |
| merges `a0f66b0` (chunk-no1 71b40b1), `9487bc5` (28e5c5a), `9ebd6f8` (32a42d8) | `spike/metal` merged three times, no conflicts. The final gates ran on binary `9ebd6f8`; later commits change only scripts, tools and docs. |

### How the adapter maps onto `Seq` / `Model` (`crates/intarsia-core/src/engine/batch.rs`, new; `engine.rs` refactor)

`engine::Seq` is the same per-request state machine `LocalEngine` drives (sampling, `min_tokens`, stop ids / strings, EOS,
streaming, speculation). What changed in `engine.rs` (all behaviour-preserving for the solo path, checked by gate 2):

* `Seq::new(sh, model, st, req, tx)` is now `check_prompt` + prefix reuse + `Seq::build(env, req, tx, verify_rows, recurrent)`
  (construction, no KV state) + `Seq::admit(cached, cap)` (prompt vs capacity, `max_new`, `cached`/`pos`; `Ok(false)` and
  `reason = Length` when nothing can be generated). The empty/too-long validation still runs before `st` is truncated. A
  `SeqEnv` (codec, telemetry, EOS ids, spec config) replaces the borrow of `Shared`.
* `Seq::cancel()` (sets the reason), `Seq::set_spec_allowed(bool)` (`spec_paused`, consulted in `plan_step`, default allowed),
  `Seq::accept_first(out)` (`token_from(out, 1)` + prefill/decode timestamps), `Seq::queued_outcome`.
* `report_done` / `report_error` extracted from `run_job`; the solo worker and the batched engine call the same two functions
  (Done event + telemetry, Error event + telemetry), so the two paths cannot drift. `build_info` is likewise shared by both engines.
* `GenEvent::Done` gets `prefill_chunk: Option<usize>` (the chunk size the prompt ran with; `LocalEngine` reports the model's).
  `intarsia serve` returns it only when the request carries `return_token_ids`, as `usage.prompt_tokens_details.prefill_chunk`.
  The scheduler records C per request (`BatchSeq::on_prefill_start(chunk, decoding)`, new default method, called at the first chunk).

`SlotModel` (`BatchModel`): owns the `Model` and `max_batch` `SequenceState`s (`new_state()` after `enable_batch`).

| `BatchModel` | Real call |
|---|---|
| `slots`, `slot_cap`, `slot_tokens` | `states.len()`, `states[0].cap`, `states[s].tokens` |
| `truncate(s, n)` | `states[s].truncate(n, backend)` |
| `prefill_chunk(s, toks, mode)` | `model.forward(&mut states[s], toks, mode.unwrap_or(HeadMode::None), None)`; `StepOut::None` -> `None` |
| `decode_step(rows)` | `model.forward_batch(refs, ids, modes)` with one `&mut SequenceState` per row taken out of the slot vector (split borrows via `Option::take`, duplicate/out-of-range slot is an error). An error here is the scheduler's fatal outer error. |
| `avoid_single_row_prefill` | `backend.avoid_single_row_prefill() && !states[0].has_recurrent()` (chunk-no1 contract) |

`SeqAdapter` (`BatchSeq<SlotModel>`) wraps `Option<Seq>`: `admit` = vocab check + `Seq::admit` + `request_started`; `first_mode` =
`head_mode(1)`; `prepare` = `set_spec_allowed(spec_allowed)` + `Seq::prepare(model, &mut states[slot])` (a speculative verify
pass runs on the sequence's own slot inside `prepare`, as designed); `accept` = `accept_first` while no token exists, else
`accept_plain`; `finish` = `Seq::cancel` if cancelled, `Seq::finish(model, state)` (or `queued_outcome` when it never got a slot),
`set_kv`, `report_done`; `fail` = `report_error` (started) or `request_rejected` + `Error` event (never started), so the
enqueued/started/finished telemetry counters balance on every path (queue full, admit reject, queued cancel, fatal;
`/health` shows `running 0, waiting 0` after every gate run).

`BatchedEngine` (`Engine` impl over `scheduler::BatchEngine<SlotModel, SeqAdapter>`): `solo_chunk = model.prefill_chunk()` (2048 on
Metal), `busy_chunk = min(256, solo_chunk)`; `submit` = `Seq::build` + `request_enqueued` + `BatchEngine::submit`.
`forward_logits_each`, `forward_logits_decode`, `forward_script` (accuracy tools) return `Unsupported("... needs the
single-sequence engine (--max-batch 1)")`. `trace_logits` requests are not run alone as design 2.3 suggested: a logits row of a
batched step is bitwise the solo one, so they batch (decision for the operator, nothing relies on it).

### Loading, server, CLI

* `load_engine_tp`: `--max-batch` absent means 1 (the `resolve_max_batch` default is bypassed by passing `supported = 1`, because
  `SERVE_MAX_BATCH` is now 8 and the qwen3 recipe now declares `max_batch = 8`; qwen3_5 recipes declaring 32 are therefore
  not silently batched). `--max-batch > 1` on a backend other than Metal fails before any weight is read ("needs the Metal
  backend (row-exact multi-row kernels ...)"; checked on `--backend cpu`), then again on `!batch_capable()`; `enable_batch`
  still refuses tensor-parallel / non-dense models. `Loaded.engine` is `Arc<dyn Engine>`, `Loaded.max_batch` new.
* Queue full: the scheduler returns `Error::Other("queue full ...")` (was `BadRequest`, which the server maps to 400), so a
  refusal that gets past the server's own limit maps to 503. The server's own admission is `max_queue + max_running`
  (`ServerConfig.max_running = --max-batch`, default 1 = unchanged) and answers 429 as before.
* `/metrics` appends `intarsia_batch_*` (decode passes by row count, queue depth, prefill chunks/tokens, EWMA step) when the
  engine batches; `/v1/intarsia/plan` has `max_batch`.
* Knobs for the harness: `SchedulerConfig.prefix_reuse` (`INTARSIA_CB_NO_REUSE=1` turns idle-slot prefix reuse off),
  `INTARSIA_CB_TRACE=1` (scheduler trace).
* CUDA / HIP: with `--max-batch 1` the serve path is the same `LocalEngine` code as before (only the shared helper extraction
  and the `Done.prefill_chunk` field changed); `--max-batch > 1` is refused with the message above.

## 2. Gate 1: build and tests

| Command | Result |
|---|---|
| `cargo check --workspace --all-targets` | pass |
| `cargo test -p intarsia-core` | pass (62 unit, scheduler suite 16 incl. randomized identity) |
| `cargo test -p intarsia-models --test batch_engine` (new, CPU) | 5 pass: batched B = 2/4/8 equals `LocalEngine` solo over 30 requests (greedy, seeded sampling, `min_tokens`, stop strings, EOS, 3..110-token prompts, chunk 16), same with speculation at B = 2 and 4 (accepted drafts asserted), cancel + out-of-vocab + too-long beside live requests, bounded queue refuses and unwinds telemetry, accuracy endpoints refused. Mutation check: reversing the token order of the batched step fails it. |
| `cargo test --workspace --release --no-fail-fast` (endor slot #2114, `INTARSIA_TEST_REAL_MODEL=1`) | 422 passed, 0 failed, 1 ignored |
| `cargo test -p intarsia-backend-metal --release` (same slot) | 102 passed, 0 failed |

(An earlier run on the pre-merge tree, #2089, found `max_batch_resolution` asserting `SERVE_MAX_BATCH == 1`; fixed, and the
workspace run now uses `--no-fail-fast` so a failing test binary cannot hide later ones.)

## 3. Gate 2: solo identity unchanged (`--max-batch 1`, default)

M0 subset (11 cases: short, 4k, grounded, stop strings, `min_tokens`, EOS, three sampled; `docs/results/raw/2026-10-15-metal-cb-m1-step2/cases-sub.txt`),
`intarsia generate --show-ids`, new binary without `--max-batch` vs the 8254311 binary, spec off (`off` vs `base`) and `--spec ngram`
(`on4` vs `bon4`): stdout and token ids identical in all **33/33** (`id-table-final.md`; pre-merge tree: also 33/33,
`id-premerge-table.md`). Slot #2114 (merged tree); #2089 and #2095 are the pre-merge / first-merge runs.

## 4. Gate 3: concurrent identity on endor (`tools/cbidentity.py`, `--token-ids on`)

**Method.** 203-case set (`tools/prompts/cb_identity_cases.jsonl`: raw/chat, greedy/sampled, EOS, `min_tokens`, stop strings,
`ignore_eos`, 20 cancellations, prompts 127 to 4500 tokens, three 4k golden prompts), unique nonce per prompt.

* References are **solo servers** (`--max-batch 1`, the `LocalEngine`) started with `--prefill-chunk 2048` and `--prefill-chunk 256`,
  every case run alone after a flush request (cold prefill, `cached = 0`).
* The batched server: `--max-batch 8` (and 4), `INTARSIA_CB_NO_REUSE=1` (idle-slot prefix reuse off: the harness cannot
  reproduce a reused prefix in a solo reference; the chat template's 17-token header would otherwise give `cached = 17`).
* Each batched request reports the prefill chunk C the scheduler chose for it (`usage.prompt_tokens_details.prefill_chunk`);
  `compare --ref ref2048 --ref-alt ref256` compares it with the reference whose chunk equals that C (new `--ref-alt`).
  Under load nearly all requests got C = 256, a few (the first requests, with nothing decoding) C = 2048 (counts below), so both
  references are exercised.
* Client in-flight bound B in {2, 4, 8}, stream on (203 streamed) and stream off (183 non-streamed + 20 cancellations,
  which are always streamed), `--arrival-scale 0.25`. Slot #2115 (references and main runs), #2116 (extras).

| Run | Server | B | requests / run | Mismatches | C = 2048 / 256 | Cancels cut short / cases |
|---|---|---|---|---|---|---|
| main, stream on x off | `--max-batch 8`, reuse off | 2, 4, 8 | 6 x 203 | **0** | 6-7 / 176-177 (B=2), 2 / 181 (B=4, 8) | 20/20 each |
| `--max-batch 4` | `--max-batch 4`, reuse off | 4, 8 (8 queues) | 4 x 203 | **0** | 2 / 181 | 20/20 each |
| sparse arrivals (`--arrival-scale 3`) | `--max-batch 8`, reuse off | 2, 4 | 4 x 203 | **0** | 10-12 / 171-173 | 20/20 each |
| prefix reuse on (informational) | `--max-batch 8`, reuse on | 8 | 2 x 203 | **0** | 2 / 181 | 20/20 each |

Token ids were compared for 201 of 203 requests per run (197 in the `--max-batch 4` B = 8 runs; the others are cancellation cases that emitted nothing before
the cancel; for the cancelled ones the emitted ids are checked to be a prefix of the reference). Server side, from `/metrics`
after the main runs: 120 requests finished `cancelled` (6 runs x 20), `intarsia_batch_decode_rows_total` = 2659 / 9434 / 1746 /
3602 / 496 / 644 / 781 / 765 passes of 1 / 2 / .. / 8 rows (real batches of every size up to 8), `queue_depth_max` 6, `/health`
`running 0, waiting 0` at the end; the `--max-batch 4` server ran 301 / 1235 / 3260 / 6648 passes of 1 / 2 / 3 / 4 rows. With reuse on, 26-27 of 183 requests reused a prefix (`cached_tokens` 17, the chat header),
`cached!=` against the cold solo reference, and still matched: no mismatch to explain. Not claimed: that 256-chunk and
2048-chunk prefill agree with each other (design 4.5 item 1, still open); the comparison is always at the same C.

Not covered: prefix reuse of decode-written KV against a reproduced-predecessor solo reference (the harness cannot do it, see
scheduler doc open point 1); the reuse-on run shows only the 17-token chat header case.

## 5. Harness fixes made on the way (`tools/cbidentity.py`)

* `--ref-alt FILE...` (compare) and `--ref-alt-url` / `--ref-alt-file` (gate): reference chosen per request by its reported chunk.
* **`cancel_after_ms` bug** (reported by the coordinator): the harness sends `Connection: close` and `intarsia serve` (like the
  mock) replies `Connection: close`; `http.client` then drops `conn.sock`, so `arm()`'s `settimeout` and the final `shutdown` did
  nothing and a time-based cancel never fired mid-stream (`cancel_after_chunks` worked because the socket was garbage collected).
  Fixed by capturing the raw socket before `getresponse()`. Selftest cases added: a slow `Connection: close` mock must be cut at the
  deadline (status `cancelled`, fewer deltas than the stream, under 1 s) and must see the disconnect; verified to FAIL with the
  old behaviour emulated. The `inflight-id` selftest now accepts the `cancelled token ids ...` reason too (cancel cases now really cancel).
* New column `cut short`: cancelled requests whose emitted tokens (ids) are fewer than the reference's, i.e. cancels that cut the
  generation. A request that "cancels" but runs to the end counts as not tested. The gate runs above use the fixed harness
  (20/20); the preliminary runs on 1bdcbca (`out/prelim-1bdcbca/`) predate the fix (16/20 cut short there, 0 mismatches).

## 6. Gate 4: throughput (localeval 0.3.0, `feat/decode-aggregate` 603ee54, PR #4 still open)

`localeval speed --sizes 1024 --gen G --reps 3 --concurrency 2,4,8`, server `--profile-every 0`, ctx 986 prompt tokens.
**Interleaved in one slot (#2117)**: server `--max-batch 1` and `--max-batch 8` alternately (`1 8 1 8 1 8` for G = 256, `1 8 1` for
G = 512), each pass a fresh server; `macmon` logs the GPU once a second; `ps` at start and end; per the coordinator's note
about a baseline that drifted 1.8x in another soak.

| `--gen 256` | N=1 decode tok/s | N=2 | N=4 | N=8 |
|---|---|---|---|---|
| `--max-batch 1` aggregate (3 passes) | 35.1 (35.0 / 35.2 / 35.2) | 34.1 (34.3 / 33.7 / 34.4) | 34.0 (34.0 / 33.9 / 34.2) | 33.7 (33.7 / 33.2 / 34.2) |
| `--max-batch 8` aggregate (3 passes) | 35.0 | 64.1 (63.7 / 63.9 / 64.6) | 109.0 (107.3 / 110.4 / 109.3) | 147.2 (144.7 / 148.6 / 148.3) |
| aggregate vs `--max-batch 1` | 1.00x | **1.88x** (gate 1.6) | **3.20x** (gate 3.0) | **4.37x** (gate 4.5: **miss**) |
| `--max-batch 8` overlap decode tok/s | - | 69.4 | 128.1 | 187.0 (gate >= 175: met) |
| `--max-batch 8` ITL p50 / p95 ms | 28 | 28.7 / 30.0 | 30.9 / 33.5 | 42.6 / 104.8 |

| `--gen 512` (design wording) | N=2 | N=4 | N=8 |
|---|---|---|---|
| `--max-batch 1` aggregate (2 passes) | 34.5 | 34.6 | 34.3 |
| `--max-batch 8` aggregate (1 pass) | 66.3 | 117.3 | 163.7 |
| ratio | **1.92x** | **3.39x** | **4.77x** (met) |
| overlap decode tok/s / ITL p50 / p95 ms | 69.1 / 28.8 / 30.0 | 128.4 / 31.1 / 32.3 | 186.7 / 42.5 / 45.5 |

* localeval's `aggregate_decode_tok_s` (result.json) is the same number as the "aggregate ... (tokens/wall)" printed in the log; the
  tables use it, and `overlap_decode_tok_s` is the decode-window figure.
* N = 1 is unchanged: `--max-batch 8` single-stream decode 35.0 vs 35.1 tok/s (0.3%, gate 2%). The `--max-batch 1` path is the
  untouched `LocalEngine`.
* Baseline drift: the three `--max-batch 1` passes agree within 2% (single-stream decode 35.0-35.2; aggregates within 3.6%),
  below the 5% bar, so the ratios stand. GPU clock while busy: median 1340 MHz (1084-1617), first third 1340 MHz, last third
  1346 MHz; GPU temperature median 73 C (max 86 / 92 C). `ps` at start and end: only the standing `kev-serve` (idle, 0% CPU) plus
  other sessions' idle shells; raw files in `out/final/gate4-*/` (`macmon.jsonl`, `ps-*.txt`).
* The N = 8 miss at `--gen 256`: aggregate is tokens / wall time including the 8 one-thousand-token prefills (about 2.8 s of a
  13.5 s run, each 256-token chunk stalls the decode step, ITL p95 105 ms); the decode window itself runs at 187 tok/s (5.3x
  one stream). With `--gen 512` the same prefill cost is diluted and the aggregate gate is met (4.77x). The task asked for `--gen 256`,
  so this is a missed gate; the `--gen 512` row is supplementary. Overlapping prefill with decode (mixed passes) is out of scope (design 4.5 item 3).
* localeval run ids (results under `/opt/code/deepeval-decode-agg/results/`, unanimis ids in the logs): gen 256 passes
  `20261015-214412 / 214718 / 214855 / 215202 / 215338 / 215643` (`-adhoc-cbwire-pN-mbM-gen256`), gen 512 passes
  `20261015-215819 / 220326 / 220552` (`-gen512`).

## 7. CUDA / HIP

`cargo check -p intarsia-cli --features cuda --tests` and `--features hip --tests` on spark1 in a scratch dir (build only, no GPU
use, no `--takes-down`): fleetq #2090 on the first wiring and **#2131 on the final tree (9ebd6f8 content: load.rs Metal-only check,
chunk-no1 / forward-no1 merges): `--features cuda --tests` EXIT=0 and `--features hip --tests` EXIT=0**, scratch dir removed.
Raw: `out/final/spark.log`, `spark1-check.log` (first run), `spark_check.sh`.

## 8. fleetq jobs

Valid results: #2089 (gates 1+2 pre-merge, found the stale `SERVE_MAX_BATCH` assertion), #2090 (spark1 cuda/hip), #2095 (gates
1+2, first merge), #2102 (gate 3 on 1bdcbca, harness without the cancel fix), #2103 (gate 4, no interleave), **#2114 (final gates
1+2), #2115 (final gate 3), #2116 (extras), #2117 (final gate 4), #2131 (final spark1 check)**. #2096 and #2097 ran with a zsh
word-splitting bug in my scripts (exit 127 / "invalid value '1 8'"), nothing measured. **#2108 and #2109 were released while queued;
fleetq's runner kept waiting and ran them later outside the lock (fleetq bug, found by the coordinator, killed, outputs
discarded: `scratchpad/step2r/gate3x*`, `gate4-gen512*` are not used anywhere).** Timings taken by other slots between 19:50 and
20:12 are suspect for the same reason; the final gates ran later.

## 9. Not done / open

* Prefix reuse is on by default in the batched engine, but the identity evidence with reuse is only the chat-header case
  (cached 17); a reproduced-predecessor reference is not implemented.
* Chunk-size identity (256 vs 2048 prefill produce the same bits) is unproven; identity is shown at equal C, with C reported per request.
* `forward_logits*` / `forward_script` unsupported with `--max-batch > 1` (accuracy tooling must use `--max-batch 1`).
* No adaptive chunk budget, no mixed prefill+decode passes (M2). `trace_logits` requests batch instead of running alone.
* `--spec ngram` together with `--max-batch > 1` ran only on the CPU fixture (`batch_engine.rs`, accepted drafts asserted at B = 2
  and 4); on Metal the speculation gate (auto-off at B >= 4) and a verify pass inside a batched iteration are untested.
* Commits without the `Co-Authored-By: Claude Opus 5.5` trailer (history not rewritten, hashes are cited by binaries, fleetq labels
  and this document): `efca589`, `1bdcbca`, `1a38892`, `1aa7c97`, `1eb3125`, `2d36908`, `f7eda13`, `7f7ae1d`, `0d9f64a`, `64df17f` and
  the three merge commits `a0f66b0`, `9487bc5`, `9ebd6f8`.
* The Qwen3-8B Metal test for `BatchedEngine` is the harness run, not a `cargo test` (so nothing new to gate behind
  `INTARSIA_TEST_REAL_MODEL`); the CPU test uses the tiny fixture, where `batch_capable()` is false and the cpu backend
  serves rows one at a time.
* localeval PR jkuepker/localeval#4 is still open; the feature branch was used.
