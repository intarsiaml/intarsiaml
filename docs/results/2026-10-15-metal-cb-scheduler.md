# Continuous batching: scheduler core (M1 step 2 + M2 policy), GPU-free

Date 2026-10-15. Branch `metal-cb-sched` off `spike/metal` 05c324d. Plan: `docs/research/continuous-batching-design.md` sections 2.1-2.5, 3.5, 6, 7 (M1, M2) and 8.1. The scheduler policy defaults follow the user's confirmation, which the coordinator relayed on 2026-10-15: `max_batch` defaults to 1 (opt-in) with an exact cap of 8, prefix reuse comes from idle slots only, admission is FIFO, and speculation turns off automatically at B >= 4 (the threshold is configurable). There are no priority or shortest-first modes.

**Scope.** This commit adds the scheduler and its tests only. It is not wired to `Model`, `engine::Seq`, `intarsia serve` or the CLI. No GPU was used and no fleet slot was needed.

New files:
* `crates/intarsia-core/src/scheduler.rs`
* `crates/intarsia-core/tests/scheduler.rs`

In `lib.rs` the only change is one `pub mod scheduler;` line. `engine.rs`, `model.rs`, `plan.rs`, `state.rs`, `admission.rs` and `intarsia-cli` are untouched. CUDA/HIP do not use the module.

## 1. Design as built

**Two traits keep the scheduler free of GPU and request types.**
* `BatchModel` owns `slots()` KV slots of uniform capacity.
* `BatchSeq<M>` is one request's state machine. It streams its own tokens and sends its own final `Done` / `Error`.

**`Scheduler<M, S>` is deterministic.** `step()` runs one iteration. `submit` queues FIFO, bounded by `max_queue`.

**`BatchEngine<M, S>` runs it on one thread that owns the model** (design 2.1).
* `submit` only sends on a channel.
* The thread blocks only when nothing is admitted or waiting.
* A panic or a fatal error fails every in-flight request.
* Dropping the engine cancels everything and joins the thread.

### Iteration (`Scheduler::step`)

```
reap:    queued requests with cancel set leave the queue (finish(None, cancelled), never get a slot);
         running ones finish(cancelled) and their slot goes idle (KV valid = the tokens it holds)
admit:   FIFO; while the queue head can get a free slot: pick the idle slot with the longest
         min(common_prefix, plen-1), ties -> least recently used -> lowest index (running slots never);
         seq.admit(cached, cap) (Err -> reject, Ok(false) -> finish at once), truncate(slot, cached)
decode:  B = #sequences in Decoding; spec_allowed = B < spec_off_at; each seq.prepare(..) -> Plain row | Ready;
         one model.decode_step(rows) (outer Err -> fatal for all; inner Err -> that request fails,
         slot invalidated); accept(out) per row; on_token per new token (admission order); finished -> slot idle
prefill: oldest Prefilling sequence: chunk size C fixed at its first chunk (busy 256 if any sequence is
         decoding now, else solo 2048); run one chunk (busy_chunks_per_iter under load); the final chunk
         carries first_mode(), its output -> accept + on_token, the sequence becomes Decoding
metrics: iterations, decode passes + batch-size histogram, queue depth (now/max), running, TTFT, ITL,
         step time + EWMA, prefill chunks/tokens, counts by outcome; optional trace log
```

**Decode-first.** Every decoding sequence gets exactly one row in each iteration until it ends. Prefill runs only after the decode step.

**No mixed passes.** Decode rows and prefill chunks never share a forward pass (design 2.4 and 4.5 item 3).

**Prefill fairness.** Prefill work always goes to the oldest prefilling sequence, so a later short prompt does not overtake a partly prefilled one.

**C = 2048 never stalls a stream.** A sequence that starts at C = 2048 began while nothing was decoding. It stays the oldest prefilling sequence until its prompt is done, and only finished prefills create decoders. So no stream can be decoding while it runs its 2048-token chunks.

**Failure isolation.**
* These fail only their own request:
  * an `Err` from a `BatchSeq` method (e.g. a codec error in `on_token`);
  * an inner row `Err`;
  * a non-fatal `prefill_chunk` `Err`.
* A failed request's slot is truncated to 0. Its KV may be one token ahead of what the sequence accounted for, so it is never offered for reuse.
* An outer `decode_step` `Err`, or any error that `BatchModel::is_fatal` classifies as fatal (default `Error::Backend`), fails every active and queued request with `Error`. After that, `submit` is refused.

**Engine failure handling.**
* The failure is published to `BatchEngine::failed()` before any request hears of it.
* `submit` holds the same lock across "check, then send".
* After a failure the thread drains the channel under that lock, so a request submitted around the failure always gets an `Error` and is never silently dropped.

### Speculation is a hook only

The scheduler passes `spec_allowed = B < spec_off_at` to `BatchSeq::prepare`. B is the number of decoding sequences that iteration; the default `spec_off_at` is 4, and 0 means never.

A sequence may run its verify pass on its own slot inside `prepare` and return `Ready`. When `spec_allowed` is false it must not draft. Verify rows are not part of the batched pass; folding them in is M3.

## 2. Trait contract (exactness)

```rust
pub struct Row<Mode> { pub slot: usize, pub token: u32, pub mode: Mode }

pub trait BatchModel {
    type Mode: Copy + Debug;   // real: HeadMode (LastArgmax / LastLogits)
    type Out;                  // real: StepOut
    fn slots(&self) -> usize;
    fn slot_cap(&self) -> usize;
    fn slot_tokens(&self, slot: usize) -> &[u32];          // tokens whose KV the slot holds, positions 0..len
    fn truncate(&mut self, slot: usize, len: usize) -> Result<()>;
    fn prefill_chunk(&mut self, slot: usize, tokens: &[u32], mode: Option<Self::Mode>) -> Result<Option<Self::Out>>;
    fn decode_step(&mut self, rows: &[Row<Self::Mode>]) -> Result<Vec<Result<Self::Out>>>;
    fn is_fatal(&self, e: &Error) -> bool { matches!(e, Error::Backend(_)) }
    fn on_admit(&mut self, _slot: usize, _id: RequestId, _cached: usize) {}   // trace hook, no effect
}

pub enum SeqStep<Mode> { Plain(u32, Mode), Ready }

pub trait BatchSeq<M: BatchModel>: Send {
    fn prompt(&self) -> &[u32];
    fn admit(&mut self, cached: usize, cap: usize) -> Result<bool>;   // Err reject, Ok(false) nothing to generate
    fn first_mode(&self) -> M::Mode;                                  // head mode of the final prefill chunk
    fn prepare(&mut self, model: &mut M, slot: usize, spec_allowed: bool) -> Result<SeqStep<M::Mode>>;
    fn accept(&mut self, out: M::Out) -> Result<()>;                  // output of its row / final chunk -> next token
    fn on_token(&mut self) -> Result<bool>;                           // true = done
    fn cancelled(&self) -> bool;
    fn finish(&mut self, model: &mut M, slot: Option<usize>, cancelled: bool) -> Result<()>;  // sends Done
    fn fail(&mut self, err: &Error);                                  // sends Error
}

pub fn chunk_schedule(cached: usize, prompt_len: usize, chunk: usize) -> Vec<(usize, usize)>;
pub struct SchedulerConfig { max_batch: 1, max_queue: 64, solo_chunk: 2048, busy_chunk: 256,
                             busy_chunks_per_iter: 1, spec_off_at: 4, trace: false }   // defaults
```

The model's half of the contract covers three things:
* **Decode rows.** Row `r`'s output is a bitwise function of slot `rows[r].slot`'s KV contents, `rows[r].token` and `rows[r].mode` only. It does not depend on other rows, the row count, the row order, or other slots' history.
* **Prefill chunks.** A chunk is exactly one pass of `tokens.len()` rows. Its output is a function of the slot's KV contents and the chunk's tokens.
* **Slot isolation.** Only a call that names a slot changes that slot.

`Model::forward_batch`, as documented on `metal-cb-m1`, promises the first point ("Row `r`'s result is bitwise the solo step's").

The scheduler's half:
* A sequence sees only its own outputs, never B or a row index.
* Its slot is truncated to `cached` at admission and used by no one else until it ends.
* Its chunk schedule is `chunk_schedule(cached, plen, C)`, with C fixed at its first chunk and never changed.

### Solo reference (precise)

**Chunk schedule.** For a request with prompt length `plen` admitted with `cached` reused tokens and chunk size `C`, the schedule is the same rule `engine::Seq::prefill_step` uses today:
* Start at `cached`.
* Emit chunks `(pos, C)` while `plen - pos > C`.
* Then emit one final chunk `(pos, plen - pos)` of 1..=C tokens. It carries the head and samples token 1.

**Choice of C.** `C = solo_chunk` (2048) if no sequence is in the Decoding phase at the moment the request's first chunk runs (after that iteration's decode step), otherwise `C = busy_chunk` (256). C is recorded as `TraceEvent::PrefillStart { chunk, decoding }`.

**The solo reference.** The reference for a batched request is the same request run alone, under three conditions:
1. Its slot is preloaded with exactly the KV the batched run reused: the same `cached` tokens, with the same values. That includes whether each row was prefill-written or decode-written, and with which chunking (design 4.5 items 2 and 4).
2. It runs with `--prefill-chunk C`, the C recorded above.
3. Nothing else is submitted.

Under that reference, its event stream (tokens, finish reason, completion and cached counts) and its chunk list must be equal. A cancelled request must be a prefix of it.

**Consequences.**
* With `max_batch = 1`, C is always 2048, which is today's behaviour and the existing golden baselines.
* A batched request prefilled under load needs a 256-schedule reference, which is what operator decision 8.1 Q2 calls for.

**Why C is fixed per sequence.** Switching C mid-prompt (for example from 256 back to 2048 when the decoders finish) would give load-dependent schedules that no single `--prefill-chunk` value reproduces. Fixed per sequence, there are exactly two possible schedules per `(cached, plen)`. The cost is that a request whose decoders all finish mid-prefill keeps 256-token chunks, which is about 7 percent slower prefill for the rest of its prompt.

## 3. Adapter the real Model / Seq must provide

This maps against `metal-cb-m1` at 05cd802 (the `Seq` and `model/batch.rs` API is the same as at 43b093c). The wiring happens after M1 step 1 merges.

### `SlotModel` (implements `BatchModel`)

It holds `&Model` (or `Arc<Model>`) and a `Vec<SequenceState>` of `max_batch` states, created by `Model::new_state` after `Model::enable_batch(max_batch)`. `Mode = HeadMode` and `Out = StepOut`.

| Trait method | Real call | Status |
|---|---|---|
| `slots`, `slot_cap` | `states.len()`, `states[0].cap` | ok |
| `slot_tokens(s)` | `&states[s].tokens` | ok (pub field) |
| `truncate(s, n)` | `states[s].truncate(n, &*model.backend)` | ok |
| `prefill_chunk(s, t, m)` | `model.forward(&mut states[s], t, m.unwrap_or(HeadMode::None), None)`; `StepOut::None` maps to `None` | ok. `forward` already runs exactly `t.len()` rows. |
| `decode_step(rows)` | `model.forward_batch(&mut refs, &ids, &modes)`, all outputs wrapped in `Ok` | Gap (small): `refs: Vec<&mut SequenceState>` needs split borrows of distinct slots, e.g. sort rows by slot, walk `states.iter_mut()`, then map the outputs back to row order. `forward_batch` has no per-row errors; every error is outer and therefore fatal, as in design 2.5. |
| `is_fatal` | default (`Error::Backend`) | Check which variant a token id >= vocab raises at upload (`model.rs` ~1121). It should be caught at admission anyway (next table). |

### `SeqAdapter` (implements `BatchSeq<SlotModel>`)

It wraps `engine::Seq` and lives inside `engine.rs` or a child module, because the `Seq` fields and `Outcome` are private.

| Trait method | Real `Seq` | Gap |
|---|---|---|
| construction | `Seq::new(sh, model, st, req, tx)` is private and validates, reuses the prefix and truncates `st` in one step. | **Split it.** `Seq::new(sh, req, tx)` builds the sampler, stop ids and decoder, with no state. A separate `admit(cached, cap) -> Result<bool>` does the plen checks, sets `max_new`, `cached` and `pos`, and returns `max_new > 0`. The scheduler computes `cached` itself (the same `common_prefix(..).min(plen-1)`). Add the token id < vocab check (design 2.5) to admission. `has_recurrent` / `restorable_len` drop out because the batch engine refuses recurrent models. |
| `prompt` | `req.prompt` | needs an accessor |
| `first_mode` | `head_mode(1)` | exists (`pub(crate)`) |
| prefill | `prefill_step(model, st)` picks `model.prefill_chunk()` itself. | The scheduler runs the chunks, so the adapter does not call `prefill_step`. The final chunk needs `accept_first(out)` = `token_from(out, 1)` plus the `prefill` / `t_dec` timestamps that `prefill_step` sets today. The cancel check between chunks becomes the scheduler's reap. |
| `prepare` | `prepare(model, st) -> Option<(u32, HeadMode)>` (None means `Ready`) | Exists, except that `spec_on` is fixed in `Seq::new`. `plan_step` needs a per-call `spec_allowed` (skip drafting when false). The verify pass runs `model.forward_verify` on that slot's state alone, as now. |
| `accept` | `accept_plain(out)` | exists |
| `on_token` | `on_token()` | exists |
| `cancelled` | `cancelled()` | private; make it `pub(crate)` |
| `finish` | `finish(self, model, st) -> Outcome` consumes the Seq and leaves sending `Done` and telemetry to `run_job`. | Two things are needed. First, a `cancel()` that sets `reason = Cancelled` for the scheduler's reap: during prefill it should report `cancel_outcome()`, during decode `finish` with Cancelled. Second, a shared `Outcome -> GenEvent::Done` plus `telemetry.request_finished` helper extracted from `run_job`, so the solo and batched paths cannot drift apart. `slot == None` (cancelled while queued) sends `Done{Cancelled, 0 tokens}`. |
| `fail` | `run_job`'s error arm (`GenEvent::Error` plus `request_finished("error")`) | extract into a helper |

### Engine, server and CLI glue (not here)

* `impl Engine for BatchEngine`:
  * `submit` builds the `SeqAdapter` from `GenRequest` and `tx`.
  * `tokenize`, `detokenize` and `apply_chat_template` go through the codec.
  * `telemetry.step()` is called once per batched pass. `request_started` is called at submit and `set_kv` per iteration.
  * The scheduler's `BatchMetrics` (batch-size histogram, queue depth, TTFT, ITL) is exported on `/metrics`.
  * `forward_logits_*` / `forward_script` need a dedicated tool slot between iterations (design 2.5). This is not designed here.
* `SERVE_MAX_BATCH = 1` (`engine.rs` on m1) must become 8 (`MAX_BATCH_CAP` here, `Model::BATCH_MAX_ROWS` there).
* `--max-batch > 1` selects `BatchEngine`. `max_queue` comes from the server admission limit (`max_batch + max_queue`, design 2.1).
* `trace_logits` requests run alone (design 2.3). The adapter can reject them in `admit` when `max_batch > 1`, or the engine can route them to `LocalEngine`. Not decided.

## 4. Tests (`cargo test -p intarsia-core --test scheduler`, 15 tests, all pass)

### The mock model

The mock's output at a position is a rolling SplitMix hash over the slot's KV entries `0..=pos`. Each entry hashes its token, its position and its writer: the prefill chunk `(start, len)` that wrote it, or "decode".

A wrong truncate, a wrong slot, a stale reused prefix, a different chunking, or decode-written rows replaced by prefill-written ones all change the output. Beyond that, the output depends on nothing else.

The mock also:
* asserts distinct slots per pass and at most `max_batch` rows;
* records the chunks run for each request, each request's reused KV at admission (a value copy) and the batch sizes;
* injects per-row errors, non-fatal prefill errors and fatal pass errors.

### The mock sequence

* Greedy or seeded-sampled tokens (one RNG draw per token), checked against the head mode.
* `min_tokens` masks the stop ids.
* EOS / `ignore_eos`, stop sequences armed after `min_tokens`, and `max_tokens` bounded by `cap - plen`.
* A cancel flag, plus detection of a dropped receiver.
* An injected error at token k.

### Test list

| Test | What it checks |
|---|---|
| `batched_equals_solo_randomized` | B in {1, 2, 3, 4, 6, 8} x 6 seeds x 24 requests (864 requests). Staggered arrivals over 120 iterations; prompts 1-3500 tokens (40 % share a prefix family, 10 % > 2048); random `max_tokens`, `min_tokens`, stop sequence, sampled / greedy, `ignore_eos`; cancel at random iterations (queued, prefilling or decoding); client disconnects; empty and too-long prompts (rejected); injected sequence errors, row errors and prefill errors. Every request equals its solo reference (cancelled ones as a prefix), and every recorded chunk list equals an independent statement of the schedule rule. Trace invariants: FIFO admission and prefill start, contiguous decode rows, prefill tokens per iteration <= budget whenever a stream decodes before and after it, only the final chunk short, spec gate = `B < 4`, the histogram matching the model's recorded pass sizes, TTFT + ITL counts = `on_token` calls. Each B reaches B rows. Coverage is asserted: in the last run, 696 exact comparisons, 88 prefix (cancelled), 80 with a reused prefix, 419 with the 256 schedule, 432 multi-chunk, 77 ending in an error, 270 sampled, 175 with `min_tokens`, 332 stopped. |
| `batched_equals_solo_other_budget` | B in {2, 4, 8} with `busy_chunk 64`, `busy_chunks_per_iter 2`, `spec_off_at 2`: same checks. |
| `leaky_model_is_detected` | Negative control: a mock whose rows depend on B is caught by the identity check. |
| `chunk_size_changes_output` | Negative control: the same request at C = 2048 and C = 256 gives different tokens, so the identity check is sensitive to the schedule. |
| `chunk_schedule_rule` | Unit test of `chunk_schedule` against the independent rule. |
| `config_limits` | `max_batch` 0 and 9 are rejected; the defaults are 1 / 2048 / 256 / 1 / 4. |
| `max_batch_one_uses_solo_chunk` | With B = 1 every prefill uses 2048, including prompts of 1.5k-4.3k tokens. |
| `busy_chunk_is_sticky` | A request arriving while another decodes gets 256 and keeps it after the decoder ends mid-prefill. |
| `prefix_reuse_from_idle_slots` | Longest prefix wins, a running slot is never taken (the same prompt goes to the LRU slot instead), LRU is used when there is no prefix. |
| `cancel_in_every_phase` | Cancel while queued (no slot taken), between prefill chunks (the slot holds exactly `prompt[..512]`) and while decoding; the other request is unaffected. |
| `failure_isolation_and_fatal` | Row, sequence and prefill errors fail only their request and invalidate its slot while the neighbour finishes its 20 tokens. A fatal pass error fails all five requests, and a later submit is refused with an `Error` event. |
| `queue_overflow` | `max_queue 3`: three submits are refused with an `Error` event; queue depth max is 3; the rest complete. |
| `spec_gate_follows_batch_size` | Thresholds 4, 2 and 0: the gate equals `B < threshold`, and with 4 both values are exercised. |
| `engine_concurrent_submit_and_cancel` | `BatchEngine` thread with B = 4. Four submitter threads, 40 requests in total, cancels from separate threads, receivers dropped mid-stream, `recv_timeout` against hangs. Every request equals its solo reference, with C taken from the model's recorded first chunk. Dropping the engine with work in flight ends that request with `Done`. |
| `engine_queue_bound_and_fatal` | Engine-level queue bound. A fatal error reaches every accepted request; `failed()` is set and later submits are refused. |

### Mutation checks

Each mutation was applied to `scheduler.rs`, the tests were run, and the file was restored from a copy. Every mutation fails tests:

| Mutation | Tests failing (of 15) |
|---|---|
| C recomputed every chunk (not fixed per sequence) | 6 |
| Row outputs handed to the wrong sequences | 4 |
| Queued cancellations not reaped | 1 |
| Busy slots offered for reuse | 10 |
| Failed slot not invalidated | 1 |
| Busy chunk ignored | 5 |

### Commands and results

* `cargo test -p intarsia-core`: pass. 62 unit, 2 gdn_hf, 15 scheduler, 1 doc test ignored.
* `cargo check --workspace --all-targets`: pass on macOS.
* Engine tests: 150 consecutive runs and the whole file 40 times, with no failure.
  * **Flaky test, fixed.** An earlier version failed about 1 run in 40. `failed()` was published after the requests had already received their `Error` events. The failure is now published first.

## 5. Open points

1. **Real-model identity harness.** Proving identity on Metal needs two things:
   * The serving path must expose each request's `cached` and `C`, e.g. as debug fields in `usage` next to `cached_tokens`, or via an `INTARSIA_CB_TRACE` log.
   * `cbidentity` must reproduce the reused prefix solo. An idle slot's prefix may have been decode-written by an earlier request, so the reference has to replay that predecessor solo first (the `accuracy --mode reuse` procedure), not cold-prefill the prefix.

   Golden baselines for the 256 schedule are still owed at M2 (design 8.1 Q2).
2. **Chunk-size identity on Metal is unproven** (design 4.5 item 1). The policy only guarantees that the batched request equals a solo run at the same C. Whether a 256-chunked prefill equals a 2048-chunked one bitwise is a separate question. It does not affect this exactness claim, but it does affect whether old 2048 golden outputs apply to requests prefilled under load. Nothing assumes they do.
3. **No EWMA-adaptive budget.** The step-time EWMA is measured and exported, but it does not steer the chunk size. An adaptive C would still be exact if fixed per sequence and reported, but the reference would then need arbitrary `--prefill-chunk` values. The fixed 256 is the operator's decision.
4. **Speculation.** Auto-off at B >= 4 preserves exactness only if verify is bitwise lossless (the G1 claim, single slot). The randomized identity tests run with the mock never speculating; `prepare` is only checked for receiving the gate. Real verify inside `prepare` serializes a verify pass before the batched step, which costs one extra weight pass per speculating sequence per iteration at B = 2-3. Splitting spare rows by acceptance EWMA is M3.
5. **Prefill completion.** One request's prefill completes per iteration at most, and under load only `busy_chunks_per_iter` chunks run. With nothing decoding, a C = 2048 prefill also runs one chunk per iteration, so the reap check happens between chunks as `prefill_step` does today. The per-iteration host overhead is assumed negligible against a 0.5 s chunk; this is unmeasured.
6. **Fatal versus per-request prefill errors.** This rests on `is_fatal` (default `Error::Backend`). Which variants the Metal backend raises for a device fault versus a bad input should be checked when wiring.
7. **Shutdown timing.** `BatchEngine::drop` cancels in-flight work (each request gets `Done{Cancelled}`). I have not compared this with `LocalEngine`'s drop behaviour; align the two when wiring serve.
8. **Queue bound counts waiting requests only.** The server limit becomes `max_batch + max_queue` (design 2.1).
