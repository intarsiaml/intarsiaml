# batch-store: the prefix store in the batched scheduler, slice 1

Lane `batch-store`, branch `batch-store` on `origin/store-settle-off-decode` (PR #352: settle after `Done`, pre-faulted host
buffers). Design: Item 2 of the ab35c plan (cutover blocker 2 of `docs/results/2026-11-12-ab35b-cutover-followup.md`):
production serves 2 streams (`--max-batch 2`) and the store was refused there, so multi-turn TTFT on the batched path was
2.7x vLLM's (5.6 s vs 1.4 s at 33k tokens, no reuse at all).

## What was built

`serve --prefix-store ... --max-batch N` (N > 1) now runs the store in the batched engine for plain decode. The plan's file:line
references were checked against this base; the serve refusal it could not locate was `crates/intarsia-models/src/load.rs`
(`meta.prefix_store.is_some() && max_batch.value > 1`), now removed.

- **Scheduler hooks** (`crates/intarsia-core/src/scheduler.rs`, `BatchSeq`, all with no-op defaults, so the mock model of
  `intarsia-core/tests/scheduler.rs` is unchanged):
  - `restore(model, slot, local)`: called at admission before `admit`; the effective cached length it returns is what the
    scheduler admits with, truncates to, starts `Phase::Prefilling` at and counts in `cached_tokens`.
  - `chunk_len(pos, remaining)`: the request's own chunk schedule; overrides the busy/solo chunk, the macro step, and keeps the
    request out of packed passes.
  - `on_chunk(model, slot, start, end, last)`: after every prefill chunk, before the final chunk's output is accepted.
- **Batched engine** (`crates/intarsia-core/src/engine/batch.rs`):
  - `BatchedEngine::new` opens the store (`StorePolicy::new`, shared with `LocalEngine`; it refuses tensor parallelism).
  - `submit` attaches the request's `SeqStore` unless it drafts with an MTP / DFlash state (`Seq::new`'s bypass rule), and refuses
    a prewarm (400): `/v1/intarsia/prefill` and `--prefix-store-pin` stay single-sequence.
  - Admission (`SeqAdapter::restore`): the slot's own prefix cut to a point of the request's store schedule
    (`SeqStore::local_reuse`, under the prefill-clean watermark), then `SeqStore::restore` of a longer stored prefix into the
    slot; deferral on.
  - Schedule (`chunk_len`): `SeqStore::chunk_end` at the model's prefill chunk, never `--busy-chunk`, so whether another slot
    decodes never changes a request's bits. Its final chunk runs whole on a recurrent state (`SlotModel::store_slot`, set at
    admission), as `Seq::prefill_step` runs it with the store on.
  - Capture (`on_chunk`): raises the slot's prefill-clean watermark (store requests only: another request's rows come from
    another schedule), then `Seq::store_at` at every non-final chunk end, and at the prompt's end when it ends at its last
    message header; `note_prefill` on the final chunk. Captures are deferred (device-to-device).
  - Settle: in `SeqAdapter::finish`, after `report_done` (the client has its `Done`), on the scheduler thread.
  - `Engine::prefix_store` (the `/v1/intarsia/prefix-store` endpoint) and the store telemetry / metrics now work for the
    batched engine.
- **Combinations** (`store_combinations`): with the store and `--max-batch` > 1, `INTARSIA_OVERLAP=1`, `INTARSIA_PACK_PREFILL=1`
  and `INTARSIA_PREFILL_MACRO` > 0 are refused with an error that names the knob; tensor parallelism by `StorePolicy::new`.
  **Interpretation (flag for review):** the plan says "refuse with pack and macro prefill", but on CUDA packed prefill is the
  backend default (`pack_prefill_default`) and the 35B recipe sets `prefill_macro = 8192`; refusing whenever they are active
  would refuse the production combination outright. So only the explicit opt-ins are refused, and the defaults are turned off
  when the store is on (pack off in `BatchedEngine::new`, macro rows 0, one stderr line). The single-sequence engine already
  skips macro prefill for store requests.
- No new flag or env variable. Docs: `docs/tuning.md` (`INTARSIA_PREFIX_STORE` row), `docs/ARCHITECTURE.md`, `serve --help`.

## Tests (CPU, this Mac)

New in `crates/intarsia-models/tests/prefix_store.rs` (tiny hybrid fixture, which is chunk-sensitive on cpu, and the dense
model): three multi-turn conversations, three turns each, every round's three requests submitted together to a 2-slot
batched engine (two run, one waits; a prompt prefills while the other slot decodes; slots change conversations), with
`--busy-chunk` set to half the model chunk. Every reply (ids and every step's logits, bitwise) equals the single-sequence
engine's with the store on; later turns report cached tokens; the store shows inserts > 0, hits > 0, tokens_saved > 0.

- `batched_hybrid_multi_turn_is_the_single_engine_store_on` (slot reuse on, the cpu default)
- `batched_hybrid_multi_turn_without_slot_reuse` (every admission restores from the store: the CUDA case, no batched snapshots)
- `batched_dense_multi_turn_is_the_single_engine_store_on`
- `batched_store_refuses_overlap_pack_and_prewarm`

Mutation check (each change reverted after): the request taking the scheduler's chunk instead of its own -> hybrid tests fail
on bits; no clean-watermark update -> every test fails on `inserts > 0`; final chunk split (lone last token) -> hybrid tests
fail on bits; restore skipped -> every test fails (no hits / nothing cached).

Results (debug build, macOS arm64, all pass):

| suite | passed |
|---|---|
| `intarsia-models --test prefix_store` | 23 (4 new) |
| `intarsia-models --test batch_engine` | 9 |
| `intarsia-models --test batch_recurrent` | 14 |
| `intarsia-core --test scheduler` | 40 |
| `intarsia-core --lib` | 200 (2 ignored) |
| `intarsia-server` (lib, api, prefix_store) | 25 + 46 + 3 |
| `intarsia-cli --test docs_env_vars` | 2 |

## CUDA (spark1)

New `prefix_store_batched_is_single_engine_cuda` in `crates/intarsia-models/tests/prefix_store_cuda.rs`: two chats on the 35B,
two turns each, each round's two requests in flight together on a `--max-batch 2` engine with the store; every reply must be
bitwise the single-sequence engine's (store on), and the second turns must restore more than half their prompt.

spark1 (GB10), build `161c0f5b4` (this branch rebased on #352's `3a25a19e3`), `cargo test --release --locked -p intarsia-models
--test prefix_store_cuda -- --test-threads=1 prefix_store_`, fleetq #9196 (`--quiet gpu --mem 45`, vLLM up), Qwen3.6-35B-A3B NVFP4,
max_seq 8192, bf16 KV: **2 passed, 0 failed** (173 s, MemAvailable never under the 10 GB watchdog; `free -g` after: 83 GB
available).

`prefix_store_batched_is_single_engine_cuda` (`--max-batch 2`, store on; startup line: "the recipe's macro prefill (8192 rows) is
off with --max-batch > 1"):

| round, chat | prompt tok | batched cached | batched wall (both in flight) | vs single-sequence (store on) |
|---|---|---|---|---|
| 1, chat 0 | 1873 | 0 | 2229 ms | bitwise (48 steps of logits + ids) |
| 1, chat 1 | 1170 | 0 (single engine: 19, its held system prompt) | 2378 ms | bitwise |
| 2, chat 0 | 2531 | 1871 (restored) | 1247 ms | bitwise, cached 1871 = 1871 |
| 2, chat 1 | 2065 | 1168 (restored) | 1349 ms | bitwise, cached 1168 = 1168 |

Batched store counters: 4 lookups, 2 hits, 6 inserts (303 MB), tokens_saved 3039, restore 46 ms in total, insert 324 ms of
which settle 150 ms over the 4 finished requests (~38 ms per request at these 1-2.5k-token prompts), 0 rejected.

The existing single-engine gate `prefix_store_restore_is_bitwise_cold_cuda` on the same build: restored vs cold (store on)
bitwise true; vs store off bitwise false from step 0 with equal ids (the documented chunk-sensitivity), restored 1871 tokens.

Not run: `batch_cuda_real.rs` / `batch_recurrent.rs` on CUDA (the batched decode path is unchanged for store-off requests;
the gate above runs it with the store on).

Process note: the first GPU attempt (#9193) ran cargo from `$HOME` (a shell `&&`/`&` grouping error) and its leftover watchdog
loop kept the job's ssh open for ~30 min, holding the GPU slot; it was stopped from inside that job's slot and the rerun is #9196.

## Open risks

- **Settle stalls the other stream (known cost, accepted for slice 1).** The settle runs on the scheduler thread right after
  a request's `Done`; the other sequence's next decode step waits for the device-to-host copies: about 70 ms per finished
  request with #352's pre-faulting (coordinator's spark2 single-engine measurement, 6.8 s over 98 settles; ab35b measured
  73.8 s over 394 inserts before it). A device tier (`device_mb=8192,host_mb=2048`) cut the single engine's TTFT sum further
  (56.8 s vs 60.5 s). Moving it off-thread later: the settle is one call at one place (`SeqAdapter::finish`, after
  `report_done`) on the shared `Arc<Mutex<PrefixStore>>`, so it can become a request to a settle worker (or a scheduler step
  taken only when no sequence decodes) without touching the restore / schedule / capture hooks; what it needs is the backend
  copies on a stream of their own, so they do not serialise with the decode stream. The GPU replay's "other stream
  inter-token p99 at capture" check is the measure.
- **One deferral flag for the whole store.** A request's settle also moves the captures another request in flight made so
  far; the hook re-arms deferral before each capture, so bits are unaffected, only where the copy cost lands.
- **A store request's prefill chunk is the model's chunk beside a decoding stream.** On CUDA the busy chunk already equals the
  solo chunk for the 35B (`load.rs`), so nothing changes there; on a backend whose recipe sets a smaller `busy_chunk` (HIP 27B
  `busy_chunk = 2048` vs solo 4096, the VL recipe 256) a store request now stalls the other streams for a whole model chunk.
- **Packed and macro prefill are off with the store** (CUDA defaults): long cold prompts lose the macro speedup (+5 to +10 %
  prefill at 4k to 16k, recipe note) when the store is on.
- **Drafting requests bypass the store** (as on the single engine). With the store configured the recipe's MTP default is
  skipped, so the batched store engine decodes plain (68 to 79 tok/s at 8k to 33k in ab35b, vLLM's range).
- A failed request's deferred captures settle at the next successful `finish` (no backstop at admission: it would stall the
  other stream); bounded by the store's 1 GiB deferral cap.

## Coordinator instructions

- 2026-11-13 ~00:05 EST, from the coordinator: PR #352 (this branch's base) got one more commit on
  `origin/store-settle-off-decode`: the after-Done settle moved out of `run_job` into the worker loop in its own
  `catch_unwind` (Fable fix), plus doc fixes; it merges to main after CI. Rebase onto origin/main once #352 is merged (or onto
  `origin/store-settle-off-decode` now). Measured on spark2 (interleaved replay, single engine): with pre-fault the settle
  costs ~6.8 s over 98 settles (~70 ms per request), TTFT sum 60.5 s, decode 75.1 tok/s; with
  `--prefix-store device_mb=8192,host_mb=2048` TTFT sum 56.8 s. For the batched store, a settle on the scheduler thread stalls
  the other stream ~70 ms per finished request: acceptable for slice 1, recorded as a known cost (Open risks), and the design
  must let it move off-thread later.
