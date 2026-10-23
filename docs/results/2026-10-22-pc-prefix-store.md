# Lane pc: a content-addressed prefix store ("rainbow table for prompt processing")

Date 2026-10-22/23 (EDT). Branch `lane-pc/prefix-store` off origin/main a92267b0. Machine for every GPU number: endor
(Apple M5 Max, Metal), fleetq slots named next to each number. Decisions: user-approved 22:09 (LOG rows).

**Result in one line:** RESULTS_LINE

![tiers and request flow](2026-10-22-pc-prefix-store/architecture.svg)

## 1. What was there (verified on origin/main a92267b0)

- Reuse was the common prefix with the one `SequenceState` the single-sequence engine holds, then `restorable_len`
  (`crates/intarsia-core/src/engine.rs:897-903` on main). One other request in between (another chat, another tab)
  replaces that state, and the next turn of the first chat prefills from scratch.
- Recurrent snapshots: at most `RECURRENT_SNAPSHOTS = 4`, device memory only (`state.rs:15`, `:159`), taken after every
  prefill chunk and at `plen - 1` (`engine.rs:1145`, `:1153`). A restore falls back to the newest snapshot before the
  divergence point, often 0.
- The Qwen3.6 chat template (`~/models/Qwen3.6-35B-A3B-NVFP4/chat_template.jinja:94-103`, `:147-152`) drops the
  `<think>` block of every assistant turn before the last user query, and the generation prompt ends in
  `<|im_start|>assistant\n<think>\n`. So turn N+1 shares turn N's prompt only up to `<|im_start|>assistant\n`: the
  `plen - 1` snapshot (after `<think>`) is never reusable by the next turn, only by a regenerate.
- Continuous batching refuses recurrent models (`engine/batch.rs:374`); the 35B is served by `LocalEngine`. The store
  is implemented there only; the batched engine refuses `--prefix-store` (`load.rs`).
- Chunk splicing (CacheBlend-style reuse of non-prefix chunks) is not possible for this model: GatedDeltaNet is a
  nonlinear recurrence over the whole prefix and partial RoPE binds K to absolute positions. Only exact prefixes are
  reusable.

## 2. Design

Code: `crates/intarsia-core/src/prefix_store.rs` (store), `engine.rs` (`StorePolicy`, `SeqStore`, `Seq::new`,
`Seq::prefill_step`, `Seq::store_at`), `crates/intarsia-server/src/prefix.rs` (endpoints), `metrics.rs`,
`web/dashboard.js`, `crates/intarsia-bench/src/reuse.rs` (workloads).

**Entries and keys.** An entry is the state after exactly `len` prompt tokens: every attention slot's K/V rows
`0..len` and every recurrent slot (all 30 GatedDeltaNet delta-rule matrices and conv states, one blob). Key = 128-bit
hash of (build fingerprint, `tokens[..len]`, `len`); a hit also compares the stored token ids in full, so a collision
can never restore a wrong state. The build fingerprint hashes: format version, model id, the parsed recipe, dims,
activation and KV dtypes, the resolved kernel plan (`plan_json` ops and layers: which kernel and fusion runs each op),
backend fingerprint, prefill chunk, tail merge, page size, the boundary/header token texts, weights bytes, and the
running executable's size and mtime. Any rebuild, recipe change, other dtype, other chunk or other backend gets a
different fingerprint, so an entry is never restored into a build that did not compute it.

**Pages.** KV is stored in pages of `page` tokens (default 256), each keyed by the hash of the whole prefix up to the
page end. Turn 1, turn 2, turn 3 of a chat (and every conversation sharing a system prompt) share their KV pages; only
the recurrent blob (~63 MiB on the 35B) is per entry. A 35B entry at 4k tokens costs 63 MiB + 80 MiB the first time
and 63 MiB + (new tokens x 20 KiB) after that.

**Tiers.** device (backend buffers, device-to-device copy on restore; default budget 0 because the memory plan already
gave device memory to the KV cache), host (default 8 GiB; `Vec<u8>`, uploaded on restore), disk (`dir=`, write-through
by a background writer thread, so host entries survive a restart of the same build; files `<dir>/<build>/o-KEY.bin` with
magic, format version, build, key, length and checksum in a 48-byte header, and `e-KEY.entry` manifests). LRU per
tier: device objects demote to host, host objects are dropped (their disk copy stays) or evicted, disk objects are
deleted; an entry missing an object is dropped with it. Pinned entries (`--prefix-store-pin FILE` at startup, or
`"pin"` on a prefill call) never leave host memory. A disk object that fails any header check or its checksum is
rejected before the state is touched, the entry is dropped, `rejected_total` counts it and the request prefills cold.
"Pinned host RAM" in the CUDA/HIP sense (page-locked) is not used: the backend trait has no pinned allocator; `upload`
from a plain host buffer is what the engine does today.

**Where entries come from (item 2).** With the store on, `prefill_step` ends a chunk at every *cut*: right before each
`<|im_start|>`, and right after the first `\n` that follows it (the end of the role header). Between cuts the plain
chunk rule applies (`chunk` rows from the previous cut). An entry is stored at every cut (and every `grid` = 2048 tokens
inside one long message, and at the end of a prewarm). The template check below decides why the header end matters:
the Qwen3.6 template drops the `<think>` block of every assistant turn before the last user query, and the generation
prompt is `<|im_start|>assistant\n<think>\n`; so turn N+1 shares turn N's prompt exactly up to
`<|im_start|>assistant\n` and no further. `crates/intarsia-models/tests/prefix_boundaries_template.rs` renders 3 turns
with the real tokenizer and template (thinking on, thinking off, with tools) and checks the shared prefix ends on a cut:
it does in all 6 transitions (0 tokens lost; e.g. turn 1 -> 2 shares 26 of 28 tokens, with tools 270 of 272).

**Why every reuse path is bitwise: the schedule rule.** A cut is a function of the tokens before it only, and the
chunks between cuts depend only on the previous cut. So any two prompts that share a prefix run *the same forward passes*
over it, and the state at a shared cut is the same bits in both runs. A request restores only an entry whose length is a
point of its *own* cold schedule (`SeqStore::points`: every chunk end, plus `plen - 1` for recurrent states). Restored
then continues with exactly the passes the cold run would run. The same rule caps the engine's local reuse (the state
it already holds) at the prefill-clean watermark (`SequenceState::clean`: decode-written rows never count) and at a
schedule point. Tail merge and 1-row avoidance are not applied with the store on (both depend on the prompt's length,
so they would make a cut depend on later tokens).

What this does *not* give: store-on prefill is bitwise equal to store-off prefill only where prefill is chunk-invariant,
because the store moves chunk boundaries. See section 3: it is not for the hybrid on cpu. `any_cut=1` restores at any
length (no schedule-point filter): only correct for a chunk-invariant build (dense models on cpu, measured).

**Prewarm (item 3).** `POST /v1/intarsia/prefill` takes `messages` (+ `tools`, `chat_template_kwargs`) or a raw `prompt`,
prefills it into the store and generates nothing (`GenRequest::prewarm`). `partial: true` drops the template's trailing
`<|im_end|>\n` and the last `margin` (default 2) tokens, for a message still being typed. `pin: NAME` pins it. Reply:
`prompt_tokens`, `cached_tokens`, `computed_tokens`, `prefill_ms`. Because only schedule points are restored, what a
partial prewarm saves is everything up to the cut after the typed message's header, plus whole chunks of the typed text
(the chunk rule counts from the message start); the last partial chunk of the typed text is recomputed.
`GET /v1/intarsia/prefix-store` returns the counters and the entry list.

**Scope.** Single-sequence engine (`LocalEngine`) only: the batched engine refuses recurrent models anyway
(`engine/batch.rs:374`), and `--prefix-store` with `--max-batch > 1` is refused at load. Tensor parallelism is refused.
Default off (`--prefix-store on` or `INTARSIA_PREFIX_STORE=on`).

## 3. Byte identity

IDENTITY

## 4. Workloads: TTFT and prefill tokens computed

WORKLOADS

## 5. Dead-tail skip in non-final chunks

DEADTAIL

## 6. Metrics

METRICS

## 7. Cache-affinity routing across two tp=1 replicas (design + prototype)

ROUTING

## 8. No-op with the store off

NOOP

## 9. Not done, limits, next

LIMITS

## Reproduce

REPRO
