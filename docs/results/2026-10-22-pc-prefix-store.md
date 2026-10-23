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

DESIGN

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
