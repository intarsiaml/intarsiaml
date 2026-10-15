# Continuous batching in `intarsia serve`: design (Metal first)

Design only, 2026-10-15, branch `metal-cb-design` off `spike/metal` c556008. No code was changed and nothing was run on a GPU for this
document. Rank 2 of the levers plan. Every number is one of: **measured** (cited to a result file), **sourced** (cited to a URL or a local
library copy, with a confidence label), or **estimate** (derived here, arithmetic shown, inputs measured unless said otherwise).

## Reference conventions

* Bare `path:line` resolves in this branch (`metal-cb-design`, c556008).
* `[spec]path:line` is `metal-spec-ngram@44c4a91` (read with `git show 44c4a91:path`). That branch contains the G0 multi-row GEMV kernels, the G0b
  multi-row attention and the G1 speculative-decoding path, none of which is merged into `spike/metal`. Line numbers are from the commit, not the
  worktree (the worktree has uncommitted edits to `engine.rs` and `spec.rs`).
* Short file names: `engine.rs`, `model.rs`, `state.rs`, `memplan.rs`, `telemetry.rs`, `ops.rs`, `registry.rs`, `plan.rs`, `spec.rs` are in `crates/intarsia-core/src/`;
  `api.rs`, `health.rs`, `metrics.rs` in `crates/intarsia-server/src/`; `lib.rs`, `attn.rs`, `kernels.rs` in `crates/intarsia-backend-metal/src/`;
  `attention_splitkv.metal`, `qk_rope_kv.metal` in `kernels/metal/`. (`[spec]` applies the same mapping.)
* Documents that are not on this branch: `[levers]` = `metal-next-levers@215faed:docs/research/metal-next-levers.md`;
  `[g0]` = `metal-g0-multirow@96460dd:docs/results/2026-10-15-metal-g0-multirow.md`;
  `[g0b]` = `metal-g0b-attn@cf735e9:docs/results/2026-10-15-metal-g0b-attn.md`;
  `[g1]` = `metal-spec-ngram@44c4a91:docs/results/2026-10-15-metal-spec-g1.md`.
* `[localeval]` = `/opt/code/deepeval/src/localeval/speed.py` (outside this repository).

## 0. Summary

**Recommendation.** Build continuous batching as an iteration-level, decode-first scheduler over **N independent `SequenceState`s** (one per slot),
with the decode rows of all running sequences going through the **row-exact multi-row GEMV kernels that G0 already built** (one weight pass for
B <= 8 rows) and **one serving `AttnDecode` launch per row on that row's own KV**. Every row is then bitwise the solo decode step by construction, so no
new attention kernel, no arena KV and no paging are needed for the first version. Cap the batch at **8**. Default stays `max_batch = 1`, and CUDA and HIP
do not change.

**The measured kernel data overturns the headline of levers section 5.3 above 6 to 8 rows.** [levers] section 5.3 assumed bandwidth-class kernels at every
batch size (B = 8 at 128 context: 290 tok/s). G0 measured the exact multi-row GEMV at 1.029x of the 1-row layer at R = 6, 1.078x at R = 7, **1.208x at R = 8**
(8-byte-aligned shard) and **2.57x at R = 16** [g0] section 3, and G0b measured attention at T(1) = 8.6 / 14.5 / 36.4 us per layer at 128 / 1k / 4k
context [g0b] section 1. Distinct sequences share no K/V, so attention scales about linearly with B. The refined estimate (section 4.6) is:

| exact path, estimate | B = 1 | B = 2 | B = 4 | B = 6 | B = 8 |
|---|---|---|---|---|---|
| 1k context, aggregate decode tok/s (x over B = 1) | 37 (1.0) | 72 (1.95) | 136 (3.7) | 193 (5.3) | 218 (6.0) |
| [levers] 5.3, 1k | 36 | - | 141 | - | 271 |

So "4-8x at 4-8 streams" becomes **about 3.7x at 4 streams and about 6x at 8 streams at 1k context, before scheduler and prefill losses** (apply 0.7-0.85 for
realised throughput). **B = 16 on the exact kernels is no better than B = 8** (R = 16 costs 2.57x; section 4.6 model: 219 tok/s at B = 16 vs 228 at B = 8 for 128 context, 209 vs 218 at 1k), so the
exact path ends at 8 streams. The non-exact alternative (NAX GEMM above 8 rows, tiled f32 logits) is estimated at 7.1x at B = 16, 1k context: **+20% over exact B = 8 for losing batch
invariance**. Recommendation: exact only; keep NAX "fast mode" as an optional late milestone, decided by the operator (section 8).

**Plan.** M0 merge the three unmerged branches and refactor the per-request token loop into a per-sequence state machine, with byte-identical output at
`max_batch = 1` (3-4 days). M1 scheduler, N slots, exact batched decode, server plumbing and a concurrent-identity harness (9-12 days). M2 chunked-prefill interleave, fairness,
cancellation hardening, metrics (3-4). M3 speculation on low load (2-3). Core path M0-M3: **17-23 working days**, versus [levers] 15-25 on top of 6-8 days of K1-K4 that now exist
(the saving from not needing an arena or a ragged attention kernel is spent on the merge and the identity harness). Optional: M4 R = 8 alignment remedy (1), M5 arena plus fused ragged
attention (5-6), M6 non-exact NAX mode for more than 8 streams (3-4). Go/no-go gates with measurable criteria are in section 7.

## 1. Current serve request path and where it serializes

Path of one request (all references on this branch):

| Step | Where |
|---|---|
| HTTP handler calls `api::start` | `crates/intarsia-server/src/api.rs:211` |
| Admission: `inflight` counter, limit `max_queue + 1` ("one running plus `max_queue` waiting"), else 429 | `api.rs:25-26`, `api.rs:70-71`, `api.rs:217`; `--max-queue` default 64: `crates/intarsia-cli/src/cmd/serve.rs:19-21` |
| Cancel flag set when the HTTP response is dropped | `api.rs:45` (`CancelGuard`) |
| `engine.submit(req)` sends a `Job` to the worker over a `std::sync::mpsc` channel | `crates/intarsia-core/src/engine.rs:512-528` |
| One bridge thread per request forwards `GenEvent`s to the async side | `api.rs:234` |
| One worker thread drains the channel with `rx.iter()` | `engine.rs:457-480` (`for job in rx.iter()`, line 462) |
| `run_job` takes the `Inner` mutex **for the whole generation** | `engine.rs:615-622` |
| `generate_one`: prefix reuse, chunked prefill, then a decode loop of one `Model::forward` per token | `engine.rs:664-843` |

Where it serializes, in order of importance:

1. **One worker, one job at a time.** The worker only picks the next `Job` after `run_job` returns (`engine.rs:462-470`). A second client waits for the first client's entire generation.
2. **One `SequenceState`.** `Inner { model, state }` holds exactly one (`engine.rs:355-358`, created at `engine.rs:415`). Prefix reuse is `common_prefix(&st.tokens, &req.prompt)` against that one held sequence (`engine.rs:679`, `engine.rs:660-662`).
3. **The state lock is held across the job**, so the accuracy entry points `forward_logits_each`, `forward_logits_decode` and `forward_script` (`engine.rs:534-594`) also queue behind a generation, and they `reset` the single state.
4. **`Model::forward` is single-sequence**: it takes one `&mut SequenceState` and one id list (`model.rs:840-856`), the decode head works on one row (`head_to_device`, `model.rs:1573-1580`; `ws.tok` is 16 bytes, `model.rs:570`), and decode-graph replay (CUDA/HIP only; Metal does not support graphs) is only for `ids.len() == 1` (`model.rs:972`).
5. **One Metal stream.** `MetalBackend` has one `Stream` behind a mutex (`crates/intarsia-backend-metal/src/lib.rs:153-165`); `upload` and `download` call `synchronize` and drain every in-flight command buffer (`lib.rs:553-570`). Each decode step therefore starts with a drain (the `ws.ids` upload, `model.rs:1120`) and ends with one (the argmax download, `model.rs:1587-1589`).
6. **Telemetry and server assume one running request** (`running` is incremented per job, `telemetry.rs:215-221`; the busy check is `running + waiting > 0`, `crates/intarsia-server/src/health.rs:44`).

Consequence (estimate, no concurrency run exists for the current build: `docs/results/2026-10-12-phase0.md:160` says "`bench speed --concurrency` not run"): N simultaneous clients get aggregate throughput equal to one stream
(about 36 tok/s decode) with per-stream rates of 1/N and queueing in TTFT. `max_batch = 1` is declared in the recipe (`recipes/qwen3.toml:46`, `crates/intarsia-core/src/recipe.rs:152,168`) and printed by `intarsia recipe`
(`crates/intarsia-cli/src/cmd/recipe.rs:2021`), but no code reads it.

Per-request logic that must survive the refactor (all inside `generate_one`): the `pick` closure with the `min_tokens` stop-id mask and the argmax-on-device fast path (`engine.rs:697-725`); stop ids (`engine.rs:787`);
incremental detokenizer and stop-string matcher with the `min_tokens` switch (`engine.rs:792-796`); streaming send, where a failed send is cancellation (`engine.rs:797-800`); `max_new` (`engine.rs:806`); the cancel flag (`engine.rs:810`);
the held-back-text flush at the end (`engine.rs:816-833`).

## 2. Scheduler design

### 2.1 Shape

A new `BatchEngine` implements the same `Engine` trait (`engine.rs:170`) and is built only when `--max-batch N` with N > 1 (flag new; the recipe default `max_batch = 1` becomes the default value of the flag).
`LocalEngine` is untouched and remains the code path of every backend and of `max_batch = 1`. This is the "keep CUDA/HIP unchanged" mechanism (section 5).

One scheduler thread owns the model and all slots; there is no lock around the model because nothing else touches it. `submit` only pushes a `Job` into a channel and bumps `waiting`.

Per-request state, one `Seq` per admitted request: `GenRequest`, `Sender<GenEvent>`, slot index, `phase` (`Prefilling { pos }` or `Decoding`), the sampler (`Sampler::new(params, req.seed)`, one RNG draw per sampled token so the stream does not depend on
batching), `IncrementalDecoder`, `StopMatcher`, `generated`, `last_tok`, `stop_ids`, timing for `Done`. The token-loop body of `generate_one` (`engine.rs:785-815`) is extracted into a method `Seq::on_token(tok) -> Continue | Finish(reason)`
that both paths call. This extraction is M0 and is the same funnel the speculative branch already uses: verified tokens go through the existing per-token body via a `pending` queue ([spec]`engine.rs:726` `generate_one`, call site [spec]`engine.rs:900`).

### 2.2 One iteration (decode-first)

```
loop {
  drain new Jobs into `waiting` (block only when nothing is running or waiting)
  reap: for each running seq, cancel flag set or event channel closed -> finish(Cancelled), slot becomes an idle cached slot
  admit: while waiting is non-empty and a slot is free (FIFO): validate prompt (length < cap, token ids < vocab),
         pick the slot (section 3.5), truncate it to the common prefix, set phase = Prefilling { pos = cached }
  decode: rows = every seq in Decoding (<= max_batch), one row each (plus draft rows if spec is on and rows allow, section 6)
          Model::forward_batch(rows) -> next token per row (argmax on device for greedy rows; logits row downloaded only for sampled rows)
          for each row: seq.on_token(tok) -> stream the event, maybe finish and free the slot
  prefill: if a seq is Prefilling and the interleave policy allows (2.4): run one chunk of the oldest one; on its last chunk, sample the first token
}
```

* **Decode-first**: every running sequence advances one token (or one verified run) per iteration, before any prefill work. A running stream is never starved by arrivals.
* **Admission** is by free slot only. Memory is pre-allocated per slot at load (section 3), so there is no preemption, no swap and no recompute in v1. A request waits in `waiting` (bounded by `max_queue`) until a slot frees.
  A prompt that cannot fit `cap` is rejected at admission with the same `BadRequest` text as now (`engine.rs:670-675`), not mid-batch.
* **Max batch** = the number of slots = `--max-batch` (default 1; recommended maximum 8 on the exact path, section 4). The server's admission limit becomes `max_batch + max_queue` (`api.rs:25,71` hard-code "one running"); `ModelInfo` (`engine.rs:145`) gains `max_batch`.
* **Fairness**: FIFO into slots; prefill chunks go to the oldest `Prefilling` seq (a later short prompt does not overtake a long one that is already partly done). No priorities in v1 ([ARCHITECTURE] mentions `priority: background`, `docs/ARCHITECTURE.md:229`; see section 8).

### 2.3 Per-sequence semantics (stop, min_tokens, EOS, max_tokens, cancel)

All of these are properties of one `Seq` and are evaluated in `on_token`, in the same order as `generate_one` evaluates them, so batching cannot change them:

* `min_tokens`: the stop-id mask applies while `k <= min_tokens` (`engine.rs:701,714-719`). A greedy row inside that window needs its logits on the host; in a batch that is a per-row download of `[1, vocab]` from the `[B, vocab]` logits workspace (`ws.logits` already holds 16 rows, `model.rs:569`), not a different kernel. The stop-string matcher is armed at token `min_tokens + 1` (`engine.rs:780,793-795`).
* EOS and `stop_ids` are not emitted but counted (`engine.rs:787-790`); `ignore_eos` empties the list (`engine.rs:692-694`).
* `max_new = min(max_tokens, cap - plen)` is per seq (`engine.rs:676`); `cap` is the slot capacity.
* Cancellation: checked at the top of each iteration (flag, or a failed `tx.send`), so a cancelled request costs at most one extra row in the step already running. Its slot's KV stays valid and becomes a prefix-cache candidate.
* Sampled rows: `HeadMode::LastLogits` semantics per row; the host sampler is unchanged. Greedy rows use the on-device argmax (`HeadMode::LastArgmax`, `engine.rs:703-708`), generalised to a row-wise argmax over `[B, V]` (the `verify_head` pattern, [spec]`model.rs:765`).
* `trace_logits` requests (accuracy tooling, `engine.rs:44-46`) are not batched: they run alone, as a one-row batch.

### 2.4 Prefill and decode interleave (chunked prefill)

A long prompt must not stall every decode stream. Measured prefill rates (overnight report, `docs/results/2026-10-15-metal-overnight-report.md:17-21`): chunk 256: 4056 tok/s at 1k and 3494 at 4k; chunk 2048: 4373 and 3719.
Derived stall of one chunk (estimate): **chunk 2048 is about 0.47-0.55 s, chunk 256 is about 63-73 ms**.

* The prefill chunk is a separate sub-pass from the decode step. The decode rows must **not** be folded into the prefill GEMM: the NAX GEMM that serves chunks is not bit-identical to the GEMV (section 4.5), and folding is exactly what vLLM and Sarathi-Serve do (one forward over decode tokens plus a prefill chunk). A mixed iteration here is therefore decode step + chunk step, two weight passes. A Sarathi-style fold would save roughly the decode pass (30-40 ms) of each ~106 ms mixed iteration at chunk 256 (estimate) and is the "fast mixed step" of M6.
* Policy (M2): when no sequence is decoding, prefill runs at the model's normal chunk (`prefill_chunk`, 2048 on Metal, `recipes/qwen3.toml:56`), i.e. exactly today's behaviour and speed. When at least one sequence is decoding, the scheduler runs at most one chunk per iteration, of at most `--prefill-budget` tokens (default 256), and records an EWMA of decode step time to keep p95 inter-token latency under a target. Example (estimate): B = 7 decoding at 1k (about 33 ms/step) + a 256-token chunk (about 73 ms) gives a ~106 ms iteration, so each running stream sees ~9 tok/s while another request prefills, and a 4k prompt takes 16 such iterations (~1.7 s, 1.5x its solo prefill of ~1.16 s at 3494 tok/s).
* **Chunk size is part of numerical identity** (section 4.5). The chunk boundaries a request is prefilled with must equal those of its reference solo run. Two consistent choices: (a) keep the model chunk (2048) and interleave only at chunk boundaries (identity with the existing golden baselines, which "were recorded with" chunk 2048, `recipes/qwen3.toml:52-53`; but ~0.5 s stalls), or (b) serve with a smaller chunk everywhere (`--prefill-chunk 256`, -7% prefill rate, ~70 ms stalls, new golden baselines). This is open question 1.
* Possible optimisation, not in the gates: today `download` drains the whole queue (`lib.rs:564-570`), so a chunk queued behind a decode step cannot overlap with host post-processing of that step's tokens. A "wait for this command buffer only" read would let the chunk execute while the host detokenizes and streams. Requires a small Metal backend addition; estimate saves the host gap (~1-2 ms/iteration) only when a chunk is due.

### 2.5 Failure isolation, non-generation entry points, telemetry

* A panic or backend error in a batch step fails the engine as now (`engine.rs:469-477`): all in-flight sequences get `GenEvent::Error`, the supervisor restarts the process. Per-request errors (bad prompt, token id >= vocab, which today is detected at upload time, `model.rs:1121-1125`) are detected at admission so they never poison a step.
* `forward_logits_each`, `forward_logits_decode` and `forward_script` (`engine.rs:534-594`) reset the single state. In `BatchEngine` they run on a dedicated extra slot and take the scheduler's exclusive "tool" lock between iterations. They are accuracy tooling, not serve traffic.
* Telemetry: `request_started/finished` per sequence (`telemetry.rs:215-231`), `running` becomes real, **one `step()` per batched forward** (`telemetry.rs:181`; both the watchdog `health.rs:44,97-100` and localeval's busy check read these), plus new batch-size and per-iteration histograms on `/metrics` (`crates/intarsia-server/src/metrics.rs:123-128`).
* Server: the bridge-thread-per-request design (`api.rs:234`) is fine at 8 + 64 requests. `AppState::acquire` limit changes as above.

## 3. KV cache

### 3.1 Today

Per sequence, per attention layer: `AttnKvState { k, v: [n_kv_heads, cap, head_dim] }`, contiguous, bf16 (`crates/intarsia-core/src/state.rs:23-33`, allocation `state.rs:93-106`), `cap` = `MemoryPlan::kv_cap_tokens`
(`model.rs:728-737`). `truncate` on an attention-only state just sets `len` (`state.rs:228-235`); rejected rows stay in the buffer and are never read, because attention reads `[0, pos0 + t]` only ([g1] section 3). Qwen3-8B: 36 layers x 2 x 8 kv heads x 128 x 2 B = **147,456 B per token** (`plan.kv_bytes_per_token`, `model.rs:1598`; derived in [levers] section 1).

### 3.2 Options

| | Layout | Kernel change | Memory | Verdict |
|---|---|---|---|---|
| **A (v0)** | N `SequenceState`s, each with its own K/V buffers, exactly as today | none: each row's `QkNormRopeKv` and `AttnDecode` launch is the serving kernel on that slot's buffers | `N x cap x 147,456 B` | **Recommended for M1.** Keeps the exactness argument trivial and prefill code unchanged. |
| B (v1, optional) | one arena buffer per layer `[slots, n_kv_heads, cap, head_dim]`, a slot is a byte offset; `AttnKvState` gains `offset` | fused ragged attention takes `(slot, pos)` per row; `QkNormRopeKv` takes a position table | same | Only to fuse the per-row launches (M5). A slot view has the same layout as today, so prefill and the 1-row kernels are unchanged. Needs `total_bytes()` to avoid double counting (`state.rs:118-120`) and 16-byte-aligned slot strides (`attn.rs` requires aligned K/V or falls back). |
| C | paged blocks plus block table | new paged attention everywhere | tight | Not needed on this machine (section 3.3); [ARCHITECTURE] line 231 plans it for dGPU backends. Do not build for Metal. |

Why A is enough: the per-row cost of fusing is bounded. [g0b] measured the shared-K/V multi-row kernel at 2.07 / 3.63 / 4.01 x T(1) for 8 rows against 8.56 / 8.07 / 6.44 x T(1) for 8 sequential 1-row launches at 128 / 1k / 4k context.
For distinct sequences the L2 sharing disappears, so the saving is at most the launch/tail latency: **upper bound** (8.56-2.07) x 8.6 us x 36 = 2.0 ms (128), (8.07-3.63) x 14.5 us x 36 = 2.3 ms (1k), (6.44-4.01) x 36.4 us x 36 = 3.2 ms (4k) per step at B = 8, i.e. 6-8% of a ~36-43 ms step (estimate).

### 3.3 Memory budget (Qwen3-8B BF16, M5 Max 128 GB)

Measured inputs: `recommendedMaxWorkingSetSize` 115.4 GB and `maxBufferLength` 86.6 GB (`docs/results/raw/2026-10-14-metal-prefill-p1/full/final/qualify-golden/logs/doctor.out`, `docs/results/2026-10-14-metal-spike.md:26`); weights 16.4 GB ([g0b] section 4);
workspace at chunk 2048: **0.27 GB** (estimate from the buffer list `model.rs:557-575`: hidden-sized 4 x 16.8 MB, q 33.6 MB, k/v 12.6 MB, three FFN buffers 151 MB, logits 9.7 MB). Default `kv_fraction` 0.5 (`model.rs:66`).
KV budget = (115.4 - 16.4 - 0.27) x 0.5 = **about 49 GB (estimate)**, or about 335k tokens in total across slots.

KV bytes (decimal GB, 147,456 B/token, slot capacity = context):

| streams | 1k tokens/slot | 4k tokens/slot | 16k tokens/slot | 32k tokens/slot |
|---|---|---|---|---|
| 4 | 0.60 | 2.42 | 9.7 | 19.3 |
| 8 | 1.21 | 4.83 | 19.3 | 38.7 |
| 16 | 2.42 | 9.66 | 38.7 | 77.3 |

At 16 streams x 4k the total resident set is about 16.4 + 0.3 + 9.7 = 26 GB of 115 GB. **Memory is not the binding constraint on this machine; the kernel cliff at R = 8 is.** Unfilled parts of a slot cost address space and maybe nothing resident (pages of a fresh shared buffer are zero-filled lazily; not verified).
Qwen3-8B's `max_position_embeddings` is 40960 (`~/models/Qwen3-8B/config.json`), so a 40k slot is 6.0 GB.

`MemoryPlan` change (`crates/intarsia-core/src/memplan.rs:78-84`): `by_mem` is sized for one sequence. With N slots, `cap = min(requested_max_seq, max_positions, by_mem / N)`, `MemoryPlan` gains `kv_slots`, and `kv_total_bytes = slots x cap x bytes_per_token` (`memplan.rs:101-102`).
Slot capacity stays uniform in v1; a mixed 8 x 4k + 1 x 32k layout is a paging-adjacent feature (open question 3).

### 3.4 Hybrid and recurrent models, tensor parallelism

Out of scope for v1. Recurrent (GatedDeltaNet) state is fixed-size, snapshot-based and cannot be truncated (`state.rs:194-226`), and the Metal rows kernels exist for the dense attention path only. `BatchEngine` refuses `has_recurrent()` models and `--tp > 1` at load, as the speculative path does ([g1] section 8).

### 3.5 Interaction with the prefix cache

Today's "prefix cache" is the one held sequence: `common_prefix(&st.tokens, &req.prompt).min(plen - 1)` then `truncate` (`engine.rs:679-686`); `cached_tokens` is reported in `usage`. With N slots it generalises without a new data structure:

* Each finished sequence leaves its slot **idle but valid** (tokens + KV). Admission picks the idle slot with the **longest common prefix**, else the least recently used free slot; running slots are never evicted. This is an N-entry LRU prefix cache with no radix tree.
* The identity consequence: the reused prefix length depends on which slot happens to be picked, and the reference solo run must have the same cached length (section 4.5).
* v1 (optional, M5-adjacent): **copy a prefix from any slot**, including a running one (shared system prompt across concurrent users). Cost (estimate): 1k tokens x 147,456 B = 151 MB read + write at a few hundred GB/s, under 1 ms, against ~0.25 s to prefill 1k tokens (TTFT 0.27 s, overnight report line 11). Needs a strided KV copy (the layout is `[n_kv_heads, cap, head_dim]`, so a prefix is `n_kv_heads` strided runs per layer and tensor); `Backend::copy` is contiguous (`state.rs:171`).
* The server-side radix tree of [ARCHITECTURE] (`docs/ARCHITECTURE.md:62,231`) is not part of this design.

## 4. Kernels

### 4.1 Op inventory for one batched decode step (B rows, Qwen3-8B attention + dense FFN)

"Exact" means: row r of the output is bitwise what the solo 1-row serving launch computes for that row, regardless of B and of the other rows.

| Op | Form needed | Exists? | Cost evidence |
|---|---|---|---|
| `Embedding` | B ids | row-parallel, exact [g0b] section 2 | T(8) = 1.10 x T(1) (1.23 us) |
| `RmsNorm`, `RmsNormAdd`, `SiluGlu`, `Add` | B rows | row-parallel, exact [g0b] section 2 | 1.04-1.52 x T(1) at 8 rows, a few us per layer |
| `Matmul` q, o, down | R-row GEMV, serving variant pinned | yes: `matmul_bf16_gemv_rows`, R = 1..8, 16, [spec]`kernels.rs:466-497` | 1.002 / 1.006 / 1.029 / 1.078 / **1.208** per layer at R = 2 / 4 / 6 / 7 / 8 [g0] section 3 |
| `MatmulPair` k+v, gate+up | R-row pair GEMV | yes: `matmul_pair_bf16_gemv_rows` | same table |
| `Logits` f32 | R-row lm_head GEMV | yes: `logits_bf16_gemv_rows`; split form 1.068 at R = 8 not wired into the launcher [g0] section 1 | 1.077 at R = 8 |
| row-wise `Argmax` over `[B, V]` | one launch | yes: `verify_head` ([spec]`model.rs:765`), 1.02 x T(1) at 8 rows [g0b] | 4B bytes back instead of B x 608 KB |
| `QkNormRopeKv` (fused qk-norm, rope, KV write) | per row: different position and different cache | **v0: one T = 1 call per row on that row's slot (the serving kernel).** v1: position table | 2.49 us per call [g0b] section 2 |
| `AttnDecode` (split-KV) | per row: different context length and cache | **v0: one serving `attn_decode_bf16_splitkv` launch per row on that row's slot.** v1: ragged fused kernel | T(1) = 8.6 / 14.5 / 36.4 us per layer at 128 / 1k / 4k, measured with cold per-layer KV pools [g0b] section 1 |

### 4.2 v0 attention: why per-row launches are exact and how much they cost

The serving decode attention picks its work partition from `visible = pos0 + 1` only (`decode_splits`, `crates/intarsia-backend-metal/src/attn.rs:47`; `splitkv_ok` requires `m == 1`, `attn.rs:54`), and for a host position uses 8 simdgroups up to 1024 keys and 4 above ([g0b] section 1).
Calling it once per row with `pos0 = that sequence's len` reproduces the solo step bit for bit; there is nothing to prove. In the model this is a loop inside the attention block (`model.rs:1310-1458`): the Q, K, V and O projections and the FFN run once on `[B, hidden]`
(`TensorRef::rows(r, 1)` views, as `head_to_device` already does, `model.rs:1575`), and only `QkNormRopeKv` (`qkv_fold`, `model.rs:1255-1306`) and `AttnDecode` (`model.rs:1424-1431`) run per row. The split-KV scratch is shared between rows (ops run in stream order, `model.rs:197`).

Cost: 3 dispatches per row-layer (qk/rope/KV-write, split-KV stage 1, combine). At B = 8 that is 756 extra dispatches on top of the 364 of a 1-row step ([levers] section 1). The GPU-side cost is already in the measured T(1) chains. The CPU-side encode cost is small and overlapped
with execution: ADR-0009 measured encode + commit of 253 GEMV dispatches at 0.107 ms (`docs/decisions/0009-native-metal-backend.md`, section 3; for simple GEMV encodes only, so treat as a lower bound on the real per-dispatch cost), buffers are committed every `commit_every` dispatches (`lib.rs:164,348`).
This must be measured in M1 (a host-bound step would show as GPU idle in `INTARSIA_METAL_PROFILE`).

### 4.3 v1: fused ragged attention (optional, M5)

Generalise G0b's `attn_rows_splitkv_t` ([spec]`kernels/metal/attention_splitkv.metal:319-431`): row r already derives its own `visible`, `nsplit`, `chunk` and simdgroup count from its own position ([spec]`attention_splitkv.metal:336-345`), which is exactly the ragged-context requirement.
Two changes: take the position from a table (`posbuf[row]` instead of `p.f + row`, line 336) and add a per-row slot base to the `kc`/`vc` pointers (lines 355-356). The latter needs one buffer per layer for all slots, hence the arena (option B). Keep the grid order `x = kvh * R + row` with rows sorted by slot, so the speculative verify rows of one
slot stay adjacent and keep the L2 sharing [g0b] measured (reversing the order was up to 32% slower). `QkNormRopeKv` uses `pos0 + t` and a single cache (`kernels/metal/qk_rope_kv.metal:61,89,111`); the table form is a few lines. Bitwise exactness survives because each row keeps the 1-row partition and arithmetic order.
Upper-bound benefit is 6-8% of a B = 8 step (section 3.2).

### 4.4 Must batched output stay bitwise equal to single-stream?

| | A. Exact (recommended) | B. Fast: NAX GEMM + tiled f32 logits above 8 rows |
|---|---|---|
| Property | row output independent of the other rows and of B; equals solo decode | depends on B and on the tile the GEMM picks for M (the NAX tile pick changes with M: "NAX tile pick for 128 <= M <= 512", `docs/results/2026-10-15-metal-overnight-report.md:77`) |
| Max batch | 8 (R = 8 costs 1.208x per layer; R = 16 costs 2.57x) | 16 and beyond; NAX is a flat latency floor up to M = 128 for the big matmuls ([levers] section 2.2, measured `bench-pick.txt`) |
| Step at B = 8 / 16, 1k (estimate, section 4.6) | 36.6 ms / 76.5 ms (16 exact rows) | n.a. / 61.2 ms |
| Aggregate at B = 16, 1k | 209 tok/s (5.7x; no better than exact B = 8 at 218) | **261 tok/s (7.1x)** |
| Gates and features that keep working | D1 identity gate, golden baselines, speculation losslessness, "same prompt gives the same bytes regardless of load" | each needs a tolerance or a per-load baseline |

Recommendation: **A.** The gain of B over A at the batch sizes that matter is about +20% (261 vs 218 tok/s at 1k), and it costs batch invariance, reproducibility under load and the lossless-speculation argument. B becomes interesting only for 16 or more concurrent streams
(NAX is flat in M, so B = 32 would approach 12x by the same arithmetic, an unmeasured extrapolation because the tiled f32 logits GEMM above M = 16 is unmeasured). It is an opt-in milestone (M6) gated on a demonstrated need.

### 4.5 What else can break bitwise identity (outside the kernels)

State these as gate conditions, not as kernel properties:

1. **Prefill chunk boundaries.** Prefill uses the NAX GEMM and flash attention, whose tile choice varies with M (the chunk length). Chunk-size invariance was demonstrated on HIP (`docs/results/2026-10-13-prefix-reuse.md`, "cold at chunk 16 vs chunk 8 / 15 / 32: 32/32 bitwise") and **has not been shown on Metal**. The reference for a batched request is the solo run with the same `--prefill-chunk` and the same cached length. The scheduler may delay chunks but must not resize them (or must use the same smaller chunk in the reference).
2. **Cached-prefix length** depends on slot choice (section 3.5); the first chunk starts at the cached length. Reference solo run: warm it to the same cached length (the `accuracy --mode reuse` procedure).
3. **Mixed steps**: decode rows never share a GEMM with prefill rows (section 2.4).
4. **Decode-written vs prefill-written KV** differ in the last bits (`docs/results/2026-10-13-prefix-reuse.md`: warm vs cold KL 1.4e-4 to 2.9e-4 on HIP). Batching does not change which rows are decode-written, so identity is judged against a solo run of the same history, not against a cold re-prefill.
5. **Speculative rows** inherit the [g1] argument; with batching the verify rows run per slot (section 6).
6. **Sampling**: the host sampler draws one random number per sampled token from a per-request seeded stream, so it is batch independent as long as the logits row is bitwise identical.

### 4.6 Throughput model (estimate)

`step(B) = step(1) + dGEMV(B) + 36 x (B - 1) x (T1_attn + 2.49 us) + dSmall`, with `step(1)` = 27.2 / 27.3 / 28.1 ms at 128 / 1k / 4k (measured, overnight report), `dGEMV` = +0.04 / 0.17 / 0.75 / 5.03 ms at R = 2 / 4 / 6 / 8 (census-weighted, measured, [g0b] section 3; with the 16-byte alignment remedy R = 8 is +3.99 ms), attention `T1_attn` = 8.6 / 14.5 / 36.4 us per layer per row (measured), `dSmall` = 0.03 ms at 8 rows (measured).
Distinct sequences share no K/V and [g0b] found the kernel instruction-bound, not byte-bound, so a bandwidth lower bound for the extra rows is not used; per-row cost is T(1). No scheduler, host or prefill loss is included.

| context | B = 1 | B = 2 | B = 4 | B = 6 | B = 8 | B = 8, R = 8 aligned |
|---|---|---|---|---|---|---|
| 128 | 27.2 ms, 37 tok/s | 27.6, 72 | 28.6, 140 | 30.0, 200 | 35.1, 228 | 34.0, 235 |
| 1k | 27.3 ms, 37 | 28.0, 72 | 29.3, 136 | 31.1, 193 | 36.6, 218 | 35.6, 225 |
| 4k | 28.1 ms, 36 | 29.5, 68 | 32.5, 123 | 35.9, 167 | 43.0, 186 | 41.9, 191 |

Speedup over B = 1 at 1k: 1.95x / 3.7x / 5.3x / 6.0x for B = 2 / 4 / 6 / 8 (4k: 1.9x / 3.5x / 4.7x / 5.2x). Fast mode (NAX, 16 rows, [g0] `step(16) existing` GEMV-only 50.10 ms, + same attention model): 57.9 / 61.2 / 73.8 ms, i.e. 276 / 261 / 217 tok/s at 128 / 1k / 4k.
The R = 8 alignment remedy (copy the 2.75 GB of 8-mod-16 shard-1 weights to aligned memory at load, +0.15 s load time, [g0b] section 4) is worth about +3% at B = 8 (estimate), which is why it is optional (M4).

## 5. Backends

* **Metal first**: everything above. New code: `Model::forward_batch` and the scheduler (backend independent, in `intarsia-core`); no new Metal kernels in v0, only registering/reusing G0 and G0b ones, and wiring the logits split form.
* **Selection mechanism: reuse the speculative branch's one.** `OpCall::exact_rows` ([spec]`ops.rs:260`) and `KernelImpl::rows_exact` ([spec]`registry.rs:90`) route a call to a row-exact kernel only when the op has one registered ([spec]`registry.rs:306`); `Phase::Verify` ([spec]`plan.rs:57,209-210`) dry-runs the verify shapes at load so a missing kernel is a **load error**, not a runtime one ([g1] section 2). Batching adds `Phase::Batch` (rows 2..=max_batch) and `Model::enable_batch(n)` next to `enable_verify` ([spec]`model.rs:714`). A batch step sets the same `exact_rows` flag ([spec]`model.rs:750-755`).
  The per-row `QkNormRopeKv`/`AttnDecode` calls have no `rows_exact` kernel registered, so `rows_only` is false for them and they select the serving kernels.
* **Capability flag**: there is none to add. Capability is "the registry has `rows_exact` kernels for Matmul, MatmulPair, Logits, Argmax". A friendlier error comes from a small `Model::batch_capable()` check before the dry run.
* **CUDA and HIP stay unchanged**: they register no `rows_exact` kernels and no call sets the flag, so their registry results and plans are literally unchanged ([g1] section 2); `--max-batch` defaults to 1 and selects `LocalEngine`; `--max-batch > 1` on those backends fails at load with "backend has no row-exact kernels". Their decode graphs bind one state ID (`state.rs:60,70-71`, `crates/intarsia-core/src/model/graph_step.rs:63`) and only replay for `ids.len() == 1` (`model.rs:972`), so batched steps would run eagerly there. Opting in later means registering `rows_exact` kernels (HIP's measured prior art: MoE is batch-variant above 16 tokens and "multi-row verify of 16 rows is not bit-equal to serial decode", [levers] section 2.1, `docs/decisions/0007-phase-1b-2a-2b-implementation-decisions.md`) and, for throughput, the paged/ragged attention plus graph buckets of [ARCHITECTURE] (`docs/ARCHITECTURE.md:221`). That is a separate design.
* **CPU reference backend**: has no `rows_exact` kernels, so a batch step just runs its ordinary row-independent kernels. The scheduler, state machine and server plumbing are therefore testable on CPU (as the speculative tests are, [g1] section 7); bitwise claims are tested on Metal only.
* **Tensor parallel and recurrent models**: refused at load (section 3.4).

## 6. Interaction with speculative decoding

Both features consume rows of one weight pass, so they share one row budget `R_max` per step. Define the step input as a list of row groups `(slot, pos0, q_len)`: a plain sequence has `q_len = 1`; a speculating sequence has `q_len = 1 + m` consecutive positions. The matmuls, norms and head run once on all `sum q_len` rows;
attention runs once per group (`AttnDecode` for `q_len = 1`, G0b's `attn_prefill_bf16_splitkv_rows` for `q_len > 1`, [spec]`kernels.rs:193,336`; both bitwise the 1-row launch per row). Rollback is per slot (`truncate`, section 3.1).

* **Budget**: rows per step <= `R_step_max` = 6 by default (1.029x layer cost) rising to 7 or 8 only with the alignment remedy. With B running sequences the drafts available are `R_step_max - B`.
* **Policy**: B = 1: as today (`--spec-k` default 4, G1 reached 3.3x on input-grounded prompts and 0.99-1.01x open-ended, [g1] section 5). B = 2-3: split the spare rows among sequences by recent acceptance (EWMA of accepted/proposed). B >= 4 (spare rows <= 2): speculation off. Reason (estimate): with B rows already sharing the weight pass, a draft row only pays if acceptance exceeds the marginal row cost, which grows with context ([g0b] break-even 1-11% at k = 4-6, i.i.d. model) while the aggregate gain of one more *sequence* row is a full token.
  Speculation is a low-load latency feature, batching is a load feature ([levers] section 5.3).
* **Per sequence**: the drafter (`spec::propose`, [spec]`spec.rs:72`) and acceptance (`spec::accept`, [spec]`spec.rs:106`) are per-`Seq` state; greedy-only, attention-only, no `trace_logits`, exactly as G1 gates it ([g1] section 1).
* `min_tokens` and EOS inside an accepted run are already handled by `spec_verify` ([spec]`engine.rs:699-725`); in the scheduler they become part of the same `on_token` funnel.

## 7. Milestones, gates, effort, risks

All timing runs go through a fleetq slot on endor (exclusive) with the GPU clock logged (`macmon`), three rounds, same-slot A/B; the overnight report documents GPU clock drops (900-1300 MHz) that otherwise swamp 5% effects
(`docs/results/2026-10-15-metal-overnight-report.md:35`, `docs/results/2026-10-15-metal-serve-prefill.md`).

### What the localeval metric measures

`aggregate_decode_tok_s` ([localeval] lines 114-139) is `sum(completion_tokens) / wall`, where wall runs from starting N threads to all joined. The N requests are the same filler prompt at the first `--sizes` entry >= 1024 (so `--sizes 4096` for 4k), differing only by a nonce prefix, with forced length
(`force_tokens`). **Wall time includes every request's prefill**, and it never exercises ragged lengths or staggered arrivals. It is an end-to-end number. So: use `--gen 512` or more, compare against the serial baseline measured on the same build with the same command, and add the decode-only numbers
(`per_stream_decode_tok_s`, and the in-repo `intarsia bench speed --concurrency` whose aggregate is "from the first token to the last stream finishing", `docs/benchmarking.md:30-32`, `crates/intarsia-bench/src/speed.rs:560-600`). Expected values (estimate, 1k context, `--gen 512`, TTFT 0.27 s):
serial baseline N = 2 / 4 / 8: about 35.9 tok/s each (aggregate equals one stream, prefill included); batched: 69 / 127 / 196 tok/s (1.9x / 3.5x / 5.5x), from `N x 512 / (N x 0.27 + 512 x step(B))` with the section 4.6 step times. The harness also needs a **concurrent-identity test** that localeval does not have.

### Milestones

| M | Content | Days (estimate) | Gate (go/no-go, measurable) |
|---|---|---|---|
| **M0** | Merge `metal-g0-multirow`, `metal-g0b-attn`, `metal-spec-ngram` into the working branch; extract `Seq::on_token` and the per-seq state machine from `generate_one`; add `--max-batch` (default 1) and read recipe `max_batch`. | 3-4 | **Go** if: `cargo check --workspace --all-targets` and `cargo test -p intarsia-backend-metal --release` pass (101 tests at G0b), `intarsia kernels test --backend metal` admits everything; at `--max-batch 1`, spec off and spec on, generate and serve outputs are **byte-identical** to `spike/metal` c556008 on the D1 identity prompts plus `golden_4k` (the G1 case list, >= 24); localeval decode at 128 / 1k / 4k within 2% of 36.7 / 36.6 / 35.6. **No-go**: any byte diff, or a merge conflict that changes serving selection (`multirow::serving_selection_unchanged`). |
| **M1** | `Model::forward_batch` (batched projections, per-row attention, row-wise head), `Phase::Batch`, N slots, scheduler (decode-only; a new request's whole prompt is prefilled on admission, stalling decode), `BatchEngine`, server limit, concurrent-identity harness. | 9-12 | **Hard identity gate**: for B in {2, 3, 4, 6, 8}, >= 200 requests with different prompt lengths (128 to 4k), staggered arrivals, a mix of greedy/sampled, `min_tokens`, stop strings, EOS and cancellations: every output (ids, finish reason, `completion_tokens`) equals its solo run at the same `--prefill-chunk` and cached length; 0 mismatches. **Throughput gate** (thresholds about 85% of the section 7 estimates; baseline measured first on c556008): localeval `--sizes 1024 --gen 512 --concurrency 1,2,4,8`: aggregate >= 1.6x / 3.0x / 4.5x of the serial baseline at N = 2 / 4 / 8; decode-window aggregate (`intarsia bench speed`) at N = 8 >= 175 tok/s (0.8 x 218); N = 1 unchanged within 2%. **No-go**: any identity mismatch (fix, never relax); N = 8 below 3.0x of baseline means host/scheduler gap or dispatch cost is the problem, investigate GPU idle before building anything else (M5 becomes the next step if per-row launch overhead is >= 8% of the step). |
| **M2** | Chunked-prefill interleave with budget and EWMA, FIFO fairness, cancellation and error paths, per-iteration and batch-size metrics. | 3-4 | Scenario: 7 decoding streams at 1k, one 4k prompt arrives at t = 5 s. Gates (estimate-derived): p95 inter-token latency of the running streams during that prefill <= 150 ms at chunk 256; steady-state p95 ITL <= 1.15 x step time; the arriving request's TTFT <= 2.5x its solo TTFT; no admitted request waits behind a later one; identity gate of M1 rerun with interleaving on. |
| **M3** | Speculation on low load: row-group step input, per-seq drafter state, spare-row allocation by acceptance, `Phase::Batch` rows include verify groups. | 2-3 | B = 1 equals G1 (grounded 3.3x, open-ended >= 0.99x); B >= 4 aggregate within 2% of M2 with `--spec ngram` on; mixed load (1 grounded + 3 open-ended) never below M2 by more than 2%; identity gate rerun with spec on. |
| M4 (optional) | R = 8 alignment remedy at load (+2.75 GB, +0.15 s) and wiring the split logits form. | 1 | Go only if N = 8 aggregate improves >= 3% (estimate +3%). Skip otherwise. |
| M5 (optional) | Arena KV, ragged fused attention and `QkNormRopeKv` position table, strided prefix copy. | 5-6 | Go only if M2 profiling shows per-row attention launches >= 8% of a B = 8 step (the upper bound is 6-8%). Gate: bitwise equal to M1 outputs, step time at B = 8 reduced >= 5%. |
| M6 (optional) | Non-exact "fast" mode: NAX/tiled path above 8 rows, optionally decode rows folded into the prefill GEMM (Sarathi-style). | 3-4 | Operator decision (section 8, Q1) and a measured need for > 8 streams. Gate: aggregate at B = 16 >= 1.15 x exact B = 8 and accuracy KL budget agreed; documented as batch-variant, off by default. |

Core path M0-M3: **17-23 working days** (estimate). Optional M4-M6: +9-11. This is about the same as the [levers] rank-2 estimate (15-25 on top of 6-8 days for K1-K4): K1-K3 already exist, which removes the arena and ragged-attention work, and the merge plus the identity harness replace it.

### Risks

| Risk | Effect | Mitigation |
|---|---|---|
| R = 8 cliff (mechanism not identified, [g0]) | exact path tops out at B = 8; 6-8 rows cost 1.03-1.21x | cap at 8; alignment remedy M4; split logits form; NAX mode M6 |
| Three unmerged, unpushed branches ([g1] section 9) with edits to `engine.rs`, `model.rs`, `registry.rs`, `ops.rs`, `plan.rs` | M0 merge conflicts; `serving_selection_unchanged` already needed an update ([g1] section 7) | M0 is first and gated on byte identity; merge in order g0, g0b, spec |
| Prefill chunk is part of numerical identity (4.5) | batched output may differ from "solo default" by last-bit flips | define the reference as solo at the same chunk and cached length; operator picks 2048 vs 256 (Q2) |
| Host/scheduler gap: per-iteration detokenize, stream, encode, and two drains (`upload`/`download` synchronize, `lib.rs:553-570`) | caps the realised fraction of the estimate (levers assumed 70-85%) | measure GPU idle in M1; backend "wait this command buffer" read in M2 if the gap is > 5% |
| GPU clock drops and run-to-run variance (overnight report) | false gate failures or passes | same-slot interleaved A/B, clock-filtered rounds, 3 passes |
| Scheduler bugs are harder to reproduce than kernel bugs | flaky identity failures | record-and-replay of arrival traces in the harness; deterministic scheduler tick for tests |
| Estimates use B x T(1) attention and measured cold KV pools; G0b's sequential timings were cache-warm | step time could be a few percent lower than modelled | measure in M1; M5 is the answer, not a redesign |
| Only helps overlapped requests | single-user chat sees nothing | default `max_batch = 1` until the operator opts in |
| Single Metal stream, `with_encoder` mutex (`lib.rs:336`) | no concurrent encode/GPU work from several threads | one scheduler thread owns the model; by design |

## 8. Open questions for the operator

1. **Exactness policy.** Exact-only (batch invariant, max 8 streams) or also an opt-in non-exact mode for more than 8? How many concurrent users do you actually expect on this machine? (If it is strictly single-user, [levers] section 6 says to swap rank 2 and rank 3.)
2. **Prefill chunk under load.** 2048 keeps the golden baselines and today's prefill speed but stalls every running stream about 0.5 s per chunk; 256 stalls about 70 ms but costs about 7% prefill speed and needs new golden baselines (`recipes/qwen3.toml:50-53`). Or a smaller chunk used only while sequences are decoding (then the identity reference must use the same schedule)?
3. **Default and maximum `--max-batch`, and slot capacity.** Proposed default 1, recommended 4-8. One uniform slot capacity (`--max-seq`, default 4096) or a mixed layout (for example 7 x 4k + 1 x 32k)? Memory allows either (section 3.3).
4. **Merge order.** May `metal-g0-multirow`, `metal-g0b-attn` and `metal-spec-ngram` be merged into `spike/metal` before M0 (they are local, unpushed)? Batching depends on all three.
5. **Prefix sharing.** Copy a cached prefix from any slot (shared system prompts across users), or only reuse an idle slot's prefix? (The first is cheap, section 3.5, but needs the strided copy.)
6. **Fairness policy.** FIFO is proposed. Do you want priorities (`background`) or shortest-prompt-first for TTFT p50?
7. **Speculation default under load.** Auto-disable at B >= 4 as proposed, or keep `--spec` as a strict opt-in that also turns batching down?
8. **Alignment remedy.** +2.75 GB resident and +3% at B = 8 (M4): wanted?
9. **Scope of models.** Batching is dense, attention-only Qwen3-8B BF16 on Metal. The 35B-A3B NVFP4 MoE (phase C) changes the weight-read model (experts) and needs its own estimate; confirm it is out of scope here.
10. **Decision record.** When the design is accepted, write `docs/decisions/0010-continuous-batching.md` (ADR numbering continues from `0009-native-metal-backend.md`) with: exact-only, N-states v0, chunk policy, `max_batch` default.

### 8.1 Operator decisions (2026-10-15 11:56 EDT)

| Q | Decision | Consequence |
|---|---|---|
| 1 | **Exact only.** | Every batched row must be bitwise equal to the solo run. The batch cap is 8. The non-exact NAX mode (M6) is out of scope. |
| 2 | **256-token prefill chunk under load.** | While other sequences are decoding, a new prompt is prefilled in 256-token chunks (about 70 ms stall per running stream instead of about 0.5 s). Solo prefill keeps the 2048 chunk from P3. The identity reference for a batched request is therefore a solo run with the same chunk schedule. Golden baselines for the 256 schedule are needed at M2. |
| 4 | **Yes, merge first.** | `metal-g0-multirow`, `metal-g0b-attn` and `metal-spec-ngram` go into `spike/metal` before M0. The merge is handled by the owner of `spike/metal`, under its own merge bar (G2). |
| 8 | **Shelve the alignment remedy.** (User decision relayed by the Metal spike session, 2026-10-15 14:20 EDT.) | `metal-weight-realign` (8b83805, opt-in `INTARSIA_METAL_REALIGN=1`, +3.8 GB, T(8)/T(1) 1.197 -> 1.152) stays unmerged. M4 is dropped unless batching at 8 streams becomes the common case. |
| 3 | **`--max-batch` default 1 (opt-in).** (2026-10-15 16:58 EDT) | Batching is off unless requested (`--max-batch 2..8`); default serve behaviour is unchanged. Slot capacity layout (uniform vs mixed) is still open. |
| 5 | **Prefix reuse from idle slots only.** | No strided KV copy from active slots. A shared system prompt across concurrent users is prefilled again per slot. |
| 6 | **FIFO admission.** | No priorities and no shortest-prompt-first. |
| 7 | **Speculation turns off automatically at B >= 4** (configurable). | `--spec ngram` stays on at low load, and the scheduler stops drafting while 4 or more sequences are active. |

| 9 | **MoE and recurrent/hybrid models out of scope.** (2026-10-15 19:29 EDT) | `BatchEngine` refuses them at load. Batching them would need its own estimate and decision. |
| 10 | **ADR-0010 accepted.** | `docs/decisions/0010-continuous-batching.md` (branch metal-cb-adr). |

Only the slot layout part of 3 is still open.

## 9. Sources and confidence

Repository (measured; read for this document): overnight report `docs/results/2026-10-15-metal-overnight-report.md` (decode, prefill, TTFT); `[levers]` (sections 1, 2, 5, 6, 7); `[g0]` (multi-row GEMV costs, bitwise tests); `[g0b]` (multi-row attention, small ops, verify-step estimate, alignment remedy); `[g1]` (verify path design, `exact_rows`, `rows_exact`, `Phase::Verify`, identity method);
`docs/results/2026-10-13-prefix-reuse.md`; `docs/results/2026-10-15-metal-serve-prefill.md`; `docs/decisions/0009-native-metal-backend.md`; `docs/ARCHITECTURE.md` lines 61-65, 221, 229-231; localeval `speed.py`; in-repo `docs/benchmarking.md`.

External references (not part of any number above):

* **mlx-lm 0.31.3, local copy** (`~/.local/share/uv/tools/mlx-lm/lib/python3.11/site-packages/mlx_lm`, read): `BatchGenerator` is its continuous batcher (`generate.py:1486`), defaults `completion_batch_size = 32`, `prefill_batch_size = 8`, `prefill_step_size = 2048` (`generate.py:1507-1509`); `step()` decodes first and returns early when the generation batch is full ("our hands full with decoding", `generate.py:1782`), then admits prompts; its KV is a left-padded contiguous batch tensor, `BatchKVCache` (`models/cache.py:912`). Same decode-first admission shape as here; different KV layout (padding instead of per-row visible length). Batch invariance of its kernels was **not checked**. Confidence high for what the code says, none claimed for numerics.
* **Orca**, iteration-level scheduling (Yu et al., OSDI 2022) and **PagedAttention / vLLM** (Kwon et al., SOSP 2023, arXiv:2309.06180): the origin of per-iteration admission and of block-table KV. Confidence high on the concepts; not re-fetched in this session.
* **Sarathi-Serve**, stall-free chunked prefill that fuses decode tokens with a prefill chunk (Agrawal et al., OSDI 2024, arXiv:2403.02310) and the **vLLM V1** scheduler, a single token budget (`max_num_batched_tokens`) over decode and prefill tokens (vLLM blog "vLLM V1", 2025). This is the design our exactness requirement forbids in v0 (section 2.4). Medium-high confidence; not re-fetched.
* **llama.cpp** server: `-np/--parallel N` slots with continuous batching over a shared KV buffer (`--cont-batching`, on by default in recent builds). Medium confidence; no source tree on this machine (only Homebrew binaries); not re-fetched.
* **Thinking Machines, "Defeating Nondeterminism in LLM Inference"** (batch invariance as a kernel property; https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/). Medium confidence; the premise of section 4.4 rests on this repo's own bitwise tests, not on that article.
