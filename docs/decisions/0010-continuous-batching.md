# ADR-0010: Continuous batching on Metal is exact, N independent sequences, opt-in

> **Update (2026-10-21, docs-accuracy pass, see `docs/decisions/LOG.md`):** C6: section 3 describes the first speculation rule. Final behaviour: `--spec-off-at` (default 4, the user's Q7) still holds, and a busy-probe gate additionally stops drafting for low-acceptance requests while other sequences decode, at B = 2 and 3 (`docs/results/2026-10-16-metal-spec-busy-probe.md`, merged `d793300e`). No record says the user changed Q7. C7: the weight-realign remedy stays shelved and unmerged (design Q14, 2026-10-16 17:50, closed M4 as no-go); the "dropped (Q8)" wording is kept as written. C16: an absent `--max-batch` means 1; a scheduler queue-full refusal is 503, the server's own admission is 429.

Status: Accepted (2026-10-15 19:29 EDT; the operator accepted this ADR, design doc question 10, and confirmed that MoE and recurrent models are out of scope, question 9). Decisions marked **[operator]** were taken by the operator (design doc `docs/research/continuous-batching-design.md` section 8.1). Decisions marked **[proposed]** are the design's recommendation, adopted with this ADR.

Context: `intarsia serve` runs one request at a time. One worker thread holds one `SequenceState` for a whole generation (`crates/intarsia-core/src/engine.rs`, design section 1), so N concurrent clients get one stream's throughput (about 36 tok/s decode, estimate). The design (`docs/research/continuous-batching-design.md`, branch `metal-cb-design`) proposes an iteration-level scheduler over N slots. Its premise is that the row-exact multi-row GEMV kernels (G0) and multi-row attention (G0b), already merged into `spike/metal` (operator Q4, done), make every batched row bitwise equal to the solo decode step. Milestones M0 and M1 step 1 are built and measured (`docs/results/2026-10-15-metal-cb-m0.md`, `docs/results/2026-10-15-metal-cb-m1-step1.md`); the scheduler core and the identity harness exist (`docs/results/2026-10-15-metal-cb-scheduler.md`, `docs/results/2026-10-15-metal-cb-identity-harness.md`). Host: endor, M5 Max, Qwen3-8B BF16.

| # | Decision |
|---|---|
| 1 | Exactness contract: each request is byte-identical to its solo run under the solo-reference chunk schedule [operator, Q1 and Q2] |
| 2 | Architecture: `Seq` state machine, FIFO scheduler over N slots, `Model::forward_batch` with shared B-row projections and per-row attention [proposed; Q6 FIFO operator] |
| 3 | Policy defaults: `--max-batch` 1 (opt-in), cap 8, 256-token chunks under load, idle-slot prefix reuse only, speculation off at B >= 4 [operator, Q1, Q2, Q3, Q5, Q7] |
| 4 | Scope: Metal only; CUDA and HIP unchanged; backends without row-exact kernels refuse batching [proposed; standing constraint] |

## 1. Exactness contract

**Decision.** Batching is exact only [operator, Q1]. Row r of a batched step is bitwise what the solo 1-row step computes for that row, independent of B, row order and the other rows. The reference for a batched request is the same request run alone with (1) its slot preloaded with the KV the batched run reused, (2) `--prefill-chunk C` for the C its first chunk used, (3) nothing else submitted (`2026-10-15-metal-cb-scheduler.md`, "Solo reference"). C is 2048 if no sequence is decoding when the first chunk runs, else 256 [operator, Q2], and is fixed per sequence for its whole prompt. Chunking follows `chunk_schedule`, which on Metal never leaves a 1-row tail: `remaining == C + 1` is cut `(C-1, 2)` via `Backend::avoid_single_row_prefill` (`2026-10-15-metal-chunk-no1.md`), because a 1-row chunk runs the decode kernels and is not bitwise a prefill row. The non-exact NAX mode is out of scope [operator, Q1].

**Evidence.** Measured: `Model::forward_batch` rows are bit-identical to the solo step on 138 (row, step) pairs at B = 2, 3, 4, 6, 8, contexts 5 to 4000, greedy and sampled rows, logits compared by bits (`2026-10-15-metal-cb-m1-step1.md`). Measured: the scheduler's randomized test (864 requests, B in {1,2,3,4,6,8}) equals the solo reference, on a mock model (`2026-10-15-metal-cb-scheduler.md`, so it proves scheduler logic, not kernels).

## 2. Architecture

**Decision** [proposed]. A new `BatchEngine` behind the existing `Engine` trait, built only when `--max-batch N` with N > 1; `LocalEngine` stays the path for `max_batch = 1` and every other backend. One scheduler thread owns the model and N independent `SequenceState`s (one KV buffer set per slot, no arena, no paging). Per-request logic lives in `Seq` (M0, `2026-10-15-metal-cb-m0.md`). Each iteration is decode-first: all Decoding sequences advance one token in one `forward_batch`, then at most one prefill chunk of the oldest Prefilling sequence. `forward_batch` runs projections, norms, FFN and head once on B rows through the row-exact multi-row kernels, and `QkNormRopeKv` plus `AttnDecode` once per row on that row's own KV, using the serving kernel. Prefill chunks are a separate pass; decode rows are never folded into the prefill GEMM. Admission is FIFO by free slot [operator, Q6], with no preemption.

## 3. Policy defaults

* `--max-batch` default 1, opt-in 2..=8 [operator, Q1, Q3]. Slot capacity layout (uniform vs mixed) is open.
* Prefill: 2048-token chunks when nothing is decoding (today's speed and goldens), 256 while others decode [operator, Q2].
* Prefix reuse from idle slots only (longest common prefix, else LRU); no KV copy from active slots [operator, Q5].
* Speculation (`--spec ngram`) runs at low load and stops drafting at B >= 4, configurable [operator, Q7].
* Alignment remedy (`metal-weight-realign`) shelved, M4 dropped [operator, Q8].

## 4. Consequences

Positive. Measured, one batched step vs B solo steps at 1k context: 1.00x / 1.96x / 3.7x / 5.69x at B = 1 / 2 / 4 / 8, i.e. 206.3 tok/s at B = 8 (38.78 ms per step) against 36.1 at B = 1 (`2026-10-15-metal-cb-m1-step1.md`); the design estimated 218 tok/s, so reality is 5 to 6 percent lower at B = 8 (per-row attention dispatches). Batch-invariance keeps the D1 identity gate, golden baselines and lossless speculation meaningful, and "same prompt, same bytes regardless of load" holds. CUDA and HIP behave as before.

Negative.
* **Throughput ceiling at B = 8 under exactness.** R = 16 exact rows cost 2.57x per layer, so B = 16 is no better than B = 8 (estimate, design section 4.6). More users than 8 queue.
* **256-chunk prefill cost.** About 7 percent slower prefill for a request prefilled under load, and a request keeps 256 chunks if the decoders finish mid-prompt (estimate from measured chunk rates; scheduler doc). Golden baselines for the 256 schedule are still owed at M2.
* **No shared-prefix copy.** A system prompt shared by concurrent users is prefilled once per slot.
* **Speculation off at load** (B >= 4).
* Realised throughput will be below the step numbers: prefill, scheduler and host gaps are not yet measured (design applies 0.7 to 0.85, estimate).

## 5. Alternatives considered

* **Non-exact NAX mode for B > 8** (NAX GEMM above 8 rows, tiled f32 logits). Estimated 261 vs 218 tok/s at B = 16, 1k (+20 percent), but output depends on B and the tile pick, which breaks the identity gate, goldens and lossless speculation. Rejected [operator, Q1].
* **Paged KV or a shared arena.** Memory is not binding (about 49 GB KV budget, estimate); per-row launches cost at most 6 to 8 percent of a B = 8 step (upper bound, estimate). Not built; the arena stays optional (M5), gated on profiling.
* **Decode and prefill rows in one GEMM** (Sarathi-style). Saves roughly one decode pass per mixed iteration (estimate), but the NAX GEMM is not bit-identical to the GEMV, so exactness forbids it.
* **Priorities or shortest-prompt-first.** Rejected for v1 in favor of FIFO [operator, Q6].

## 6. Open items

* Q9 (decided, operator 2026-10-15): the 35B-A3B NVFP4 MoE and recurrent/hybrid models are out of scope (batching is dense, attention-only; `BatchEngine` refuses them at load). Batching them would need its own estimate and decision.
* Q10 (decided): this ADR is accepted.
* Slot capacity layout (uniform `--max-seq` vs mixed), part of Q3.
* 256-schedule golden baselines (M2); chunk invariance of prefill is shown on Metal only for chunks of at least 2 rows.

## 7. Verification gates

* **Bitwise batch test**: `Model::forward_batch` rows equal the solo step by bits (met at step 1: 138 pairs).
* **Concurrent identity harness**: at least 200 requests per B in {2, 3, 4, 6, 8}, staggered arrivals, greedy and sampled, `min_tokens`, stop strings, EOS, cancels; every output (token ids, finish reason, `completion_tokens`) equals its solo reference, 0 mismatches. The harness (`tools/cbidentity.py`, 203 cases, token-id comparison) is built; against a serial server it passed 49 cases at B = 2, 4, 8 and caught injected differences. It has not yet run against a batched server (`2026-10-15-metal-cb-identity-harness.md`).
* **Throughput** (localeval `--sizes 1024 --gen 512 --concurrency 1,2,4,8`): aggregate at least 1.6x / 3.0x / 4.5x of the serial baseline at N = 2 / 4 / 8; decode-window at N = 8 at least 175 tok/s (met by the bare step, 206 tok/s, measured); N = 1 unchanged within 2 percent. Thresholds are about 85 percent of the design's estimates.
