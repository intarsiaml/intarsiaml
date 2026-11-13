# Lane interleave: cap the 2-stream ITL tail (prefill vs decode interleave)

Date: 2026-11-13. Branch `prefill-decode-interleave`, off origin/main `10b235785` (#353).

## Brief (copied verbatim-in-substance so it survives compaction)

**Goal.** With 2 streams (`--max-batch 2 --prefix-store on`), Qwen3.6-35B-A3B-NVFP4 on a DGX Spark GB10 matches vLLM on TTFT and decode, but the inter-token tail is worse.

Window 3 baseline (48-request interleaved replay, seed 0, 2 streams, spark2):

| | ITL p99 median | ITL p99 worst | TTFT sum | decode/stream | combined | wall |
|---|---|---|---|---|---|---|
| vLLM | 304 ms | 884 ms | 77.8 s | 39.8 tok/s | 58.9 tok/s | 129 s |
| intarsiaml | 370 ms | 1412 ms | 73.4 s | 39.8 tok/s | 57.4 tok/s | 132 s |

Data: branch `ab35c`, `docs/results/2026-11-13-ab35c-tools-store-sampled.md` "Window 3", `raw/D-store-*`, `raw/serve-D-store.log` (worktree `/opt/code/intarsiaml/.worktrees/ab35c-lane`).

Cause: worst gaps = the other stream's uncached prefill running while this stream waits (c2t1 p99 1412 ms vs c3t1 8k-uncached prefill TTFT 1.49 s; c1t3 1312 ms vs ~5k uncached TTFT 1.31 s). Second source: store settle after Done, 170-410 ms per settle (256-602 MiB at ~1.5 GB/s).

**Target:** worst ITL p99 <= ~900 ms, median p99 <= ~310 ms; TTFT sum no worse than +5%; decode tok/s not lower; batched store-on output bitwise equal to single-engine store-on.

**Scope.**
1. Find how the batched engine schedules prefill chunks vs the other slot's decode; measure chunk size / per-chunk time with store on (packed + macro prefill are off with store on).
2. Cheapest fix capping the stall (decode step between prefill chunks / smaller chunk while another slot decodes / decode rows in the chunk). Keep chunk boundaries deterministic (bitwise gate; CUDA is chunk-sensitive). Document trade-off.
3. Reduce settle stall (split across decode steps or move download off scheduler thread). Fold in Fable #353 follow-ups: (a) `fail_active` (scheduler.rs ~905-916) never settles when it empties `active`; (b) request cancelled while queued (`slot == None`) still triggers settle (batch.rs ~683-691): skip it.
4. Conflicts: `prefix_store.rs` is being changed on `store-writer-bound` (keep edits small; rebase if it merges first). Lane `batch-fu` changes batch.rs startup/macro workspace, serve.rs, docs.

**Gates.**
- CPU: `cargo test -p intarsia-models --test batch_engine`, `--test prefix_store`, `-p intarsia-core --test scheduler`, `-p intarsia-core --lib`. New CPU test: decoding slot advances during other slot's long prefill.
- CUDA spark1: build via `REPO=/opt/code/intarsiaml/.worktrees/interleave ABL='~/intarsia-interleave' BHOST=spark1 FLEETQ_OWNER=claude-interleave fleetq run --host spark1 --load cpu --mem 24 --expect 20 --label "interleave: build" -- bash /opt/code/intarsiaml/.worktrees/ab35c/build.sh`; then `INTARSIA_TEST_REAL_MODEL=1 INTARSIA_PREFIX_STORE_DEBUG=1 cargo test --release -p intarsia-models --test prefix_store_cuda` under `fleetq run --host spark1 --quiet gpu --mem 45` (see `docs/results/2026-11-13-batch-store.md`). Never stop spark1 vLLM.
- Replay window spark2: <=35 min vLLM down, template `scripts/ab35c/window3.sh` (branch ab35c) copied to `scripts/interleave/window.sh`. Arms: main `10b235785` vs this branch, both `--max-batch 2 --prefix-store on`, replay2 interleaved seed 0 K=2 (K=3 if time). `FLEETQ_OWNER=claude-interleave fleetq run --host spark2 --takes-down retro --quiet gpu,cpu --mem 58 --expect 45 ...`; trap restores vLLM. Standing OK for <=60 min windows.

**Rules.** Real checks; tool calls <=4 min; fleetq status first, always `--expect`, no `--force`/`--no-queue`; GB10 memory watch (kill if avail <10 GB); commit trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; PR body ends with Claude Code line. Finish: DRAFT PR against main, no merge; report before/after table, gates, PR link.

## Progress log

### 1. What the scheduler does today (orientation, 2026-11-13 morning)

- `Scheduler::step` (crates/intarsia-core/src/scheduler.rs) already runs one decode step, then one prefill chunk, per iteration (`busy_chunks_per_iter` = 1). With the store on, the chunk length is the request's own store schedule (`SeqAdapter::chunk_len`: the model chunk 2048, cut at message boundaries). So decode **does** run between chunks; the ab35c note "the batched engine does not run a decode step between prefill chunks" is wrong.
- Window 3 server metrics (`raw/serve-D-store.metrics` minus `.metrics.before`, branch ab35c): 253 chunks, 271,856 tokens (mean 1074 rows per chunk); chunk *call* time 20–50 ms for most (the call only enqueues: CUDA runs it asynchronously and the next decode step's sync absorbs it); `intarsia_batch_itl_seconds` (server, between consecutive tokens of one request): 0 gaps above 0.5 s, 104 in 0.25–0.5 s, 61 in 0.15–0.25 s, 7383 at or below 0.15 s.
- **The 1.2–1.4 s client "p99"s are a metric artifact, not 1.4 s stalls.** localeval's `itl_stats` (`/opt/code/deepeval/src/localeval/client.py`) divides each SSE chunk's delta by a character-count weight (`n_tokens * chars / total_chars` when chunks != tokens), and its p99 is `ordered[int(0.99 n)]`, the second-largest of ~159 values. A ~330 ms stall that lands on a 1-character chunk (weight ~0.23) reads as ~1.4 s. So worst-request p99 ≈ (largest real gaps) / (weight of the token they land on): to get worst <= ~900 ms reliably, every server-side gap must be about 200 ms or less.
- The real gap sources: (1) one store chunk's GPU time (2048 rows ≈ 330 ms; boundary-cut chunks shorter), (2) settle after `Done` on the scheduler thread (170–410 ms), which also sits in front of the next admission (TTFT).

### 2. Design (chosen)

- **Layer-boundary pause of a prefill chunk** while another slot decodes: the chunk's thread parks between layers (residual stream saved and restored, model pass flags saved/cleared/restored), the scheduler runs the partner's decode step on the same stream, then the chunk resumes. Chunk boundaries, kernels and inputs are unchanged, so bits cannot change; pause points may even depend on timing without breaking reproducibility. Rejected: smaller busy chunk for store requests (changes bits vs single-engine store-on and splits chunk-keyed entries), decode rows in the chunk (changes GEMM row counts on a chunk-sensitive backend), the overlap twin (refused with the store; rb3 measured chunks ~55% slower beside decode, would break the TTFT gate).
- **Settle in slices**: `finish` only marks a settle pending; the scheduler demotes a bounded number of bytes per iteration after a decode step's sync (stream idle) and only in iterations with no prefill in progress, and settles fully when idle (covers Fable follow-up (a): `fail_active` emptying `active`). Queued-cancel (`slot == None`) never marks one (follow-up (b)).

