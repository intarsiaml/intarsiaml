# Captured spec step at tp>1 (lane gt2, 2026-11-07)

Coordinator go-ahead 2026-11-07 02:15 EST (user: 'keep building overnight'; radiance downtime approved 2026-11-05, max 45 min per window, radiance up by 06:30): captured spec step at tp>1, HIP first, opt-in knob, no default flips; merge on passing gates.

## Why

`--spec mtp` at tp=2 ran its draft and verify passes eagerly: `Model::spec_step_ok` required `tp.world == 1`. Measured tonight:
HIP 35B tp2 MTP k2 only 1.04-1.07x over tp2 spec off at 4k (`2026-11-07-hip35-mtp-tp2.md`, rule wanted 1.15x); CUDA Spark tp2
1.16x (`2026-11-07-mtp-moe-tp.md`). At tp=1 the captured step is engaged (lane ag1). Plain decode at tp2 on HIP already replays
graphs with the collectives inside (`model/graph_step.rs`), so the HIP all-reduce / all-gather are capturable.

## Design (knob `INTARSIA_SPEC_GRAPH_TP=1`, default off)

One recording per rank per (sequence, catch-up rows n, verify rows q = k + 1), all made by rank 0's step, replayed in lockstep:

* **Rank 0** records what it records at tp=1 (`capture_spec_step`): head catch-up over the n queued rows -> draft 1 (argmax into
  `ws.ids[1]`), k - 1 chained head passes, scripted-draft overwrite (tests), the verify pass over `[next, d_1..d_k]` on the row-exact
  kernels (its all-reduces recorded), the verify rows into the head's queue.
* **The drafts reach the peers inside the graph.** The eager path hands draft ids to the peers over a host channel (`DraftLink`),
  which a recording cannot contain. Instead every rank records `spec_gather_ids`: an all-gather of `ws.ids[0..g]` (g = q rounded
  up to 4 ids = 16 bytes, the HIP gather quantum) into spare entries 80..96 of the position table `ws.pos` (fixed address, no new
  allocation), after which a peer copies rank 0's segment over its own `ws.ids[0..g]`.
  * head whole on rank 0 (MoE, 35B, lane mt2): one gather, after the drafts (the peers hold no head and draft nothing);
  * sharded head (dense, 27B): one gather before every chained pass (the peers' shards of that pass embed the draft rank 0's
    argmax just produced) plus the final one: k gathers. Rank 0's head passes and the peers' pair their all-reduces as eagerly.
  Collective count and order are identical on every rank.
* **A peer** records (`capture_spec_peer`): [sharded head: catch-up and chained passes on its shard, gathers between], the final
  gather, the verify pass's layers on the row-exact kernels (its slice of a vocab-parallel head), [sharded head: rows into its
  queue]. Exactly `mtp_draft_tp` (Follow) + `verify_peer`'s launches.
* **Lockstep and failure.** Rank 0 decides (`spec_step_tp_ok`: knob, in-process peers only, every peer graph-capable, peer state
  at rank 0's length, sharded head: same queue and head-cache length), captures its own recording, then each peer on its own thread
  prepares its per-row recurrent copies, uploads the same position table and captures its share if missing; a barrier; then all
  replay, or none if any capture failed (captures execute nothing, collectives included, so nothing ran anywhere and the step runs
  eagerly, counted as a capture failure). A failure after the barrier aborts the communicator (the group goes down, ADR-0006).
* **Host mirrors / rollback.** After rank 0's one download (outputs + the verify ids), every peer gets what its eager passes leave:
  `commit_verify(ids)`, recurrent `spec_rows.saved = q - 1`, sharded head: head cache commits the catch-up, queue = q rows. So the
  existing `spec_rollback` (all ranks, eager) and acceptance run unchanged. Recordings are keyed by rank 0's sequence id and dropped
  with the spec cache (96 entries) as at tp=1.
* Remote ranks (CUDA Sparks, `tp-worker`) stay eager (`remote.is_some()` -> not applicable). Knob unset: `spec_step_tp_ok` is false
  at tp>1 exactly where `tp.world == 1` was, and nothing at tp=1 changes (no gathers recorded at one rank).

CPU evidence: `crates/intarsia-models/tests/tp_spec_graph.rs` runs tp=2 on a graph-recording cpu backend (`RecCpu`) with a
recording communicator (`RecComm`, the host rendezvous runs at replay on each rank's thread): one captured step's drafts and
verify outputs equal tp=1's eager `mtp_draft` + `forward_verify` (k 1..3, replicated and vocab-parallel head, catch-up n = 1 and
n = q), and engine greedy ids with the step equal `--spec off`, with acceptance counts equal to the eager tp=2 engine (own and
scripted drafts, every acceptance pattern through the GatedDeltaNet rollback). The cpu fixture is dense (sharded head); the
whole-head (MoE) path is checked only by the 35B HIP ids gate below, so that gate is hard.

## Pre-registered (written before the window)

(a) **ids (gates merge):** 256 greedy ids, `--spec mtp --spec-k 2 --tp 2` with `INTARSIA_SPEC_GRAPH_TP=1` == `--spec off --tp 2`,
for 35B and 27B, prompts short-0 and 4k-1; the knob runs must show `spec: mtp graph step` trace lines (engaged) and no capture
failure. Plus: plan show JSON identical between main (7489a48aa) and this branch with the knob unset for 8B tp1 off, 27B tp1/tp2
off + mtp2, 35B tp1/tp2 off + mtp2.

(b) **speed (decides recommending a default flip in the morning; no flip tonight):** 35B tp2 MTP k2 with the knob >= 1.05x
without it at 4k, and the knob arm >= 1.15x tp2 off at 4k. Arms ABAB (tp2off / tp2m / tp2mg, then reversed), fresh
`serve --max-batch 1` per arm, `localeval speed --sizes 1024,4096,16384 --gen 128 --reps 3`; 27B tp2 mtp2 vs mtp2 + knob at 4k
(reported, no rule).
