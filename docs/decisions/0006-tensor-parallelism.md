# ADR-0006: Tensor parallelism (tp=2) and the comm abstraction

> **Update (2026-10-21, lane G1c, `docs/results/2026-10-21-tp2-cuda-graphs.md`):** "CUDA graphs are off when `world > 1`" (below) is
> superseded. Decode graphs run at world > 1 when the group's communicator is capturable (`Comm::capture_safe`: NCCL yes;
> the host comm and HIP's `P2pComm` no, so they stay eager). Rank 0 decides per step and the remote rank follows
> (`Forward` flag `FWD_GRAPH`); the NCCL f32 landing buffer is reserved at load (`Comm::reserve`). Fused decode ops under
> TP are a backend capability (`Backend::fusion_under_tp`), opt-in with `INTARSIA_TP_FUSION=1`; the GatedDeltaNet
> out-projection fold stays off under TP (it would swallow the all-reduce).

> **Update (2026-10-21, docs-accuracy pass, see `docs/decisions/LOG.md`):** C15: item 2 of "Accuracy gate" (the self-golden against tp=1) is NOT a TP gate. Its follow-up paragraph (2026-10-14, `docs/results/2026-10-14-tp2-budget.md`, `docs/accuracy.md` "TP self-golden budget") stands: the fp32 floor and the f32 Mac sharding test are the TP gates.

Status: Accepted (Phase 2b, 2026-10-13)
Context: ARCHITECTURE section 5 lists `Comm` implementations (`nccl`, `rccl`, `p2p_oneshot`) and TP preconditions checked at plan time. This ADR records how the first implementation is built. Results: docs/results/2026-10-13-phase2b-tp.md.

## Decision

**One process, one thread per rank.** A tensor-parallel model is `world` full `Model` shards, each with its own backend (device), weights, workspace and KV cache. `Model::forward` on rank 0 runs ranks 1.. on scoped threads (`std::thread::scope`, one per forward call) and runs rank 0 itself on the calling thread. No multi-process launcher, no IPC handle exchange; the engine, server, accuracy and bench code see one `Model`.

**The comm abstraction lives in core and is backend neutral** (`intarsia_core::comm`).

```rust
trait Comm: Send + Sync {            // one rank's handle onto a group
    fn rank(&self) -> usize;  fn world(&self) -> usize;  fn name(&self) -> &str;
    fn all_reduce_sum(&self, x: &TensorRef, stream: StreamId) -> Result<()>;   // in place, bf16/f16/f32
    fn all_gather(&self, send: &TensorRef, recv: &TensorRef, stream: StreamId) -> Result<()>; // rank-major
    fn abort(&self);  fn stats(&self) -> CommStats;  fn note_timed(&self, d: Duration);
}
```

* Collective: all ranks make the same calls in the same order. Stream ordered: on return the result is visible to later work on `stream` (the caller never synchronises).
* Reduction is `f32(a) + f32(b)`, one rounding to the tensor dtype, rank order for more than two ranks. Addition of two operands commutes, so every rank gets the same bits, which is what keeps the per-rank copies of the residual stream identical.
* A rank that fails calls `abort()`; the others return an error instead of waiting. `forward_tp` does this for errors and panics. After an abort the group is unusable (the ranks no longer agree where they are), so a failed TP forward fails every later one.
* `HostCommGroup` (core) is the reference implementation: download, exchange through host memory under a mutex and condvar, sum in rank order, upload. It works on any backend, including the cpu reference, so the whole sharding path is tested on a machine without a GPU. `--comm host` selects it on any backend.
* Device implementations come from the backend crate as a constructor that returns the backends and comms together (`intarsia_backend_hip::create_group`), so the core never downcasts a `dyn Backend`. The HIP implementation is described in the HIP section below.

**Collectives are not ops.** They bypass the kernel registry, admission and `Op::ALL`. During plan recording (`Model::dry_run`) the all-reduce is counted (`PlannedComm`) and not run; that is also what keeps the load-time dry run from waiting on peers that are not running. The plan reports `all_reduce_calls_per_decode_step` and the bytes per rank.

**Sharding (Megatron style).**

| weight | split | note |
|---|---|---|
| q, k, v | rows (column-parallel) | q heads `[nh/w * r, ...)` meet kv heads `[nkv/w * r, ...)`: GQA groups stay on one rank |
| o | columns (row-parallel) | partial `[T, hidden]`, then all-reduce |
| gate, up | rows | |
| down | columns | partial, then all-reduce |
| q_norm, k_norm, layer norms, embedding | replicated | |
| KV cache | by KV head | each rank holds `nkv/w` heads; capacity is the minimum over ranks |
| lm_head | **replicated on rank 0 only** | see below |

`ModelDims::sharded(w)` gives the per-rank dims (`Model::ldims`; `Model::dims` stays the checkpoint's). `n_heads`, `n_kv_heads` and `intermediate` must divide by `w`, checked at load with a message naming the offender. Row slices are a contiguous byte range of the mmap; column slices are gathered row by row on the host and only the slice is uploaded, so a rank never holds the other's part. The shape check against the checkpoint is implicit: the loader divides the checkpoint's shape and `Model` compares with the per-rank expectation.

**lm_head: replicated, on rank 0 only.** The simpler of the two options: no all-gather in the decode path, `HeadMode`s and the accuracy logits path unchanged, rank 1 stops after the last layer (it needs no final norm result and holds no head). Cost: rank 0 reads the whole head (1.24 GB for Qwen3-8B) on every token while rank 1 idles for that time, about 2 ms of a ~13 ms floor. A vocab-parallel head with an `all_gather` of the logits (the comm already has it, tested) is the follow-up if that gap matters; it needs a row-interleave step for multi-row logits.

**Partials are f32 by default.** A row-parallel matmul (o, down) writes an f32 `[T, hidden]` partial (the existing f32-output `logits` op, same kernels), and `all_reduce_sum_into(partial, out)` sums the ranks' partials and rounds once into the bf16 activation. One unsharded matmul accumulates in f32 and rounds once; this keeps that, so the only tp=1 / tp=2 difference is the order of an f32 sum. Cost: the collective moves 16 KiB per decode call instead of 8 KiB. `INTARSIA_TP_PARTIAL=bf16` reduces bf16 partials in place (two roundings, what vLLM does); both are measured in the results doc and neither changes the accuracy verdict.

**State.** `SequenceState.peers` holds the other ranks' slots (`PeerState{backend, state}`); `Model::new_state` creates them with the common capacity; `truncate`/`snapshot` recurse; each rank commits its own tokens in its forward. A forward that failed on some ranks leaves them at different lengths; the next forward truncates peers that are ahead and refuses to run if one is behind.

**Not supported yet.** GatedDeltaNet and MoE blocks (Qwen3.6-35B-A3B), FP8 and NVFP4 weights: sharded loading refuses with `Unsupported` instead of loading unsharded. CUDA graphs are off when `world > 1` (the HIP backend has none anyway). `--tp` on cuda uses the host comm.

## Accuracy gate

TP is gated in two parts, both on the HF fp32 Qwen3-8B goldens (`qwen3-8b-fp32`, `qwen3-8b-long-fp32`):

1. **The fp32 floor.** tp=2 must pass the absolute bf16-class floor (`--gate-class bf16`: prefill and decode top-1 >= 99.5%, full-vocabulary KL <= 1e-3, tail excess <= 0.02) on both goldens. The bf16 goldens were retired as the TP gate: they sit within the engine's own bf16 noise, and tp=1 itself is at 99.51% on the long one, so one extra flip failed tp=2 (99.44%, 8 flips against 7 allowed) without a defect (results doc, section 4).
2. **The self-golden against tp=1** (`--write-self-golden` on tp=1, `--self-golden` on tp=2, same weights, default budgets, no loosening). This is the measure of what sharding itself changes. The floor shows tp=2 is as close to the reference as tp=1; the self-golden shows how far apart tp=1 and tp=2 are from each other.

Measured result (results doc, "Re-gate against fp32 goldens"): the floor passes on both goldens; the self-golden gate with default budgets **fails** (mean 21-way KL 5.9e-4 prefill and 5.1e-4 decode against 1.0e-4). The default budgets were chosen for same-numerics comparisons (one kernel change on one device); a different summation order across two ranks is not that. The gate is kept at the default budgets and the failure is recorded, not tuned away. Any budget for TP needs its own measured noise floor and a separate decision.

Follow-up (2026-10-14, `docs/results/2026-10-14-tp2-budget.md`, `docs/accuracy.md` "TP self-golden budget"): the 5.9e-4 is the size of an equally correct reordering (switching tp=1 to SIMT attention gives 6.1e-4), and a TP budget derived from tp=1's equivalent variants (k = 2) exists, but it does not reliably catch a 1% partial-scale bug, so it is not shipped. The self-golden gate is not a TP gate; the f32 Mac test (sharding and reduction logic with the host communicator, catches 1%) and the fp32 floor (catches 2% and more) are; nothing on the GPU gates catches a 1% model-level bug.

## HIP all-reduce

See the section added to `docs/backends/hip.md` and the results doc for the measured cost.
