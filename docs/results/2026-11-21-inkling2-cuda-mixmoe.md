# Inkling-2-Mini CUDA in-projection, mix / conv and MoE kernels (lane ink-cuda-mixmoe, spark2 GB10)

Branch `ink/cuda-mixmoe` (worktree `.worktrees/ink-cuda-mixmoe`, base `1dacac3a8`). Not pushed or merged into main.

## 0. Briefs received

- (coordinator, during the lane) Report status: which of AttnResMix, LoopBlend, MoeExpertsMxfp8 (+NVFP4 situ), InkInProj, ConvMix
  are ADMITTED on CUDA, with an ETA for the rest. As soon as all are admitted, merge `ink/cuda-attn` (tip at or after `db002e2fa`)
  into `ink/cuda-mixmoe` and report the ref; when merging, extend `inkling2_tiny_no_kernel_refusal_names_the_op` (cuda-attn's
  `db002e2fa` expects CUDA kernels for exactly the inkattn TABLE ops) to this lane's ops.

- (coordinator, 2nd) Merge plan approved; keep the uncommitted perf work out of the merge. After the merge and the tiny test,
  without waiting: (1) tiny-model gate for cuda-attn's `d536ff099`: logits / tokens bitwise with the CUDA fused InkAttnDecode rule
  on vs forced to the pair, reported with the ref; (2) the first real Inkling run on CUDA on spark2 (checkpoint copy there is
  hash-identical to endor's): G1 / G2 / G3 per the spec gates, then greedy decode tok/s at short and 5k with profiling off; budget
  memory per the DGX rules (weights + KV + ~10 GB OS + build headroom against `fleetq status` free), `--mem` = that peak, poll
  `free -g` during the first load, kill below 10 GB available; correctness jobs at default load, timing `--quiet gpu,cpu`;
  (3) report G-gate results and tok/s next to Metal's 63 tok/s. Do not take any vLLM role down.

- (coordinator, 3rd) `ink/metal` `e3f92beff` guards the two `intarsia_backend_metal` uses in `crates/intarsia-models/tests/inkling2.rs`
  behind `cfg(target_os = "macos")`; if this branch merges `ink/metal` at or after `07589bd9c`, take `e3f92beff` too; pass it on to
  ink-cuda-attn with the merged ref. (This branch has its own equivalent guard, commit after the cuda-attn merge; no ink/metal merge
  here.)

(sections below filled in as the lane finishes)
