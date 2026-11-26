# Inkling-2-Mini HIP, package B (mix + projection), lane ink-hip-mix, 2026-11-25

Branch `ink/hip-mix`, merged `ink/hip` (skeleton `6c5ee5d67`, on origin/main `d8c3c7148`).
Ops: SconvRes 69, AttnResMix 67, LoopBlend 68, ConvMix 74, InkInProj 73 (+ `_rows` twin).

## Briefs received (kept against context loss)

- Shared brief: `/opt/code/intarsiaml/.worktrees/hip-swarm/COMMON.md` (ownership, correctness bar, tatooine rules: windows <= 20 min,
  <= 3 per lane, none after 06:00 EST, radiance up by 07:00, report once at pass or 06:30 EST).
- Lead update 22:00 EST: `git merge ink/hip` now; module in the exact shape `TABLE` / `OPS` / `all()` /
  `#[cfg(ik_hip)] pub(crate) mod launch { extern "C" {..} pub(crate) unsafe fn launch(kernel, &InkArgs) -> Option<c_int> }`;
  per-file `extern "C"`; do not touch `UNIMPLEMENTED` or shared tests; **skip MoeRoute** (lead wrote it); correctness forms first
  (`ink_row_kernel` / blend, `in_proj_gemv_kernel` bf16 arm); no `.row_table()`, no `.act_quant()`.

## Status

(open)
