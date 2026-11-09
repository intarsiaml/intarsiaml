# DSpine tap replay: our port vs Tencent's MLX reference on real 27B taps (lane dspine, 2026-11-08)

Question (from `2026-11-08-dspine-cats.md`): is DSpine's lower short-prompt acceptance than DFlash2 a port bug?
Method: our engine (Metal, endor, Quark 27B MXFP4, `--spec dflash --spec-k 7`, greedy, 256 tokens) dumps every tap row the draft ingests and every
draft pass (`INTARSIA_DSPINE_DUMP`, opt-in). `tools/dspine/replay_taps.py` rebuilds the draft context from those taps in Tencent's `dspine_mlx`
(@6450af7, bf16 embed/lm_head from the target checkpoint) and runs the reference block for each pass. Acceptance for both drafts is counted
against the target's real greedy tokens, capped at k=7. One prompt per category file plus the 4420-token prompt, fp8 head (default) and bf16 head.

## Verdict: the port matches the reference

| head | passes | row agreement | passes identical | t/p ours | t/p reference | engine counter |
|---|---:|---:|---:|---:|---:|---|
| fp8 (default) | 614 | 98.5 % | 95.0 % | 3.318 | 3.316 | equal on all 8 prompts |
| bf16 | 599 | 98.0 % | 95.2 % | 3.392 | 3.389 | equal on all 8 prompts |

* Per prompt the t/p difference between ours and the reference is at most 0.017; the 1.5-2 % differing rows are bf16 noise (layer rel-L2
  0.65-1.0 % on the layer dump, the same as MLX GPU vs MLX CPU on the same pass).
* An independent recompute from the raw dumps (Opus verifier, own script) reproduced every cell and found the engine's spec counters equal to
  the replay on all 16 runs.
* So the short-prompt gap to DFlash2 is the model on this target (27B, MXFP4, k=7), not our code. Tencent's DSpine-over-DFlash2 result is for
  Qwen3-8B, both drafters retrained on Open-PerfectBlend, block 16, bf16 target, SGLang; it makes no 27B k=7 claim, so there is no contradiction.
* fp8 vs bf16 head moves per-prompt t/p by -0.30..+0.30 with mixed sign (the target's head switches too, so the token streams differ): no
  consistent head effect.

Not covered: the tap point and layer numbering are checked by code reading only (engine `model.rs:3629-3634` residual after 0-based layer li;
reference hooks `layers[lid]` for [1,16,31,46,61]); the replay feeds our own taps. In the bf16 tcp-handshake run 6 tokens came from target
forwards outside any DSpine pass; both counters exclude them. Per-prompt tables and the raw report: `2026-11-08-dspine-tap-replay/`.
Dumps (2.6 GB) were not kept.

## Consequence

No port fix will recover DSpine's short-prompt acceptance. DSpine stays opt-in; DFlash2 k7 stays the better drafter for short prompts on this
target. The speed levers and tp=2 plan in `docs/plans/2026-11-08-dspine-speed-tp2.md` only pay where DSpine already wins: long prompts, which
need a long-prompt served sweep before any further DSpine work.
