# lm_head fp8 becomes the default for the Qwen3.8-27B on Metal and HIP

> User 2026-11-06: 27B fp8 default.

> User 2026-11-06: CUDA too.

Lane `lane-lh/lmhead-fp8-default`, worktree `.worktrees/lmhead-fp8-default`, base `origin/main` `af6aa5226`. This is the follow-up to `docs/results/2026-11-05-lmhead-fp8-27b.md`, which shipped the quantised head opt-in. The user's decision overrides the 1.0 % top-1 flip bound that doc set (fp8 measured 2.0 to 2.5 % flips in prefill on both backends and 1.0 to 1.6 % in the Metal decode window; MMLU 0.8368 against 0.8316 and GSM8K 0.52 against 0.48 for bf16).

**Status: in progress.**
