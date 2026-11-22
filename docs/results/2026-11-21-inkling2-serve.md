# Inkling-2-Mini through the OpenAI server (lane ink-serve, 2026-11-21)

Status: in progress. Branch `ink/serve` (not pushed, not merged).

## Briefs received

- 2026-11-21 (launch): serve Inkling-2-Mini through `intarsia-server` like vLLM does (`--reasoning-parser inkling`): chat template, `reasoning_content`, tool calls, `response_format` json_schema, effort knob; plus the confident-row metric in `intarsia accuracy`. Worktree `.worktrees/ink-serve`, branch `ink/serve` from 1dacac3a8; merge `ink/metal` before Metal checks; no push, no merge to main; every endor run through fleetq (real model `--mem 25`).
