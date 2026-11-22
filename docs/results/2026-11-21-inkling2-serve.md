# Inkling-2-Mini through the OpenAI server (lane ink-serve, 2026-11-21)

Status: in progress. Branch `ink/serve` (not pushed, not merged).

## Briefs received

- 2026-11-21 (launch): serve Inkling-2-Mini through `intarsia-server` like vLLM does (`--reasoning-parser inkling`): chat template, `reasoning_content`, tool calls, `response_format` json_schema, effort knob; plus the confident-row metric in `intarsia accuracy`. Worktree `.worktrees/ink-serve`, branch `ink/serve` from 1dacac3a8; merge `ink/metal` before Metal checks; no push, no merge to main; every endor run through fleetq (real model `--mem 25`).
- 2026-11-21 (coordinator, relaying ink/skel 46349d879): the goldens' chat prompts are ChatML text, not Inkling turn tokens, so `accuracy` falls back to the recipe's `chat_turns` and masks no rows. Keep the goldens comparable (same token ids as before); add the Inkling-template path separately. The confident-row metric in `accuracy` must match the skeleton lane's side script: G2 short 0/363, long 1/1206; G3 4k 2/8229.
