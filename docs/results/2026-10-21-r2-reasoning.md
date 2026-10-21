# R2: reasoning parser for thinking models (2026-10-21)

Lane r2, branch `lane-r2/reasoning-parser`, base `0294b778`. Design and behaviour: docs/structured-output.md#thinking-models.

## Summary

| Check | Result |
|---|---|
| e2e, Metal (endor, Qwen3-8B BF16), thinking on + `json_schema`, 50 requests at concurrency 4 (fleetq #3226) | **50 / 50** valid JSON after `</think>` (jsonschema Draft 2020-12), **50 / 50** non-empty `reasoning_content` (1052 / 2617 / 15260 chars min / median / max) |
| Same, Anthropic `thinking` enabled + `tool_choice: any`, 5 requests (fleetq #3248, rerun) | **5 / 5**: `thinking` block first, then valid `tool_use` input(s) |
| Thinking-off greedy ids vs main `0294b778`, same prompt, 200 tokens (#3226) | **identical** x4: thinking off (`da5fa413`), no kwargs (`67e00b4b`), thinking on unconstrained (`67e00b4b`), constrained thinking-off ids + content (`68639274`) |
| Unit + server API tests (`cargo test -p intarsia-server -p intarsia-core`, endor, non-exclusive) | pass: server api 37, structured unit 16, core 89 |

Raw files: `raw/` (`e2e-c4.*`, `e2e-anthropic.*`, `ids-*.txt`, `run.log`, `anth.log`, `sample-thinking.json`).

## What changed

- `structured.rs`: `ReasoningParser` (splits at `</think>`, any token chunking, trims the whitespace around the tags,
  drops a leading `<think>`); `compile_with(plan, think)` gives the llguidance constraint a gate on the single
  `</think>` token: inactive (device argmax, nothing masked, `<tool_call>` ignored) until it is committed, then the
  normal grammar (or the tool trigger). 400 if the tokenizer has no single `</think>` token.
- `openai.rs`: `message.reasoning_content` / `delta.reasoning_content`; tool calls are parsed from the content part only.
- `anthropic.rs`: `thinking` blocks (non-stream and `thinking_delta` + empty `signature_delta` streaming).
- On only when the request sets thinking on AND the recipe has a `reasoning_parser`. Everything else takes the old code
  path (the no-op is unchanged; see the ids above). J1's no-spec-under-constraint and T1's `feedback_ok` constraint
  exclusion are untouched: a gated constraint is a constraint from step 0.

## Notes and limits

- **Opt-in, not default.** A Qwen3 request that does not say `enable_thinking: true` (template default is thinking on)
  keeps today's output: `<think>` stays in `content`, constrained requests answer directly. Parsing by default (as vLLM's
  `--reasoning-parser` does) would change `content` for existing clients, so it is a decision for the user (LOG.md row).
- The e2e `ids` rows for "no kwargs" and "thinking on" are equal because Qwen3's template defaults to thinking.
- `budget_tokens` is accepted, not enforced. A think section that never closes before `max_tokens` leaves empty `content`
  (a 1500-token `17*23` request did exactly this: `finish_reason: length`, reasoning only; `raw/sample-thinking.json`).
  Thinking requests need a generous `max_tokens`; the e2e used 6000.
- The first Anthropic e2e run (#3226) got 3 / 5: the two streamed cases failed in the e2e script, which concatenated the
  `input_json_delta` of two parallel tool calls into one JSON string. The script now accumulates per block and the 5
  requests were rerun against the same code (#3248, head `c9c40d46`; only the script and a test-only `#[cfg]` differ
  from the binary of #3226, `06be1f1e`).
- Not done: `/v1/completions` reasoning (chat only), `/v1/responses`, per-delta `token_ids` in a streamed reasoning
  request, the cross-backend no-op checks (the coordinator runs those), CUDA / HIP e2e (Metal only, as asked).
- Fleet use: endor only, no role taken down, no downtime.
