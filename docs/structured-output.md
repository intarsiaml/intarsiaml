# Structured output (constrained decoding)

intarsia enforces OpenAI `response_format`, plus tool schemas on both the OpenAI and the Anthropic API, by masking the
logits at every step. The masks come from [llguidance](https://github.com/guidance-ai/llguidance), pinned at
`=1.9.1` (MIT). That is the engine llama.cpp and vLLM use. Requests intarsia cannot honour get **HTTP 400**; they
are never silently ignored.

## API

| Endpoint | Fields | Output |
|---|---|---|
| `POST /v1/chat/completions` | `response_format`: `{type: "text" \| "json_object" \| "json_schema", json_schema: {name, schema, strict}}` | `message.content` is JSON matching the schema |
| `POST /v1/chat/completions` | `tools` (`type: "function"`), `tool_choice`: `"none" \| "auto" \| "required" \| {type: "function", function: {name}}`, `parallel_tool_calls` | `message.tool_calls[]` (`arguments` is a JSON string matching the function's `parameters`), `finish_reason: "tool_calls"`; streamed as `delta.tool_calls` |
| `POST /v1/completions` | `response_format` (as above) | `text` is the JSON; `tools` returns 400 |
| `POST /v1/messages` | `tools` (`name`, `description`, `input_schema`), `tool_choice`: `{type: "auto" \| "any" \| "tool" \| "none", name, disable_parallel_tool_use}` | `tool_use` content blocks (`input` matches `input_schema`), `stop_reason: "tool_use"`; streamed as `content_block_start` (tool_use) + `input_json_delta` |

Notes:
- `strict` is accepted. The schema is enforced either way.
- `tool_choice` defaults to `auto` when `tools` are given.
- On `/v1/messages`, assistant `tool_use` blocks and user `tool_result` blocks in the history are passed to the chat
  template as `tool_calls` and `tool` messages.

**How `auto` works.** Under `tool_choice: auto` the model writes free text. A call is constrained once the model emits
the `<tool_call>` token, and only until `</tool_call>`; after that the text is free again. Without
`parallel_tool_calls`, only EOS may follow the call. While no call is open, the request decodes exactly like an
unconstrained one, on-device argmax included.

**JSON style.** By default JSON is written compactly with `json.dumps` separators (`", "` and `": "`) and no free
whitespace: a greedy model can otherwise pad with whitespace until it hits `max_tokens`. A schema's own `x-guidance`
object (llguidance's JSON options) overrides this.

**Tool-call format.** The recipe's `tool_parser` picks the dialect, which is the one the model's chat template teaches
and renders history in. Both share the `<tool_call>` / `</tool_call>` tags (and the streaming hold-back); they differ
in the body:
- `hermes` (Qwen3): `<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>`, the body constrained by the call's
  JSON schema.
- `qwen3_xml` (Qwen3.5, Qwen3.6, Qwen3.8, as vLLM's parser of the same name; `recipes/qwen3_5*.toml`):
  `<tool_call>\n<function=NAME>\n<parameter=ARG>\nVALUE\n</parameter>\n</function>\n</tool_call>`, several calls
  separated by `\n`. A string parameter is the raw text up to the first `\n</parameter>` (a lazy llguidance lexeme with
  that stop; an `enum` or `const` string is its members). Any other parameter is JSON constrained to its property
  schema (the template writes it with `tojson`). Parameters come in the schema's property order; `required` ones
  must be there, the others may be skipped. A property with no type is a string. The parser reads a string as it is
  (the `\n` after the opening tag is the template's), and parses a non-string value as JSON by its property's schema, keeping the
  raw text when it is not JSON (Python's `True`, `False` and `None` count as JSON). A string or null union
  (`type: ["string", "null"]`, `anyOf`) is raw text, where the text `null` is null. A tool with no `properties`
  (`{"type": "object"}`) or `additionalProperties` that is not `false` may also carry any other parameter, a raw
  string, which the parser types as JSON when it parses. A string ends at the first `\n</parameter>` followed by a newline
  (or the end of the call), so it may hold a bare `</parameter>`. A parameter name with `>` or a newline is a 400. Not
  enforced: string `maxLength` / `pattern`.

A call that the assistant made earlier in the conversation is rendered by the template in the same format. The
OpenAI API carries its `arguments` as a JSON string; the server hands the template an object when the template needs one
(`ChatTemplate::render`), so the history round-trips byte for byte (a test renders it with the real Qwen3.8 and
Qwen3.5 templates). On other models:
- `required` or a named function returns 400;
- `auto` behaves as before: the tools reach the template and the output stays text.

### HTTP 400 cases

- an unknown `response_format.type`, or a missing `json_schema` or `schema`;
- a schema that is not an object;
- a schema llguidance rejects: an unknown type, a bad regex `pattern`, an unresolvable `$ref`, an unsatisfiable
  schema, or an unsupported keyword;
- a tool that is not a function, or that has a bad name, a duplicate name or a bad `parameters` / `input_schema`;
- `tool_choice` without `tools`, or naming a tool that is not in `tools`;
- `response_format` together with tools (unless `tool_choice` is `none`);
- `ignore_eos` or `min_tokens` together with a constraint (the grammar ends with EOS);
- a model without `tokenizer.json`, or `required` / named tools on a model without a tool-call format (`tool_parser`).

## Design

- **Token table.**
  - Built from the model's HF `tokenizer.json` with `llguidance::token_bytes_from_tokenizer_json`, on the first
    constrained request (about 0.2 s for Qwen3's 151k vocabulary).
  - Padded to the logit width (151936 for Qwen3), so the padding ids are never allowed.
  - Special tokens carry llguidance's marker and never match text.
  - The model's EOS ids (`generation_config.json`) end the grammar.
  - Forced text is tokenized with the model's own tokenizer.
  - Code: `crates/intarsia-server/src/structured.rs`.
- **Engine hook** (`crates/intarsia-core/src/constrain.rs`, `engine.rs`).
  - `GenRequest.constraint` carries a compiled `ConstraintSpec`; each sequence starts its own `TokenConstraint`.
  - A constrained step runs its head in `LastLogits` mode, the host-logits path that sampled requests already use, on
    every backend. Only constrained rows take it. In a batch, each row has its own head mode: the pass computes device
    argmax for every row and downloads logits only for `LastLogits` rows (`Model::forward_batch`). Unconstrained rows
    therefore keep their device argmax next to constrained ones. The cost each constrained row adds to the shared step
    is one row download (V x 4 bytes, about 600 KB for Qwen3), the same cost a sampled row already adds.
  - **Behaviour change for `tools`.** On a model with a tool-call format `tool_choice` defaults to `auto`, so a request with `tools` now
    carries a lazy constraint. It therefore gets no speculative drafts, and it gets a 400 when combined with
    `ignore_eos` or `min_tokens`, even before any `<tool_call>` appears. A request without `tools` or `response_format`
    is unaffected.
  - The mask is applied after the `min_tokens` mask and before sampling (greedy, or temperature / top-k / top-p).
  - The chosen token is committed in `on_token`, after the stop-id check, so EOS is never fed to the grammar.
  - A mask that allows nothing ends the request with an error; the sampler never falls back to token 0.
- **Overlap.** The next step's mask is computed on a helper thread while the forward pass runs:
  - it starts in `Seq::plain_input` for decode steps;
  - it starts in `Seq::admit` for the first token, overlapping prefill.
  - Measured on Qwen3: the worst step per sequence is 15-50 µs (`structured.rs::qwen3_token_table`).
- **Device sampling.** No backend needed a device kernel: the Metal, CUDA and HIP decode paths all have per-row host
  logits. An unconstrained request never touches any of this code, so its device argmax and its greedy ids are
  unchanged.
- **Speculative decoding** (`--spec`, Metal, off by default) is **disabled for constrained requests**: they take plain
  steps only. Drafts are not verified against the mask.
- **Prefix cache.** Unaffected: the constraint starts after the prompt and changes no KV state.

## Unbounded numbers

A schema `{"type": "integer"}` (or `number`) without `maximum` / `minimum` has no length limit in the grammar. A greedy
model that starts repeating digits can then run until `max_tokens`, and the cut-off JSON does not parse. It happened in
3 of 50 GSM8K first calls (docs/results/2026-10-21-j1-structured.md). intarsia leaves such schemas as the client sent them,
as llama.cpp and vLLM do.

## Thinking models

Models with a reasoning parser in their recipe (`reasoning_parser = "think-tags"`: Qwen3, Qwen3.5, Qwen3.6) write
`<think>...</think>` before the answer. intarsia splits it out and keeps the constraint off until the think block is
closed. Whether a request is split follows the template, as vLLM's `--reasoning-parser qwen3` does for every request:
the Qwen3.5+ templates open the block themselves (the rendered prompt ends with `<think>\n`) unless thinking is turned
off, so a stock OpenAI client that sends no `chat_template_kwargs` gets `reasoning_content` and a clean `content`.

**When it applies.** The model has a reasoning parser, and
- `enable_thinking: false` (or `thinking: {"type": "disabled"}`): never split;
- `enable_thinking: true` (`chat_template_kwargs` on `/v1/chat/completions`, or `thinking: {"type": "enabled",
  "budget_tokens": N}` or the same `chat_template_kwargs` on `/v1/messages`): split;
- nothing said: split when the rendered prompt ends with `<think>\n`. A template that closes the block in the prompt
  (`<think>\n\n</think>\n\n`, as the Qwen3.5-0.8B-Base and 2B templates do unless thinking is asked for) is answered as
  generated, and so is a template that does not mention `<think>` (Qwen3 with no flag: the model writes the block
  itself; send `enable_thinking: true` to split it).

The server never fills in `enable_thinking` for the template: the prompt is what the template makes of the request as sent
(and `/v1/intarsia/prefill` renders the same). `budget_tokens` is accepted and not enforced: the think section is
bounded only by `max_tokens`. On a model without a reasoning parser the flag only reaches the chat template.

**Output.** The text before `</think>` is the reasoning, the text after it is the answer. A leading `<think>` is
dropped (newer templates put it in the prompt), and the whitespace around the tags is trimmed. Output without a
`</think>` (cut off by `max_tokens`, or the model stopped) is all reasoning, with empty `content`.

**A call before the close.** With tools parsed, complete `<tool_call>` envelopes that end the reasoning, with `</think>`
after them and nothing but whitespace after that, are the reply's tool calls instead of reasoning (`finish_reason`
`tool_calls` / `stop_reason` `tool_use`), whole and streamed. vLLM's `qwen3` parser returns such calls too. A call followed by
more reasoning or by an answer, an incomplete envelope, and a reply with no `</think>` stay reasoning. While streaming,
reasoning from a `<tool_call>` that could still be such a tail is held back until it is decided: the answer starts, other
text follows the call, or the reply ends.

| Endpoint | Reasoning | Answer |
|---|---|---|
| `/v1/chat/completions` | `message.reasoning_content` (`null` when empty), streamed as `delta.reasoning_content` | `message.content` / `delta.content` |
| `/v1/messages` | a `thinking` content block (`{type, thinking, signature: ""}`) first; streamed as `thinking_delta`, closed by an empty `signature_delta` | `text` and `tool_use` blocks |

This matches vLLM's `--reasoning-parser qwen3` field names. The signature is empty because the thinking is not signed.
Streams send every reasoning delta before the first content delta. `/v1/completions` is not parsed. With
`return_token_ids`, a streamed reasoning request does not report per-delta `token_ids` (the choice-level `token_ids` of
a non-streamed response is unchanged).

**Constraints.** With reasoning split (as above), `response_format`, `tool_choice: required` / a named tool, and `auto` are all gated:
nothing is masked until the model emits the `</think>` token, so the think section is free text (a `<tool_call>` inside
it does not open a call). Then the grammar starts from the first answer token. This is the same lazy-trigger mechanism
as `tool_choice: auto` (`TokenConstraint::active`): until `</think>` the request decodes like an unconstrained one.
The template's usual `\n\n` after `</think>` may come before the constrained text (the grammar takes an optional `"\n\n"`
first); the reasoning parser trims it, so `content` starts with the JSON or the call itself.
- The tokenizer needs a single `</think>` token (Qwen3 has one); otherwise a constrained thinking request returns 400.
- If the model never closes the think block, the output is reasoning only and never constrained.
- J1's rules are unchanged: a constrained request never uses speculative decoding, and `Seq::feedback_ok` still excludes
  constraints. Gated requests count as constrained from the first step.

With the reasoning not split (`enable_thinking: false` / `thinking: {type: "disabled"}`, a closed or absent think block
in the prompt, or no reasoning parser) the grammar applies from the first generated token. Tool calls are parsed only
from the content after the split, never from inside the think block.

## Tests

- `crates/intarsia-server/src/structured.rs` unit tests: random-logit "models" under the mask.
  - JSON schema outputs parse and validate.
  - EOS is allowed only when the JSON is complete.
  - required / named / auto / single-call tools.
  - 400 parse errors and the streaming tool parser.
  - `ReasoningParser` (any chunking of the tags) and the `</think>` gate on json / required / auto constraints.
  - The real Qwen3 token table, when the model is on the host.
- `crates/intarsia-models/tests/constrain.rs` (the tiny Qwen3-shaped model on the cpu backend, single-sequence and
  batched engines):
  - a permissive mask keeps the greedy ids;
  - the mask decides the tokens, and the grammar end stops the request;
  - sampled requests are masked too;
  - lazy triggers;
  - no speculation on constrained requests;
  - constrained and unconstrained requests batch together;
  - failures are reported.
- `crates/intarsia-server/tests/api.rs`: every endpoint shape, streaming and not, and the 400 paths, against a mock
  engine that generates under the real llguidance constraint.
- `scripts/j1-structured/e2e.py`: 50 varied requests against a real server, with jsonschema validation. Results are in
  `docs/results/2026-10-21-j1-structured.md`.
- `crates/intarsia-server/tests/api.rs` (mock engine): `reasoning_content` and `thinking` blocks, streaming and not, alone, under `response_format`, and under required / auto tools; thinking off is unchanged.
- `scripts/r2-reasoning/endor-slot.sh`: the Metal e2e (docs/results/2026-10-21-r2-reasoning.md).
