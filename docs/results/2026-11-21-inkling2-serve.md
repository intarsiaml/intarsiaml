# Inkling-2-Mini through the OpenAI server (lane ink-serve, 2026-11-21)

Status: chat, reasoning, tools, json_schema and the accuracy metric done and unit / mock-engine tested; Metal real-model smoke: see "Real model". Branch `ink/serve` (not pushed, not merged).

## Briefs received

- 2026-11-21 (launch): serve Inkling-2-Mini through `intarsia-server` like vLLM does (`--reasoning-parser inkling`): chat template, `reasoning_content`, tool calls, `response_format` json_schema, effort knob; plus the confident-row metric in `intarsia accuracy`. Worktree `.worktrees/ink-serve`, branch `ink/serve` from 1dacac3a8; merge `ink/metal` before Metal checks; no push, no merge to main; every endor run through fleetq (real model `--mem 25`).
- 2026-11-21 (coordinator, relaying ink/skel 46349d879): the goldens' chat prompts are ChatML text, not Inkling turn tokens, so `accuracy` falls back to the recipe's `chat_turns` and masks no rows. Keep the goldens comparable (same token ids as before); add the Inkling-template path separately. The confident-row metric in `accuracy` must match the skeleton lane's side script: G2 short 0/363, long 1/1206; G3 4k 2/8229.

## What was built

| Step | Where | What |
|---|---|---|
| 1 Chat | `crates/intarsia-models/src/inkling_chat.rs`, `tokenizer.rs` (`HfCodec`), `Engine::chat_ids` / `TextCodec::encode_chat` (default `None`: other models unchanged) | A native port of vLLM's `render_inkling2_messages`, as ids. Text of every block is encoded by a copy of the tokenizer with its added tokens removed (vLLM's `split_special_tokens=True`), so a user typing `<|end_message|>` gets characters, not the framing token. Chosen over the jinja template because the template cannot do: literal specials as text, `developer`, `tool_choice` (none hides tools, named declares one, required / parallel false = instruction text), `continue_final_message`, string tool arguments, the reference's float spelling. The recipe still names `chat_template = "hf"` (the reference, `intarsia generate`). Detected by the vocabulary (all framing tokens present). |
| 2 Reasoning | `crates/intarsia-server/src/inkling.rs`, `openai.rs` | The server's text stream drops special tokens, so a state machine over (token id, released text) with the transitions of vLLM's `inkling_config()`: message header (the optional function name), reasoning, content, tool arguments; text after a call is kept, whitespace-only content before a call and trailing reasoning whitespace are dropped, raw / error tool blocks are content. `reasoning_content` in deltas and in the full reply (`null` when none); always split (effort `none` still writes a short thinking block). Non-streaming parses the ids (runs between markers decoded). Effort: request `reasoning_effort` or `chat_template_kwargs.reasoning_effort` (kwargs win, as in vLLM), name or number in [0, 0.99] or numeric string, default 0.9, quantized as the reference does (milli-effort, two decimals, `:g`); other models never see it. |
| 3 Tools | same | `{"name":..,"args":{..}}` after `<|content_invoke_tool_json|>` parsed into `tool_calls` (streaming: one complete call per delta, `index`, `finish_reason` `tool_calls`); the call name falls back to the block's header; a payload that is not a call stays as content. `tool_choice` as vLLM's Inkling renderer: declarations, instruction texts, **no grammar** for required / named (vLLM: `supports_required_and_named = False`). Tool results come back as `tool` messages (author name from `tool_call_id`), string arguments are parsed. |
| 4 json_schema | `structured.rs` (`GrammarEnv.ink`) | When the vocabulary has the block markers, a JSON constraint is framed like vLLM's `adjust_request`: `think* <\|message_model\|> <\|content_text\|> %json{schema} <\|end_message\|>` with free-text thinking blocks (llguidance lark, `<[id]>` special-token refs), then EOS. Not applied when a tool is forced. Compiled and masked against the real 200058-token vocabulary in a test: only `<\|message_model\|>` may start a reply. |
| 5 accuracy | `crates/intarsia-cli/src/cmd/accuracy.rs`, `accuracy/baseline.rs` | `PosStat.confident` (golden p1 - p2 >= 0.5; probabilities of the reference: the reported log-probs of a top-k golden, the renormalised softmax of a full one), `Agg.confident` / `confident_misses` (strict top-1 miss, a non-finite row of a confident row counts), printed per mode ("confident rows ... top-1 misses N (reported, not gated)") and written as `confident_rows` / `confident_misses` on `ModeStats` and `PromptStats` (new optional JSON keys, skipped when absent; no existing field changed). |

Not changed: model math, kernels, op contracts, other models' templates and parsers (`think-tags`, hermes, `qwen3_xml` paths untouched; the Qwen server tests pass), the accuracy tool's token ids (the goldens' chat prompts stay ChatML text through the recipe's `chat_turns`; `encode` and `apply_chat_template` are unchanged, the Inkling renderer is a separate path, `Engine::chat_ids`).

## Checks

- Reference harness `tools/inkling2_chat_ref.py` (transformers + the real tokenizer; vLLM's `inkling_encoding.py` loaded by path with `split_special_tokens=True`) wrote `crates/intarsia-models/tests/fixtures/inkling2_chat_cases.json`: 59 cases (system, multi-turn, reasoning, 14 effort spellings, tools, tool round trips, parallel calls, tool_choice none / required / named, developer role, literal specials, content parts, unicode, continue_final_message, errors). `cargo test -p intarsia-models --test inkling_chat`: 55 cases equal vLLM's ids token for token, the 3 cases vLLM rejects (effort 1.5, effort "loud", role `narrator`) are errors here too, and the image part is refused by design; **37 of the 55 also equal the HF jinja template's ids**. The template differs on: `tool_choice` (8 cases: no instruction text, no filtering), literal specials (2), `content: null` assistant, effort 0.125 (the template prints 0.125, the reference 0.12), floats in tool arguments (`2.5`, `1e-7`, `1e22`), `continue_final_message`; it raises on `developer`, a numeric-string effort and `tool_error` parts.
- Rust unit tests: float spelling, effort quantization (half-even), tool declaration / argument ordering (declaration sorted at every level, arguments sorted only at the top, nested keys in the client's order), the marker state machine (reasoning, blocks in any order, text after a call, whitespace rules, broken / cut-off calls, the repeated-last-id event, whole-ids parse equals the stream).
- `crates/intarsia-server/tests/inkling.rs` (mock engine with a byte vocabulary plus framing tokens, markers arrive as `Token` with empty text): prompt text, effort precedence and errors, streamed and full reasoning split, tool call parse / stream / follow-up turn with string arguments, `required` and `none` choices, json_schema framing (masks at step 0 and 1), prose cannot escape the grammar, images and the Anthropic route refused; plus the grammar against the real vocabulary. The full `intarsia-server` suite (lib 68, api 59, prefix_store 3, inkling 10), `intarsia-core` / `intarsia-models` lib tests and `intarsia-cli` (198 lib) pass.
- accuracy: golden-only confident row counts through `compare_pos` equal an independent Python count and the skeleton lane's: vLLM goldens short 367 / 569, long 1211 / 1569, 4k 8229 / 13096; oracle goldens short 363, long 1206 (misses need the model: the skeleton lane's G2 short 0 / 363, long 1 / 1206, G3 4k 2 / 8229 use the same definition). Unit test with sharp, flat and non-finite rows.
- `docs_env_vars` passes for `INTARSIA_INKLING2_DIR` (new, tests only); it still lists seven variables of the Metal lanes' branches, not mine.

## Differences from vLLM kept on purpose

- Tool-call arguments are re-serialized (`serde_json`, compact, the model's key order kept) instead of the verbatim span of the model's JSON.
- Images / audio: refused (400); vLLM emits the image marker for its vision tower, this server loads the text model only.
- An unparseable tool block is returned as content; a call cut off by `max_tokens` is returned as content, not as a partial call.
- Streaming tool calls arrive complete in one delta (as every tool parser of this server), not as argument fragments.
- Only `/v1/chat/completions` (Anthropic route: 400).

## Real model (Metal, endor): not run in this lane

ink/metal (8f23e15ec, merged into this branch, clean) runs the real checkpoint end to end and passes G1-G3 on Metal. The server smoke
on it was queued (fleetq #10750, `--host endor --expect 25 --mem 25`) but endor's queue had 10 lane jobs ahead (estimated start
about 2 h away, growing), so the job was cancelled rather than left holding a place with nobody to read it. Everything the server
does with the model's output is covered by the mock-engine tests and the real-vocabulary grammar test; what is still unmeasured is
how the real model behaves under the parser (does it emit the block layout, the tool-call JSON, thinking at effort none) and the
localeval smoke cells.

To run it (one slot: server on the real checkpoint, `tools/inkling2_serve_smoke.py` API smoke with effort none / high, tool call,
tool-result follow-up, required choice, json_schema full and streamed, then the two localeval smoke cells like the oMLX runs
`20261120-222246-endor-inkling-omlx-none-smoke-json_schema` / `...-smoke-none`, inside the same slot):

    cargo build --release -p intarsia-cli --features metal      # in .worktrees/ink-serve
    fleetq run --host endor --expect 25 --mem 25 --label "intarsiaml inkling serve: Metal server smoke + localeval cells" -- scripts/inkling2-serve-smoke.sh

Results land in `.scratch/smoke-out.json`, `.scratch/server.log` and `/opt/code/deepeval/results/*ink-serve-metal-smoke-*`.
