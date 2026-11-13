# Whole-repository review, 2026-11-12 (Fable)

Scope: `origin/main` at `2939d30ac` (about 183k lines). Requested by the user on 2026-11-12 ("have fable review the whole
repo and make suggestions/fixes also"). Branch `fable/repo-review`.

## How this was done, and its limits

- Read in full by the reviewer: the HTTP layer (`crates/intarsia-server/src/{lib,api,openai,anthropic,prefix}.rs`, the
  parsers and request parsing of `structured.rs`), `crates/intarsia-models/src/{chat_template,safetensors,recipe}.rs`, the
  `[tuning]` part of `crates/intarsia-core/src/recipe.rs`.
- Mapped by five read-only helper agents (Sonnet), then spot-checked by the reviewer at the cited lines: the remote-rank
  protocol, graph capture and replay, speculative decoding and the request lifecycle, the `INTARSIA_*` knobs and the
  recipes, unsafe/FFI and duplication in the backends. A finding marked *(helper, lines checked)* was not re-derived
  end to end.
- Not read: the `.cu` / `.hip` / Metal kernels, `intarsia-cli` beyond the worker entry point, `intarsia-tui`, `-ui`,
  `-engram`, `-bench`, `tools/`, most of `scripts/`, the RDMA path, DSpine.
- Nothing was run on a fleet box. Everything was checked on the Mac (CPU and Metal build here; the CUDA and HIP crates
  compile only as stubs here). Findings that need a GPU to confirm are marked **needs a fleet gate** with the backend.
- The production chat templates were checked against the local checkpoints `Qwen3.5-4B` and `Qwen3.8-27B`
  (`chat_template.jinja`); the Qwen3.6-35B-A3B template is not on this Mac and is assumed to be of the same lineage.

Status values: **fixed** (commit), **suggested**, **needs a fleet gate**, **needs a decision**.

## Findings, by severity

### 1. Follow-up turns of an OpenAI tool conversation were a 400 on the Qwen3.5+ templates. Fixed (`82d0090f6`)

- Evidence: `crates/intarsia-server/src/openai.rs:184-191` passes each message to the template as sent, and
  `crates/intarsia-models/src/chat_template.rs` (`render`) did the same. OpenAI clients send an assistant turn's
  `tool_calls[].function.arguments` as a JSON **string**; the server returns it as a string too (`openai.rs:115-117`).
  The Qwen3.5+ templates iterate it as a mapping: `{%- for args_name, args_value in tool_call.arguments|items %}`.
- Failure: any agent loop (call a tool, send the result back) fails on its second request with
  `chat template: invalid operation: cannot convert value into pairs`. Reproduced on this Mac with the real Qwen3.8-27B
  template: object arguments render, string arguments fail. vLLM parses the string before templating, so the same client
  works there.
- Fix: `ChatTemplate::render` renders once more with string arguments that hold a JSON object parsed, only when the
  template rejected the messages as sent. A conversation that renders today gives the same prompt (the Qwen3 template
  prints a string argument verbatim; a test pins that). It covers every caller (chat completions, `/v1/intarsia/prefill`,
  the CLI).
- Left open (**needs a decision**): parse always, as vLLM does. It also fixes templates that `tojson` the string a
  second time (Llama 3.1), but it changes the prompt of Qwen3 requests that work today, so it was not done here.

### 2. Tool calls and thinking do not match what the Qwen3.5+ templates ask of the model. Needs a decision, then a fleet gate (CUDA and HIP, tool-call accuracy)

Three seams between the structured-output lane and the templates of the models now served:

- **Format.** `docs/structured-output.md:32-33` says the Qwen3.5 and Qwen3.6 templates use the hermes format
  (`<tool_call>{"name": ..., "arguments": ...}</tool_call>`), and `recipes/qwen3_5.toml:10`, `qwen3_5_moe.toml:12` and
  `qwen3_5_vl.toml:13` set `tool_parser = "hermes"`. The Qwen3.5-4B and Qwen3.8-27B templates instead instruct, and
  render history as, `<tool_call>\n<function=name>\n<parameter=arg>\nvalue\n</parameter>\n</function>\n</tool_call>`.
  With `tools` present the grammar (`structured.rs`, `calls_schema`) forces hermes JSON after the `<tool_call>` token.
  The call parses, but the model writes a format its prompt did not show it, and the history it sees next turn is in the
  other format. vLLM serves this lineage with an XML tool parser. Effect on tool-call quality: not measured.
- **Thinking is on by default in the template, off by default in the server.** The templates open a think block unless
  `enable_thinking` is false (`{%- if enable_thinking is defined and enable_thinking is false %}` ... `{%- else %}
  <think>\n`). The server splits reasoning only when the request says `enable_thinking: true` (`openai.rs:204`,
  `anthropic.rs:328`; documented as opt-in in `docs/structured-output.md:95-97` and `:129-131`). A client that sends no
  `chat_template_kwargs` (every stock OpenAI SDK) therefore gets the reasoning and the `</think>` tag inside
  `message.content`. vLLM with `--reasoning-parser qwen3` splits it for every request.
- **Same default, with tools or a schema.** For such a request the constraint is not gated on `</think>`
  (`structured::attach(..., think)` gets `false`): a `<tool_call>` the model writes while thinking opens a real call, and
  a `response_format` grammar starts inside the open think block.
- Suggested: decide "thinking" from the rendered prompt (it ends with `<think>\n`) instead of from the request flag, and
  add a `qwen3_xml` tool parser and grammar for the 3.5+ lineage. Both change responses, so neither was done here.

### 3. `logprobs` on `/v1/completions` keeps one full-vocabulary row per generated token. Suggested

- Evidence: `openai.rs:477-482` turns `trace_logits` on for any `logprobs` value; `crates/intarsia-core/src/engine.rs:1631-1633`
  pushes a clone of the whole logits row per token; nothing bounds `max_tokens` for such a request beyond the context.
- Failure: at a 151,936-token vocabulary a row is 608 KB. `{"logprobs": 1, "max_tokens": 100000}` holds about 60 GB of
  host memory until the request ends; the default 4096 tokens is 2.5 GB, times the queue. On a Spark (unified memory, swaps
  instead of OOM-killing) one request can hang the box.
- Suggested: reduce each row to the chosen token and the top 20 at trace time (an engine hook that maps the row before
  it is stored), or refuse `logprobs` above a token bound. The first needs an `Engine` API change, the second a number.

### 4. HIP: a graph exec can be destroyed while a peer rank still runs it. Needs a fleet gate (HIP, tp 2) *(helper, lines checked)*

- Evidence: `crates/intarsia-core/src/model/graph_step.rs:114-119` (`invalidate`) clears the solo, batched, spec and verify
  caches without waiting for the device; `crates/intarsia-backend-hip/src/device.rs:122-134` syncs before the destroy only
  with `INTARSIA_GRAPH_SYNC_DESTROY=1` (off by default), and its own comment describes the hazard. The DSv4.1 runtime
  synchronizes before it replaces a recording.
- Failure: rank 0 captures a new row count and clears the cache while rank 1's enqueue-only replay is queued; HIP hands the
  freed kernel-argument memory to the next instantiate. It fits the "lost kernels" reports. `docs/tuning.md` records that
  the knob alone "did not cure the hang", so this is one cause at most.
- Suggested: synchronize in `invalidate()` (in the backend-neutral layer, as the DSv4.1 runtime does), then delete the knob.

### 5. The remote-rank link has no deadline while idle and the worker port takes the first connection. Needs a fleet gate (CUDA, both Sparks) *(helper, lines checked)*

- Evidence: `crates/intarsia-core/src/tp_remote.rs:205` and `:713` set only `TCP_NODELAY` (no read or write timeout, no
  keepalive); `reply_timeout()` (`:84-86`) is 600 s and starts only after rank 0's own pass.
  `crates/intarsia-cli/src/common.rs:513-521` binds `--listen` as given, accepts one connection and reads 128 bytes with no
  timeout and no authentication.
- Failure: (a) a Spark that loses power or its link mid-idle leaves the peer blocked in `read`; the next request waits
  600 s before the group is marked failed. (b) A port scan or health probe that connects first either makes the worker
  exit or occupies it for good, and any host that reaches the port supplies the NCCL id and drives the GPU.
- Suggested: `SO_KEEPALIVE` with short probes on both ends, a read timeout on the 128-byte id, a shared token in the
  handshake, and an accept loop that drops a peer failing the handshake. Loopback tests can cover all four on a Mac.

### 6. DSv4.1: a failed graph capture turns graphs off on one rank only. Needs a fleet gate (CUDA, tp 2) *(helper, lines checked)*

- Evidence: `crates/intarsia-core/src/dsv41/runtime.rs:1079-1082` sets this rank's `graphs_on` to false and goes eager.
  The other rank keeps replaying. The `Model` path fails the step under TP instead, and the spec-graph path agrees on the
  outcome through a barrier.
- Failure: nothing checks that the eager rank and the replaying rank issue the same collectives in the same order. If
  they do, the cost is a silent slow rank; if they ever do not, the group deadlocks.
- Suggested: make the capture outcome a group decision as `spec_graph.rs` does, and add a mock-backend test.

### 7. A misspelt `[tuning.<backend>]` key or backend was ignored. Fixed (`2e4df51cc`)

- Evidence: `BackendTuning` (`crates/intarsia-core/src/recipe.rs`) was the one recipe table without
  `deny_unknown_fields`, and `Tuning.backends` is a flattened map, so `[tuning.cudda]` was accepted as a backend nobody
  asks for. `docs/recipes.md` says "An unknown key is rejected".
- Failure: `prefil_chunk = 512` under `[tuning.cuda]` loads cleanly and the tuning never applies; the run is then
  measured, or served, with the default.
- Fix: `deny_unknown_fields` on `BackendTuning`; `Recipe::validate` rejects a table that is not `cpu`, `cuda`, `hip` or
  `metal`. All ten shipped recipes still load (pinned by the test).

### 8. A corrupt safetensors header panicked the loader. Fixed (`0fb1dbd3a`)

- Evidence: `crates/intarsia-models/src/safetensors.rs` (`parse_header`): `8 + hlen` on a file-supplied `u64`, `off[0]` /
  `off[1]` on a file-supplied array, `base + e`, an unchecked shape product, and non-integer dimensions dropped by
  `filter_map`.
- Failure: a truncated download or a bad header aborts `serve` with an index or overflow panic instead of naming the
  file. No out-of-bounds read was found: the slices are built only after the checks.
- Fix: checked arithmetic and `get`; a bad dimension is an error.

### 9. Speculation and drafting stop without a trace. Suggested *(helper, lines checked)*

- `crates/intarsia-core/src/model/dflash.rs:438-441`: when the draft cache falls out of step or the sequence outgrows its
  capacity, `synced = false` and the rest of the request decodes plainly. Output is right; throughput drops and nothing
  says so (no log, no counter). A benchmark that crosses the draft cache's capacity measures plain decode.
- `crates/intarsia-core/src/engine.rs:1943`: the lookup-draft branch does not cap `k` by the tokens left to `max_tokens`
  as the MTP branch does; the extra rows are computed and rolled back. Wasted work only.
- Suggested: one counter per reason speculation was skipped, exported on `/metrics`.

### 10. Server robustness, smaller items. Suggested

- **Body limit.** No `DefaultBodyLimit` is set, so axum's 2 MB applies. A prompt inside a 262k-token context can exceed
  it (whitespace-heavy JSON or logs), and the rejection is reported as a 400 with axum's text
  (`openai.rs:166`: every `JsonRejection` becomes `bad_request`, so 413 and 415 are lost too).
- **Work before admission.** The chat template and the tokenizer run before `start()` takes a queue place
  (`openai.rs:205-212`), on tokio's blocking pool (512 threads). A burst of large prompts is tokenized in full and
  then mostly answered 429.
- **Engine errors are the client's fault.** `api.rs:204` maps every error of a `blocking` call to 400, an internal
  tokenizer failure included.
- **No API key.** There is no `--api-key`; `deploy/` examples bind `0.0.0.0`. vLLM has one.
- **Ignored fields.** `logprobs` / `top_logprobs` on chat, `logit_bias`, `frequency_penalty`, `presence_penalty` and
  `repetition_penalty` are dropped silently (`CommonParams` has no `deny_unknown_fields`, which is right for
  compatibility, but a client asking for chat logprobs gets none and no error).
- **Tool parser.** `ToolParser::push` (`structured.rs:613-624`) never gives up on a block that does not parse: after one
  malformed `<tool_call>...</tool_call>`, later valid calls are swallowed into the text, and a stream shows nothing
  until the end.
- **Template errors echo the context.** minijinja's debug rendering puts the whole message list into the 400 body.
- **Bridge spawn failure** (`api.rs:265-280`): the request is already submitted and its cancel flag is never set.

### 11. Smaller engine and backend items. Suggested *(helper)*

- `NcclComm` has no `Drop` (`crates/intarsia-backend-cuda/src/nccl.rs`): a TP group rebuilt in one process leaks the
  communicator. `CudaBackend` has no `Drop` either, where `HipBackend` has one. Harmless at process exit.
- `RankBuf::sized` (`crates/intarsia-backend-hip/src/comm.rs`) leaks earlier allocations when a later one fails.
- `tp_remote.rs` mixes `lock().unwrap()` with poison-tolerant locking; a panic under one lock cascades.
- The prefix-store key has the model id and weight bytes but no content hash: two checkpoints with the same id and size
  collide (low likelihood).

## Knobs and recipes

From the helper audit; the recipe part was re-read by the reviewer, the env read sites were not all re-read.

- 435 `INTARSIA_*` names in code, 444 rows in `docs/tuning.md`. The docs-sync test keeps the two in step; the drift is
  elsewhere.
- **No shared parser.** Three truthiness rules coexist: exactly `"1"`, anything but `"0"`, and "is set" (so `=0` means
  on: `INTARSIA_PREFIX_STORE_DEBUG`, `INTARSIA_METAL_DEBUG`, `INTARSIA_METAL_ATTN_HS` at
  `crates/intarsia-backend-metal/src/attn.rs:112`). The kernels add `atoi` and `v[0]=='1'`.
- **Bad values fall back silently** where they select kernels or schedules: `INTARSIA_METAL_ATTN_HS` / `_HSB`,
  `INTARSIA_PACK_MIN_ROWS`, `INTARSIA_TP_OVERLAP`, `INTARSIA_HIP_FP8_Q`, the DSv4.1 `env_usize`
  (`crates/intarsia-models/src/dsv41_engine.rs:39`). `INTARSIA_GDN_STATE`, `INTARSIA_TP_PARTIAL`, `INTARSIA_BATCHED_MTP`
  and `INTARSIA_PREFILL_MACRO` reject bad input and are the pattern to copy.
- **Same name, different rule:** `INTARSIA_MOE_BM` takes 16 in two kernels and silently not in `moe_w4a16.cu`;
  `INTARSIA_DSV41_HC_SIDE` takes `"2"` in the backend and `"1"` or `"2"` in the runtime; `INTARSIA_HIP_FP8_Q` is parsed
  twice (Rust and C++) by hand-mirrored rules.
- `docs/tuning.md` cites `path:line` for read sites and many are stale (for example `INTARSIA_DSV41_GRAPHS`).
  `docs/recipes.md` omits `prefill_peel`, `prefill_peel_period` and `prefill_macro` from its key table.
- Precedence (flag, then env, then recipe, then default) held at every site checked.

## Deletion candidates (listed, not deleted)

From the helper audits; confirm each has no user before deleting.

| What | Where | Why |
|---|---|---|
| `pair_entry_enabled` and `INTARSIA_METAL_MULTIROW` | `crates/intarsia-backend-metal/src/multirow.rs:114-119` | no caller; its own comment calls it a former switch |
| `ik_rdma_available`, `is_read_only` | `intarsia-backend-cuda/src/ffi.rs`, `intarsia-backend-metal/src/buffer.rs` | no caller |
| `INTARSIA_GRAPH_SYNC_DESTROY` | `crates/intarsia-backend-hip/src/device.rs:128` | once finding 4 is settled either way |
| `INTARSIA_OVERLAP` | `crates/intarsia-core/src/engine/batch.rs` | LOG.md: latency up, aggregate down; no script or test sets it |
| `INTARSIA_HIP_MOE_Z`, `INTARSIA_HIP_FP8_ROWS` | `kernels/hip/moe.hip`, `kernels/hip/fp8.hip` | LOG.md says drop; only old scripts still set them |
| 14 `docs/tuning.md` rows for names no code reads (`CUDA_GDN_CHUNKED`, `FP8_REG_CFG`, `GDN_EXP`, `GDN_HOIST`, `GDN_RING`, `HIP_A8_ILP`, `HIP_FP8_UN`, `L7M_NEG`, `MUT_DIR`, `REPLAY_OUT`, `STAMP`, `TP_BATCH_SYNC`, `W4_SIM`, `WQ_EXP`) and the scripts that set them | `docs/tuning.md`, `scripts/cuda-round6`, `scripts/cuda-prefill2`, `scripts/dh`, ... | finished experiments |
| `docs/adding-a-model.md.bak` | `docs/` | backup file in the tree |
| `crates/intarsia-backend-{cuda,hip}/src/disable.rs` | both | identical after renaming; belongs in core |

## Test gaps on the riskiest paths

- `engine.rs`, `state.rs`, `model/mtp.rs` and `model/dflash.rs` have no unit tests of their own: `spec_accept`,
  `Seq::on_token` and `finish` (the rollback arithmetic) are covered only through whole-model tests.
- No test for: a scratch grow after a graph capture, a graph destroyed while in flight, a DSv4.1 capture failure under
  TP, a silent peer on the remote-rank socket, a first connection to the worker that is not rank 0.
- A plain `cargo test` on a laptop passes without running any GPU kernel (`gpu_skipped_report.rs` fails only with `CI=1`
  or `INTARSIA_REQUIRE_GPU=1`). That is right for a laptop, but a lane that only ran it has not tested its kernel.
- The server tests use a mock engine whose template is a stub, so no test renders a real 3.5+ template through the HTTP
  path. Finding 1 lived there.

## Top 5 suggestions

1. **Serve the 3.5+ lineage the way its template expects** (finding 2): reasoning split by default, an XML tool
   parser, and one HTTP-level test per shipped recipe that renders the real template for a tool conversation. This is
   the gap between "passes our tests" and "replaces vLLM".
2. **One `env` module** in `intarsia-core` (`flag`, `usize`, `choice`), which rejects a bad value and records what was
   read, so `/v1/intarsia/plan` can print the effective knobs. Move the kernels' `getenv` reads behind values passed from
   Rust. It removes the three truthiness rules and makes a run's configuration reportable.
3. **Give TP failure one owner.** Capture failure, replay failure, a lost peer and a destroy while in flight are each
   handled per lane (findings 4, 5, 6). A small "group decision" helper (barrier plus outcome, as `spec_graph.rs` has)
   and a mock two-rank backend test for each would end the class.
4. **Bound what one request can cost before it is admitted**: body size, tokenization concurrency, `logprobs` memory, an
   API key (findings 3 and 10).
5. **Retire finished experiments on a schedule.** A knob whose LOG.md entry says "lost" or "superseded" gets deleted with
   its scripts in the same PR; the 14 dead doc rows and the list above are the backlog. Share `device.rs` / `ffi.rs` /
   `disable.rs` scaffolding between CUDA and HIP while doing it (about 800 duplicated lines).

## What was changed in this branch

| Commit | Change | Check that exercised it |
|---|---|---|
| `82d0090f6` | `ChatTemplate::render` parses string tool-call arguments when the template rejects them | `cargo test -p intarsia-models --lib` (2 new tests), `-p intarsia-server` |
| `2e4df51cc` | unknown `[tuning.<backend>]` keys and backends are rejected | `cargo test -p intarsia-models --lib` (1 new test, all shipped recipes), `-p intarsia-core`, `recipes_doc` |
| `0fb1dbd3a` | safetensors header parsing cannot panic on file contents | `cargo test -p intarsia-models --lib` (1 new test) |

Run on the Mac for the three commits together: `cargo test --locked -p intarsia-models -p intarsia-core -p intarsia-server
-p intarsia-cli` (all suites pass; CPU backend, the GPU tests print SKIPPED), `cargo check --workspace --locked --all-targets`
(the CUDA and HIP crates type-check only here) and `scripts/prepush.sh`. No kernel, plan or sampling code was touched.

Commit trailers name Claude Fable 5.2, the model that wrote them.
