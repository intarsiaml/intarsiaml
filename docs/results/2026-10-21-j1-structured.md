# J1: structured output (constrained decoding), 2026-10-21

Lane J1 (requirement 6), approved by the user at 13:58 ("json 1"). Branch `lane/j1-structured-output`.

- **Gated code rev:** head `d8ff9b0b`, against base `origin/main` `c4ea00bd`.
- **Later commits:** scripts, docs and raw results only.
- **Design:** [docs/structured-output.md](../structured-output.md).

## Summary

| Check | Result |
|---|---|
| e2e, CUDA (spark1, Qwen3-8B BF16, vLLM up), 50 varied requests | **50 / 50** parse and validate (jsonschema Draft 2020-12) |
| e2e, Metal (endor, Qwen3-8B BF16), 50 sequential + 50 at concurrency 4 | PENDING (fleetq #3144) |
| localeval GSM8K n=50, `--thinking off`, json_schema (default) vs `--structured prompt`, Metal | PENDING (fleetq #3144) |
| No-op, CUDA (fleetq #3145) | **PASS**: KL21 short + long outputs byte-identical, 256 greedy ids identical (short `36901faa`, 4k `b91c2745`), HTTP chat greedy ids identical (`f7d4c284`) |
| No-op, HIP (fleetq #3150, run by the coordinator) | **PASS**: 256 greedy ids identical for Qwen3-8B short, Qwen3-8B 4k and Qwen3.6-35B-A3B-NVFP4 short |
| No-op, Metal (fleetq #3144) | PENDING |
| Unconstrained decode overhead, Metal A/B x3 | PENDING (fleetq #3144) |
| Constrained vs unconstrained decode, same length | CUDA, shared GPU: 12.8 / 11.6 / 11.4 vs 12.5 / 12.2 / 14.4 tok/s. This is noise with vLLM on the same GPU, not an overhead measurement. Metal: PENDING |
| Unit, engine and server tests | pass (see Tests) |

## e2e: 50 requests

`scripts/j1-structured/e2e.py` sends 50 requests and validates each answer with `jsonschema`. The mix:

| Requests | Endpoint | Shape |
|---|---|---|
| 20 | `/v1/chat/completions` | `response_format` json_schema over ten schemas |
| 5 | `/v1/chat/completions` | `response_format` json_object |
| 5 | `/v1/completions` | `response_format` json_schema |
| 5 | OpenAI tools | `tool_choice: required` |
| 5 | OpenAI tools | named function |
| 5 | OpenAI tools | `tool_choice: auto` |
| 5 | Anthropic `/v1/messages` tools | `any` / named `tool` |

- **Schemas exercised:** enums, numeric bounds, regex `pattern`, nested objects, arrays with min/max items, `anyOf`, `$ref`/`$defs`, `const`, and nullable types.
- **Request variants:** streaming and non-streaming, greedy and temperature 0.7.
- **Pass rule:** every JSON answer, and every tool call's arguments, must parse and validate against its schema, and `finish_reason` / `stop_reason` must be `stop` / `tool_calls` / `tool_use`.

**CUDA (spark1, fleetq #3145):** 50 / 50 valid. All 5 `auto` requests produced a call.
- Raw files: `raw/cuda/e2e-seq.json`, `raw/cuda/e2e-seq.log`.
- Sample answers: `{"answer": 26}` and `get_weather {"city":"Tokyo","unit":"celsius","days":3}`.

**Metal:** PENDING.

## No-op (unconstrained requests unchanged)

For an unconstrained request `GenRequest.constraint` is `None`, so no new code runs. Recipes:

| Backend | Recipe | Same |
|---|---|---|
| CUDA | `docs/results/2026-10-21-m1-qwen38-drill/raw/cuda-noop/m1cuda.sh` | KL21 short + long, and 256 greedy ids on the short and 4k prompts, base then head |
| Metal | `docs/results/2026-10-21-m1-qwen38-drill/raw/metal-noop.sh` | as CUDA, plus a plan comparison and `cargo test --release --workspace` |
| HIP | `scripts/j1-structured/hip-window.sh` | 256 greedy ids only |

**CUDA, spark1 (Qwen3-8B BF16, fleetq #3145, non-exclusive with vLLM up):**

| Gate | base | head |
|---|---|---|
| golden gate, short (KL21) | PASS | PASS (`acc-short.out` byte-identical) |
| golden gate, long | PASS | PASS (`acc-long.out` byte-identical) |
| 256 greedy ids, short | `36901faa` | `36901faa` |
| 256 greedy ids, 4k | `b91c2745` | `b91c2745` |
| HTTP chat, 256 tokens, greedy (`return_token_ids`) | `f7d4c284` | `f7d4c284` |

Raw files are in `raw/cuda/`.

**HIP, tatooine (fleetq #3150, run by the coordinator, radiance down 14:39:15-14:40:00, 45 s):**

| Ids | base = head |
|---|---|
| Qwen3-8B short, 256 ids | identical (md5 `3fc146f7`) |
| Qwen3-8B 4k, 256 ids | identical (`8aca363b`) |
| Qwen3.6-35B-A3B-NVFP4 short, 256 ids | identical (`1cdd1af4`) |

- The ids were compared from the raw files: in that run the script's own `[noop]` lines did not print. Cause: docker inside `gen()` read the rest of the `ssh ... bash -s` heredoc from stdin. Fixed in `7e56fdec` by redirecting `dev.sh` stdin from `/dev/null`.
- Raw files: `raw/hip/`, `raw/downtime.log`.

**Metal:** PENDING.

## Mixed batches

- **Engine design.** Within a batched step, each row has its own head mode (`Model::forward_batch`): device argmax is computed for every row, and logits are downloaded only for constrained (`LastLogits`) rows.
- **CPU fixture test.** `crates/intarsia-models/tests/constrain.rs::constrained_and_unconstrained_batch_together` checks that unconstrained rows batched beside constrained ones keep their solo greedy ids.
- **Metal check (PENDING).** `endor-slot.sh` runs one unconstrained greedy chat request alone, then again while 3 constrained requests decode beside it, and compares the ids.

## Cost of a constrained step

- **Token table and parser factory:** 0.2 s, built once on the first constrained request. Measured with the Qwen3-8B `tokenizer.json`, 151936 logits (`structured.rs::qwen3_token_table`).
- **Schema compile:** 0.6-0.8 ms per request.
- **Mask per step:** 15-50 µs worst case per generated sequence on the endor CPU, computed while the forward pass runs. A Qwen3-8B BF16 decode step takes about 30 ms on Metal.
- **Speed harness.** `e2e.py --constrained-speed` runs one long constrained generation, then the same prompt unconstrained at the same length (`ignore_eos`), in interleaved A/B x3.
  - It is a script rather than localeval because `localeval speed` never sends `response_format` (its `--structured` flag only shapes deepeval quality calls).
  - localeval also forces the length with `ignore_eos`, which a grammar that ends in EOS cannot honour.

## GSM8K (localeval, Metal)

PENDING (fleetq #3144). Both runs use n=50 and `--thinking off`: one with `--structured json_schema` (the default), one with `--structured prompt`. The baseline before this lane is the json_schema score of 0.0 in `docs/results/2026-10-21-standings.md` section 10.

## Tests

| Suite | Result |
|---|---|
| `cargo test -p intarsia-server --lib` | 14 pass. Includes 9 structured tests: JSON schema validity under random logits, EOS only once JSON is complete, invalid schemas, required / named / auto / single-call tools, the streaming tool parser, and the real Qwen3 token table |
| `cargo test -p intarsia-core --lib` | 88 pass. Includes the mask and prefetch tests |
| `cargo test -p intarsia-models --test constrain` | 7 pass, cpu backend. Covers: permissive mask keeps the greedy ids; the mask decides tokens and the grammar end stops the request; sampled requests are masked; lazy trigger; no spec; mixed batch; failures |
| `cargo test -p intarsia-server --test api` | 32 pass. Includes 6 structured tests: every endpoint shape, streaming and not, and 17 of the 400 paths |
| `cargo check --workspace --all-targets` | clean |

## Notes

- **Thinking.** The grammar starts at the first generated token. A constrained request on a thinking model with thinking left on answers without a `<think>` block. See docs/structured-output.md#thinking-models.
- **HIP build script.** `hip-build.sh` first shared one cargo target volume between the base and head trees, both mounted at `/work`. The `git archive` mtimes predate the earlier build, so cargo reused base's `intarsia-core` and the head build failed (fleetq #3148). Each tree now has its own volume (#3149, OK).
