# Metal spec G1: greedy n-gram speculative decoding, byte-identical to plain decode

> **Update (2026-10-21, docs-accuracy pass, see `docs/decisions/LOG.md`):** Merged to main in `56ec2d57` (Merge metal-spec-ngram); "not pushed" / "nothing merged" wording in this page describes the state when written.

Date 2026-10-15. Branch `metal-spec-ngram`: `spike/metal` c556008 with `metal-g0b-attn` cf735e9 merged in (ca18d27), which also brings `metal-g0-multirow`. Machine endor (Apple M5 Max), Qwen3-8B BF16. Plan: `docs/research/metal-next-levers.md` (on `metal-next-levers`) section 7, gate G1. Kernels: `docs/results/2026-10-15-metal-g0-multirow.md`, `docs/results/2026-10-15-metal-g0b-attn.md`.

**Verdict: G1 is met.** I ran 24 cases:
* short and ~4k prompts at 48 and 256 tokens: the D1 gate's synthetic 4k filler, plus natural 4k text and code from the files `tools/prompts/golden_4k.jsonl` names;
* an input-grounded code prompt and an open-ended prompt;
* the max_tokens, stop-string, min_tokens and EOS-inside-a-run cases. In every case `generate --show-ids` gives byte-identical stdout, ids and finish/completion counts across four variants:
* `spike/metal` (c556008);
* this branch with `--spec off`;
* this branch with `--spec ngram` (k = 4);
* this branch with `--spec ngram --spec-k 6`.

That is 96 runs, all with the final code binary 47e6d0d (later commits are docs and data only).

* A `max_tokens`, stop-string or EOS-masked ending lands inside an accepted run in 22 of the 38 spec-on runs. In those runs 1 to 6 verified tokens were dropped and the KV cache was rolled back.
* The `min_tokens` boundary lands inside an accepted run. A masked EOS inside the window also goes through the cut path.
* An EOS accepted as a draft, with drafts after it, ends the request as without spec.
* Streaming through `serve` emits the same text with spec on and off. A second request reusing the KV prefix that a mid-run stop left behind matches too. `/metrics` carries the counters.
* Acceptance over the first 19 cases is 61.1 % at k = 4 and 49.9 % at k = 6.
  * Input-grounded code prompt: 72-78 %, 3.9-4.1 tokens per verify pass.
  * Repetitive 4k prompt: 65-70 %.
  * Open-ended story: 3 %.
  * Natural 4k text: 6-7 %. Natural 4k code: 12-33 %.
* Rough speed, one `generate` each at 256 tokens:
  * Grounded prompt: 3.3-3.4x faster (35.0 -> 114.7 / 120.9 tok/s).
  * Open-ended prompt: 0.99-1.01x in the clean speed runs, 0.93-0.95x in the traced identity runs.

## 1. What was built

| Piece | Where | What |
|---|---|---|
| Drafter | `crates/intarsia-core/src/spec.rs` `propose` | Prompt lookup over prompt + emitted ids. Takes the longest suffix (3, then 2, then 1) that occurred earlier and returns up to k tokens that followed its most recent earlier occurrence. Two refinements: an occurrence overlapping the suffix counts, and when the newest occurrence has fewer than k followers (repetition runs), the next older one with k followers is taken. Host side, O(len x 3) comparisons per pass, no allocation beyond the draft. |
| Acceptance | `spec.rs` `accept` | `j` = longest prefix where `argmax(row i) == d_{i+1}`. The pass emits `out[0..=j]`: d_1..d_j plus the correction or bonus token. |
| Verify pass | `crates/intarsia-core/src/model.rs` `Model::forward_verify` | One forward of `[last, d_1..d_m]` (q_len = 1 + m, 2..=7). The `exact_rows` flag is set on every op call. The head is the f32 lm_head over the q rows followed by a row-wise `Argmax`, and 4q bytes come back. All rows are committed; the caller truncates. |
| Rollback | `engine.rs` `spec_verify` | `st.truncate(len0 + 1 + j)`. The rejected rows' K/V stay in the buffer and are overwritten by the next append before any read (section 3). At the end of a request the KV is truncated to the sequential length `prompt + generated - 1` (section 4). |
| Engine loop | `engine.rs` `generate_one` | Verified tokens go into a queue that feeds the existing per-token loop body. EOS / stop ids, stop strings (from `min_tokens + 1`), `min_tokens`, `max_tokens`, cancel and streaming therefore act on each token exactly as before. No draft (no match, or no room in the context) means the unchanged plain `pick` step. |
| min_tokens | `spec_verify` | Inside the window, plain decode masks the stop ids before its host argmax. A verify row whose device argmax is not a stop id has the same argmax under the mask: both take the lowest index of the maximum (`argmax_f32_v4` and `sampler::argmax` checked). A row whose argmax is a stop id ends the pass before that row; that row's KV goes too, and the next token comes from the plain masked step. |
| Gate | `generate_one` | Spec runs only for greedy requests without `trace_logits`, on attention-only state with verify enabled. **Sampled requests (temperature > 0) fall back to plain decode** (the seeded RNG stream is unchanged, see test `spec_sampled_requests_fall_back`). |
| Load / CLI | `intarsia-models/src/load.rs`, `intarsia-cli/src/common.rs` | `--spec off|ngram` (default off) and `--spec-k` (default 4, 1..=6) on every model command (`generate`, `serve`, `plan show`, ...). `load_engine` refuses spec on any backend but Metal, and under tensor parallelism. It calls `Model::enable_verify(k + 1)` before `configure_plan` and logs the verify kernels. |
| Counters | `telemetry.rs`, `intarsia-server/src/metrics.rs`, `generate.rs` | `spec_steps`, `spec_proposed`, `spec_accepted` in the telemetry snapshot. On `/metrics` they are `intarsia_spec_steps_total`, `intarsia_spec_proposed_tokens_total` and `intarsia_spec_accepted_tokens_total`. `generate` prints a `spec:` stats line. `INTARSIA_SPEC_TRACE=1` prints one line per verify pass and one when a request ends with verified tokens unemitted (the evidence column below). |

## 2. Kernel selection for the verify step (T = 2..8)

**How it worked before.** `Model::run_chunk` takes `Op::AttnDecode` for t = 1 and `Op::AttnPrefill` otherwise, with host `pos0 = st.len`. `OpCall::check` rejects `AttnDecode` with T != 1 and allows a device position (`OpCall::pos`) only on `Rope`, `KvAppend`, `AttnDecode` and `QkNormRopeKv`. `Registry::select` takes the highest-priority admitted kernel whose predicate accepts the shape. For T = 2..8 that is, from `plan show` with prefill t = 2..16:

| Op | T = 1 decode | T = 2..8 ordinary pick | Verify pick (this work) |
|---|---|---|---|
| `Matmul` | `matmul_bf16_gemv` | `matmul_bf16_nax` (priority 5; `NAX_MIN_M` = 2 since the tail-chunk fix) | `matmul_bf16_gemv_rows` |
| `MatmulPair` | `matmul_pair_bf16_gemv` | `matmul_pair_bf16_nax` | `matmul_pair_bf16_gemv_rows` |
| `Logits` | `logits_bf16_gemv` | `logits_bf16` (tiled f32-out GEMM) | `logits_bf16_gemv_rows` |
| attention | `attn_decode_bf16_splitkv` (host position) | `attn_prefill_bf16_flash_nax` (priority 12) | `attn_prefill_bf16_splitkv_rows` |
| norms, embedding, `QkNormRopeKv`, `SiluGlu`, `Argmax` | row-parallel kernels | same kernels | same kernels (proven bitwise per row in G0b `g0b_small`) |

**What was needed.** The row kernels were registered at priority -1 and, in the pair's case, unlisted. Raising their priority would have moved ordinary prefill tails onto them, and gating them through their predicate would have given them zero admission cases. The minimal change is a selection flag (exact text in section 6):
* `OpCall::exact_rows` (false in `OpCall::new`);
* `KernelImpl::rows_exact` (false in `KernelImpl::new`, builder `.rows_exact()`);
* in `Registry::select`, a call with `exact_rows` set may only take a `rows_exact` kernel if the op has any, and fails (`NoKernel ... [row-exact verify kernels only]`) instead of falling back to NAX, tiled GEMM or flash. Calls without the flag skip `rows_exact` kernels and otherwise select exactly as before.

On CUDA and HIP no kernel is `rows_exact` and no call sets the flag, so selection there is literally unchanged.

**Attention position.** No change to `OpCall::check`. Metal has no graph capture, so its decode step runs `attn_decode_bf16_splitkv` with the host position. G0b built the rows kernel to reproduce that host-position launch per row, including the 8 -> 4 simdgroup switch at 1024 keys, and the verify pass gives it host `pos0`. A device position on `AttnPrefill` would only be needed to replay a verify pass from a captured graph (CUDA/HIP, or a future Metal graph). That would mean adding `AttnPrefill` to the `pos` list in `check`; it is not done here.

**How the flag reaches every call.** `Model` holds an `AtomicBool` that is set only inside `forward_verify` and the verify dry run. `exec_pos` copies it into every `OpCall`, so the fused paths are covered too. `fused_ok`, which `try_fused` (k+v and gate+up pair) and `qkv_fold` use to decide on fusion, selects with the flag set. Plan resolution has a new `Phase::Verify`. With spec on, it dry-runs every verify row count (2..=k+1) at both ends of the context. As a result:
* a missing row kernel fails the load;
* `serve` admission (require_admitted) covers the verify kernels with their exact shapes: 18 instead of 14 kernels, 36 extra cases;
* the scratch is sized for the rows attention (1.06 MB -> 5.32 MB).

With spec off the probe list, the plan and the scratch are unchanged: `plan show` output of base and spec off is byte-identical (`raw/.../plan/`). The verify kernels logged at every spec-on load:
`matmul=matmul_bf16_gemv_rows matmul_pair=matmul_pair_bf16_gemv_rows logits=logits_bf16_gemv_rows attn_prefill=attn_prefill_bf16_splitkv_rows`, plus the row-parallel `embedding_bf16 rmsnorm_bf16 rmsnorm_add_bf16 qk_norm_rope_kv_bf16 silu_glu_bf16 argmax_f32`.

## 3. KV rollback against the causal mask

* **What a verify row reads.** Row t of `attn_prefill_bf16_splitkv_rows` reads keys `[0, pos0 + t]` only. Its visible length is `pos0 + t + 1`, and the split partition is that of the 1-row launch at that position. G0b's `g0b_attn` test fills the cache with random data beyond `pos0 + r` and still gets the 1-row bits, so nothing past the visible length is read.
* **What a verify pass writes.** It appends K/V for its q rows at `pos0..pos0+q-1` (fused `QkNormRopeKv`) before the attention of the same layer, so every key a row sees was written by this pass or committed earlier.
* **Rollback.** Rollback is `SequenceState::truncate`, which for attention-only state sets `len`. The stale rows beyond `len` are never visible to a later decode row (visible = len + 1) or verify row: the next pass writes position `len` onward before its attention reads them.
* **Stale data and ordinary prefill.** Stale rows exist without spec too: a shorter request after a longer one leaves them. Prefill kernels mask positions beyond the causal bound. Spec only changes which finite values sit there. The follow-up-request tests (section 7, cpu) and the two-turn serve check on Metal (section 4) show identical results.
* **End of a request.** When a request ends inside an accepted run, the KV is truncated to what sequential decode holds: every emitted token but the last. A following serve request therefore reuses the same prefix and prefills the same chunks. In the two-turn serve check (section 4), turn 1 stops with 4 verified tokens dropped (KV 182 -> 178), and turn 2 reports `cached_tokens 178` and identical text in both modes.

## 4. Identity results (G1 acceptance)

Command (D1 identity-gate procedure): `intarsia generate --model ~/models/Qwen3-8B --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --show-ids --max-tokens N [case flags] [--spec ...]`.
* Raw completion cases use `--ignore-eos`.
* Chat cases use `--chat --no-think`.

The prompts are in `raw/.../prompts/`:
* `short` and `4k`: the D1 gate's prompt files.
* `grounded`: "repeat this Rust struct and Default impl with `max_tokens` renamed to `max_new_tokens`".
* `open`: a 300-word story.

"==" compares stdout bytes, the `ids:` line and `finish`/`prompt`/`completion`. Run with the final binary (47e6d0d) in slot #1998; base runs from slots #1995/#1996. `raw/.../identity-final-table.md` is the table below; the full per-variant table is in `identity-final-vs-base.md` and `identity-final-vs-off.md` (both "ALL IDENTICAL").

| case | prompt tok | finish / tokens | off == base | on4 == base | on6 == base | acceptance on4 / on6 | tokens per pass on4 / on6 | verified tokens dropped at the stop, on4 / on6 | min_tokens cuts on4 / on6 | decode tok/s base / off / on4 / on6 |
|---|---|---|---|---|---|---|---|---|---|---|
| short-48 | 122 | length / 48 | yes | yes | yes | 38% / 30% | 2.50 / 2.81 | 0 / 0 | 0 / 0 | 35.8 / 35.5 / 81.6 / 87.1 |
| short-256 | 122 | length / 256 | yes | yes | yes | 42% / 30% | 2.66 / 2.82 | 0 / 1 | 0 / 0 | 35.6 / 35.8 / 89.9 / 84.7 |
| 4k-48 | 4062 | length / 48 | yes | yes | yes | 69% / 58% | 3.77 / 4.45 | 2 / 2 | 0 / 0 | 33.7 / 34.0 / 115.8 / 121.4 |
| 4k-256 | 4062 | length / 256 | yes | yes | yes | 70% / 61% | 3.81 / 4.64 | 0 / 0 | 0 / 0 | 33.8 / 33.7 / 114.3 / 99.8 |
| grounded-256 | 371 | length / 256 | yes | yes | yes | 73% / 64% | 3.92 / 4.81 | 0 / 0 | 0 / 0 | 31.5 / 34.6 / 105.8 / 104.1 |
| grounded-512 | 371 | stop / 329 | yes | yes | yes | 72% / 60% | 3.89 / 4.61 | 0 / 0 | 0 / 0 | 32.0 / 34.0 / 100.5 / 95.2 |
| open-256 | 40 | length / 256 | yes | yes | yes | 3% / 2% | 1.11 / 1.12 | 0 / 0 | 0 / 0 | 31.7 / 34.8 / 33.0 / 30.8 |
| grounded-max102 | 371 | length / 102 | yes | yes | yes | 78% / 65% | 4.12 / 4.90 | 2 / 2 | 0 / 0 | 35.3 / 36.0 / 122.7 / 118.3 |
| grounded-max188 | 371 | length / 188 | yes | yes | yes | 73% / 63% | 3.94 / 4.80 | 2 / 5 | 0 / 0 | 35.3 / 35.3 / 114.2 / 100.1 |
| 4k-max9 | 4062 | length / 9 | yes | yes | yes | 38% / 33% | 2.50 / 3.00 | 2 / 4 | 0 / 0 | 33.6 / 34.3 / 56.4 / 45.9 |
| 4k-max44 | 4062 | length / 44 | yes | yes | yes | 69% / 58% | 3.75 / 4.45 | 2 / 6 | 0 / 0 | 34.2 / 34.4 / 97.3 / 95.3 |
| grounded-stop-topp (`--stop top_p:`) | 371 | stop / 115 | yes | yes | yes | 70% / 60% | 3.81 / 4.62 | 4 / 6 | 0 / 0 | 35.9 / 35.1 / 108.6 / 97.5 |
| grounded-stop-vecstring (`--stop Vec<String>`) | 371 | stop / 186 | yes | yes | yes | 73% / 62% | 3.94 / 4.74 | 4 / 0 | 0 / 0 | 35.6 / 35.0 / 111.6 / 108.6 |
| grounded-stop-atomic (`--stop AtomicBool>>,`) | 371 | stop / 224 | yes | yes | yes | 75% / 67% | 4.02 / 5.04 | 2 / 4 | 0 / 0 | 34.8 / 34.4 / 116.0 / 105.2 |
| 4k-stop250 (`--stop 250.`) | 4062 | stop / 85 | yes | yes | yes | 65% / 55% | 3.58 / 4.30 | 2 / 2 | 0 / 0 | 34.1 / 34.1 / 101.5 / 95.0 |
| short-stopkettles (`--stop kettles`) | 122 | stop / 57 | yes | yes | yes | 41% / 32% | 2.64 / 2.89 | 4 / 1 | 0 / 0 | 35.1 / 35.6 / 79.7 / 76.2 |
| grounded-min331 (`--min-tokens 331`, max 400) | 371 | length / 400 | yes | yes | yes | 74% / 62% | 3.98 / 4.69 | 0 / 1 | 1 / 1 | 34.0 / 34.9 / 118.4 / 112.4 |
| grounded-min335 (`--min-tokens 335`, max 400) | 371 | length / 400 | yes | yes | yes | 74% / 62% | 3.98 / 4.69 | 0 / 1 | 1 / 1 | 35.0 / 35.5 / 113.6 / 95.3 |
| grounded-min320-stopfence (`--min-tokens 320 --stop` three backticks) | 371 | stop / 328 | yes | yes | yes | 73% / 61% | 3.93 / 4.66 | 0 / 0 | 0 / 0 | 35.2 / 34.8 / 115.3 / 99.1 |

**Mid-run evidence** (`INTARSIA_SPEC_TRACE` lines in `raw/.../identity-final/on*-*.err`):
* **max_tokens:** grounded-max188 on4 ends with "finished at token 188 with 2 verified tokens unemitted; KV 560 -> 558". 4k-max44 on6 drops 6 tokens.
* **Stop strings:** grounded-stop-topp on4 ends with "finished at token 115 with 4 verified tokens unemitted; KV 489 -> 485". 4k-stop250 drops 2.
* **min_tokens.** Unconstrained, the grounded output ends with EOS at token 329.
  * With `--min-tokens 331`, the verify pass that covers token 329 is cut at row 0, because its device argmax is the masked EOS ("token 329 ... accepted 0 emits 0 (min_tokens cut: next token by a plain step)"). The masked plain step then emits the next-best token.
  * The pass "token 330 ... accepted 4 emits 5" emits tokens 330..334. The min_tokens boundary (EOS and stop strings allowed from token 332) lies inside that run.
  * In min335 the boundary (336) lies inside the run 335..339.
  * In min320-stopfence the boundary (321) lies inside the run 320..324, and the stop string then matches the closing fence at token 328.
* The "0" entries in the dropped column are cases where the stop fell on the last token of a pass, or on a plain step.

**Natural 4k prompts and EOS inside a run** (slot #1999, `raw/.../identity-c/`, `identity-c-table.md`).
* The D1 4k prompt above is synthetic filler, close to best case for n-gram lookup. These rows use the first 15,000 characters of the two files `tools/prompts/golden_4k.jsonl` names (`corpus_docs.txt`, `corpus_code.txt`), as raw completions with `--ignore-eos`.
* `repeat` is a chat prompt that asks for one sentence back verbatim.
* Binary: new 47e6d0d code (same code as at HEAD 8c1668a), same base binary.

| case | prompt tok | finish / tokens | off == base | on4 == base | on6 == base | acceptance on4 / on6 | tokens per pass on4 / on6 | verified tokens dropped at the stop, on4 / on6 | min_tokens cuts on4 / on6 | decode tok/s base / off / on4 / on6 |
|---|---|---|---|---|---|---|---|---|---|---|
| golden-docs-48 | 4326 | length / 48 | yes | yes | yes | 7% / 5% | 1.27 / 1.27 | 1 / 1 | 0 / 0 | 34.2 / 34.4 / 38.5 / 35.3 |
| golden-docs-256 | 4326 | length / 256 | yes | yes | yes | 6% / 4% | 1.25 / 1.25 | 0 / 0 | 0 / 0 | 34.0 / 33.7 / 36.1 / 31.8 |
| golden-code-48 | 3596 | length / 48 | yes | yes | yes | 12% / 8% | 1.48 / 1.48 | 0 / 0 | 0 / 0 | 33.4 / 33.2 / 43.4 / 39.7 |
| golden-code-256 | 3596 | length / 256 | yes | yes | yes | 33% / 25% | 2.30 / 2.48 | 1 / 3 | 0 / 0 | 33.6 / 33.4 / 66.1 / 54.5 |
| repeat-eos (`--chat --no-think`, max 64) | 35 | stop / 16 | yes | yes | yes | 100% / 100% | 5.00 / 7.00 | 1 / 0 | 0 / 0 | 34.6 / 34.4 / 131.2 / 162.8 |

* **EOS inside an accepted run.** In repeat-eos on4, the pass at token 13 accepted 4 drafts. The fourth draft, token 16, is the EOS (`<|im_end|>`, proposed from the chat template's own `<|im_end|>` in the prompt).
  * The trace then shows "finished at token 16 with 1 verified tokens unemitted; KV 51 -> 50": the bonus token after the EOS is dropped.
  * Result: finish `stop`, completion 16, the same ids as base.
  * In on6 the EOS was the bonus token of the second pass, so it did not land mid-run.
* Over these 5 cases: k = 4 ran 335 verify passes, 1340 proposed, 213 accepted (15.9 %); k = 6 ran 327 passes, 1962 proposed, 222 accepted (11.3 %).

**Spec off is byte-identical to spike/metal** on all 24 cases (column "off == base"), and `plan show` output is identical too.

**Totals of the first 19 cases:**

| k | verify passes | drafts proposed | drafts accepted | acceptance |
|---|---|---|---|---|
| 4 | 1001 | 4003 | 2446 | 61.1 % |
| 6 | 866 | 5193 | 2590 | 49.9 % |

**Serve / streaming** (`raw/.../serve/`, binary 5a5162d, slot #1996). `intarsia serve --spec off|ngram`, grounded prompt, `/v1/chat/completions`, greedy, `enable_thinking: false`, max 400:
* the streamed text (328 deltas) and the non-streamed text are identical between spec off and spec on, and equal to `generate`'s grounded-512 stdout (1303 bytes);
* finish `stop`, completion 329 in both;
* `/metrics` with spec on: `intarsia_spec_steps_total 164`, `intarsia_spec_proposed_tokens_total 656`, `intarsia_spec_accepted_tokens_total 490` (two requests); with spec off all three are 0.

**Two-turn serve check: prefix reuse after a mid-run stop** (`raw/.../serve2/`, `serve_check2.py`, binary 47e6d0d, slot #1999). Each mode runs one server:
* Turn 1: raw `/v1/completions` on the short prompt with `stop: ["kettles"]`. With spec on, the trace shows "finished at token 57 with 4 verified tokens unemitted; KV 182 -> 178".
* Turn 2 (streamed): the turn-1 prompt + turn-1 text + "kettles on day 99. The", 64 tokens. Its prefill reuses the 178-token prefix, and the positions past it held rejected-draft K/V in the spec-on run.

| | turn 1 | turn 2 |
|---|---|---|
| text, spec off vs spec on | identical | identical |
| finish, spec off and spec on | `stop` | `length` |
| usage | 57 completion tokens | prompt 186, cached 178, completion 64 (identical) |

Spec on counted 42 verify passes, 168 proposed, 79 accepted.

## 5. Rough speed check (not the G2 benchmark)

Single `generate --max-tokens 256 --chat --no-think` runs, no trace, two repetitions each, binary 5a5162d, slot #1996 (`raw/.../speed/`). The later 47e6d0d change touches only the step after a min_tokens cut, which these runs never reach.

| prompt | spec | decode ms (rep 1 / 2) | decode tok/s (rep 1 / 2) | vs off |
|---|---|---|---|---|
| grounded (371-token prompt) | off | 7281 / 7240 | 35.0 / 35.2 | 1 |
| grounded | ngram k=4 | 2223 / 2109 | 114.7 / 120.9 | **3.3x / 3.4x** |
| open-ended (40-token prompt) | off | 7224 / 7139 | 35.3 / 35.7 | 1 |
| open-ended | ngram k=4 | 7299 / 7072 | 34.9 / 36.1 | **0.99x / 1.01x** |

The traced identity runs agree on the grounded side, 3.0-3.5x. On the open-ended side they were slower than the clean runs: 0.95x (k = 4) and 0.89x (k = 6) in slot #1998, and 0.93x / 0.81x in #1995 (single runs with per-pass stderr tracing, so noisier).

On natural 4k text (golden docs file, traced identity runs) spec is 1.07x at k = 4 and 0.94x at k = 6 (6 % acceptance); on the natural 4k code file it is 2.0x / 1.6x (33 % / 25 %).

The open-ended prompt gets a draft at almost every step: 106 passes for 256 tokens, but only 3 % acceptance. Each pass costs a few percent over a 1-token step (G0b estimate 1.01-1.04x at k = 4 up to 1k context), so the drafter needs a gate before G2. Options: a minimum match length of 2, or turning drafting off after repeated zero-acceptance passes. That is G2 work and is not changed here, since the task fixed the 3..1 suffix rule.

## 6. Shared-code changes (exact)

`intarsia-core`:
* **`ops.rs`**: new `pub exact_rows: bool` field in `OpCall`, set to `false` in `OpCall::new`. Every construction in the workspace goes through `new`; the only struct literal is inside `new`.
* **`registry.rs`**:
  * new `pub rows_exact: bool` field in `KernelImpl`, `false` in `KernelImpl::new`, builder `rows_exact()`;
  * in `Registry::select`, `rows_only = call.exact_rows && <some entry of this op is rows_exact>`, and `continue` for every kernel with `k.rows_exact != rows_only`;
  * one extra suffix in the `NoKernel` message.

  With no `rows_exact` kernel registered (CUDA, HIP, cpu), `rows_only` is false and the skip never triggers, so selection is unchanged.
* **`model.rs`**:
  * new private fields: `exact_rows: AtomicBool`, `verify_tok: Option<Buffer>` and `verify_rows: usize`, default false / None / 0;
  * `exec_pos` copies the flag into the call; `fused_ok` selects with the flag while verifying;
  * `dry_run` handles `Phase::Verify`;
  * new pub fns `enable_verify`, `verify_rows`, `forward_verify` and const `VERIFY_MAX_ROWS`, plus private `verify_head`;
  * the workspace is untouched. `ws.tok` stays 16 bytes; the verify argmax buffer is allocated only by `enable_verify`, so spec off has the same allocations and the same memory plan.
* **`plan.rs`**: new `Phase::Verify` variant; `record()` adds verify probes only when `model.verify_rows() > 0`.
* **`engine.rs`**:
  * new `EngineMeta::spec` (`SpecConfig`, `Default` = off);
  * `LocalEngine::new` validates it;
  * the decode loop is unchanged when spec is off. The added `pending`, `plain_next` and `ctx` stay empty / false; the end-of-request truncate is skipped.
* **`telemetry.rs`**: three counters, three snapshot fields and `Telemetry::spec_step`. `TelemetrySnapshot` literals elsewhere use `..Default::default()`.
* **`spec.rs`** (new) and `lib.rs` (`pub mod spec`).

Elsewhere:
* **`intarsia-models/src/load.rs`**: `EngineOptions::spec`, the Metal-only check, `enable_verify` before `configure_plan`, and the verify-kernel log line.
* **`intarsia-server/src/metrics.rs`**: three counters.
* **`intarsia-cli`**: `--spec` and `--spec-k` in `ModelArgs`, and the `spec:` line in `generate`.

Metal backend:
* the four rows kernels get `.rows_exact()`;
* `Backend::kernels()` now always lists `matmul_pair_bf16_gemv_rows`. As a `rows_exact` kernel it is never an ordinary pick, and `INTARSIA_METAL_MULTIROW` is no longer needed;
* comments updated.

**Builds.**
* `cargo check --workspace --all-targets` (default features) passes; re-run at HEAD 8c1668a, `raw/.../cargo-check-workspace.log`. It covers `intarsia-backend-cuda` and `intarsia-backend-hip` as crates, including their tests.
* `cargo check -p intarsia-cli --features cuda` and `--features hip` both pass.
* Not built here: a linking CUDA or HIP build, and their GPU tests. macOS has no nvcc/hipcc and no NVIDIA/AMD device.

## 7. Tests

* **`cargo test -p intarsia-backend-metal --release`**: **101 passed, 0 failed** (slot #1997, `raw/.../cargo-test-metal-final.log`).
  * The first run in #1996 had 1 failure, `multirow::serving_selection_unchanged` (`raw/.../cargo-test-metal.log`). It is a G0 test that expected the tiled GEMM for M = 2..15. `spike/metal`'s tail-chunk fix (`NAX_MIN_M` 16 -> 2) moved that to NAX, so the test already fails on the plain merge ca18d27 without any change of this branch (same assertion, `raw/.../mergebase-ca18d27-serving-selection.log`, slot #1997).
  * It now checks `Registry::select` itself:
    * ordinary calls get NAX / NAX pair / tiled logits as before;
    * `exact_rows` calls of M = 2..8 get the three rows GEMVs;
    * M = 15 verify calls fail.
* **`intarsia kernels test --backend metal -m ~/models/Qwen3-8B`**: 27 kernels, 321 cases, all ADMITTED (`raw/.../kernels-test-metal.log`, #1996). There is one more kernel than in G0b because `matmul_pair_bf16_gemv_rows` is now listed; its 12 cases were all admitted, max_abs 0.
* **CPU-side crates**, `cargo test --release -p intarsia-core -p intarsia-models -p intarsia-server -p intarsia-cli`: all passed (core 55 unit tests + 2; models: every test binary, including the new `spec.rs`; server 21; cli 76). These run on the cpu backend only, outside a slot.
* **New unit tests:**
  * `intarsia-core` `spec::tests` (7): drafter longest-suffix / most-recent / fallback / overlap / no self-match, acceptance prefix, config parse.
  * `intarsia-models/tests/spec.rs` (4): the engine on the cpu backend with verify enabled. The cpu kernels compute each row with the 1-row loop, so the property is the same one the Metal row kernels provide.
    * `spec_identity_greedy`: 3 seeds x k 1/3/6 x 3 prompts.
    * `spec_stops_inside_accepted_runs`: `max_tokens` 1..48 each with a follow-up request that reuses the KV prefix, and a stop string ending at every position of the output.
    * `spec_min_tokens_and_eos_inside_runs`: EOS = each token of the reference stream, `min_tokens` 0..24, with and without a stop string; the trace shows the cut path taken 184 times.
    * `spec_sampled_requests_fall_back`: temperature 0.8, seed 7: same stream, 0 verify passes.

## 8. Limits and open points

* **Greedy only.** Seeded sampling (temperature > 0) falls back to plain decode with spec on, and the result is the same seeded stream. Exact-match sampling as described in the plan (section 4.1) needs per-row logits and is not built.
* **Metal only, single rank, attention-only models.** `load_engine` refuses other backends and TP. The engine skips spec on recurrent state, and a load with spec on such a model fails.
* **Open-ended text.** The drafter fires at almost every step and is accepted 3 % of the time, so spec is 0.9-1.0x there (section 5). A gate is G2 work.
* **Long context.** At k = 6 and 4k context the verify pass costs more than at k = 4: the attention rows are the dominant extra cost (G0b). k = 6 had more tokens per pass but often a lower decode speed here.
* The speed numbers are single `generate` runs, not the G2 localeval benchmark.

## 9. Commits (on `metal-spec-ngram`, not merged, not pushed)

* `cf74ff5` row-exact kernel selection (`OpCall::exact_rows`, `KernelImpl::rows_exact`), Model verify pass, n-gram drafter
* `5a5162d` engine loop, `--spec`/`--spec-k`, counters in telemetry, /metrics and generate stats
* `e5bc3a1` `serving_selection_unchanged` expects the NAX picks for M = 2..15
* `be4478c` raw data of identity A/B, speed, serve check, first test run
* `47e6d0d` a min_tokens cut that emits nothing is served by the plain step right after it (before: two plain steps; identity unaffected, rerun in #1998)
* `8c1668a` results doc, final identity data
* `ff2ed75` phase D additions to this report (golden 4k, EOS inside a run, two-turn serve, workspace check log)

## 10. fleetq

Every GPU run (generate, serve, cargo test of the Metal crate, kernels test, plan show) ran inside an exclusive endor slot taken with `fleetq acquire --host endor --exclusive --mem 20 --label "metal spec G1: ..." --wait` through `raw/.../slot.sh`.
* Before running anything, the script checks `fleetq show <id>` for "running on endor", asserts `DEVELOPER_DIR`, and exports `FLEETQ_JOB`.
* It releases by trap.

| slot | what | running | released |
|---|---|---|---|
| #1995 | identity A (7 cases x base/off/on4/on6) | 10:46:25 | 10:49:10 |
| #1996 | identity B (mid-run cases), speed, serve check, metal cargo test, kernels test | 10:50:33 | 10:58:02 |
| #1997 | metal cargo test after the test fix, merge-base selection test, plan show | 10:59:23 | 10:59:46 |
| #1998 | final identity rerun, 19 cases x off/on4/on6 with 47e6d0d | 11:01:17 | 11:05:19 |
| #1999 | golden 4k files, EOS inside a run (5 cases x base/off/on4/on6), two-turn serve check | 11:13:16 | 11:16:37 |

All five were released; `fleetq status` afterwards shows endor with 0 jobs. Nothing ran unslotted on the GPU. The CPU-only work ran outside slots: builds, `cargo check`, and the cpu-backend tests.

## 11. Files

Code: section 6. Raw data `docs/results/raw/2026-10-15-metal-spec-g1/`:
* scripts: `slot.sh`, `run_identity.sh`, `phase_b.sh`, `phase_c.sh`, `serve_check.py`, `compare.py`, `make_doc_table.py`;
* case lists: `cases-a.txt`, `cases-b.txt`, `cases-all.txt`, `cases-c.txt` (phase D, plus `phase_d.sh`, `serve_check2.py`);
* `prompts/`;
* run outputs:
  * `identity/` (A, binary 5a5162d), `identity-b/` (B, 5a5162d), `identity-final/` (all cases, 47e6d0d, plus the base runs);
  * `identity-final-table.md`, `identity-final-vs-base.md`, `identity-final-vs-off.md`;
  * `identity-c/` with `identity-c-table.md` (phase D);
  * `speed/`, `serve/`, `serve2/`, `plan/`;
* test logs: `cargo-check-workspace.log`, `cargo-test-metal.log` (first run), `cargo-test-metal-final.log`, `mergebase-ca18d27-serving-selection.log`, `kernels-test-metal.log`;
* `slots.log`.
