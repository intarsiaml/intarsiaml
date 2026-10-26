# sk2: macro prefill under `serve --max-batch` (2026-10-25)

Lane `sk`, branch `lane-sk/serve-macro`, base origin/main `b6c01f08` (#162, macro prefill for the single-sequence engine). spark2 (GB10), `Qwen3.6-35B-A3B-NVFP4`,
default recipe (chunk 2048, macro 8192). Raw logs `2026-10-25-sk2-serve-macro/raw/`, scripts `scripts/sk/`. Follow-up of
`2026-10-25-sk-spark-prefill.md` section 6 item 2: #162 did not touch `serve --max-batch`.

## What changed

The batched scheduler (`scheduler.rs`) gets `BatchModel::macro_step(slot, remaining, chunk)` (default `None`; the real adapter in `engine/batch.rs` answers
with `scheduler::macro_step_rows`, the same step rule #162 uses). `prefill_plan` takes a macro step instead of one chunk only when ALL hold:
no sequence is decoding, this request is the only one prefilling, it runs the solo chunk, and the model has `macro_rows > 0` and a recurrent state. Then one
`prefill_chunk` call carries several whole chunks (the last token still runs alone with the head); `Model::forward` runs them as ONE layer-by-layer pass
exactly as in #162, vendor GEMMs per chunk. With a decoder running, a second prompt prefilling, or a busy chunk, the schedule is the plain one chunk per
iteration, so a decoding stream never waits for a long pass. Recurrent snapshots (prefix reuse) are taken after each call. No new knob: it follows
`INTARSIA_PREFILL_MACRO` / the recipe (`=0` turns it off for both engines).

## Speed (localeval, spark2, 3 interleaved rounds, `serve --max-batch 4`, `raw/ab1/`, slot #5527)

| prefill tok/s, median of 3 (min-max) | 1k | 4k | 16k |
|---|---|---|---|
| main `b6c01f08`, `--max-batch 4` | 5236 (5226-5254) | 6569 (6477-6588) | 6043 (6015-6051) |
| branch, `--max-batch 4` | 5199 (5079-5200) | **6887 (6651-6929)** | **6611 (6567-6615)** |
| vLLM (spark2) | 5375 (5224-5377) | 6364 (6259-6398) | 6275 (6270-6278) |
| branch vs main | -0.7 % (noise band) | **+4.8 %** | **+9.4 %** |
| TTFT s main / branch | 0.190 / 0.191 | 0.623 / 0.594 | 2.756 / 2.521 |

Decode unchanged (84.2 / 81.9 / 74.3 vs 83.8 / 81.3 / 73.7 tok/s). The batched engine now matches the single-sequence engine of #162 (+5.6 / +9.4).

## Gates (spark2; base = origin/main `b6c01f08`)

| gate | result | raw |
|---|---|---|
| 256 greedy ids through `serve --max-batch 4` vs main at 1k / 4k / 16k (1004 / 4074 / 16255 tokens), each alone and all three at once; branch default and `INTARSIA_PREFILL_MACRO=0` | **identical** (12 of 12) | `raw/ids1/`, `ids1.log` |
| `real_35b_batched_macro_prefill_is_bitwise_the_plain_local_engine` (new): BatchedEngine with the recipe macro vs LocalEngine with `prefill_macro: Some(0)`, prompts 2.6k / 4k / 8.3k / 16.4k (one sampled), alone and together | pass; 6 prefill calls instead of 17 chunks alone | `g-bmacro.log` |
| 35B batched == solo: `real_35b_batched_decode_is_bitwise_the_solo_step`, `real_35b_batched_engine_equals_the_local_engine` | 2 passed | `g-batch.log` |
| `real_35b_batched_engine_long_prompts_equal_the_local_engine` | 1 passed | `g-long.log` |
| `intarsia kernels test --backend cuda --quick` | 77 kernels, 531 cases, 0 FAIL (no kernel changed) | `g-ktest.log` |
| CPU: scheduler mock `macro_steps_only_for_a_sole_prefill_with_nothing_decoding` (sole prefill: 3 balanced passes; decoder running: 256 chunks; second prompt arrives: single chunks), `batched_engine_with_macro_prefill_equals_plain_solo` (3 slots, reuse off / on, alone and together) | pass | `cargo test` |
| KL goldens | not applicable (bitwise) | |

## Risks

* A request that arrives during a macro pass is admitted after it (up to ~1.2 s for 8192 rows); decoding streams are never affected, because a pass runs only
  with nothing decoding. Cancellation is checked per pass.
* The batched scheduler's trace/metrics count a pass as one chunk (`prefill_chunks` falls: 6 vs 17 in the test).

Fleet: builds #5518, #5519, #5521 (`--load cpu`); ids #5524; test #5525; A/B #5527 (`--quiet gpu,cpu`); gates #5528-#5530. Build dirs `~/intarsiaml-sk-main`, `-dev` deleted after the last gate.

`cargo test --workspace` (Mac, CPU): 932 passed, 0 failed (`raw/cargo-test-workspace-mac.log`).
