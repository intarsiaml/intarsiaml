# Metal: no 1-row prefill tail (chunk invariance of the first token)

Branch `metal-chunk-no1` (off `spike/metal` d4e6c32), Qwen3-8B BF16 on endor. Raw: `docs/results/raw/2026-10-15-metal-chunk-no1/`.

## 1. Problem

On Metal, prefill rows are bitwise chunk-invariant for chunks of >= 2 rows (`flash_nax.rs`, `gemm_nax.rs`). A 1-row chunk is not:
`m == 1` matmuls go to the decode GEMV (`NAX_MIN_M = 2`) and `t == 1` attention to `AttnDecode`. A prompt whose chunking ends
in a 1-row tail (remaining = k*C + 1, k >= 1) therefore computes its last row, the one that picks the first token, with decode
kernels, while a schedule that keeps that row inside a bigger chunk uses the GEMM/flash kernels. That breaks "2048 goldens cover
requests prefilled at 256 under load" (continuous-batching design 8.1) and makes solo `L % 2048 == 1` prompts differ from
chunk-256 runs.

## 2. Change

* `scheduler::next_prefill_chunk(remaining: usize, chunk: usize, avoid_one_row: bool) -> usize`, the one chunk-size rule, used by
  `scheduler::chunk_schedule(cached, prompt_len, chunk, avoid_one_row)`, by the scheduler core's `prefill_one`, and by
  `engine::Seq::prefill_step` (attention-only path).
  * `remaining <= chunk`: `remaining` (final chunk). Otherwise `chunk`, except with `avoid_one_row`, `chunk >= 3` and
    `remaining == chunk + 1`: `chunk - 1`, so the tail is `(C-1, 2)` instead of `(C, 1)`. Every chunk stays <= C (workspace is sized for C).
  * Unavoidable, left as is: a whole uncached remainder of 1 row (prefix reuse capped at `prompt_len - 1`, nothing precedes it),
    `C <= 2` (the split would itself make a 1-row chunk).
  * Recurrent/hybrid states (`st.has_recurrent()`): path unchanged (plain chunking, last token alone for the snapshot rule).
* Metal only. New default trait method `Backend::avoid_single_row_prefill(&self) -> bool { false }` (`intarsia-core/src/backend.rs`),
  overridden to `true` in `MetalBackend`. `BatchModel::avoid_single_row_prefill` (default false) carries the flag into the
  scheduler core; the real adapter (not written yet) should return the backend flag and false for recurrent models.
  `Seq::prefill_step` reads `model.backend.avoid_single_row_prefill()`, so `model.rs` is untouched.
* **CUDA/HIP/CPU behavior is unchanged on this branch**: flag false, chunking bit for bit `C, C, .., rest`. Turning the flag on for
  CUDA/HIP (the row would then be computed in a 2-row chunk there too) is a follow-up for the user to decide.

Files: `crates/intarsia-core/src/{scheduler,engine,backend}.rs`, `crates/intarsia-backend-metal/src/lib.rs`,
`crates/intarsia-core/tests/scheduler.rs`. Merge notes: `engine.rs` change is one hunk in `Seq::prefill_step`; `git merge-tree`
of this branch with `metal-cb-m1` is clean. `backend.rs` and Metal `lib.rs` are not touched by `metal-cb-m1` (diff vs d4e6c32 empty
for both); `model.rs` (heavily edited there) is not touched here.

## 3. Tests

`cargo test -p intarsia-core`: pass (62 unit, 16 scheduler integration). `chunk_schedule_rule` checks flag off equals today's
schedule for every (cached, length, C) case incl. C = 1, 2, 3, and flag on against an independent statement of the rule
(partition, chunks <= C, no 1-row chunk unless C <= 2 or the whole remainder is one row). `next_prefill_chunk_rule`: L = C+1,
2C+1, 257 @ 256, 2049 @ 2048, 2049 @ 256, 4097 @ 2048, cached offsets, C = 1 and 2, remainder exactly 1. The randomized
batched-equals-solo tests run with the flag off and on.
`cargo test -p intarsia-backend-metal --release -- --test-threads=1` (slot #2082): pass.

## 4. Real model (slot #2082, exclusive endor)

Greedy, 48 tokens, `--ignore-eos --show-ids`, prompts of N words (token counts verified by `generate`: exactly 257 / 2049 / 4097),
two prompts each (s1, s2). Compared with the base run where the whole prompt is one chunk (`--prefill-chunk L`, no tail, unaffected
by the change). `base` = d4e6c32 binary, `after` = this branch. (`identity/matrix.txt`)

| L | prompt | base @256 | base @2048 | after @256 | after @2048 | after @L |
|---|---|---|---|---|---|---|
| 257 | s1 | same | same | same | same | same |
| 257 | s2 | same | same | same | same | same |
| 2049 | s1 | **DIFF at token 1** | **DIFF at token 1** | same | same | same |
| 2049 | s2 | same | same | same | same | same |
| 4097 | s1 | same | same | same | same | same |
| 4097 | s2 | same | same | same | same | same |

Before the fix a 1-row tail changes the ids (2049 s1: both 256 and 2048 chunking diverge from the single-chunk run at generated
token 1, because both end in a (C, 1) tail; so 2048 vs 256 look equal to each other but both differ from a no-tail schedule).
After: all identical. The other cases agree before and after, which only means the 48-token greedy ids did not move there (the
prompts are repetitive word lists and the output degenerates into a loop, so ids are an insensitive probe; bitwise logits for
those rows were not compared, see 7). The 257 and 4097 base @256/@2048 cases are not a counterexample to the finding (the same
last-row kernels differ, the argmax just did not flip).

## 5. Accuracy and qualify (slot #2082)

`intarsia accuracy` (spec off, same args as the cb-m0 `qualify_acc.sh`), base vs after: logs byte-identical (`diff` 0 lines) for
short, long and 4k.

| Golden | Prefill top-1 / KL | Decode top-1 / KL |
|---|---|---|
| short | 100.00% / 2.832e-4 | 100.00% / 3.313e-4 |
| long | 99.72% / 3.699e-4 | 99.80% / 3.538e-4 |
| 4k (21-way) | 99.28% / 2.274e-4 | 98.96% / 2.548e-4 |

These match the expected values. The accuracy tool goes through `Model::forward` (one call, chunked inside `model.rs`), not
`Seq::prefill_step`, so it cannot move with this change. Note: the standalone `accuracy` on the long golden exits 1 in both
binaries (`reuse gate: SKIPPED ... needs --prefill-chunk 2048`, floor verdict PASS); that is the script's arguments, not the change.
`intarsia qualify --backend metal --golden-dir /opt/code/intarsiaml/golden`: exit 0, all run stages PASS (acc-short, acc-long,
acc-4k incl. the 4k golden, bench, bench-long; `ab` and `tp` skipped as before). `qualify/full-report.md`.

## 6. CUDA / HIP compile proof

`fleetq run --host spark1 --mem 8` (#2083), scratch dir `~/chunk-no1-check-<pid>` (removed afterwards, `spark1-check.log`):
`cargo check -p intarsia-cli --features cuda --tests` EXIT=0, `--features hip --tests` EXIT=0. Build only, no model, no `--takes-down`.

## 7. Not done / open

* `Model::forward` itself still splits a long input with `ids.chunks(max_tokens)` (`model.rs`, `forward_inner` and `run_peer`).
  Callers that pass a whole prompt in one call (accuracy tooling, `generate --show-logits`, `forward_logits_each`) therefore still
  end in a (C, 1) tail for `L % C == 1`. The serving path goes through `Seq::prefill_step` with slices <= C and is fixed. Fixing the
  accuracy path means editing `model.rs` (out of scope here, heavily edited on `metal-cb-m1`).
* A prompt whose whole uncached remainder is 1 row (e.g. prefix reuse capped at `prompt_len - 1`) still runs as a 1-row chunk.
* The scheduler core's real `BatchModel` adapter does not exist yet; it must forward the flag (and false for recurrent models).
* Bitwise logit comparison of the first token was not done; only greedy ids.
* fleetq: endor #2082, spark1 #2083, both released.
