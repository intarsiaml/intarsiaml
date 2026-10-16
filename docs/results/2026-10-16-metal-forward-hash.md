# Metal: why `Model::forward` hashes differ between chunk 256 and 2048 at L = 1 mod 256

Branch `metal-forward-hash` (off `spike/metal` 9ffa531), Qwen3-8B BF16, Metal, endor M5 Max. Raw: `docs/results/raw/2026-10-16-metal-forward-hash/`
(`base/` = 9ffa531, `fix/` = this branch; `fwdrows/`, `chunkgolden-harness/` standalone harnesses, `compare.py`, `gates.sh`, `mk_trunc_golden.py`).
fleetq jobs (all `--exclusive` on endor): #2160 first per-row probe, #2165 gates on base, #2166 gates on fix, #2167 `cargo test -p intarsia-backend-metal --release`.

## Answer

**The 32a42d8 rule is applied and works for the layers. The remaining difference is in the lm_head: `AllLogits` runs the head in groups of
`LOGIT_ROWS = 16` rows that restart at every forward chunk, and a 1-row group runs the decode GEMV, which is not bitwise the same row inside a
>= 2-row group.** At L = 257 and chunk 2048 the single 257-row chunk ends in a 1-row head group (16 * 16 + 1); at chunk 256 the rule makes the
last chunk 2 rows, so the same row goes through a 2-row head group. Only the last row of the prompt differs (rows 256 / 512 / 768 at L = 257 / 513 / 769).
Serving is unaffected: it computes the head for the sampled row alone (M = 1), always.

Fix (6 lines, `crates/intarsia-core/src/model.rs`): the `AllLogits` head groups use the shared prefill rule
`scheduler::next_prefill_chunk(remaining, LOGIT_ROWS, backend.avoid_single_row_prefill())`, so a 17-row remainder becomes (15, 2) instead of (16, 1).
Gated on the backend flag (default false), so CUDA / HIP take `LOGIT_ROWS.min(remaining)` exactly as before.

## Q1. What the hash covers, and which path

`chunkgolden` (and `fwdrows`) call `Engine::forward_logits_each` -> `Model::forward(.., HeadMode::AllLogits, Some(cb))` -> `forward_inner`.
`forward_inner` cuts `ids` with `forward_chunks` (the 32a42d8 rule, `ws.max_tokens` = `--prefill-chunk`), so the rule is not bypassed. The callback sees
**every position's** logits row: `all_rows_hash` is FNV-1a over all L rows, `last_row_hash` over row L-1. Per chunk, `AllLogits` calls
`head_logits(&hn, r, c)` for `c = min(16, rest)` rows at a time (the `LOGIT_ROWS` workspace), and `proj` picks the kernel by M: M = 1 is the decode
GEMV, M >= 2 the batched path (`Backend` doc comment in `backend-metal/src/lib.rs`: `NAX_MIN_M = 2`; the kernel names were not profiled here).

## Q2. Which rows differ at L = 257, 256 vs 2048 (base, `base/fwdrows-compare.txt`)

| prompt | L | forward rows differing | max abs diff | logits differing | argmax | serving row0 chunk 256 vs 2048 |
|---|---|---|---|---|---|---|
| code257 | 257 | 1 of 257 (row 256, the last) | 5.15e-5 (|v| 17.5) | 141922 of 151936 | same (2204) | identical |
| code513 | 513 | 1 of 513 (row 512) | 3.82e-5 | 138105 | same (330) | identical |
| code769 | 769 | 1 of 769 (row 768) | 2.96e-5 | 145687 | same (1372) | identical |

All other lengths tested (17, 33, 255, 256, 258, 273, 2049): 0 rows differ. The sampled (serving) last row is identical at both chunks, and at chunk 2048
the forward last row equals it bit for bit (`serve0 == fwd_last` True at 257 / 513 / 769 only for chunk 2048).

## Q3. Root cause, with the evidence chain

1. The layers are chunk-invariant: serving row0 at chunk 256 (body (255, 2), head M = 1) equals serving row0 at chunk 2048 (body 257, head M = 1), so the hidden state
   of row 256 is the same under both schedules. Forward rows 0..255 are bitwise equal, including row 255 (head group M = 2 at chunk 256, M = 16 at chunk 2048), so
   the NAX head is row-invariant across M >= 2.
2. Chunk 2048 forward last row == serving row0 (M = 1 head); chunk 256 forward last row (M = 2 head) differs by ~5e-5. The only difference is the head group size, 1 vs 2.
3. Why 2049 and 4097 matched: both schedules end in a 2-row chunk ((2047, 2) at chunk 2048), so both use M = 2.
4. Discriminating prediction, confirmed: L = 273 (= 1 mod 16, not 1 mod 256, chunk 256 runs (256, 17), chunk 2048 one 273-row chunk): both end in a 1-row head group,
   0 rows differ, and forward last == serving row0 at both chunks.
5. The fix removes the difference: after it, 0 rows differ at all ten lengths (`fix/fwdrows-compare.txt`).

Why 32a42d8's probe missed it: it used a 300-token prompt at chunk 299 vs a single chunk. 300 = 18 * 16 + 12, so neither schedule ends in a 1-row head group.

Side finding (not a bug here): forward rows are not serving's bits for the head in general. Serving samples with an M = 1 GEMV head; forward uses M >= 2 groups. Serving row0 vs
forward last row differ by 3e-5..7e-5 at L = 255, 256, 258, 2049 (chunks 256 and 2048 alike). So "accuracy scores serving's bits" holds for the layers, not the head. Matching it would
need serving's 1-row head bits per row (per-row GEMV launches, or the `rows_exact` multi-row logits shader `gemv_f32o_v_mr<R>` in `multirow.rs`, whose contract is row r == the 1-row
launch; not checked here whether it admits the Qwen3-8B lm_head) and would change every golden log; not done.

## Q4. Does it change `intarsia accuracy`?

Yes, in the last position of a prompt with L = 1 mod C, by a relative ~1e-4 of that one position's KL; invisible in the printed tables. Built as prefix-truncated copies of
`golden/qwen3-8b-4k-fp32-top20.safetensors` (`mk_trunc_golden.py`; the reference is one causal forward, so the first n rows are exact): lengths 257, 513, 769, 2049 and a control 258, 3846 positions.
`intarsia accuracy --mode prefill --prefill-chunk {256,2048}` with `INTARSIA_ACC_POS_DUMP`:

| position | KL(21-way) chunk 256 | chunk 2048 | top-1 |
|---|---|---|---|
| first257 pos 256 | 7.759656e-5 | 7.761125e-5 | same |
| first513 pos 512 | 6.828792e-9 | 6.828816e-9 | same |
| first769 pos 768 | 7.554660e-5 | 7.553065e-5 | same |
| other 3843 positions | identical | identical | |

The printed per-prompt and OVERALL rows (`base/trunc/c256.log` vs `c2048.log`) are identical apart from the "prefill chunk" header lines. After the fix: 0 positions differ.
The existing goldens are not affected (no prompt length or chunk remainder is 1 mod 16 for them: 38, 39, 80, 79, 18, 32, 58, 90, 300, 700, 4064, 4557): short / long / 4k accuracy logs at chunk
256 and 2048 are byte-identical base vs fix.

## Fix verification (`fix/`, base = 9ffa531)

| gate | result |
|---|---|
| `cargo test -p intarsia-backend-metal --release -- --test-threads=1` (#2167) | 103 passed, 0 failed (`fix/cargo-test-metal.txt`) |
| `cargo test -p intarsia-core --release` (CPU) | 82 passed, 0 failed |
| serving path, 33 prompts, chunk 256 and 2048: ids + all 1056 sampled-logits row hashes, base vs fix | identical at both chunks |
| `Model::forward` chunk 256 vs 2048 (33 prompts, last-row and all-rows hash) | base: `code257`, `code513` differ; fix: 0 differ |
| `Model::forward` base vs fix | chunk 256: unchanged for all 33 prompts; chunk 2048: only `code257`, `code513` change (they now equal chunk 256) |
| fwdrows base vs fix, rows changed | only final 1-row head groups: L = 17, 33, 273 (both chunks), 257 / 513 / 769 (chunk 2048). These rows move from M = 1 to M = 2 bits |
| accuracy short / long / 4k at chunk 256 and 2048, base vs fix | 6 logs byte-identical |
| truncated-golden accuracy logs, base vs fix | byte-identical; per-position dump: chunk 256 vs 2048 differing positions 3 -> 0 |

Known and unchanged: `accuracy` long golden at chunk 2048 exits 1 on `prefix-reuse FAIL` in both base and fix (pre-existing, reuse gate; cause not investigated here; also reported by the cbgap gates).

Cost of the fix: for lengths with a final head group of 1 row (L = 1 mod 16) the last row of `forward_logits_each` is now an M = 2 row instead of an M = 1 row, so
it no longer equals serving's sampled row at those lengths (it did only by coincidence of the group size; at all other lengths it never did). No CUDA / HIP change (flag defaults to false).

## Left undone

* Row-invariance across M >= 2 was shown only for the BF16 lm_head of Qwen3-8B on this M5 Max; other head formats (fp8, nvfp4, quantised) were not tested, and the fix relies on it.

* The head is still not serving's M = 1 bits for accuracy scoring (side finding above).
* `2026-10-15-metal-chunk-golden.md` still attributes the forward-path diff to the 1-row tail; this doc supersedes that explanation.
* Not run: CUDA / HIP builds (no change reachable: `avoid_single_row_prefill()` is false there).
