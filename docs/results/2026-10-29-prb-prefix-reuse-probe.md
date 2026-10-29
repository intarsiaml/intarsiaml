# PRB: Metal has the BI8 prefix-reuse bug; HIP probably, cpu does not (2026-10-29)

Lane `prb`, branch `lane-prb/prefix-reuse-probe`, base `origin/main` `f2bb3410f` (#231). endor (M5 Max, macOS 27.0), Qwen3-8B BF16
(`~/Models/Qwen3-8B`), Metal. Raw: `2026-10-29-prb-prefix-reuse-probe/raw/`.

BI8 (`2026-10-29-bi8-serve-batch-invariance.md`) found that on CUDA a request resuming from a slot's KV prefix did not compute the
bits of its cold run: reuse resumes at `min(common, plen - 1)`, so a repeated prompt's last token is prefilled alone as a 1-row pass,
and a prefill pass's bits depend on how many rows it has. It fixed CUDA (`Model::reuse_len`, `Backend::prefix_reuse_any_cut` false)
and left cpu, Metal and HIP on the old rule, "probably affected, not measured". This lane measures them.

## Result

| backend | affected | evidence | fix |
|---|---|---|---|
| **Metal** | **yes**, the same size as CUDA | repeated-prompt reuse: first logits row differs from cold in **16 / 16** equal.py prompts (max abs 0.106-0.237); greedy tokens flip within 64 steps in 6 / 16 (prompts 3, 4, 5, 9, 13, 15 at tokens 37, 49, 63, 9, 11, 28); served, old rule: 13 / 28 requests differ on `--max-batch 4`, 2 / 28 on the single-sequence engine | `Backend::prefix_reuse_any_cut` false for Metal (BI8's exact rule), one function in `crates/intarsia-backend-metal/src/lib.rs` |
| **cpu** | **no**, by construction and measured | every matmul output is one fixed-order `dot` of an activation row with a weight row, attention is per (row, head); 30-token prompt split at every k (1-row head and tail included) is bit-identical in bf16 and f32 (`cpu_prefill_is_bitwise_row_count_invariant`) | none |
| **HIP** | **probably yes, not measured** | code reading below; the 8B does not fit beside Radiance (31 of 32 GB used on both R9700s), so no job was run on tatooine | none; the test is ready |

## Metal: what differs

Probe `prefix_reuse_probe` (`crates/intarsia-models/tests/prefix_reuse_real.rs`, new; the Metal / HIP / cpu twin of BI8's
`real_qwen3_8b_prefix_reuse_probe`), on the 8B, 16 equal.py prompts (5-15 tokens), recipe prefill chunk 2048
(`raw/metal-probe.log`, `raw/metal-probe-splits.log`):

| case | result |
|---|---|
| repeated prompt, old rule: cold state truncated to `plen - 1`, last token alone | logits row 0 **differs 16 / 16**, max abs 0.106-0.237; greedy tokens flip in 6 / 16 within 64 steps |
| prefill split (k, plen - k), k = plen - 1 (a 1-row tail) | **16 / 16 differ** from the one-pass prefill |
| prefill split with a 1-row head (k = 1) | **16 / 16 differ** |
| every other split (86 of 118; head and tail both 2 or more rows) | **0 differ** |
| cross-prompt reuse at the shared prefix (1 token, 34 pairs) | 0 / 34 differ |

So the shape matches Metal's own note (`avoid_single_row_prefill`, `crates/intarsia-backend-metal/src/lib.rs`: `m == 1` goes to the
decode GEMV and `t == 1` to `AttnDecode`; "rows of chunks of >= 2 rows are bitwise chunk-invariant"): a 1-row pass is the one pass
whose bits are different. The note was right but its consequence was not drawn: the repeated-prompt rule makes exactly that pass.

`prefix_reuse_long_splits` (`raw/metal-long-splits.log`) checks the other half on a 700-token prompt: 44 splits (heads
and tails of 1 to 9, 15-17, 31-33, 63-65, 127-129, 255-257, 300 and 511-513 rows, and seven three- to nine-pass splits) against the
one-pass prefill. The splits that contain a pass of exactly 1 row differ (`[1, 699]`, `[699, 1]`, `[5, 1, 294,
400]`, `[128, 1, 271, 300]`: 0.12-0.20); **every split whose passes all have 2 or more rows is bitwise the one-pass prefill** (40 of 40).

## Fix

`Backend::prefix_reuse_any_cut` returns false for Metal. With it `Model::reuse_len` applies BI8's rule unchanged: a dense request resumes
only at a point of its own cold chunk schedule, over rows written by exactly those passes (`SequenceState::ladder_reuse_len`). A stale
doc comment each in `backend.rs`, `model.rs`, `state.rs`, `exact_reuse.rs` and `docs/tuning.md` ("cpu / Metal / HIP keep their reuse")
is corrected; `plan show` is not touched.

**Cost, and why it is large on Metal.** Metal's prefill chunk is 2048 (`recipes/qwen3.toml`), so a dense prompt of up to 2048 tokens
has no non-final chunk boundary and is **never resumed**: a multi-turn chat on Metal re-prefills its whole history (up to 2k tokens)
on every turn, instead of only the new message. Not measured here; it is bounded by the Metal prefill speed on the history length.
`INTARSIA_PREFIX_REUSE=any` restores the old reuse (not bitwise), `--prefix-store` reuses chat prefixes exactly.

**Cheaper rule, not shipped (the task asked for BI8's rule).** The measurements above support a Metal-specific rule that keeps almost
all reuse: resume at any cut `r <= min(want, plen - 2)` such that no row below `r` was written by a pass of 1 row (a decode step, a
batched decode step, or a 1-row prefill). That needs one watermark in `SequenceState` (the lowest row a 1-row pass wrote, lowered by
`truncate`, kept by `commit`) and about 10 lines in `Model::reuse_len` for backends with `avoid_single_row_prefill`. It rests on the
8B and on prompts up to 700 tokens; it has not been run through the served check. Left as an option for the coordinator.

## Gates

All on endor, Metal, 8B, one binary per pair; the test files read `INTARSIA_PROBE_DIR` / `INTARSIA_PROBE_BACKEND` / `INTARSIA_PROBE_CHUNK`.

| gate | old rule (`INTARSIA_PREFIX_REUSE=any`) | new rule |
|---|---|---|
| `reuse_rule_resumes_with_cold_bits`: per prompt, a slot that ran the prompt (or another prompt sharing a prefix) resumes at `reuse_len`; logits of the resumed prefill and of 4 teacher-forced steps bitwise the cold run's | **FAILS**, 16 repeats differ (`raw/metal-gates-mutant.log`) | passes (`raw/metal-gates-fix.log`) |
| `engine_repeats_equal_the_cold_reference`: single-sequence engine (22 requests, repeats of 3, 4, 5, 9, 13, 15) and 4-slot batched engine (6 rounds) vs each prompt cold, 64 greedy tokens | **FAILS**, 6 of 22 single-sequence and 10 of 24 batched requests differ, cached 5-14 tokens | passes |
| the same two with `INTARSIA_PROBE_CHUNK=4` (cold chunk boundaries exist, so reuse is not 0: cached 4, 8, 12) | **FAILS** (16 repeats; `raw/metal-gate-chunk4-mutant.log`, `raw/metal-engine-chunk4-mutant.log`) | passes, cached 4 / 8 / 12 and equal to cold (`raw/metal-gate-chunk4-fix.log`, `raw/metal-engine-chunk4-fix.log`) |
| `cpu_prefill_is_bitwise_row_count_invariant` (CI: runs on cpu) | n/a | passes |
| served (`intarsia serve --backend metal`, `scripts/prb/served.sh`, `served.py`): prompts 3, 4, 5, 9, 13, 15 repeated on one server, sequential then 3 rounds of 4 at once; reference = `INTARSIA_CB_NO_REUSE=1` | `--max-batch 4`: 15 / 28 identical (13 differ, cached 5-14); single-sequence: 26 / 28 | **28 / 28 and 28 / 28** (`raw/served/`) |
| `plan show` (8B, Metal, `--max-seq 8192`), base vs fix binary | n/a | byte-identical, json 8371 B and text 1342 B (`raw/plan/`) |
| Mac `cargo test --workspace --release --no-fail-fast` | n/a | 1090 passed, 0 failed, 28 ignored (`raw/mac-test.log`) |

The default-chunk (2048) rows of the table pass because nothing is reused (`reuse cuts` are all 0 in `raw/metal-gates-fix.log`); the
chunk-4 rows are the ones that test a nonzero cut on the real Metal kernels. The served check could not show `cached > 0` with the new
rule for the same reason. Repeats of 9, 13, 15 and 3 are in the orders because those are the prompts whose greedy tokens flip on Metal
(BI8's repeat orders were chosen for CUDA's flips, 4, 5 and 8; of Metal's six they repeat only 4 and 5). The first full-length run of the
old rule is in `raw/metal-probe.log`; `INTARSIA_PROBE_VERBOSE=1` prints every split.

## HIP: code reading, not measured

Not run: `fleetq status` shows `radiance-rank0` / `rank1` holding 31 of 32 GB on tatooine's two R9700s, and the 8B in BF16 needs about
16 GB. `prefix_reuse_any_cut` and `avoid_single_row_prefill` are both default (true / false) on HIP. What the code says:

* a prefill pass of 1 to 4 rows picks `matmul_bf16_gemv` (`crates/intarsia-backend-hip/src/kernels.rs:222-228`, rows 1 to 4 unless the lane-wg WMMA rows path takes them,
  priority 10), a pass of more rows the WMMA GEMM (`matmul_bf16`, tile by M: `kernels/hip/matmul.hip:615`); `Model::forward` runs
  `Op::AttnDecode` for `t == 1` and `Op::AttnPrefill` otherwise (`crates/intarsia-core/src/model.rs:3596`), and HIP has several
  prefill attention kernels chosen by row count (`attn_prefill_bf16`, `_flash`, `_flash_v2`, `_splitkv`).
* so the repeated-prompt rule (last token alone, 1 row) takes a different matmul and a different attention kernel than the cold run's
  pass of the same row, and a tail of 2 to 4 rows a different matmul kernel as well. Whether the f32 accumulation orders differ in a way
  that changes a bf16 output is exactly what the probe measures; the CUDA and Metal results say it does at 1 row.
* HIP does not set `avoid_single_row_prefill`, so its cold schedule can itself end in a 1-row tail (it is the same hazard as BI8's
  "tail N" case).

To measure: `INTARSIA_PROBE_BACKEND=hip INTARSIA_PROBE_DIR=<Qwen3-8B> cargo test --release -p intarsia-models --test prefix_reuse_real
-- --nocapture --test-threads=1` (all four tests; `prefix_reuse_probe` prints the table) on a tatooine slot with Radiance down (about 16 GB of
free GPU memory for the 8B). If it confirms, the fix is the same one function in `crates/intarsia-backend-hip/src/device.rs`.

## cpu

`crates/intarsia-backend-cpu/src/fast.rs`: `matmul` is `par_chunks_mut` over output columns with `dot(x_row, w_row)` per (row, column),
`attention` is parallel over (row, head) and reads K and V per head: no output element depends on the number of rows or on the first row of
the pass. `cpu_prefill_is_bitwise_row_count_invariant` (`crates/intarsia-models/tests/exact_reuse.rs`) pins it on the tiny fixture: the
30-token prompt in one `forward` (chunks 8, 8, 8, 6) against every split (k, 30 - k), k = 1..29, with 3 decode steps after, in bf16 and
f32; all 29 splits are bit-identical in both. cpu keeps `prefix_reuse_any_cut` true.

## Files

* `crates/intarsia-backend-metal/src/lib.rs`: the one-function fix. Comments in `crates/intarsia-core/src/{backend,model,state}.rs`, `docs/tuning.md`.
* `crates/intarsia-models/tests/prefix_reuse_real.rs` (new): `prefix_reuse_probe`, `prefix_reuse_long_splits`,
  `reuse_rule_resumes_with_cold_bits`, `engine_repeats_equal_the_cold_reference`; backend `metal` / `hip` / `cpu`, opt-in by `INTARSIA_PROBE_DIR`.
* `crates/intarsia-models/tests/exact_reuse.rs`: `cpu_prefill_is_bitwise_row_count_invariant`.
* `scripts/prb/served.sh`, `scripts/prb/served.py`: the served repeat check.
