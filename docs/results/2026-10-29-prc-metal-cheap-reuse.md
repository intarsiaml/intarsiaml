# PRC: exact prefix reuse on Metal at any row of a wide-pass prefix (2026-10-29)

Lane `prc`, branch `lane-prc/metal-cheap-reuse`, base `main` `200be3f05` (#233). endor (M5 Max, macOS 27.0), Qwen3-8B BF16
(`~/Models/Qwen3-8B`), Metal. Raw: `2026-10-29-prc-metal-cheap-reuse/raw/`.

PRB (`2026-10-29-prb-prefix-reuse-probe.md`, #233) made Metal resume a dense request only at a cut of its own cold chunk schedule.
Metal's prefill chunk is 2048, so a prompt under 2048 tokens was never resumed and a multi-turn chat re-prefilled its whole history
every turn. PRB's data said why a cheaper exact rule exists: on Metal only a pass of exactly 1 row differs (86 of 86 short-prompt
splits and 40 of 40 on a 700-token prompt whose passes are all >= 2 rows are bitwise the one-pass prefill). This lane implements it.

## Result

| | #233 rule (`INTARSIA_PREFIX_REUSE=exact`) | this rule (Metal default) |
|---|---|---|
| repeated prompt (8B, 16 equal.py prompts, plen 5-15) resumes at | 0 (nothing under 2048 tokens) | `plen - 2` (16 / 16), bitwise cold |
| next chat turn (turn 1 prompt + 24-token answer + new text) resumes at | 0 | the end of turn 1's prompt (16 / 16), bitwise cold |
| turn-2 time to first token, ~1.8k-token chat, 3 runs | 458 / 456 / 455 ms (cached 0 of 1853 / 1804 / 1802 tokens) | **57 / 57 / 57 ms** (cached 1781 / 1732 / 1730) |
| served identity to the cold reference, `--max-batch 4` and single-sequence | 60 / 60 | 60 / 60 (28 / 28 completions + 32 / 32 chats, both engines) |

Turn 2 of a 1.8k-token chat starts its first token **8x sooner** (456 to 57 ms): it prefills the answer and the new message (about
70 tokens), not the history.

## The rule

`Model::reuse_len` (`crates/intarsia-core/src/model.rs`), attention-only models, on a backend whose `prefix_reuse_any_cut` is false and
that reports `Backend::prefix_reuse_min_pass_rows() = Some(n)` (new; default `None`; Metal `Some(2)`):

1. **cached rows**: the longest `c <= want` such that rows `0..c` were written by passes of >= `n` rows only, in an unbroken run from
   row 0 (`SequenceState::wide_reuse_len`);
2. **remainder**: back `c` off until none of the passes the prompt's chunk schedule runs from `c` has fewer than `n` rows (a
   1-token remainder becomes 2, `plen = 2` resumes nothing).

A decode step is a 1-row pass, so a previous turn's generated tokens are never reused (below). Everything else is unchanged: cpu and HIP
(`prefix_reuse_any_cut` true) still cut anywhere; CUDA reports no minimum and keeps #231's cold-schedule rule.
`INTARSIA_PREFIX_REUSE=exact` is the #233 rule on every backend, `=any` the old rule (`docs/tuning.md`); no new value.

`SequenceState::passes` is now `Vec<Pass { start, end, rows }>`: `rows` is the row count whose kernels' bits the rows carry. A truncate
**clips** the pass it cuts through (it used to drop it) and keeps `rows`, so a repeated prompt (cut at `plen - 2`, inside its one prefill
pass) is reusable again on the next repeat. Row-exact speculative-verify passes (`Model::forward_verify`, its TP and DFlash variants,
the HIP spec graph) write the bits of 1-row kernels whatever their row count; they are committed with `commit_rows(ids, 1)` and
`advance_rows` records 1. The CUDA ladder rule also rejects them. Falls back to the ladder when a prefill tail peel is on
(`prefill_peel > 0` can end a remainder in a 1-row pass the schedule check does not see).

## Are decode rows reusable? No

`multi_turn_resumes_with_cold_bits` also resumes each turn 2 *over* the answer's 24 decode rows (cut `plen2 - 2`, the remainder rule
kept, the cached-range rule removed), so the only difference from the passing run is that the cut is over decode-written rows:
**16 / 16 differ from cold at the first logits row** (max abs 0.14-0.25) (`raw/metal-gates-fix-multi_turn_resumes.log`, `DECODEROWS`).
A decode step (GEMV, `AttnDecode`) does not write what a 2-or-more-row prefill pass writes. So the rule excludes them, and a chat's
answer is prefilled again on the next turn (about 70 tokens of the 1.8k above). With `INTARSIA_PREFIX_REUSE=any` the same turn-2
requests (remainder 10 rows, so no 1-row remainder is involved) **fail 16 / 16**, which isolates the decode rows as the cause.

## Gates (endor, Metal, 8B, one binary; one test per process, `raw/metal-gates-*.log`, `raw/exits.log`)

| gate | default rule | `INTARSIA_PREFIX_REUSE=any` | chunk 4 (`INTARSIA_PROBE_CHUNK=4`) | `=exact` (#233) |
|---|---|---|---|---|
| `reuse_rule_resumes_with_cold_bits` (repeats + cross-prompt, logits of the resumed prefill + 4 decode steps) | pass; repeat cuts `plen - 2` asserted | **FAILS**, 16 repeats differ (max abs 0.106-0.237) | pass | pass (cuts 0) |
| `multi_turn_resumes_with_cold_bits` (new) | pass; cuts = turn 1 prompt length asserted | **FAILS**, 16 / 16 differ | pass | pass (cuts 0) |
| `engine_repeats_equal_the_cold_reference` (22 single-sequence + 6 batched rounds x 64 greedy tokens; repeats must have cached > 0) | pass; every repeat cached `plen - 2` (asserted) | **FAILS**, 16 requests differ | pass | not run |
| `engine_multi_turn_equals_cold_reference` (new; single-sequence + 4-slot, turn 2 cached = turn 1 prompt asserted) | pass (8 / 8 cached 13, 15, 8, 6) | passes (no greedy flip in 32 tokens; the logits gates are the sensitive ones) | pass | pass (cached 0) |
| `prefix_reuse_probe` (diagnostic) | tail-1 split 16 / 16 differ; any other split 16 / 102 differ (all 16 are the k = 1 split, a 1-row head); cross-prompt 0 / 34 | | | |
| `prefix_reuse_long_splits` (700 tokens, 44 splits) | 0 of 44 not as the 1-row-pass rule says | | | |

Reuse happens under 2048 tokens: every repeat in `engine_repeats` after the first of a prompt reports cached tokens `plen - 2` (single-sequence
requests 1, 6, 12, 14, 16, 18, 20, 21: 13, 11, 11, 4, 4, 4, 11, 11) and, at the default chunk (2048), the model-level repeat cuts are
`[(7, 5), (13, 11), (7, 5), ...]` (`raw/metal-gates-fix-reuse_rule_resumes.log`).

Unit tests (cpu, `crates/intarsia-models/tests/exact_reuse.rs`, a `Wide` wrapper reporting Metal's flags): `wide_reuse_len_reads_the_pass_history`
(a decode pass inside the range, a clip then re-prefill, the 1-row remainder back-off, `plen = 2` gives 0, a row-exact verify pass,
`load_prefix` leaves nothing, CUDA's `None` keeps the ladder, cpu keeps any-cut), and the single-sequence and batched engines over the
same request sequence as the CUDA tests.

## Served (`scripts/prc/served.sh`, `raw/served/`)

Repeats (`scripts/prb/served.py`: 16 sequential + 3 concurrent rounds of 4, 64 greedy tokens) and 2-turn chats (`scripts/prc/chat.py`:
4 chats x 2 turns sequentially, repeated, then 2 concurrent rounds; 48 forced tokens, thinking off) on `serve --max-batch 4` and on the
single-sequence engine, each against the cold reference (batch-4 server with `INTARSIA_CB_NO_REUSE=1`; the single-sequence engine's
reference is the same, bit for bit):

| server | completions identical | chats identical | turn-2 chat requests with cached > 0 |
|---|---|---|---|
| `--max-batch 4`, this rule | 28 / 28 | 32 / 32 | 16 / 16 |
| single-sequence, this rule | 28 / 28 | 32 / 32 | 16 / 16 |
| `--max-batch 4`, `exact` (#233) | 28 / 28 | 32 / 32 | 0 / 16 |
| single-sequence, `exact` | 28 / 28 | 32 / 32 | 0 / 16 |
| `--max-batch 4`, `any` (old) | **15 / 28** | 32 / 32 | |
| single-sequence, `any` | **26 / 28** | 32 / 32 | |

(Old-rule chats are identical at 48 tokens: the flips need the 64-token completions' near ties. The served chat set does not
discriminate; the logits gates above do.) The chat turn 2 prompts re-render the history, so a turn 2 resumes before the previous turn's
`<think>` block (cached 15-21 of 87-93 tokens here, 1730-1781 of 1802-1853 in the TTFT run).

## Time to first token (`scripts/prc/ttft.sh`, `raw/ttft/`)

`serve --max-batch 4`, one 6000-character passage per conversation (1.7-1.8k tokens), turn 1 then turn 2 ("Please continue, ..."),
streamed, TTFT = first content delta; 3 conversations per rule after a warm-up chat, the same binary:

| run | #233 rule: turn 2 prompt / cached / TTFT | this rule |
|---|---|---|
| 0 | 1853 / 0 / 458 ms | 1853 / 1781 / 57 ms |
| 1 | 1804 / 0 / 456 ms | 1804 / 1732 / 57 ms |
| 2 | 1802 / 0 / 455 ms | 1802 / 1730 / 57 ms |

## Other checks

* `plan show` (8B, Metal, `--max-seq 8192`), binary of `main` 200be3f05 against this branch: byte-identical, text 1342 B, json 8371 B (`raw/plan/`).
* Mac `cargo test --workspace --release --no-fail-fast`: 1095 passed, 0 failed, 28 ignored (`raw/mac-test.log`).

## Limits

* Measured on Qwen3-8B BF16 on an M5 Max only. The claim "passes of >= 2 rows agree" rests on PRB's and this lane's splits (prompts up to
  700 tokens at the default chunk; chunk 4 for passes at many cut points); a prompt over 2048 tokens (chunked passes) is covered by the
  chunk-4 runs, not by a real long prompt. Another Metal chip or model family would need `prefix_reuse_probe` first.
* Recurrent (hybrid) models keep their snapshot rule; `--prefix-store` keeps its own rule (`SeqStore::local_reuse`, cuts at any
  prefill-clean row; not changed here, and it can cut at `plen - 1`).
* Spec decoding on Metal is untested here; its verify passes are recorded as 1-row, which only costs reuse.
* The turn-2 gain is the history; the answer's tokens (about 70 of 1.8k here) are prefilled again.

## Files

`crates/intarsia-core/src/{backend,model,state}.rs`, `crates/intarsia-core/src/model/spec_graph.rs`,
`crates/intarsia-backend-metal/src/lib.rs`, `crates/intarsia-models/tests/{exact_reuse,prefix_reuse_real}.rs`, `docs/tuning.md`,
`docs/decisions/LOG.md`, `scripts/prc/{served.sh,chat.py,ttft.sh}`.
