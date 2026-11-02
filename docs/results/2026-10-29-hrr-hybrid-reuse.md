# HRR: hybrid prefix reuse resumed off the cold schedule (2026-10-29)

Lane `hrr`, branch `claude/bold-jackson-f249a3`, base `main` `694b519cf` (#258). endor (M5 Max, macOS 27.0), Qwen3.6-35B-A3B-NVFP4
(`~/models/Qwen3.6-35B-A3B-NVFP4`, 30 GatedDeltaNet + 10 gated-attention layers), Metal, prefill chunk 4096 (recipe). Raw:
`2026-10-29-hrr-hybrid-reuse/raw/`.

**Status (user decision, 2026-11-02): the strict rule below is opt-in, `INTARSIA_PREFIX_REUSE=exact`. The default is main's: a hybrid
model resumes at any snapshot (`restorable_len`) on every backend, and can give text that differs from a fresh server's.** Read
"Fix" as what `exact` does. The numbers in "Cost" are default (the old rule) against `exact`.

## Report (lane mmtp)

`serve --backend metal` (LocalEngine, no prefix store). A = `prompt-4096.txt` (4096 tokens), 256 greedy tokens. B = A +
" Summarize the above in one paragraph." (4105 tokens): `cached_tokens` 4095, and B's text differs from a fresh server's B
from about character 152, with `--spec off`, `--spec mtp` and the main binary alike.

## Which rule resumed at 4095

With no prefix store `Seq::new` takes `Model::reuse_len`. For a state with recurrent slots that returned
`SequenceState::restorable_len(want)` before any backend rule: the largest snapshot at or below `want`. The single-sequence
prefill snapshots a recurrent state after every chunk and before the prompt's last token, so A left one at 4095. Lane prc's
Metal rule (`prefix_reuse_min_pass_rows`, `wide_reuse_len`) and the bi8 ladder never run for a hybrid model.

The passes (recurrent schedule, `Seq::prefill_step`: plain chunks, the last token alone):

| run | passes |
|---|---|
| B warm (A's snapshot) | [0, 4095) (A's pass) + [4095, 4104) + [4104] |
| B cold | [0, 4096) + [4096, 4104) + [4104] |

The cold run never holds a state at 4095: that row is inside its first pass.

## Measurement (`tests/hybrid_reuse_real.rs`, `hybrid_reuse_shapes`)

Logits of the 9- / 8-row pass, the last token and 4 teacher-forced decode steps, plus the 30 recurrent slots' bytes at 4105,
against cold B (`raw/diag-run1.log`):

| run | vs cold B |
|---|---|
| cold B again | EQUAL (deterministic) |
| warm shape from a fresh state | DIFFER: last-token logits max abs 1.14, recurrent slots 1-29 of 30 |
| A as the engine runs it, snapshot, 32 decode steps, truncate to 4095, B's rest | EQUAL to the warm shape (restore is exact), so DIFFER from cold |
| [0, 2048) + [2048, 4096) + [4096, 4104) + [4104] | EQUAL |
| [0, 4095) + [4095, 4096) + [4096, 4104) + [4104] | DIFFER |
| [0, 4094) + [4094, 4096) + [4096, 4104) + [4104] | DIFFER |
| tail [4096, 4104) split 4+4, 3+5, 1+7, 2+3+3 | DIFFER (all) |

The snapshot and restore are exact. The pass shape changes the bits. Splitting at 2048 gives the cold bits, as the
chunk4k lane measured. Splits that are not on a chunk boundary do not, even when every pass has at least 2 rows.

The Metal delta-rule kernel (`kernels/metal/gdn.metal`, `gdn_delta_rule_t`) walks the tokens one at a time with f32 state.
The conv state holds the bf16 inputs in f32. So the recurrence itself gives the same bits wherever a pass is cut, as long
as its inputs are the same. The inputs are not: the first differing slot is GatedDeltaNet layer 1, 3, 6 or 27 (slot index), depending on the split. That
points to an op before the GatedDeltaNet whose rows depend on the pass's row count. Candidates are the NVFP4 GEMMs and the
MoE path choice (`DECODE_MAX_PAIRS` GEMVs vs grouped tiles, tile rows from the mean rows per expert). Not bisected here.

So the GatedDeltaNet state at 4095 cannot be the state a cold prefill of B writes: cold B has no state there, and warm B's rows
0..4094 come from a 4095-row pass where cold B's come from a 4096-row one. **It is a bug**: reuse has to refuse, or cut where B's cold schedule cuts.

## Fix

`Model::reuse_len`, recurrent branch: with `INTARSIA_PREFIX_REUSE=exact` (opt-in; unset and `any` keep `restorable_len`),
resume only at `SequenceState::recurrent_reuse_len(want, cuts)`. That is the longest snapshot
`L <= want` that is one of `cuts` = `scheduler::recurrent_schedule_cuts(plen, chunk, tail_merge)`, the pass ends of the
prompt's own cold recurrent schedule. In addition, the rows `0..L` must have been written by prefill passes (`Plain` /
`RowInvariant`) in an unbroken run from row 0, each ending at one of `cuts`. Rows written by a decode step, a verify pass or
a prefix-store restore end the run. The default keeps `restorable_len` on every backend. The batched engine calls the same
function (`engine/batch.rs`). Neither engine uses 1-row avoidance for recurrent states (`Seq::prefill_step`,
`engine/batch.rs` `avoid_single_row_prefill`), so the cuts match both. A batched prefill that runs at the busy chunk (2048
beside decoders) writes passes ending at multiples of 2048, which are not 4096-chunk cuts, so it resumes nothing: this is conservative.

| case (35B, chunk 4096) | default (before) | `exact` (after) |
|---|---|---|
| B = A + suffix after A | cached 4095, text differs | cached 0, = cold |
| A repeated | cached 4095, = cold | cached 4095, = cold |
| A after B | cached 4095 | cached 0 (B never held 4095), = cold |
| prompt > 4096 sharing the first 4096 with the held one | snapshot at 4096 | snapshot at 4096 (a cut of both) |
| next chat turn, history < 4096 | held sequence incl. decode rows | 0 |

Cost of `exact`: without a prefix store a hybrid model no longer resumes a next chat turn or an extended prompt
inside a chunk. The prefix store (`--prefix-store`) cuts recurrent prefill at message boundaries
(`SeqStore::chunk_end`) and resumes there (its own bitwise gate: `tests/prefix_store_metal.rs`; not re-measured here). It is the way to get bitwise multi-turn reuse on these models.

## Tests

* `crates/intarsia-core/tests/scheduler.rs` `recurrent_schedule_cuts_follow_the_solo_schedule`: the cuts, with and without
  tail merge.
* `crates/intarsia-models/tests/hybrid.rs` `chunk_sensitive_backend_resumes_a_recurrent_state_only_at_cold_cuts` (strict rule switched on): tiny hybrid
  fixture on a cpu backend reporting Metal's flags. Repeat 19, A + 3 tokens 16 (not 19), next chat turn 16, short prompt
  extended 0. All give the same tokens as a fresh engine. The cpu backend itself still gives 19.
* `crates/intarsia-models/tests/hybrid_reuse_real.rs` (opt-in `INTARSIA_PROBE_DIR`): `hybrid_reuse_shapes`, the diagnostic
  above (`INTARSIA_PROBE_ASSERT=1` makes it a gate on the shapes). `engine_hybrid_reuse_equals_cold` is the gate: A then B on
  one engine equals a fresh engine's B. A after B and A repeated equal cold A, with the cached counts above.

* Rework (opt-in `exact`, default = main): endor `cargo test --workspace --release --no-fail-fast` (fleetq #7888, `raw/cargo-test-workspace-rework.log`): 1200 passed, 0 failed, 29 ignored. The strict test turns the rule on with `Model::set_prefix_reuse_exact(true)`; `default_resumes_a_recurrent_state_at_any_snapshot_on_a_chunk_sensitive_backend` pins the default (B after A resumes at A's `plen - 1` = 19). `plan show` for the 35B and Qwen3-8B on Metal identical to main, `kernels test --backend metal` 0 FAIL (#7893).
* Before the merge: endor `cargo test --workspace --release --no-fail-fast`: 1112 passed, 1 failed (`docs_env_vars`: the two new probe variables had no row in docs/tuning.md; rows added, re-run passes), 28 ignored. 35B engine gate: B after A cached 0 = cold; A after B cached 0 = cold; A repeated cached 4095 = cold (`raw/tests-run2.log`).

## Cost of the strict rule and the qualify gate (2026-11-02, after the merge with main `e72ce1964`)

endor, Metal, 35B, chunk 4096 (recipe), `serve --spec off`, greedy, raw `/v1/completions`, the old rule (the default now) is `old`/`default` below, the strict rule is `new`/`exact`.
Probe: `raw/hrr_reuse_probe.py` (run by `raw/hrr_run.sh`); fleetq #7834 (chat), #7839 (A = 4096 tokens), #7835 (A = 4098), all endor, `--quiet gpu,cpu`.

**The reported case, reproduced and fixed** (A = 4096 tokens, B = A + " Summarize the above in one paragraph.", 256 tokens; `raw/probe-ab4096-*`):

| | B after A cached | B text vs a cold B | A after B cached | B-after-A TTFT |
|---|---|---|---|---|
| old rule | 4095 | DIFFERS from char 152 (same char as lane mmtp's report) | 4095 | 54 ms |
| new rule | 0 | equal | 0 | 1058 ms (cold 1031 ms; this run was ~45 % slow on every request, a concurrent build: the same cold B took 735 ms in #7835) |

With A = 4098 tokens (`raw/probe-ab4098-*`) A's pass ends at 4096 and B's cold schedule cuts at 4096 too: new rule cached 4096, TTFT 49 ms, equal to cold.

**Cached prefix lost in a chat continuation** (turn 1 = L tokens + 32 greedy reply tokens, turn 2 = turn 1 + reply + a 20-token question; `raw/probe-chat-*`; TTFT = first streamed token):

| turn 1 prompt | held | old: cached / TTFT | new: cached / TTFT | cold TTFT | tokens lost | text new = cold |
|---|---|---|---|---|---|---|
| 1443 | 1475 | 1474 / 48 ms | 0 / 294 ms | 296 ms | 1474 | yes |
| 2954 | 2986 | 2985 / 53 ms | 0 / 631 ms | 572 ms | 2985 | yes |
| 4960 | 4992 | 4991 / 56 ms | 4096 / 234 ms | 1055 ms | 895 | yes |
| 8481 | 8513 | 8512 / 61 ms | 8192 / 137 ms | 1838 ms | 320 | yes |
| 11992 | 12024 | 12023 / 67 ms | 8192 / 1016 ms | 2864 ms | 3831 | yes |

The loss is the held tokens above the last multiple of 4096 inside them: 0 to 4095, about half a chunk on average. Cold prefill costs about 0.2 ms per token here (1055 ms for 4960 tokens), so a turn pays up to about 0.8 s more TTFT. A chat shorter than one chunk reuses nothing, so it pays the full cold prefill each turn (0.3 to 0.6 s at 1.5k to 3k tokens). Greedy text over 32 to 48 tokens was equal under the old rule too in these cases: the old rule is wrong only where the resume point is a snapshot the cold schedule does not hold (the first table), not on every continuation.

The rule trades prefix for exactness; the lost prefix is what `--prefix-store` (message-boundary cuts) recovers.

## The qualify gate, default against strict (user decision: strict is opt-in)

qualify's `acc-4k-reuse` command (`accuracy --mode reuse --prefill-chunk 1024 --reuse-require-all`, golden `qwen36-35b-a3b-nvfp4-vllm-4k-top20`, Metal 35B), run three times: this branch with the default rule (`raw/accuracy-reuse-35b-metal-default.txt`), this branch with `INTARSIA_PREFIX_REUSE=exact` (`...-exact.txt`), and a build of `origin/main` (`...-main-binary.txt`; fleetq #7893, #7896). The default rows equal main's line for line (cached, agree, verdict): the default is main's behaviour.

`ModelInfo::reuse_cold_cuts_only` is true only under `exact`, so the default gate keeps `cached >= s` and `exact` uses `s / chunk * chunk`.

| prompt (split) | default: cached, warm = cold tokens | default verdict | exact: cached, warm = cold tokens | exact verdict | oracle warm/cold KL ratio (limit 1.5x + 3e-3), both rules |
|---|---|---|---|---|---|
| long_docs_4k (2000) | 2015, 1 of 32 | ok (oracle) | 1024, 32 of 32 | ok (oracle) | 1.71x (1.54e-2 vs 1.65e-2): pass |
| long_code_4k (2000) | 2015, 3 of 32 | **FAIL**: streams diverge at token 3, KL up to 1.6e-1 | 1024, 32 of 32 | ok | none (committed oracle does not match this prompt's ids) |
| chat_reuse_4k (4521) | 4536, 32 of 32 | ok (oracle) | 4096, 32 of 32 | ok (oracle) | 0.71x (1.08e-2 limit): pass |

Reading it:

* The oracle ratio does not separate the rules. `judge_continue_oracle` scores a scripted resume at the split, not the state the engine resumed from, so the ratio is the same (1.71x, 0.71x) under both and under main. The default passes the oracle limit on the two prompts that have one, and 1.71x is not new to this PR (the same case read 0.99x on 2026-10-22, before main's later changes; not bisected).
* What does separate them is the stream against a cold run, which the oracle path replaces on long_docs_4k: under the default, warm and cold agree on 1 of 32 tokens there, and long_code_4k FAILS. That failure is on main too (docs/results/2026-10-29-chunk4k-metal.md; main binary run above: same rows). `exact` agrees on 32 of 32 on all three and the gate passes (3 judged, 0 skipped).
* So the default fails qualify's reuse gate on this golden at chunk 1024 on Metal, as main does. Under the default, qualify's `acc-4k-reuse` for the 35B stays red.

Bug repro, A = 4096 tokens, B = A + " Summarize the above in one paragraph." (`raw/probe-ab4096-default.txt`, `-exact.txt`, fleetq #7893): default B after A cached 4095 (TTFT 49 ms), text differs from the cold B; A after B cached 4095. `exact`: both cached 0, B equals the cold B.

CUDA and HIP: read from the code only. Under `exact` they take the same branch and floor; by default they resume at any snapshot as on main. Not run (no spark run, tatooine untouched). DSv4.1 (`dsv41_engine.rs`) does not use `Model::reuse_len` (its own `restorable_len(common)` in `generate`); the merge is independent of it and its `ModelInfo` sets `reuse_cold_cuts_only: false`.

## Follow-up (not done here)

The small-pass splits above differ with 2+ rows per pass on this NVFP4 MoE model. Lane prc's Metal
`prefix_reuse_min_pass_rows = Some(2)` was measured on dense BF16 Qwen3-8B. It still applies to attention-only models, so an
attention-only MoE or NVFP4 checkpoint on Metal may break its premise. Bisecting the op (NVFP4 GEMM vs MoE grouping) would
settle whether `Some(2)` needs a per-model check.
