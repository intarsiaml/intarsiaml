# HRR: hybrid prefix reuse resumed off the cold schedule (2026-10-29)

Lane `hrr`, branch `claude/bold-jackson-f249a3`, base `main` `694b519cf` (#258). endor (M5 Max, macOS 27.0), Qwen3.6-35B-A3B-NVFP4
(`~/models/Qwen3.6-35B-A3B-NVFP4`, 30 GatedDeltaNet + 10 gated-attention layers), Metal, prefill chunk 4096 (recipe). Raw:
`2026-10-29-hrr-hybrid-reuse/raw/`.

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

`Model::reuse_len`, recurrent branch: on a backend that is not `prefix_reuse_any_cut` (Metal, CUDA, HIP), and unless
`INTARSIA_PREFIX_REUSE=any`, resume only at `SequenceState::recurrent_reuse_len(want, cuts)`. That is the longest snapshot
`L <= want` that is one of `cuts` = `scheduler::recurrent_schedule_cuts(plen, chunk, tail_merge)`, the pass ends of the
prompt's own cold recurrent schedule. In addition, the rows `0..L` must have been written by prefill passes (`Plain` /
`RowInvariant`) in an unbroken run from row 0, each ending at one of `cuts`. Rows written by a decode step, a verify pass or
a prefix-store restore end the run. The cpu backend (any cut) keeps `restorable_len`. The batched engine calls the same
function (`engine/batch.rs`).

| case (35B, chunk 4096) | before | after |
|---|---|---|
| B = A + suffix after A | cached 4095, text differs | cached 0, = cold |
| A repeated | cached 4095, = cold | cached 4095, = cold |
| A after B | cached 4095 | cached 0 (B never held 4095), = cold |
| prompt > 4096 sharing the first 4096 with the held one | snapshot at 4096 | snapshot at 4096 (a cut of both) |
| next chat turn, history < 4096 | held sequence incl. decode rows | 0 |

Cost: without a prefix store a hybrid model on Metal, CUDA or HIP no longer resumes a next chat turn or an extended prompt
inside a chunk. The prefix store (`--prefix-store`) cuts recurrent prefill at message boundaries
(`SeqStore::chunk_end`) and resumes there bitwise. It is the way to get bitwise multi-turn reuse on these models.

## Tests

* `crates/intarsia-core/tests/scheduler.rs` `recurrent_schedule_cuts_follow_the_solo_schedule`: the cuts, with and without
  tail merge.
* `crates/intarsia-models/tests/hybrid.rs` `chunk_sensitive_backend_resumes_a_recurrent_state_only_at_cold_cuts`: tiny hybrid
  fixture on a cpu backend reporting Metal's flags. Repeat 19, A + 3 tokens 16 (not 19), next chat turn 16, short prompt
  extended 0. All give the same tokens as a fresh engine. The cpu backend itself still gives 19.
* `crates/intarsia-models/tests/hybrid_reuse_real.rs` (opt-in `INTARSIA_PROBE_DIR`): `hybrid_reuse_shapes`, the diagnostic
  above (`INTARSIA_PROBE_ASSERT=1` makes it a gate on the shapes). `engine_hybrid_reuse_equals_cold` is the gate: A then B on
  one engine equals a fresh engine's B. A after B and A repeated equal cold A, with the cached counts above.

* endor `cargo test --workspace --release --no-fail-fast`: 1112 passed, 1 failed (`docs_env_vars`: the two new probe variables had no row in docs/tuning.md; rows added, re-run passes), 28 ignored. 35B engine gate: B after A cached 0 = cold; A after B cached 0 = cold; A repeated cached 4095 = cold (`raw/tests-run2.log`).

## Not run: qualify's reuse gate

`accuracy --mode reuse` (qualify's `acc-4k` / `acc-4k-reuse` on the 35B) has a `continue` case that requires
`cached >= split`: turn 2 must reuse the whole of turn 1. The rule here resumes a hybrid at the last chunk cut below turn 1's
end. At chunk 1024 with a 4k prompt split near the middle that is 1024 < ~2032. Read from the code, that case will report
"reuse covered X < the S tokens of turn 1" on Metal, HIP and CUDA. The `restore` / `aligned` cases (snapshots at chunk
boundaries) keep passing. On `main` the 35B Metal `acc-4k` already FAILS on `long_code_4k continue`, for another reason
(docs/results/2026-10-29-chunk4k-metal.md). Not re-run here. Either the `continue` case accepts the last cold cut on a
chunk-sensitive backend (as it would have to for a dense model under the bi8 ladder), or this rule needs an opt-out for it.

## Follow-up (not done here)

The small-pass splits above differ with 2+ rows per pass on this NVFP4 MoE model. Lane prc's Metal
`prefix_reuse_min_pass_rows = Some(2)` was measured on dense BF16 Qwen3-8B. It still applies to attention-only models, so an
attention-only MoE or NVFP4 checkpoint on Metal may break its premise. Bisecting the op (NVFP4 GEMM vs MoE grouping) would
settle whether `Some(2)` needs a per-model check.
