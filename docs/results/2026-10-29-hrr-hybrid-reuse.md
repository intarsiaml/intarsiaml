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
function (`engine/batch.rs`). Neither engine uses 1-row avoidance for recurrent states (`Seq::prefill_step`,
`engine/batch.rs` `avoid_single_row_prefill`), so the cuts match both. A batched prefill that runs at the busy chunk (2048
beside decoders) writes passes ending at multiples of 2048, which are not 4096-chunk cuts, so it resumes nothing: this is conservative.

| case (35B, chunk 4096) | before | after |
|---|---|---|
| B = A + suffix after A | cached 4095, text differs | cached 0, = cold |
| A repeated | cached 4095, = cold | cached 4095, = cold |
| A after B | cached 4095 | cached 0 (B never held 4095), = cold |
| prompt > 4096 sharing the first 4096 with the held one | snapshot at 4096 | snapshot at 4096 (a cut of both) |
| next chat turn, history < 4096 | held sequence incl. decode rows | 0 |

Cost: without a prefix store a hybrid model on Metal, CUDA or HIP no longer resumes a next chat turn or an extended prompt
inside a chunk. The prefix store (`--prefix-store`) cuts recurrent prefill at message boundaries
(`SeqStore::chunk_end`) and resumes there (its own bitwise gate: `tests/prefix_store_metal.rs`; not re-measured here). It is the way to get bitwise multi-turn reuse on these models.

## Tests

* `crates/intarsia-core/tests/scheduler.rs` `recurrent_schedule_cuts_follow_the_solo_schedule`: the cuts, with and without
  tail merge.
* `crates/intarsia-models/tests/hybrid.rs` `chunk_sensitive_backend_resumes_a_recurrent_state_only_at_cold_cuts`: tiny hybrid
  fixture on a cpu backend reporting Metal's flags. Repeat 19, A + 3 tokens 16 (not 19), next chat turn 16, short prompt
  extended 0. All give the same tokens as a fresh engine. The cpu backend itself still gives 19.
* `crates/intarsia-models/tests/hybrid_reuse_real.rs` (opt-in `INTARSIA_PROBE_DIR`): `hybrid_reuse_shapes`, the diagnostic
  above (`INTARSIA_PROBE_ASSERT=1` makes it a gate on the shapes). `engine_hybrid_reuse_equals_cold` is the gate: A then B on
  one engine equals a fresh engine's B. A after B and A repeated equal cold A, with the cached counts above.

* After the merge with main (`e72ce1964`) and the gate change: endor `cargo test --workspace --release --no-fail-fast` (fleetq #7841, before the one-line `hybrid.rs` assertion; `-p intarsia-models --test hybrid` and `-p intarsia-cli` re-run after it, #7844): 1199 passed, 0 failed, 29 ignored. Plan show for the 35B and Qwen3-8B on Metal identical to main, `kernels test --backend metal` 0 FAIL (#7846).
* Before the merge: endor `cargo test --workspace --release --no-fail-fast`: 1112 passed, 1 failed (`docs_env_vars`: the two new probe variables had no row in docs/tuning.md; rows added, re-run passes), 28 ignored. 35B engine gate: B after A cached 0 = cold; A after B cached 0 = cold; A repeated cached 4095 = cold (`raw/tests-run2.log`).

## Cost of the rule and the qualify gate (2026-11-02, after the merge with main `e72ce1964`)

endor, Metal, 35B, chunk 4096 (recipe), `serve --spec off`, greedy, raw `/v1/completions`, `INTARSIA_PREFIX_REUSE=any` is the old rule.
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

## The qualify gate: decision

`accuracy --mode reuse` `continue` required `cached >= split` (turn 2 reuses all of turn 1). On the 35B at qualify's chunk 1024 the new rule resumes at 1024 for split 2000 (old rule: 2015), so the gate would print "reuse covered 1024 < the 2000 tokens of turn 1" on Metal, HIP and CUDA (read from the code; the unmodified gate was not run on this head).

Options:

1. **Accept the last cold cut** (implemented). `ModelInfo::reuse_cold_cuts_only` (`Model::reuse_recurrent_at_cold_cuts_only`: recurrent slots, the backend is not `prefix_reuse_any_cut`, `INTARSIA_PREFIX_REUSE` is not `any`) lowers the `continue` bar from `s` to `s / chunk * chunk`. Every other model (dense, cpu, DSv4.1) keeps `cached >= s`. The gate still compares the stream or logits with a cold run, so a wrong-bits resume still fails.
2. An opt-out that makes the gate test the old rule (`INTARSIA_PREFIX_REUSE=any` in qualify). It passes `cached >= s` only by validating the behaviour that served wrong text, and a serving default that qualify does not exercise.

Option 1 is recommended: it gates what the server does and keeps the bit-exact `aligned` and `restore` checks. Result on Metal with option 1 (`raw/accuracy-reuse-35b-metal.txt`, fleetq #7846; qualify's `acc-4k-reuse` command at chunk 1024 with `--reuse-require-all`): prefix-reuse gate PASS, 3 judged, 0 skipped: long_docs_4k continue cached 1024 (floor 1024), long_code_4k 1024, chat_reuse_4k split 4521 cached 4096 (floor 4096); restore and aligned unchanged and bit-exact. The long_docs_4k continue oracle check passes with warm/cold 1.71x against a 1.5x + 3e-3 limit (1.54e-2 vs 1.65e-2): thin, and above the 0.99x the same case gave at cached 2015 (docs/results/2026-10-22-gc2-4k-continue-oracles); not investigated here (the resume now recomputes rows 1024..2000 with prefill kernels where the old rule kept rows the decode kernels wrote).

CUDA and HIP: read from the code only. Both report `prefix_reuse_any_cut` false (`backend-cuda/src/device.rs:201`, `backend-hip/src/device.rs:799`), so the 35B there takes the same branch and the same floor applies. Not run (no spark run, tatooine untouched).

DSv4.1 (`dsv41_engine.rs`) does not use `Model::reuse_len`: `generate` resumes at `SequenceState::restorable_len(common)` of its own state. This PR neither changes nor tests that path; the merge is textually and semantically independent of it (its `ModelInfo` sets `reuse_cold_cuts_only: false`, so its gate bar is unchanged).

## Follow-up (not done here)

The small-pass splits above differ with 2+ rows per pass on this NVFP4 MoE model. Lane prc's Metal
`prefix_reuse_min_pass_rows = Some(2)` was measured on dense BF16 Qwen3-8B. It still applies to attention-only models, so an
attention-only MoE or NVFP4 checkpoint on Metal may break its premise. Bisecting the op (NVFP4 GEMM vs MoE grouping) would
settle whether `Some(2)` needs a per-model check.
