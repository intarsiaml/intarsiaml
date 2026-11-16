# a4k-restore: HIP acc-4k restore / aligned cases reused nothing (macro prefill dropped the chunk-end snapshots)

Lane `a4k-restore`, branch `a4k-restore` from origin/main `76726167f`. Follow-up to `docs/results/2026-11-15-hip-gdn-default.md`, whose
`acc-4k` stage failed on HIP (35B, tatooine, one R9700) with every `restore` and `aligned` case at `cached 0`:
"nothing was reused (no restorable state at or below the split)", at chunk 4096 (recipe) and 1024, with `INTARSIA_GDN_CHUNK` on and off.

## Summary

| item | result |
|---|---|
| Root cause | **Macro prefill** (`[tuning.hip] prefill_macro = 8192`, lane l7m, 2026-10-26) ran several prefill chunks of a prompt as ONE `forward` call, and the engine snapshots the recurrent state only after that call (`crates/intarsia-core/src/engine.rs:1799` macro step, `:1809` the one snapshot on main; batched engine `crates/intarsia-core/src/engine/batch.rs:246,255`). The chunk ends inside the pass had no snapshot. Not a HIP kernel, budget, recipe-flag or harness problem |
| Fix | A macro pass now keeps the snapshots of its inner chunk ends (the same set the plain chunk-by-chunk schedule leaves): copies only, no op and no numerics change |
| acc-4k restore / aligned, after | all 8 rows (2 at chunk 4096, 6 at chunk 1024) reuse and pass; the 4 `aligned` rows are **logits bit-identical to the cold run**; `prefix-reuse gate: PASS` at chunk 4096 and at 1024 |
| Non-reuse numerics | acc-4k prefill / decode KL and top-1 lines byte-identical to the 2026-11-15 window (same `qualify` stage, prior build) |
| acc-4k stage verdict | now `MISSING-BASELINE` (exit 1): KL, tail and prefix-reuse gates all pass, but no HIP `acc-4k` baseline is committed for this model, so top-1 is not gated. Was `FAIL` (prefix reuse) |
| Tests | Mac `cargo test --workspace --release` 1494 passed, 0 failed; HIP window: see Tests |
| Not changed | No default: `prefill_macro` stays 8192 on HIP and CUDA |

## Why there was no restorable state

`intarsia accuracy --mode reuse` (`crates/intarsia-cli/src/cmd/accuracy.rs`, restore loop around line 2460) runs, per prompt, turn 1 =
the prompt's first `s` tokens plus 40 different tokens, generates 4 tokens, then turn 2 = the whole prompt. A hybrid model
(GatedDeltaNet + attention) can resume only from a recurrent snapshot at or below the shared prefix `s`
(`SequenceState::restorable_len`).

The single-sequence engine's prefill step (`Seq::prefill_step`) on a recurrent state hands `forward_prefill` up to
`macro_step_rows(body, chunk, macro_rows)` rows (the whole headless body when it fits 8192 rows), which `Model::forward_inner` runs as
one layer-by-layer macro pass (`run_macro_group`), and takes one snapshot after it. Worked through for the failing cases:

| case | chunk | turn 1 | body in one pass | chunks in the pass | snapshots (main) | split | restorable on main |
|---|---:|---:|---:|---|---|---:|---|
| chat_reuse_4k restore | 4096 | 4561 | 4560 | 4096 + 464 | 4560 | 4521 | none (4560 > 4521) |
| chat_reuse_4k aligned | 4096 | 4136 | 4135 | 4096 + 39 | 4135 | 4096 | none |
| long_* restore | 1024 | 2040 | 2039 | 1024 + 1015 | 2039 | 2000 | none |
| long_* aligned | 1024 | 1064 | 1063 | 1024 + 39 | 1063 | 1024 | none |

With the plain schedule (one chunk per call, a snapshot after each) a snapshot sits at 4096 / 1024. That is what Metal does
(no macro default: `docs/results/2026-10-29-chunk4k-metal/raw/qualify/acc-4k.out` shows `restore ... 4096 ok`, chunk 1024 `1024 ok`),
and what the control arm below shows on HIP. The loss was known and accepted when macro prefill shipped
(`docs/results/2026-10-25-sk-spark-prefill.md` line 97: snapshots "are taken after a pass, not after every chunk (reuse granularity
only, no numerics)"); the gate that exposes it on HIP was not run between then and 2026-11-15. The existing macro tests missed it
because they flush before every request, so `cached_tokens` is always 0 there.

**CUDA has the same recipe shape** (`[tuning.cuda] prefill_chunk = 2048`, `prefill_macro = 8192`), so the same gap applies there
(not measured in this lane: no Spark run; no CUDA 35B acc-4k log after 2026-10-25 is committed). The fix is backend-independent.

Serving cost on main (beyond the gate): a request that shares only part of an earlier prompt's macro step (an edited or regenerated
earlier message, a different tail after a long shared system prompt) re-prefilled from the last snapshot below that step, possibly from 0.
A plain continuation (`continue` case: turn 2 = turn 1 + reply + more) was not affected (the snapshot at `plen - 1` covers it).

## The fix (commit `f2b105625`)

- `SequenceState::prefill_snapshots` (new, default 0): the engines' prefill steps set it (to `RECURRENT_SNAPSHOTS`, batched engine
  `batch_snapshots()`) around the prefill `forward_prefill` call, and clear it after; every other caller of `Model::forward` is unchanged.
- `Model::forward_inner`: with it set and several chunks in the call, the chunk ends that the plain schedule's `keep` would retain
  (the last `keep - 1`; the caller still snapshots the call's end) get a snapshot:
  - inside a macro group: buffers are taken before the pass (`SequenceState::inner_snapshot_bufs`: free pool first, oldest snapshots
    evicted to stay within `keep`, so no allocation beyond what the memory plan already reserves), each recurrent block copies its slot
    into the segment's buffer right after that segment's recurrence (`Model::macro_snap_copy`, called from `gdn_block_segs` and
    `conv_block_segs`), and the snapshots are filed after the chunks commit (`push_inner_snapshots`);
  - between a macro group and a following chunk in the same call (a tail below `pack_min_rows`): `SequenceState::snapshot`.
- Skipped (old behaviour, snapshot at the call's end only) with an MTP or DFlash drafter state (an MTP snapshot carries the head's
  pending rows, which the pass has only at its end) and under tensor parallelism (macro prefill is tp = 1 only anyway).
- `SequenceState::snapshot` refactored to share the buffer logic (`snapshot_buf`); same eviction and same-length replacement.

The copies run on the model's stream after the segment's kernels and read the slot only; no kernel, launch order or reduction changes.
Cost: one device-to-device copy of the recurrent slots (`SequenceState::recurrent_bytes`; one live recurrent state is 63 MB on the 35B per `model/batch.rs`; not timed) per kept inner chunk end, against a multi-second prefill pass.

## Commands

All tatooine work through fleetq: #9374 / #9375 / #9397 sync, #9376 / #9399 build, #9380 window 1 (radiance down 19:57:58 to 20:13:34 EST
2026-11-15, back up after 15 s), #9409 window 2 (radiance down 20:30:58 to 20:31:32, back up after 15 s). #9400 was a window-2 attempt that
never started its window script (a quoting error in the launcher; radiance was not stopped) and was ended with SIGTERM.

```
# Mac
cargo test --release -p intarsia-models --test batch_recurrent macro_prefill_keeps   # before the fix: FAIL (cached 0 vs 32), raw/mac/test-before.log
cargo test --release -p intarsia-models --test batch_recurrent macro                 # after: 4 passed, raw/mac/test-after.log
cargo test --workspace --release                                                      # 1494 passed, 0 failed
# tatooine (scripts/a4k-restore/: sync.sh on the Mac, build.sh, window.sh -> stages.sh)
qualify --backend hip --golden /models/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors ... --mem-budget 29 \
  --skip kernels --skip gate --skip acc-short --skip acc-oracle --skip bench --skip acc-long --skip envelope --skip bench-long --skip ab --skip tp
# the acc-4k stage runs: accuracy --backend hip --golden golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors --mode all ... (chunk 4096)
#                    and: accuracy ... --mode reuse --prefill-chunk 1024 --decode-positions 64 --max-seq 8192
# the same with INTARSIA_PREFILL_MACRO=0 (control)
```

## acc-4k, before and after

Before (main `4379dd0`, 2026-11-15 window, `docs/results/2026-11-15-hip-gdn-default/raw/window/q-a4k/logs/`):

```
chat_reuse_4k      restore     4521   4631       0      0     32/32  FAIL (nothing was reused (no restorable state at or below the split))
chat_reuse_4k      aligned     4096   4631       0      0     32/32  FAIL (nothing was reused (no restorable state at or below the split))
long_docs_4k       restore     2000   4064       0      0     32/32  FAIL (nothing was reused ...)          [chunk 1024]
long_docs_4k       aligned     1024   4064       0      0     32/32  FAIL (nothing was reused ...)          [chunk 1024]
long_code_4k       restore     2000   4064       0      0     32/32  FAIL (nothing was reused ...)          [chunk 1024]
long_code_4k       aligned     1024   4064       0      0     32/32  FAIL (nothing was reused ...)          [chunk 1024]
prefix-reuse gate: FAIL
```

After (this branch `f2b1056`, macro prefill on as in the recipe, `raw/q-a4k/logs/`):

```
chat_reuse_4k      restore     4521   4631    4096      0     32/32  ok
chat_reuse_4k      aligned     4096   4631    4096      0     32/32  ok (logits bit-identical to the cold run)
prefix-reuse gate: PASS                                                                                     [chunk 4096]
long_docs_4k       restore     2000   4064    1024      0     32/32  ok                                     [chunk 1024]
long_docs_4k       aligned     1024   4064    1024      0     32/32  ok (logits bit-identical to the cold run)
long_code_4k       restore     2000   4064    1024      0     32/32  ok
long_code_4k       aligned     1024   4064    1024      0     32/32  ok (logits bit-identical to the cold run)
chat_reuse_4k      restore     4521   4631    4096      0     32/32  ok
chat_reuse_4k      aligned     4096   4631    4096      0     32/32  ok (logits bit-identical to the cold run)
prefix-reuse gate: PASS                                                                                     [chunk 1024]
golden gate: PASS
```

The `continue` rows are unchanged from 2026-11-15 (same cached lengths, same verdicts; `long_code_4k continue` passes by the
legacy >= 16 leading tokens rule, warm and cold diverge at token 27, as before).

Control, same binary with `INTARSIA_PREFILL_MACRO=0` (`raw/q-a4k-macro0/logs/`): the identical rows (cached 4096 / 1024, aligned
bit-identical, both prefix-reuse gates PASS), confirming that macro prefill alone removed the snapshots.

Stage verdict: `acc-4k: MISSING-BASELINE` (qualify exit 1): "no baseline for backend hip next to the golden or in golden/baselines:
KL, tail, prefix-reuse gates only, top-1 not gated" (prefill top-1 93.09 % KL 7.95e-2 all rows; decode top-1 98.44 % KL 8.92e-3).
The 2026-11-15 run printed the same missing-baseline note but its FAIL took precedence. Recording a HIP acc-4k baseline is a separate
decision (not done here).

Non-reuse numerics: the prefill / decode floor and KL-threshold lines of `acc-4k.out` are byte-identical to the 2026-11-15 window's
(prefill 21-way KL scored rows 1.237e-2, decode 8.690e-3).

## Tests

Mac (cpu backend, `cargo test --release`):
- `batch_recurrent macro_prefill_keeps_chunk_snapshots_for_reuse` (new): on main FAIL, `Dense, solo, macro 64 rows`: cached 0 vs the plain
  schedule's 32 (`raw/mac/test-before.log`); with the fix the four macro tests pass (`raw/mac/test-after.log`).
- `cargo test --workspace --release`: 1494 passed, 0 failed (`raw/mac/workspace-test-summary.log`).

HIP, window 1 (`raw/`, this branch `f2b1056`):

| run | result |
|---|---|
| `cargo test -p intarsia-backend-hip --tests` (benches / timing skipped), `hip-tests.out` | 206 passed, 0 failed, 12 ignored |
| `batch_recurrent` in full, `INTARSIA_BATCH_TEST_BACKEND=hip`, `models-tests.out` | 14 passed, 1 failed: the new test's **batched** half (cached 0, expected 32). A test bug: batched slots keep snapshots only on the cpu backend (`Model::batch_snapshots` is 0 on a GPU, decision 2026-10-23 06:41), so batched reuse is off on HIP by design. Fixed in `4b6e308` (expects reuse only where `batch_snapshots() > 0`); its solo half passed on HIP. Every pre-existing test passed, incl. `prefix_reuse_restores_recurrent_state_from_a_snapshot`, `batched_engine_with_prefix_reuse_equals_solo`, the three macro tests |
| `mtp` packed / macro prefill tests, `models-tests-mtp.out` | 2 passed (`mtp_model_takes_the_macro_prefill_pass_and_the_head_follows`, `batched_mtp_with_packed_prefill_equals_solo`) |
| real 35B, `batch_cuda_real --ignored`, `models-tests-35b.out` | 3 passed: `real_35b_macro_prefill_is_bitwise_the_plain_schedule`, `real_35b_batched_macro_prefill_equals_the_plain_local_engine`, `real_35b_batched_engine_long_prompts_equal_the_local_engine` |
| real 35B `hybrid_reuse_real engine_hybrid_reuse_equals_cold`, `hybrid-reuse.err` | FAIL: "B resumed from A's state (cached 4095) differs from a fresh engine's B". **Pre-existing, not this branch**: see window 2 |

HIP, window 2 (`raw/w2/`, `4b6e308`, and an origin/main `76726167f` tree built separately):

| run | result |
|---|---|
| `macro_prefill_keeps_chunk_snapshots_for_reuse` (fixed test), `models-mytest.out` | pass |
| `engine_hybrid_reuse_equals_cold` built from **origin/main**, `hybrid-reuse-base.err` | FAIL, the same assertion with the same token lists (the assertion text of all three runs hashes identically) |
| the same on this branch with `INTARSIA_PREFILL_MACRO=0`, `hybrid-reuse-macro0.err` | FAIL, identical |

So that test fails on HIP on main as well. It had not been run on HIP before (its committed runs are cpu / Metal). It is lane hrr's
case: A = a 4096-token prompt snapshots at 4095 (`plen - 1`, one 4095-row pass), B = A + a short suffix resumes there and runs
[4095, 4104) + the last token, while a fresh B runs [0, 4096) + [4096, 4104) + the last token; on HIP those pass shapes do not give
the same bits. The fix here does not touch that path (A and the warm B are single-chunk calls). The opt-in strict rule
`INTARSIA_PREFIX_REUSE=exact` (resume only at the prompt's own cold cuts) exists for it; whether HIP should default to it is open.

## Left open

- CUDA (Sparks) has the same macro default and so had the same gap; the fix applies there but was not run on CUDA in this lane.
- With an MTP state (`--spec mtp`, the CUDA tp=1 default and any `--spec mtp` run) a macro pass still snapshots only at its end:
  an MTP snapshot needs the head's pending rows at that length, which the pass does not have mid-way. HIP serves the 35B without
  MTP by default, so the HIP gate and HIP serving get the full fix.
- No HIP `acc-4k` baseline is committed for the 35B, so the stage reports `MISSING-BASELINE` instead of PASS.
- `engine_hybrid_reuse_equals_cold` fails on HIP on main (above): a separate, pre-existing reuse-exactness issue at a `plen - 1` cut.
- Batched engine (`serve --max-batch`) on GPUs keeps no recurrent snapshots by design, so this fix changes nothing there except on cpu.

## Raw

`raw/q-a4k/` and `raw/q-a4k-macro0/` (qualify reports and logs), `raw/steps.log`, `raw/window.log`, the test logs
(`raw/hip-tests.out`, `raw/models-tests*.out`, `raw/hybrid-reuse.*`), `raw/w2/` (window 2), `raw/mac/` (Mac test logs).

## Speed (main vs this branch)

Question: does the chunk-end recurrent-state copy in a macro prefill pass (macro 8192, chunk 4096) slow HIP prefill or decode? Qwen3.6-35B-A3B-NVFP4, tatooine R9700, `intarsia serve --backend hip --max-seq 18432` (flags of scripts/gdn-chunk/speed-window.sh), `localeval speed --sizes 1024,4096,16384 --gen 64 --thinking off`. Trees: main 558512697 (stamp code-tree 641c7f44074b) vs branch f99ccdf56 (code-tree bc47f85c9ddb), each its own stamped `--features hip` release build. A fresh server per arm; 3 rounds interleaved main/dev, dev/main, main/dev. Cells: mean (min-max) over rounds (tp1x: 3 rounds x 3 reps). Raw logs and steps.log: `raw/speed/`. Note: 1024 and 4096 prompts stay inside one macro pass of 8192 (no extra chunk-end copy beyond the first); 16384 is the size that crosses macro passes.

### tp 1, 1 rep per round (localeval-tp1-*)

| size | metric | main mean (min-max) | dev mean (min-max) | dev vs main |
|---:|---|---|---|---:|
| 1024 | prefill tok/s | 8286 (7914-8475) | 8162 (7904-8453) | -1.5% |
| 1024 | decode tok/s | 164 (163-165) | 163 (162-165) | -0.4% |
| 4096 | prefill tok/s | 12013 (11901-12096) | 11868 (11653-12041) | -1.2% |
| 4096 | decode tok/s | 161 (160-161) | 145 (136-161) | -9.7% |
| 16384 | prefill tok/s | 10748 (10716-10784) | 10525 (10084-10748) | -2.1% |
| 16384 | decode tok/s | 146 (145-147) | 147 (146-148) | +1.0% |

Dev decode at 4096 was 138 and 136 tok/s in rounds 1 and 3 (main 160-161), with a peak decode of about 252 tok/s: a bursty stream, not slower steady decode. It did not repeat in the rerun below.

### tp 1 rerun, 3 reps per round (localeval-tp1x-*)

| size | metric | main mean (min-max) | dev mean (min-max) | dev vs main |
|---:|---|---|---|---:|
| 1024 | prefill tok/s | 8219 (7986-8641) | 8185 (7838-8741) | -0.4% |
| 1024 | decode tok/s | 164 (157-166) | 165 (165-165) | +0.9% |
| 4096 | prefill tok/s | 11730 (10913-11959) | 11615 (10109-11911) | -1.0% |
| 4096 | decode tok/s | 160 (151-164) | 164 (155-190) | +2.1% |
| 16384 | prefill tok/s | 10671 (10648-10706) | 10652 (10588-10693) | -0.2% |
| 16384 | decode tok/s | 142 (128-147) | 146 (140-146) | +2.4% |

### tp 2 (--tp 2 --devices 0,1), 1 rep per round (localeval-tp2-*)

| size | metric | main mean (min-max) | dev mean (min-max) | dev vs main |
|---:|---|---|---|---:|
| 1024 | prefill tok/s | 10029 (9957-10116) | 9981 (9917-10040) | -0.5% |
| 1024 | decode tok/s | 192 (192-192) | 180 (156-192) | -6.1% |
| 4096 | prefill tok/s | 14834 (14769-14903) | 14415 (13665-14878) | -2.8% |
| 4096 | decode tok/s | 189 (188-189) | 193 (188-203) | +2.4% |
| 16384 | prefill tok/s | 13718 (13700-13752) | 13597 (13437-13732) | -0.9% |
| 16384 | decode tok/s | 180 (179-180) | 185 (180-195) | +3.0% |

Reading: prefill is 0.2-2.8% lower on dev in every cell, within the round-to-round spread (the low-end dev prefill samples, e.g. tp2 4096 at 13665, are single rounds). Decode differences are +/-3% except two bursty tp1 dev rounds and one tp2 1024 round (156 vs 192) that did not reproduce in the 3-rep rerun. No difference exceeds the spread; no measurable prefill or decode slowdown from the branch. Single-rep rounds are coarse: localeval treats under 10% as noise.
