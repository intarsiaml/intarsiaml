# BI8: served Qwen3-8B batched != solo on CUDA was prefix reuse, not the batch width (2026-10-28 evening, doc dated 10-29)

Lane `bi8`, branch `lane-bi8/serve-batch-invariance`, base `origin/main` `db1d81d4b` (#227). spark2 (spark-fa00, GB10 sm_121),
`spark2-vllm` (retro, 38 GB) up and idle. Bug from SB8 (`2026-10-29-sb8-spark-8b-multistream.md`): served greedy on the 8B
differed from a solo server once 5+ streams ran, on `--max-batch 8` and 16, with or without the #224 GEMV knobs and CUDA graphs.
Raw: `2026-10-29-bi8-serve-batch-invariance/raw/`.

## Result

**Root cause: KV prefix reuse resumed a dense prefill at a row its cold schedule never cuts at; on CUDA that changes the bits.**
Both engines reuse the KV prefix a slot holds: `common_prefix(held, prompt).min(plen - 1)` (`crates/intarsia-core/src/engine.rs:1206`
for the single-sequence engine, `crates/intarsia-core/src/scheduler.rs:857-858` for the batched engine's `pick_slot`), and for an
attention-only state any length was accepted (`crates/intarsia-core/src/state.rs:436`, `restorable_len` returns `want`). A slot that
had already served the same prompt resumed it at `plen - 1`: the prompt's last token ran as a 1-row pass. On CUDA a prefill pass's
bits depend on its row count, so that request computed other logits than its cold prefill, and near-tie argmaxes flipped.

- **Why "5 or more streams":** the prompts that flip are equal.py's prompts 4, 5 and 8, which only exist at n >= 5 (n >= 9 for 8). At
  n <= 4 the reused prompts (0-3) happened not to be near ties.
- **Why reps differed:** a prompt flipped only when it found its own earlier slot idle (rep 2 onward) or a slot sharing its first
  token; which slot a request lands in depends on arrival order. SB8's log: prompt 4 is clean the first time a server sees it
  (n=5 rep 0) and differs at token 48 in every later run.
- **Why the model-level gate passed:** it prefills every sequence cold and compares decode steps; it never resumes a prompt.
- The 35B (hybrid) never had it: a recurrent state reuses only its snapshots, and its cold schedule always runs the last prompt token
  alone, so a repeated prompt resumes at a point of its own schedule.

The batch width was never the problem: batched decode is bitwise the solo step at every N 2..16 (gate below).

### Evidence (before the fix)

Model level, `real_qwen3_8b_prefix_reuse_probe` (`raw/run1/probe.log`, 8B on CUDA, the 16 equal.py prompts):

| case | result |
|---|---|
| repeated prompt resumed with its last token alone (cold state truncated to `plen - 1`) | logits differ for **16 / 16** prompts (max abs 0.11-0.23); greedy tokens flip for prompts **4 at 48, 5 at 3, 8 at 3**: exactly SB8's served flips |
| every split `forward(P[..k])` + `forward(P[k..])`, k = 1..plen-1 | **every** split of every prompt differs from the one-pass prefill (first row) |
| another prompt's state truncated to the shared first token ("The", "In", "A") | 14 / 34 pairs equal, 20 differ: bitwise only for some row-count pairs, so no "tail of at least N rows" rule is safe |

Serve level, the SB8 setup (`--max-batch 16`, cap-16 recipe), reps on one server, compared with the model-level cold reference
(`raw/run1/`):

| server | n=5 x3 | n=8 x3 | n=16 x3 |
|---|---|---|---|
| reuse on (as shipped) | 1 / 2 identical (rep 2: (4, 48)) | 0 / 2: (4, 48), (5, 3) | 0 / 2: (4, 48), (5, 3), (8, 3) |
| `INTARSIA_CB_NO_REUSE=1` | 3 / 3 | 3 / 3 | 3 / 3 |

SB8's solo reference (`ident/solo.json`) equals the cold reference for all 16 prompts.

## Fix

`Model::reuse_len` (`crates/intarsia-core/src/model.rs:654`), called by the single-sequence engine (`engine.rs:1214`) and by the
batched engine's slot choice (`BatchModel::reuse_len`, `scheduler.rs:863`, `engine/batch.rs:204`):

- recurrent state: unchanged (`restorable_len`, snapshots);
- attention only, on a backend whose prefill may resume anywhere (`Backend::prefix_reuse_any_cut`, default true: cpu, Metal, HIP):
  unchanged;
- attention only on CUDA (`prefix_reuse_any_cut` false, `crates/intarsia-backend-cuda/src/device.rs`): only a point `L` of the
  prompt's own cold chunk schedule (`scheduler::chunk_schedule_tm`), and only if the held sequence wrote rows `0..L` with exactly
  those passes, full `chunk`-row passes `(0, C), (C, 2C), .., (L - C, L)` (`SequenceState::ladder_reuse_len`, `state.rs:453`).
  `SequenceState::passes` records every committed pass (`commit`, `state.rs:622`); a truncate drops the passes past the new length, a
  prefix-store restore leaves none. Rows a decode step, a busy 256-row chunk or a resumed prefill wrote never qualify. Exact by
  construction: the resumed request runs the remaining chunks of its cold schedule on bit-identical KV.

`INTARSIA_PREFIX_REUSE=any|exact` overrides the backend's choice (`docs/tuning.md`). Cost: on CUDA a dense prompt shorter than one
prefill chunk (4096 on the 8B) is never resumed; a repeated or follow-up prompt resumes at the last full 4096-row chunk its earlier run
prefilled. Requests whose prompts start with a nonce (localeval) were never reused, so the speed sweep does not see this. The prefix
store (`--prefix-store`) keeps its own exact rule and is unchanged.

Also in the PR: SB8's `raw/qwen3-cap16.patch` (`recipes/qwen3.toml` `max_batch` 8 -> 16, `tests/batch.rs` 9 -> 17).

## Identity after the fix (spark2, `raw/run2/`)

| check | result |
|---|---|
| `real_qwen3_8b_batched_decode_is_bitwise_the_solo_step` (N 2, 3, 4, 8, 16 x 130 steps, then every other N 5..15 x 32 steps) | **pass**: 4290 + 3264 (row, step) pairs bitwise |
| `real_qwen3_8b_batched_engine_equals_the_cold_reference` (new; single-sequence engine over repeats and shared first tokens, then 8 and 16 slots: 3 rounds of all 16 without a restart, a round of 5, a round of repeats; greedy + seeded sampling) | **pass**: 13 + 59 + 59 requests equal the cold reference |
| served greedy vs cold, solo server, 1 / 4 / 16 streams x 3 reps on one server | **9 / 9** identical |
| served greedy vs cold, `--max-batch 16`, 1 / 4 / 8 / 16 streams x 3 reps on one server | **12 / 12** identical |
| served greedy vs cold, `--max-batch 8`, 8 / 16 streams x 2 reps | **4 / 4** identical |
| equal.py 21 cells (7 sampling settings x 1 / 4 / 16 streams), solo vs `--max-batch 16` | **21 / 21** identical, 9408 ids (SB8: 7 / 21; the sampled cells were the same reuse effect) |
| mutant: the engine gate with `INTARSIA_PREFIX_REUSE=any` (the old rule) | **caught**: fails at the single-sequence engine's 2nd request (prompt 4 repeated, ids differ from token 48; `raw/run3/mutant.log`) |

## Speed at 16 streams (localeval A/B, ABAB, one window)

`localeval speed --sizes 1024 --gen 256 --reps 3 --concurrency 16 --thinking off`, server `--max-batch 16 --max-seq 8192
--mem-budget 36`; base = origin/main `db1d81d4b` with the cap-16 recipe passed as `--recipe` (SB8's 163.0 setup), fix = this branch.

| arm | round 1 | round 2 | median |
|---|---:|---:|---:|
| base | 162.0 | 161.5 | 161.8 |
| fix | 161.6 | 161.6 | **161.6** |

Fix vs base: -0.1 % (noise; decode noise is about 5 %). Against SB8's 163.0 (another window): -0.9 %, inside the 2 % bar. localeval
runs `20261028-221552-adhoc-bi8-base-r1`, `-221854-...-fix-r1`, `-222157-...-base-r2`, `-222501-...-fix-r2`
(`/opt/code/deepeval/results/`; `raw/run2/speed-*.log`). The sweep's prompts start with a nonce, so neither arm reuses a prefix.

## Other models (`raw/run3/`)

| model | path | affected before? | after |
|---|---|---|---|
| Qwen3.6-35B-A3B (hybrid) | recurrent state: `reuse_len` returns `restorable_len`, as before | no: snapshots only, and the cold schedule runs the last token alone | `real_35b_batched_*` pass (4290 pairs; batched engine == local engine); served equal.py 21 / 21 cells base vs fix at `--max-batch 16`, 9408 ids |
| Gemma-4-26B-A4B (dense attention, recipe `max_batch = 1`) | single-sequence engine only | yes in kind: the solo engine resumed any prefix on CUDA, so a repeated or prefix-sharing request could leave its cold bits | the exact rule applies (no reuse for prompts under one chunk); `plan show` unchanged; not re-measured at serve level |
| Qwen3-8B, other dense Qwen3 / Llama / Qwen2 on CUDA | both engines | yes | exact rule |

`plan show` (json and text, `--max-seq 8192 --mem-budget 40`) is byte-identical base vs fix for the 8B (8758 / 1478 B), the 35B
(15336 / 3946 B) and Gemma-4 (8401 / 1790 B) (`raw/run3/plan.log`).

## Gates

| gate | result |
|---|---|
| served greedy == solo on the 8B at 1 / 4 / 8 / 16 streams, 3 reps | pass (solo 9 / 9 and `--max-batch 16` 12 / 12 vs the cold reference; equal.py 21 / 21 solo vs batched) |
| model-level batched == solo at every N up to 16 (8B) | pass (`real_qwen3_8b_batched_decode_is_bitwise_the_solo_step`, now N 2..16 on CUDA) |
| new engine-level batched == cold reference (8B) | pass; catches the old rule |
| shared path touched: CUDA `rows_exact` | pass, 9 passed, 2 ignored (`raw/run3/rows_exact.log`) |
| shared path touched: `batch_cuda_real` on the 35B | pass (`raw/run3/batch35.log`) |
| `plan show` byte-identical, 8B / 35B / Gemma-4 | pass (6 / 6 files) |
| 35B 21-cell served identity (#224's set) at 16 streams | pass, 21 / 21 base vs fix |
| Mac `cargo test --workspace --release` | pass, 1085 passed, 0 failed (`raw/mac-test.log`) |
| cpu policy tests (`crates/intarsia-models/tests/exact_reuse.rs`) | pass (exact rule: short prompt never resumed, repeat at 24 of 30, shared 20 -> 16, follow-up turn 24; any-cut backend unchanged) |
| 16-stream speed within 2 % of 163.0 | pass, 161.6 (-0.9 %; same-window base 161.8) |
| CI | see the PR |

## Not covered / follow-ups

- **HIP and Metal keep the old rule** (`prefix_reuse_any_cut` true). Metal's own backend note says a 1-row prefill chunk is not bitwise
  the same row inside a bigger chunk (`crates/intarsia-backend-metal/src/lib.rs:680`), so a repeated prompt resuming with its last
  token alone is the same bug there; HIP prefill is likely chunk-sensitive too (recipes/qwen3.toml, the HIP reuse judge
  tolerances). Neither was measured here (spark2 only); flipping either is a one-line override once a probe like
  `real_qwen3_8b_prefix_reuse_probe` shows it.
- A request that prefills while others decode runs 256-row chunks, which are not bitwise the solo 4096-row chunk for prompts above
  256 tokens (known, `2026-10-23-rb2-cuda-batching.md` item 9): unchanged, and not exercised by the short equal.py prompts.
- Multi-turn chat on CUDA now re-prefills the previous turn's tail and reply instead of reusing them (time to first token, not
  decode); `--prefix-store` restores message prefixes exactly and is the tool for that case.

## Commands

```
# builds (spark2, CPU slot)
fleetq run --host spark2 --load cpu --mem 8 --expect 40 --label "bi8: build ..." -- scripts/bi8/build.sh        # fix -> ~/intarsiaml-bi8/bin/intarsia
fleetq run --host spark2 --load cpu --mem 8 --expect 25 --label "bi8: build base" -- scripts/bi8/build-base.sh  # origin/main -> bin/base
# GPU jobs (quiet slot)
OUT=... fleetq run --host spark2 --quiet gpu,cpu --mem 36 --expect 25 --label "bi8: probe + served reuse on/off" -- scripts/bi8/run1.sh
OUT=... REF=raw/run1/cold.json fleetq run --host spark2 --quiet gpu,cpu --mem 36 --expect 50 --label "bi8: gates + identity + A/B" -- scripts/bi8/run2.sh
OUT=... fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 45 --label "bi8: 35B + plan + mutant" -- scripts/bi8/run3.sh
# cold reference from the probe's COLD lines; served runs compared with it
python3 scripts/bi8/cold-ref.py raw/run1/probe.log raw/run1/cold.json
python3 scripts/bi8/greedy.py compare raw/run1/cold.json raw/run2/greedy-mb16.json
```
