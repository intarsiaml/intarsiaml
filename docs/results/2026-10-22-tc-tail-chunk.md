# Lane tc: opt-in prefill tail merge (`INTARSIA_PREFILL_TAIL_MERGE`)

2026-10-22. Branch `lane-tc/tail-chunk` from origin/main `62c759b7`. Raw logs: `2026-10-22-tc-tail-chunk/raw/` (windows `wA` .. `wE`, `cuda`).

## Headline

* **What it does.** A prompt whose remainder after a full chunk is at most `<rows>` tokens runs that remainder inside the last chunk instead of as a separate short chunk. localeval's 16k prompt (16654 tokens) at the HIP 35B recipe chunk of 4096 runs `4096 x 3, 4366` (4 passes) instead of `4096 x 4, 270` (5 passes). Off by default; with the knob unset every chunk plan, workspace size and plan probe is unchanged (tests below, and base == off bitwise on the GPU).
* **Saving (R9700, 35B NVFP4, exclusive slot, fresh process per run, n=5, arms interleaved, `INTARSIA_PREFILL_TAIL_MERGE=1024`).** Prefill of the 16654-token prompt: 1930 ms (1917 to 1943) base, 1933 off, **1871 ms (1862 to 1884) on**: -59 ms, 8629 -> 8901 tok/s (+3.2 %). Prefill of a 4408-token prompt (`4096 + 312`): 506 ms base, 505 off, **460 ms on**: -46 ms, 8711 -> 9583 tok/s (+10.0 %). The 4089-token prompt (localeval's 4k, shorter than a chunk) is unchanged: 428 ms on all three arms, because there is nothing to merge.
* **Bitwise.** 256 greedy ids after the 16654 and the 4408-token prompts are identical on base, off and on. `accuracy` self-golden, base vs off vs on, on both HIP goldens (4k: prompts 4064 / 4064 / 4631 tokens; long: 5 prompts up to 12064), prefill and teacher-forced decode: off and on print the same statistics to every digit (mean KL21 2.345e-13 and 8.168e-12 on the 4k golden, 2.680e-13 and 4.202e-12 on the long golden, 0 top-1 flips, verdict PASS). That 1e-13 is the comparator's floor (log-probs stored as f32), the same for off and on, so the merged 4631-token prompt differs from the unmerged one by less than the printed precision. Measured on HIP 35B only.
* **Not claimed.** The saving is the fixed cost of one extra pass over the weights (the MoE experts are re-read for 270 rows: +47 ms of PF's +106 ms), not the per-row work, which the merged chunk still does. PF's "about 111 ms" was therefore the upper bound for a free tail; the lever recovers about 60 of it (about half). It does not move the 4k number, and it only helps prompts whose length is within `<rows>` above a multiple of the chunk.

## Design

* `scheduler::next_prefill_chunk_tm(remaining, chunk, avoid_one_row, tail_merge)`: with `tail_merge > 0`, `chunk < remaining <= chunk + min(tail_merge, chunk)` returns `remaining` (one pass); everything else is `next_prefill_chunk`. `tail_merge = 0` is exactly `next_prefill_chunk` (tested for every `remaining` up to three chunks, chunks 1, 2, 3, 256, 2048, 4096, both `avoid_one_row`). `chunk_schedule_tm` is the same for a whole prompt. `next_prefill_chunk` and `chunk_schedule` are untouched.
* **Fold, not an even split.** An even split of `4096 + 270` into `2183 + 2183` keeps five passes and saves nothing: the cost being removed is the pass count (each pass re-reads the weights). Only the fold removes a pass.
* **Memory plan.** The activation workspace and the MoE scratch were sized for `prefill_chunk` rows, so a longer chunk would overrun them. `Model::load` reads the knob, clamps it to `chunk` (a larger value is not an allocation of that many rows; it is the same as `chunk`), and sizes the workspace for `chunk + rows`; `ws_bytes` feeds `MemoryPlan::compute_budgeted`, so the KV budget shrinks by the extra rows. `Model::prefill_chunk()` still returns `chunk`, so callers that split on it, speculative verify rows and the batch scheduler's `solo_chunk` are unchanged. The kernel scratch is sized from the plan's dry-run probes, which only covered `chunk`; the first GPU window failed with `memory plan: kernel needs 178932736 scratch bytes but the plan reserved 167905792` and the plan now also probes `chunk + 1` and `chunk + rows` rows, at the start and the end of the context, when the knob is set (nothing is added when it is off). A malformed value is refused at load. With `--tp-remote` the knob must be set on every rank (a remote rank reads its own environment; if it is unset there, rank 0 would run one 4366-row pass while the remote rank splits it 4096 + 270 into a smaller workspace and the collectives diverge); in-process TP peers share the environment and are fine. Not tested.
* **Callers.** `engine::Seq::prefill_step` (also for recurrent models, whose last token still runs alone: a merged remainder runs as `remaining - 1` rows and then one row), `Model::forward_chunks` (so `accuracy` and every direct `forward` see the same plan), the batch scheduler (`BatchModel::tail_merge`, default 0; the adapter returns the model's value) and `chunk_schedule_tm`. The head groups of `AllLogits` (`LOGIT_ROWS`) keep the plain rule.
* **Bitwise is a measurement, not a property.** Chunk invariance was only established for chunks of 256 / 1024 / 2048 / 4096 rows. A 4366-row or 4631-row chunk is outside that set, so the knob stays opt-in for every backend; the result above is HIP only.

## Tests

* `cargo test -p intarsia-core --test scheduler` (25 tests, 4 new): knob parse and clamp, tm = 0 equals the plain rule, the 16654 plan (`4096 x 3, 4366`; unchanged below 270; with a prefix hit; 4097 with `avoid_one_row`), every planned chunk fits `chunk + clamped rows`, a merged chunk only ever ends the prompt, an absurd knob cannot exceed `2 * chunk`.
* `cargo test -p intarsia-models --test tail_merge` (CPU, end to end, own test binary because it sets the environment): the model loads with the workspace for `chunk + rows`, a 13- and a 14-token prompt at chunk 4 run merged (`4,4,5` / `4,4,6`) and match the unmerged logits (rel L2 < 1e-5), a huge value is clamped to the chunk, a malformed one is refused. Negative control: with the workspace left at `chunk` rows the same test fails with `Backend("cpu write out of bounds")`.
* `cargo test -p intarsia-core --lib` (97), `cargo test -p intarsia-models --test plan` (7), `cargo test -p intarsia-cli --test docs_env_vars` (2, new row in `docs/tuning.md`), `cargo check --workspace`: pass.

## HIP 35B on the R9700 (tatooine, GPU 0 unless noted)

Arms: **base** = origin/main `62c759b7` (volume `intarsia-cargo-target-tcbase`); **off** = lane branch, knob unset; **on** = lane branch, `INTARSIA_PREFILL_TAIL_MERGE=1024` (exported inside the container command: `dev.sh` forwards only `INTARSIA_HIP_*` / `INTARSIA_TP_*`). Recipe chunk 4096, `--mem-budget 29`, `--max-seq 18432` (generate) / 17408 (serve; the planned context was 17408 on every arm, so the larger workspace did not cost context).

### Prefill time, fresh process (window E, `tc-stages.sh pf`, exclusive slot, n=5, order reversed on even reps)

| prompt tokens | passes base / on | base ms | off ms | on ms | on vs base |
|---:|---|---|---|---|---|
| 16654 | 5 / 4 | 1930 (1917 to 1943) | 1933 (1919 to 1943) | 1871 (1862 to 1884) | -59 ms, +3.2 % tok/s |
| 4408 | 2 / 1 | 506 (505 to 507) | 505 (504 to 506) | 460 (459 to 462) | -46 ms, +10.0 % tok/s |
| 4089 | 1 / 1 | 428 (427 to 430) | 428 (427 to 428) | 428 (428 to 430) | 0 (no tail) |

Medians and ranges of `prefill N ms` from `generate --max-tokens 1`. The `steps=` count of the 16654 run drops from 261 to 260 and of the 4408 run from 258 to 257 on the `on` arm only: the positive control that the knob engaged (and that the other two arms did not fold).

### localeval speed (served; windows B and D, exclusive slots, `localeval speed --sizes 4096,16384 --reps 3`, 9 interleaved rounds)

`scripts/hip-decode/tc_speed.py raw/wB raw/wD`. localeval cuts its own prompt, so the token count varies by a few tokens (about 4k: 4084 to 4111; about 16k: 16631 to 16680, always `4 x 4096 + 247 to 296`, so always merged on the `on` arm).

| size | arm | prefill tok/s median (min to max) | TTFT s median |
|---|---|---|---|
| ~16k | base | 8470 (8181 to 8560) | 1.97 |
| ~16k | off | 8466 (8287 to 8502) | 1.97 |
| ~16k | **on** | **8721** (8055 to 8776) | **1.91** |
| ~4k | base | 9297 (7055 to 9351) | 0.44 |
| ~4k | off | 8523 (8036 to 9385) | 0.48 |
| ~4k | on | 8239 (7210 to 9428) | 0.50 |

Read with care. Window B alone (5 rounds) is clean at 16k: on 8724 (8687 to 8776) against base 8470 (8458 to 8549), +3.0 %, -60 ms TTFT, the same size as the fresh-process numbers. Window D (4 rounds) is noisy and its 16k medians overlap (base 8359, off 8466, on 8416). At about 4k the two lane-branch arms had low rounds in B and D (11 of 18 rounds at 7210 to 8635 against 9230 to 9430 otherwise; base had one low round, 7055, of 9). The knob cannot act there (4088-token prompts have no tail, the 4111-token ones are 4096 + 15), the fresh-process run above shows 428 ms on all three arms, and the served startup logs of base and off are identical. Window F (served base vs off only, no `on` arm, 6 rounds alternating) settles whether the default path regressed: it did not (below). The cause of the low rounds in B and D is **unexplained** (they appear on `off` as well as `on`, only in the windows where the `on` arm also ran, and I did not reproduce them in-process or in F). The standings number stays the fresh-process one.

Window F (`raw/wF`, job 4098, 08:00:45 to 08:06:32): served base vs off, 6 rounds, order reversed on even rounds. About 4k: base 9368 (9223 to 9542) tok/s, off 9396 (9275 to 9428); about 16k: base 8482 (8473 to 8568), off 8495 (8482 to 8519). The unset knob is the same speed as main, served as well as in-process.

### ids and accuracy (windows A2 and C)

* 256 greedy ids (`generate --show-ids --ignore-eos`), 16654 and 4408 tokens: **base == off == on** at both sizes (`raw/wA2/window.log`; the id files are `raw/wC/ids-*.ids`).
* `intarsia accuracy --mode both --self-golden` (base writes the self-golden on one GPU, off and on read it on the two GPUs in parallel; `--prefill-chunk 4096 --decode-positions 64`; `--allow-unstamped --allow-dirty` as the HIP harness does). 4k golden, prefill (12759 positions) mean KL21 **2.345e-13** for both off and on, max KL21 2.562e-10, 0 flips; decode (192) 8.168e-12, 0 flips; long golden, prefill (36887) 2.680e-13 both, decode (320) 4.202e-12 both; per-prompt rows identical, including `chat_reuse_4k` (4631 tokens = 4096 + 535, merged on `on`). All PASS. The accuracy command of the `on` arm starts with `export INTARSIA_PREFILL_TAIL_MERGE=1024;` (`acc-on-4k.sh`, `acc-on-long.sh` on tatooine, read back after the run), so the process had the knob, and the path it takes (`engine.forward_logits_each` -> `Model::forward` -> `forward_chunks`) is the one the unit tests and the `generate` positive control exercise; the accuracy run itself prints no pass count, so this is by construction, not by a counter. The `off` and `on` logs are byte-identical per golden (`diff acc-off-4k.log acc-on-4k.log` prints nothing; same for the long golden), so within the precision of that comparator the knob changes no logit; the stronger statement is the ids. No prompt in either golden has a tail below 535 tokens, so the 270-token case is covered by the ids only; there is no golden at 16k.

### Radiance downtime (one `fleetq run --takes-down radiance --exclusive` per window, EXIT trap restores and waits for `/v1/models`)

| window | job | radiance down | up | minutes | what |
|---|---|---|---|---|---|
| A | 4022 | 06:58:23 | 06:59:14 | 0.9 | ids; stopped by me after the `on` arm failed the kernel-scratch plan (fixed, see Design) |
| A2 | 4031 | 07:01:37 | 07:12:25 | 10.8 | ids (good), then `localeval run` ran the full benchmark suite instead of `speed`; stopped by me, localeval script fixed |
| B | 4054 | 07:17:42 | 07:25:02 | 7.3 | localeval speed, 5 rounds (stopped by me) |
| C | 4072 | 07:31:13 | 07:35:39 | 4.4 | accuracy self-golden, both GPUs |
| D | 4085 | 07:41:09 | 07:47:07 | 6.0 | localeval speed, 4 rounds |
| E | 4091 | 07:52:16 | 07:55:26 | 3.2 | fresh-process prefill A/B |
| F | 4098 | 08:00:45 | 08:06:32 | 5.8 | served base vs off only, 6 alternating rounds |

Total 38.4 min over seven windows, none above 20 min. **The gap between A and A2 was 2.4 min, below the 5 min the brief asks for**: A was the aborted first attempt and I restarted too soon. Every other gap is at least 5.1 min (E to F 5.3). Radiance answered 15 s after each start. All slots released.

## CUDA (spark1)

See the section at the end of this file if the CUDA run happened; otherwise it did not (the queue held an exclusive and a 40 GB job until about 08:48).

## fleetq jobs

CPU/builds (non-exclusive): 4008, 4011, 4016, 4019 (negative control, expected exit 101), 4020 (HIP builds base and dev on tatooine), 4021, 4029, 4030, 4087 (file copies and the dev rebuild), 4028 (unit tests), 4082 (CUDA builds on spark1). GPU windows: 4022, 4031, 4054, 4072, 4085, 4091, 4098; 4097 and 4096 (read-only script check, final tests). CUDA: 4092 (see below).

## Files

`crates/intarsia-core/src/{scheduler,model,engine,plan}.rs`, `crates/intarsia-core/src/engine/batch.rs`, `crates/intarsia-core/tests/scheduler.rs`, `crates/intarsia-models/tests/tail_merge.rs`, `docs/tuning.md`, `docs/decisions/LOG.md`, `scripts/hip-decode/{tc-window.sh,tc-stages.sh,tc_speed.py}`, `scripts/cuda-decode/tc-cuda.sh`. Outside `docs/`, `crates/<backend>/`, `kernels/`, `scripts/` the diff touches the shared core (`crates/intarsia-core`, one test file in `intarsia-models`): the coordinator's cross-backend no-op check applies (knob unset: `tm = 0` paths are tested equal to the old rule and the Metal / CUDA code paths are not edited).

## Reproduce

```
# builds (CPU slots): the dev tree rsynced to ~/intarsiaml-tc and origin/main to ~/intarsiaml-tcbase, then on tatooine
#   INTARSIA_REPO=~/intarsiaml-tc INTARSIA_TARGET_VOL=intarsia-cargo-target-tc bash deploy/tatooine/dev.sh cargo build --release -p intarsia-cli --features hip
fleetq run --host tatooine --exclusive --takes-down radiance --mem 30 --label "tc: ..." -- \
  env OUT=<raw dir> STAGES="pf" TM=1024 PFREPS=5 BUDGET=780 bash scripts/hip-decode/tc-window.sh      # also: STAGES="ids", "acc", "speed"
python3 -I scripts/hip-decode/tc_speed.py <raw dir> ...
```
