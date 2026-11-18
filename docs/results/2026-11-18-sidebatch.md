# sidebatch: `INTARSIA_SIDE_EAGER=0` at `--max-batch 8` (CUDA, Qwen3.6-35B-A3B NVFP4, spark2)

Lane `sidebatch`, branch `sidebatch` from origin/main `7cfe3d885` (has #447 / `fabc570a`, the knob's PR). Follows `docs/results/2026-11-18-postchain2.md`, whose gates covered
`--max-batch 1` greedy only. spark2 (DGX Spark GB10, vLLM `retro` up beside it), fleetq `--quiet gpu,cpu` for the runs and `--load cpu` for the build, build with
CARGO_BUILD_JOBS=3 MAX_JOBS=3 FLASHINFER_NVCC_THREADS=1 and fleetq `--mem 55`, free-memory watchdog `scripts/sidebatch/wd.sh` (kill below 15 GB available; it never
fired, 48-51 GB available at every server start). Raw output: `docs/results/2026-11-18-sidebatch/raw/`; scripts: `scripts/sidebatch/`. No engine change, nothing default-on.
**spark2's GPU runs about 3.6x slow for a reason not found, so no absolute time here means anything; the on/off wall ratio below is secondary and only valid within this box.**

**Result: the knob changes nothing a client can see at `--max-batch 8`.** Reply shas and token counts are identical, knob unset vs `INTARSIA_SIDE_EAGER=0`, in every one of
30 requests per arm, for both server configs (default recipe MTP N = 2, and `INTARSIA_BATCHED_MTP=8`), greedy and sampled (seeded), 8 concurrent streams, a 16.8K-token prompt
alone and a 16.8K-token prompt beside 4 decoding streams. Both configs also equal the `--spec off` server (30 of 30). The changed eager path ran in the batched cells (counter below).
One gate of the brief could not run: `INTARSIA_SPEC_COPY` is refused with `--max-batch` > 1 (see "Not run").

## 1. Every eager path #447 changes

The diff (`git diff fabc570a^1 fabc570a`) has one behaviour change: `Model::side_fork` (`crates/intarsia-core/src/model.rs`) returns `None` (no fork) when `INTARSIA_SIDE_EAGER=0`
and no graph capture is in progress in the process (`CAPTURES_ACTIVE == 0`; five `CaptureGuard` lines around every `graph_capture` call in `graph_step.rs` and `spec_graph.rs`).
Everything else is a counter, a doc row and the lane's scripts. `side_fork` is called from three places (the MoE block's shared expert through `ExecRunner::fork`;
the attention block's k / v projections in `attn_block`; the GatedDeltaNet z / b / a projections in `gdn_block_segs` and `gdn_batch`), each with the rows of the pass; it forks only for a
decode step (1 row) or a row-exact verify pass (`verifying()`, at most 16 rows), single rank, no profile, no plan recorder. So the paths where an eager pass loses its fork with the knob:

| eager path | where | batched? | ran in this lane's cells |
|---|---|---|---|
| solo decode step not replayed from a graph (graph disabled, not yet captured, or the request's last 1-draft MTP step) | `forward` / `graph_decode_step` fallback | `--max-batch` 1; also a 1-sequence batch (`forward_batch` with b = 1 is the solo step) | yes: 32-64 fork points ("1row" below) |
| solo verify pass (a copy chain's verify, a 1-sequence MTP verify) | `forward_verify` | `--max-batch` 1; in the batching engine for a lone drafting sequence | yes (inside the "verify" count) |
| batched ragged MTP verify pass: one pass over the drafting sequences' rows, split at 16 rows | `forward_verify_batch*` (`INTARSIA_BATCHED_MTP`, `INTARSIA_BATCHED_VERIFY`; the 35B CUDA recipe has both on, N = 2) | **yes, the batched eager path** | yes: ran in every batched-MTP server (98-142 passes, below); its share of the 4.8K-7.2K verify fork points is not separated from solo verifies |
| batched plain decode step not replayed from a graph (`INTARSIA_BATCH_GRAPHS=0` or a capture that failed) | `forward_batch` eager fallback | yes | no: `intarsia_decode_eager_steps_total` 0 in every server; not exercised |
| the MTP drafter head's own steps and a 1-row prefill tail chunk | head layers and chunk through the same blocks | yes | inside the "1row" count, not separated |
| inside a graph capture (a batched step's first use of a row count, the MTP step, the spec / catch-up captures) | all `CaptureGuard` sites | yes | forks kept in captures by design; the replays are not eager |

So at `--max-batch` > 1 the knob's eager paths are, in practice, the MTP verify passes (ragged multi-sequence and solo) and the few 1-row eager steps. A sequence drafts while at most N sequences decode, so with 8 streams of different
lengths it runs at the tails of the waves (N = 2: last 2 sequences; N = 8: all of them, up to 5 sequences per 16-row pass).

## 2. Method

Binary: `intarsia serve --backend cuda` built on spark2 from this tree plus an **uncommitted observational counter** in `side_fork` (`raw/counter.patch`; atomic increments and a
stderr line on every power-of-two count and every 4096th; no change to what forks), stamped `sidebatch-7cfe3d885-counter`. Every arm below ran on that one binary, env only.
The committed tree has no counter.

Servers (`scripts/sidebatch/serve.sh`): `--max-batch 8 --max-seq 20480 --mem-budget 40 --no-supervisor`, 35B NVFP4, one server per arm, warm-up of 8 short streams, then the waves of
`scripts/sidebatch/gate.py` (non-streaming; reply sha1 and token count per request; thinking off; prompts from `scripts/mcopy/prompts.py`: copy3k, agent-edit, agent-batch,
json-xform, chat-short, reason, code-gen x2; every request has a nonce prefix, the same in every arm):

| wave | requests |
|---|---|
| `g8` | 8 concurrent greedy streams, max_tokens 300-700 (so they finish at different times and the tails draft) |
| `g8s` | 8 greedy, starts staggered by 0.35 s, max_tokens reversed |
| `s8` | 8 sampled, T 0.7, top_p 0.95, seeds 1000 + i |
| `long` | one prompt of 16799 tokens (docs/tuning.md, cut), 512 tokens, greedy (all four servers: 512 tokens generated, `length` finish; `longmix`: prompt 16800) |
| `longmix` | the same long prompt beside 4 decoding streams (the prefill parks between layers while they decode) |

Arms: `u` (knob unset, recipe default: `--spec mtp --spec-k 2`, batched MTP N = 2, batched verify on), `e0` (`INTARSIA_SIDE_EAGER=0`), `off` (`--spec off`, knob unset), `ubm` /
`e0bm` (`INTARSIA_BATCHED_MTP=8`, `--spec mtp --spec-k 2`, knob unset / 0). Comparison: `scripts/sidebatch/compare.py` (sha and token count per request).

## 3. Gates (all must be identical)

| gate | cells | result |
|---|---|---|
| default MTP (N = 2), `--max-batch 8` x 8 streams, greedy: `u` = `e0` = `off` | `g8`, `g8s`, `long`, `longmix` (8 + 8 + 1 + 5 = 22 requests) | **PASS** identical |
| same, sampled (T 0.7, top_p 0.95, seeds 1000 + i): `u` = `e0` = `off` | `s8` (8 requests) | **PASS** identical (spec on = spec off, as the seed-exact sampled drafting says) |
| batched MTP N = 8: `ubm` = `e0bm` = `off`, greedy | 22 requests | **PASS** identical |
| same, sampled | 8 requests | **PASS** identical |
| one long-context request, ~16.8K prompt tokens, 512 out, greedy, on / off (alone and with 4 streams) | `long` (1) + `longmix` (5) in each config | **PASS** identical |
| total | `u` vs `e0`, `off` vs `u`, `off` vs `e0`, `ubm` vs `e0bm`, `off` vs `ubm`, `off` vs `e0bm`: each 30 of 30 | **PASS**, no mismatch (`raw/compare-g1.txt`, `raw/compare-g3.txt`) |

All five arms generated 12511 completion tokens in total (`stop` and `length` finishes both occur).

## 4. The changed eager path ran (counter)

Last `sidebatch:` line of each server log (printed at power-of-two counts and every 4096th, so each count is a lower bound within a factor 2 of the final; the four counts per
line are the MoE-expert and projection fork points, in a 1-row pass and in a row-exact multi-row pass; "taken": the fork happened; "SUPPRESSED": the knob stopped it):

| arm | 1-row MoE / proj | verify-pass MoE / proj | reading |
|---|---|---|---|
| `u` | 32 / 32 | 4814 / 4814 (taken) | default N = 2 |
| `e0` | 32 / 32 | 4772 / 4772 (SUPPRESSED) | same path, fork suppressed |
| `ubm` | 64 / 64 | 7172 / 7172 (taken) | N = 8 |
| `e0bm` | 32 / 32 | 6009 / 6009 (SUPPRESSED) | same path, fork suppressed |

The counter's "verify" bucket counts every row-exact multi-row pass, solo (`forward_verify`, a lone drafting sequence) and ragged (`forward_verify_batch`) alike, so it does not by itself
count the ragged passes. The server's own `intarsia_batch_verify_passes_total` / `_rows_total` do (summed over their labelled series; the labels were stripped when the metrics were saved): `u` 98 passes /
583 rows, `e0` 96 / 569, `ubm` 140 / 981, `e0bm` 142 / 1001, `off` none. About 6-7 rows per pass is two or more sequences of 3 rows each, so multi-sequence ragged passes ran in every batched-MTP server,
with the knob unset and set. That they ran inside the knob-suppressed region is an inference from the code (the pass is eager, `verifying()`, at most 16 rows, so `side_fork` is reached and, with the knob, returns
`None`), not a separate count.

Server counters (`raw/g1/metrics-*.txt`, `raw/g3/metrics-*.txt`): `u` ran 530 MTP steps (1057 drafts proposed, 931 accepted), `ubm` 675 (1342 / 1187), `e0bm` 680 (1354 / 1196);
`intarsia_decode_eager_steps_total` is 0 in every server (no plain batched step ran eagerly). So in each batched cell the knob suppressed thousands of eager fork points (at
`--max-batch` 8, in verify passes of both kinds), and the replies did not change. Fork points inside graph captures are not counted.

## 5. Secondary: on / off wall time (spark2 is slow and has live vLLM traffic: ratios only, never absolute)

Sum of the five waves' wall times, single run per arm, no repeats: `u` 181.3 s, `e0` 181.0 s (0.998x); `ubm` 179.9 s, `e0bm` 180.7 s (1.005x); `off` 187.6 s. Run-to-run noise was not
measured (no ABBA); read as "no visible difference", not as a measurement of speed.

## Not run / unverified

- **`INTARSIA_SPEC_COPY=20` at `--max-batch 8`: refused by the engine**, so the brief's gate "batched MTP + `INTARSIA_SPEC_COPY=20`" cannot exist: `unsupported: INTARSIA_SPEC_COPY needs --spec mtp
  with --max-batch 1, one rank, on the cuda or metal backend (max batch 8, tp 1, backend cuda)` (`raw/g2-spec-copy-refused.out`; the server died at start, nothing ran). The batched cells therefore
  ran batched MTP N = 8 without copy drafts. Copy chains with the knob are covered at `--max-batch 1` by postchain2's gates (`generate` ids, 5 cells), not re-run here.
- Not exercised: eager plain batched steps (`INTARSIA_BATCH_GRAPHS=0`), tensor parallelism (no fork there anyway), HIP and Metal, `--max-batch` 16, more than 8 streams, a profiled pass (it never forks).
- One server per arm, one run: this is a determinism / equality gate, so one run is the gate, but a rare data-race difference would need repeats. With the knob on, an eager pass runs on one stream,
  strictly ordered, so it cannot introduce a race the two-stream path did not have; a difference would show as a mismatch, and there was none in 180 request pairs.
- The counter is a lower bound and does not separate the 1-row pass types or the waves.
- `INTARSIA_SPEC_COPY` and `INTARSIA_SIDE_EAGER` have docs rows (`docs/tuning.md`); this lane adds no env var. It does not measure speed (the `sideeager` lane does).

## Verdict and revert

At `--max-batch 8` on CUDA, `INTARSIA_SIDE_EAGER=0` is output-neutral in every cell run: greedy, sampled, default MTP N = 2, batched MTP N = 8, long context. Nothing flipped; the knob stays
opt-in and flipping it is the user's call. Revert: revert this PR (docs and scripts only; there is nothing to revert in the engine).

## Instructions received

None beyond the brief and the lane rules (`/opt/code/intarsiaml/.worktrees/scan/lane-rules.md`). Deviation: the brief's `INTARSIA_SPEC_COPY=20` batched cell was refused by the engine and replaced by
batched MTP N = 8 alone (above). Run history: build #10160 (18 min), gates #10183 (arms `u`, `e0`, `off`), #10186 (batched MTP arms with `INTARSIA_SPEC_COPY=20`, died at start, exit 1), #10190 (arms `ubm`, `e0bm`).
spark2 `~/intarsiaml-sidebatch` is removed; no fleetq slot is held.

## Files

`scripts/sidebatch/`: `build.sh`, `wd.sh`, `serve.sh`, `gate.py`, `compare.py`. Raw: `raw/g1/` (arms `u`, `e0`, `off`: jsonl per request, metrics, client and server logs), `raw/g3/` (`ubm`, `e0bm`, and a copy of `off.jsonl`),
`raw/compare-g*.txt`, `raw/g1.out`, `raw/g3.out`, `raw/g2-spec-copy-refused.out`, `raw/build.out`, `raw/counter.patch`.
