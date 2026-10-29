# SB8: Qwen3-8B BF16 multi-stream on one DGX Spark (spark2, GB10, CUDA), re-measured (2026-10-28 evening, doc dated 10-29)

Lane `sb8`, branch `lane-sb8/spark-8b-ms`, base `origin/main` `318d508fb` (#224). spark2 (spark-fa00, GB10 sm_121), `spark2-vllm` (role retro, 38 GB) left up and idle.
Build: `cargo build --release --features cuda -p intarsia-cli` in `~/intarsiaml-sb8/src` (stamp `head=318d508 code-tree=8e10697e60d6 dirty=no`). Raw: `2026-10-29-sb8-spark-8b-multistream/raw/`.

## Result

The scoreboard cell "15 / 15 tok/s at 4 / 16 streams" was a **serial** run, not a batching failure of the engine. `serve` without `--max-batch` decodes one sequence at a time
(`crates/intarsia-models/src/load.rs:509-513`: `supported = 1` when the flag is absent), so 4 streams give the 1-stream rate (15.8 tok/s, TTFT 24 s at 4 streams, section 2).
With batching on, the 8B scales to 163 tok/s at 16 streams. But `--max-batch 16` was **refused** for Qwen3 by the recipe cap (`recipes/qwen3.toml:49`, `max_batch = 8`:
"--max-batch 16 exceeds the max_batch 8 of recipe qwen3", `crates/intarsia-core/src/recipe.rs:528`), although the CUDA backend proves 16 rows
(`crates/intarsia-backend-cuda/src/device.rs:503`). With the cap at 8 the 16-stream cell is a queue of 8 (103.6 tok/s, TTFT 10.9 s).

**Proposed fix, not applied** (`raw/qwen3-cap16.patch`, one command to apply: `git apply`): `recipes/qwen3.toml:49` `max_batch = 8` -> `16` (plus a comment) and the unit test `max_batch_resolution`
(`crates/intarsia-models/tests/batch.rs`) moved from "9 exceeds" to "17 exceeds" with a 16 case added. Mac `cargo test --workspace --release` passes with the patch (1081 passed, 0 failed). Metal is unaffected: the backend cap (8) still applies
(`resolve_max_batch` checks `supported` first). The other recipes copied from qwen3 (`llama`, `qwen2`, `lfm2`: 8) are not touched; they were not measured.
It is not applied because the served-identity check fails (section Identity); whether to accept that caveat is the coordinator's call. This PR is docs only.

### Table (localeval `speed`, 1k prompt (988 tokens), 256 generated, `ignore_eos`, greedy, `--thinking off`, 3 reps + 1 warmup, median)

**With the recipe cap raised to 16** (the patch above, passed as `--recipe` file; `serve --backend cuda --max-batch 16 --max-seq 8192 --mem-budget 36`, 3 rounds, fresh server each, raw `mb16-r1/2/3.out`). Aggregate decode tok/s. The 1/4/8-stream cells do not depend on the cap (the cap-8 run agrees: 15.9 / 57.0 / 102.2). Today's main gives 103.6 at 16 streams (`mb8-r1.out`, one round).

| streams | r1 | r2 | r3 | median | per-stream | TTFT s | vLLM (R51, **old**) | vs vLLM | SGLang (R54) | llama.cpp (R47) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.9 | 15.9 | 15.8 | **15.9** | 16.1 | 0.20 | n/a | n/a | n/a | n/a |
| 4 | 57.1 | 57.1 | 56.9 | **57.1** | 14.8 | 0.60 | 52.2 | **1.09x** | 52.1 | 48.1 |
| 8 | 102.8 | 102.5 | 102.2 | **102.5** | 13.7 | 1.14 | n/a | n/a | n/a | n/a |
| 16 | 163.0 | 163.1 | 163.0 | **163.0** | 11.4 | 2.19 | 156.9 | **1.04x** | 153.6 | 114.3 |

Rank among the four engines on the scoreboard (ours, vLLM, SGLang, llama.cpp), **today's main (cap 8): 1/4 at 4 streams (1.09x vLLM), 4/4 at 16 (103.6 = 0.66x vLLM, 0.91x llama.cpp)**; with cap 16: 1/4 at 16 (1.04x vLLM). Was 4/4 and 4/4 with 15 / 15 (0.29x / 0.10x vLLM). The 4-stream cell was identity-clean in both greedy reps (section Identity); the 8- and 16-stream cells run on the path that is not. Caveat: the competitor numbers are
the old MB records (10-22, other window, other harness runs; vLLM R50 is flagged **old** in the scoreboard); they were not rerun here. The 4-stream lead over vLLM and SGLang is 9-10 %, the 16-stream lead
is 4-6 %; a same-window vLLM rerun could change either by a few percent. Single-stream decode (15.9) is close to the scoreboard decode cell (15.1 at the 1k point, 8B BF16 is bandwidth-bound at about 16 GB per step).

### Comparison arms (1 round each, same harness)

| arm | 1 | 4 | 8 | 16 | note |
|---|---:|---:|---:|---:|---|
| no `--max-batch` (serial; the old row) | 15.8 | 15.8 | not run | not run | TTFT 24.4 s at 4 streams: requests queue one at a time; 8/16 skipped (would take 20 min) |
| `--max-batch 8` (the largest the recipe allows today: **main**) | 15.9 | 57.0 | 102.2 | 103.6 | 16 streams queue behind 8: TTFT 10.9 s; one round |
| `--max-batch 16` (recipe cap 16, patch, not applied) | 15.9 | 57.1 | 102.5 | 163.0 | median of 3 |

## Identity (batched == solo)

**Batched greedy output is NOT bitwise equal to solo for the 8B on CUDA at 5 or more streams. It is there on today's main at `--max-batch 8` too (so the cap change does not cause it), but it is the reason the cap patch is not applied: the brief's condition for a fix was identical greedy output, and this check failed.**

`scripts/htk/equal.py` (7 sampling settings x 1/4/16 streams, 64 tokens, seeds 100+i, token ids compared), a solo server (no `--max-batch`) against `--max-batch 16`
(`raw/ident/`, `compare.log`): **7 of 21 cells identical**. Greedy: n=1 and n=4 identical, n=16 differs on 3 of 16 streams (streams 4, 5, 8; first differing token at position 48, 3, 3 of 64).
Sampled cells differ already at n=1 for two settings (`t1.0-k40-p0.9`, `t0.8-p0.9-nok`: seeded sampling is not comparable between the solo and the batching engine, so only the greedy rows are evidence).
Localisation, greedy only against the solo token ids (`raw/greedy-localisation-1.log`, `-2.log`; 2 reps per n):

| server | n=4 | n=5 | n=8 | n=9 | n=12 | n=16 |
|---|---|---|---|---|---|---|
| `--max-batch 16` (recipe cap 16) | identical x2 | identical, differs | differs x2 | differs x2 | differs x2 | differs x2 |
| `--max-batch 16`, the four msd2 CUDA knobs `=0` | n/a | n/a | identical, differs | n/a | n/a | differs x2 |
| `--max-batch 8` (today's main cap) | identical x2 | n/a | identical, differs | n/a | n/a | n/a |
| `--max-batch 16`, `INTARSIA_BATCH_GRAPHS=0` | n/a | identical, differs | differs x2 | n/a | n/a | differs x2 |
| `--max-batch 16 --no-graphs` | n/a | identical, differs | differs x2 | n/a | n/a | differs x2 |

So: not the msd2 knobs, not CUDA graphs, not the cap (the old cap-8 server already differs at n=8). The flipped streams are the same few (4, 5, 8), and the same n gives different results
between reps (stream 5 flips at position 3 in one rep and 62 in another), which points at something that depends on the batch composition at each step (which rows share a step) or at a race, not at a
fixed numeric difference. Not isolated further here. The engine-level gate (`crates/intarsia-models/tests/batch_cuda_real.rs:515`, `real_qwen3_8b_batched_decode_is_bitwise_the_solo_step`,
4,290 pairs, docs/results/2026-10-23-rb2-cuda-batching.md section on Qwen3-8B) passes at the model level; this finding is at `serve` level (the scheduler, chunked prefill beside decode, the sampler path), where no 8B gate
exists. Short-prompt, 64-token requests only; whether the flips are near-tie argmaxes amplified by a small numeric difference was not checked (no logits compared).

Consequence for the scoreboard row: the 8- and 16-stream throughputs are real throughputs of a batched path that is not bitwise-gated for the 8B at `serve` level (the 35B rows are: 4,290 pairs, 21/21 cells in #224). The row says so.

One candidate from the rb2 doc, section 9: busy-chunk prefill is not bitwise when the busy chunk differs from the solo chunk, and "dense Qwen3-8B on CUDA has not been measured for this". It is probably not the cause here: the serve log shows busy chunk 4096 = the 8B's CUDA solo chunk, and the prompts are 5-12 tokens. Left open.


## Commands

```
# build (spark2, CPU slot)
fleetq run --host spark2 --load cpu --mem 8 --expect 45 --label "sb8: build main 318d508 CUDA on spark2" -- sb8-build.sh   # rsync worktree -> ~/intarsiaml-sb8/src, cargo build --release --features cuda -p intarsia-cli
# measure (quiet slot)
fleetq run --host spark2 --quiet gpu,cpu --mem 36 --expect 45 --label "sb8: localeval speed Qwen3-8B BF16 ..." -- sb8-run.sh
#   per arm: intarsia serve --backend cuda --model <Qwen3-8B snapshot b968826d9c46> --port 8191 --served-model-name intarsia --max-seq 8192 --mem-budget 36 --profile-every 0 \
#            --recipe ~/intarsiaml-sb8/qwen3-mb16.toml --max-batch 16          # recipes/qwen3.toml with max_batch = 16 (the one-line change)
#   then:   (cd /opt/code/deepeval && uv run localeval speed --url http://spark-fa00:8191/v1 --model intarsia --thinking off --sizes 1024 --gen 256 --reps 3 --concurrency 1,4,8,16 --no-unim)
```

`--mem-budget 24` was tried first and fails localeval's calibration (`the prompt has 4465 tokens but the context window is 3751 tokens`: the KV pool left in a 24 GB budget with 16 slots gives a 3.7k-token window; cause inferred, not isolated); 36 GB is used for all
rows. The measured binary is main `318d508` with the recipe passed as a file (`--recipe`), which equals the patch's one-line change. This PR changes no code.
The scripts used (`sb8-build.sh`, `sb8-run.sh` / `sb8-run2.sh` (run 1 = mb16 r1 + mb8 r1, killed in its default arm after 16-stream serial would take 20 min; run 2 = mb16 r2, r3 + default at 1/4), `sb8-ident.sh`, `sb8-greedy*.sh/.py`) are in `raw/`; they follow `scripts/msd/build.sh` and `le-window.sh` retargeted to spark2.

## Checks

| check | result |
|---|---|
| Mac `cargo test --workspace --release` with the patch applied (the PR itself is docs only) | pass: 1081 passed, 0 failed (`max_batch_resolution` ok) |
| served-token identity, solo vs `--max-batch 16` (`equal.py`, 21 cells, 9408 ids) | **fail**: 7/21 cells; greedy n=16 differs on 3 of 16 streams; same at `--max-batch 8` (pre-existing), see Identity |
| CI | see PR |
