# DSv4.1 draft round on the device, NCCL measurement (lane dsv41/draftdev, 2026-11-02)

Branch `dsv41/draftdev`, merged with `dsv41/plan` 189d61e91 (graphs, split-K verify, chooseK on). Spark dir `~/intarsia-dsv41-draftdev`.
Raw: `docs/results/2026-11-02-dsv41-draftdev/` (nccl_bench output, window steps log).

## Task 1: draft round (Fable report item 3)

| step | change | numerics |
|---|---|---|
| (i) | `Dsv41Runtime::draft_steps(st, first, k)`: k Markov steps, not `block_size` (5); the call site passes `sp.k` (the engine's `draft(first, steps)`); `hx` download k rows | bitwise: step i of a k-step round is step i of the full round (test: `draft_steps` k=1,3 equal the prefix of `draft`, ids and confidence bits) |
| (ii) | the verify graph ends in an `Argmax` step (`Dsv41Graph::ids`); `forward_rows_ids` downloads `rows` ints instead of `rows x 129280` f32; the engine reads ids unless `min_tokens` still masks (`n < min_tokens`: the mask edits logits). Rank 1's `verify_rank` no longer downloads the logits either (it did, and the round waits for it in `finish()`) | bitwise: same first-strict-max kernel (`argmax.cu`) as `sampler::argmax`; test compares both on every verify pass |
| (iii) | Markov embed, head GEMV, f32 `logits + bias`, argmax inside the draft graph (rank 0 only; rank 1 builds the graph without them), so one sync; ids, the k embeddings and `hx` come back as three small copies. New `add_f32` kernel (`Op::Add` had bf16 only on CUDA) | bitwise: one IEEE f32 add per element, same order `logits + bias` as the host loop |

Gates (3-layer cache, spark1): cpu f32 `speculative_greedy_equals_plain_greedy` 48/48 (also with `INTARSIA_DSV41_DPOS=1`, and again on the merged head),
`verify_rows_track_decode_through_rollback` cpu f32 worst rel_rms 0, argmax differs nowhere, plus the two assertions above and the two-rank variant (rank 1 takes the no-Markov graph);
`device_position_passes_equal_host_passes` cuda, spec, eager and graphs: 52 passes, first bitwise difference none (15 draft replays).
`dspark_rounds_match_the_official_drafter` could not run: `layers_bf16.safetensors` is not in `~/intarsia-dsv41-tp2/fixtures`.
`cargo test --workspace --no-fail-fast` on endor: all pass (it found the missing `docs/tuning.md` row for `INTARSIA_NCCL_GATHER`, added).

Full model (window #7784, both Sparks, TP=2, `INTARSIA_DSV41_SPEC=1`, ab0 = dsv41/plan 189d61e91 binary, ab1 = head `f4ae35c84`):

| | ab0 base | ab1 head |
|---|---:|---:|
| golden 10 prompts x 64 tokens vs `tok-intarsia-tp2-w12-specoff.st` | 640/640 | 640/640 |
| golden ab1 vs ab0 | | 640/640 identical |
| tokens a round (golden / lspeed sweep) | 2.177 / 2.176 | 2.177 / 2.187 |
| decode tok/s at prompt 137 / 1028 / 4074 / 16308 (localeval median of 3) | 39.2 / 38.6 / 39.6 / 38.5 | 42.8 / 44.2 / 42.1 / 40.5 |
| mean | 39.0 (25.6 ms/token) | 42.4 (23.6 ms/token) |

-2.0 ms/token (+8.7 %), more than Fable's -1.0 to -1.3: the host argmax (4 x 129280 f32 on rank 0, and the same download on rank 1), the five Markov syncs and the
idle between verify and draft all go. Spread within a size is 5-10 % (ranges in the localeval logs under `.scratch/win1/ab*/`), so read the mean, not one row.
Memory: min MemAvailable 6.4 / 6.9 GiB. vLLM restored on both Sparks and `fleetq mark-up`.

## Task 2: NCCL all-reduce vs all-gather (nccl-tests could not be used)

nccl-tests exist only on spark1 (MPI build; spark2 has no MPI, no `nccl.h`, so no build without sudo). `crates/intarsia-backend-cuda/examples/nccl_bench.rs`
runs the engine's own `Comm` (same libnccl 2.30.7, stream and `NCCL_*` as the decode) between the two Sparks; `run-full.sh` `PRE=ncclbench`.
us per call, `pipe` = 300 calls queued back to back, `sync` = a device sync after each call; NCCL_MAX_NCHANNELS=2 (the engine's setting):

| bytes per rank (f32) | all-reduce in place pipe / sync | `all_reduce_sum_into` (+ f32->bf16) pipe / sync | all-gather pipe / sync |
|---:|---:|---:|---:|
| 16384 | 24.2 / 33.9 | 23.4 / 29.5 | 16.9 / 21.3 |
| 32768 | 24.2 / 29.5 | 26.2 / 32.0 | 19.7 / 30.0 |
| 49152 | 26.2 / 31.4 | 28.0 / 40.3 | 22.5 / 32.1 |
| 65536 | 28.6 / 33.7 | 30.4 / 36.4 | 25.5 / 28.4 |
| 98304 | 32.5 / 37.9 | 38.4 / 53.2 | 36.9 / 41.5 |
| 131072 | 50.2 / 57.5 | 49.8 / 55.8 | 38.1 / 46.4 |

One row of the model's partial is 20 KB, four rows 82 KB. The all-gather is 3-7 us under the all-reduce at 16-65 KB, but the engine's all-reduce already includes the
bf16 round (the `ar_into` column) and the gather path needs the rank-order sum-and-round kernel (about 2-3 us in a graph), so the net is about 2-4 us a collective, 0.2-0.3 ms
a pass of 32-53 ms. The measured collectives in the decode (36-50 us each) are mostly launch gaps and rank skew, not wire time: the isolated all-reduce is 24-33 us. **Not clearly faster; Task 3 skipped.**
NCCL_MAX_NCHANNELS=1 is no better (23-58 us all-reduce, worse at 98-128 KB; `ncclbench-r0-c1.txt`).

Task 3 code is in the branch anyway, off by default (`INTARSIA_NCCL_GATHER=1`: `ncclAllGather` + `ik_rt_f32_sum_round`); window ab2/ab3 measured it for the record:

| decode tok/s at 4 prompt sizes | head | `INTARSIA_NCCL_GATHER=1` | `NCCL_MAX_NCHANNELS=1` |
|---|---:|---:|---:|
| 137 / 1028 / 4074 / 16308 | 42.8 / 44.2 / 42.1 / 40.5 | 43.7 / 41.1 / 40.6 / 40.0 | 38.7 / 37.6 / 40.0 / 40.6 |

Gather: golden 640/640 identical to base (same 294 rounds, same accepts), speed inside the noise (mean 41.4 vs 42.4). Channels 1: slower (mean 39.2). MMLU on the 570 was not run (no speed case for it).
