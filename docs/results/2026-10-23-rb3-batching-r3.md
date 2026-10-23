# rb3: batched GDN and attention, device sampling, graph buckets, prefill overlap (R3)

Lane rb3, 2026-10-23, DGX Spark GB10 (spark1 for speed, spark2 for gates), Qwen3.6-35B-A3B NVFP4, `serve --max-batch 16`. Follows
[rb2](2026-10-23-rb2-cuda-batching.md) (PR #117, hybrid continuous batching on CUDA). Branch `lane-rb3/r3-batched-graphs`, PR #121 (draft).
Machine-readable trace: [trace.json](2026-10-23-rb3-batching-r3/trace.json).

## 1. Result

R3 (a) and (c) are in and bitwise: every decode row is still exactly the solo step. 35B aggregate decode in the composer cell (1024-token
prompt, 128 generated, localeval, three separate bursts per cell, same machine, same session, main = `6ca51a19`):

| streams | main (rb2, #117) | rb3 | change | target |
|---|---|---|---|---|
| 1 | 73.2 | 73.3 | noise | no regression |
| 4 | 122.4 (121.8 to 122.7) | **141.2** (139.3 to 144.5) | +15.4% | above 140 |
| 16 | 167.1 (166.8 to 167.4) | **194.6** (194.2 to 195.5) | +16.5% | about 200 |
| 4, 4096-token prompts | 83.0 (82.4 to 84.0) | **91.3** (90.5 to 91.8) | +10.0% | (reported) |

The 4-stream figure clears 140 on the mean of three bursts but one burst is 139.3; call it at the line. 16 streams is 2.7% under 200.
Inter-token latency (server histogram, buckets are upper bounds, so p99 of 500 means "in the 250 to 500 ms bucket"): mean ITL at 4 streams
25.0 to 21.7 ms, at 16 streams 74.9 to 62.5 ms; **p99 is 35 ms at 4 streams (unchanged) and 500 ms at 16 streams (unchanged)**: that tail is
the one decode step that waits behind a 2048-token prefill chunk (about 0.5 s), which only (d) removes, and (d) costs aggregate speed (section 4).

Sub-steps, measured one after another (pre-merge tree, one burst per cell from the same job, #4686; the three-burst means above and in
section 3 agree to within 1.5%):

| config | 1 | 4 | 16 | 4k at 4 | mean ITL at 4 / 16 (ms) | p99 at 4 / 16 |
|---|---|---|---|---|---|---|
| main (#117) | 73.5 | 122.8 | 167.0 | 83.6 | 21.3 / 68.2 | 35 / 500 |
| (c) alone, row tables off (`INTARSIA_ROW_TABLES=0`) | 73.1 | 128.6 | 167.2 | 82.6 | 20.5 / 68.2 | 35 / 500 |
| **(a) row tables**, graphs off (`INTARSIA_BATCH_GRAPHS=0`) | 73.6 | 131.6 | 190.7 | 86.9 | 20.0 / 58.6 | 35 / 500 |
| **(a)+(c) row tables + graphs** (default) | 73.0 | 140.5 | 193.5 | 91.7 | 18.9 / 57.5 | 35 / 500 |
| (d) overlap (`INTARSIA_OVERLAP=1`) | 73.0 | 130.4 | 177.1 | 77.2 | 17.7 / 40.2 | 35 / 150 |

- (a) is the big step (+7% at 4, +14% at 16): the per-row GDN and attention launches were 25% and 42% of the step.
- (c) adds +7% at 4 and +1.5% at 16 on top of (a). Graphs without row tables do nothing at B >= 2 (a per-row launch is bound to one
  sequence's state, so the step stays eager); the 5% at 4 streams in the "row tables off" row comes from the per-state solo graphs.
- (b) cannot show in a greedy localeval cell (greedy rows take the on-device argmax); its own measurement is section 2.
- (d) trades aggregate throughput for latency: the chunk schedule is unchanged and outputs are bitwise, but a chunk that shares the GPU
  with decode runs about 55% slower (last TTFT at 4 streams 0.88 s to 1.38 s), and staggered stream starts lower the average batch. It
  stays opt-in.

![One batched decode step, before and after](2026-10-23-rb3-batching-r3/step.svg)

## 2. What changed

**(a) Row tables.** One launch per op for all rows of a batched step, replacing the per-row loops of rb2. Kernels (all CUDA, `row_table`
kernels selected only for an `OpCall::rows` call): `gdn_conv_bf16_rows`, `gdn_delta_rule_bf16_split_rows`, `kv_append_bf16_rows`,
`attn_decode_bf16_splitkv_rows`, plus `rope_bf16` with a position per row. Each row reads and writes its own recurrent state or KV cache in
place through a table of device addresses (no gather or scatter); the model uploads the table (and the row lengths) once per step. A row-table
kernel runs the one-row kernel's arithmetic per row, and is only selected where the solo step would run the kernel it twins, so a batched
row cannot differ from the solo step. Launches per step of the 35B: 814 at any B (op listing `real_35b_batch_op_listing`); rb2 launched
about 814 + 130 per extra row (B = 16: about 2760). Per-row part of the profiled step (`real_35b_batch_step_profile`):

| ms per step | B = 4 rb2 | B = 4 rb3 | B = 16 rb2 | B = 16 rb3 |
|---|---|---|---|---|
| gdn_conv + gdn_delta_rule | 3.92 | 2.40 | 17.51 | 9.31 |
| rope + qk_norm + kv_append + attn_decode | 2.97 | 1.04 + qk_norm | 12.06 | 2.29 + qk_norm |
| model-loop step (eager in rb2, graph in rb3) | 25.60 | 21.65 | 65.70 | 52.31 |
| model-loop tok/s | 156.2 | 184.8 | 243.5 | 305.9 |

(qk_norm is a row-parallel batch op now and below the profile's top-14 cut.) What is left in the step is weight traffic:
moe_experts 8.0 / 18.0 ms and matmul_fp8 7.4 / 14.0 ms at B = 4 / 16, plus the delta rule at 8.6 ms for 16 rows.

**(b) Device-side top-k sampling.** `HeadMode::LastTopK(k)`: a sampled row with 1 <= top_k <= 64 gets its k best logits from one kernel
(`topk_f32`, in the sampler's order: value descending, ties by ascending index) and three small downloads instead of a `[B, V]` f32 download
(V = 248,320) and a host sort; `Sampler::sample_topk` draws from those candidates. **Parity is exact by construction and measured**: the
host `sample_row` selects exactly these k candidates in this order and touches nothing else, so one draw gives one token. Evidence:
unit test against `sample_row` for every draw (V up to 248,320, k up to 64, temperature, top-p, ties, -0.0 and -inf entries); kernel test
against a host sort (8180 pairs, NaN rows flagged and sent back to the host path); the 35B engine test with seeded sampled requests at 4
and 16 slots equals `LocalEngine` token for token; and `serve` returns identical text for 16 concurrent sampled requests with the knob on and
off (section 5). Rows with a constraint, `min_tokens`, trace logits, `top_k` outside 1..=64, or a NaN row use the host path. A single row
(B = 1) is the solo step and also samples on the host. Cost of a sampled batched step including the host work (`real_35b_batch_step_sampled_timing`,
#4698):

| rows | host sampling | device top-k | change |
|---|---|---|---|
| 4 | 33.80 ms, 118.3 tok/s | 22.66 ms, 176.5 tok/s | -33% step time |
| 16 | 72.69 ms, 220.1 tok/s | 54.86 ms, 291.6 tok/s | -25% step time |

This is Fable's B2 (12 to 22% at N = 16 sampled); measured here it is larger because the host download and sort sit in the serialized step.
Because parity holds, it is **on by default** (`INTARSIA_DEVICE_SAMPLING=0` turns it off); the localeval cells above are greedy and unaffected.

**(c) Graph buckets and the m3 fix.** A batched step of 2 to 16 rows is recorded once per row count as a CUDA graph and replayed (the row
tables are rewritten before each replay, which is why a recording is valid for any set of sequences). Solo decode graphs are now kept per
sequence state instead of one slot, so alternating single rows no longer re-record (Fable m3; test `alternating_rows_and_batches_do_not_recapture`).
Server counters after the whole matrix: 41 captures for thousands of replays.

**(d) Prefill/decode overlap (opt-in `INTARSIA_OVERLAP=1`).** A decode twin of the model (same weights and plan, its own workspace, scratch,
graphs, row tables and sampling buffers) enqueues batched decode on stream 1, the highest-priority non-blocking stream, while a prefill chunk
of another request runs on stream 0 in its own thread. The chunk schedule is the scheduler's unchanged one, so outputs stay bitwise (long-prompt
engine test, 2.6k to 16k tokens, overlap on). Measured effect in section 1: lower ITL (p99 at 16 streams 500 ms to 75 to 150 ms; mean 62 ms to 38 to 45 ms)
and +8% per-stream decode speed at 4 streams (41.4 to 44.6 tok/s median), at the price of 8 to 15% aggregate throughput on this short-output benchmark.

## 3. Method

localeval (`/opt/code/deepeval`, `localeval speed --sizes 1024 --gen 128 --concurrency N`, `--thinking off`), server started by
`deploy/localeval/serve-and-bench.sh --served-model-name intarsia --max-seq 8192 --max-batch 16`, other models' page cache evicted before each
server (the loader finds MemFree, not MemAvailable; PR #119). One slot per job, `--quiet gpu,cpu`, fleetq owner `claude-f31ccf91-rb3`.
localeval aggregates one burst per invocation, so each multi-stream cell is three invocations (`bench-inner2.sh`); one stream is 3 reps in
one invocation. ITL is the server's `/metrics` histogram, scraped around each burst. Final matrix: #4699 (merged head, main `6ca51a19`);
sub-step attribution: #4686; first matrix after the admission fix: #4661; three-burst matrix before the merge: #4695. Baseline from rb2's doc
for comparison: 73.4 / 123.3 / 166.6 / 83.2; this session's main reproduces it.

## 4. Why (d) does not pay here

At 4 streams the burst is 0.9 s of four serialized 1k prefills plus 127 steps. With overlap the first stream decodes while the others
prefill, but the chunk and the decode both stream the expert and projection weights, so each slows the other; the four prefills end at 1.38 s
instead of 0.88 s, and the early rows run at B = 1 to 3 (smaller batches, more steps per token). The GPU is the shared resource, not the
schedule. Overlap helps when the objective is decode latency under arrivals, not burst throughput.

## 5. Gates

All on the final head unless stated; fleetq jobs in brackets.

| gate | result |
|---|---|
| G-B new kernels (`row_tables`: attention, conv, delta rule, kv_append, rope, topk) | pass, byte-identical to the solo launches for M 1..16 [#4676, #4702] |
| G-B R0 regression (`rows_exact`, `rows_registry`) | pass [#4676, #4702] |
| G-C `batch_recurrent` (tiny hybrid, engine, overlap, decode twin beside a running prefill) | 10 of 10 [#4645, #4698, #4702] |
| G-C real Qwen3-8B, N 2..16 x 130 steps | 4290 (row, step) pairs bitwise [#4645 pre-merge, #4702] |
| G-C real 35B, N 2..16 x 130 steps | 4290 pairs bitwise [#4638, #4645, #4676, #4702] |
| 35B engine == `LocalEngine`, greedy and seeded-sampled, 4 and 16 slots; overlap off and on; device sampling on and off | pass [#4687, #4698, #4702] |
| long prompts 2.6k to 16k tokens (several busy chunks), engine == `LocalEngine`, overlap off and on | pass [#4687, #4698, #4702] |
| `serve` parity: 16 requests (48 to 2675 prompt tokens), each alone vs all at once, greedy and sampled, default / device sampling / overlap / both | all identical, also across configs [#4692, #4701] |
| CUDA no-op, batching off (35B and 8B `plan show` json and txt, 256 greedy ids) vs main | IDENTICAL [#4686, #4699] |
| Metal no-op on endor (shared core changed): plan show, 256 ids short and 4k, goldens, kernels test, workspace tests | identical to main `6ca51a19` on the final head `af1bc4d3`: plan show, ids (short, 4k), both goldens; kernels test 41 kernels / 778 cases; workspace tests 817 passed, 0 failed [#4703; also #4685 on `c4947850`] |

## 6. Things found on the way

1. **Admission hid the speedup.** The first speed run showed no gain: under `serve` only admitted kernels are selectable, the row-table
   twins are never swept (no sweep case has a table of device addresses), so every batched step logged "row-table path unavailable" and fell
   back to per-row launches. Tests run without admission, so G-B and G-C could not see it. Fix: a row-table kernel takes the admission of the
   kernel it twins (`Registry::effective_admission`, unit test). Lesson: run the speed matrix through `serve` before believing a kernel PR.
2. **Capture beside a prefill.** The decode twin captures graphs on stream 1 while the prefill thread works on stream 0. The first overlap
   test failed with "operation not permitted when stream is capturing" because the test itself ran its solo reference steps on the prefill
   stream, in the same process; a micro-experiment (#4659) showed no ordinary call from a second thread breaks a thread-local capture except a
   device-wide sync. The test now runs only batched steps beside the chunk and replays the solo references afterwards.
3. **Hard-coded stream 0.** `topk_rows` was launched on stream 0 from the twin, which raced with the prefill and produced garbage tokens under
   overlap + device sampling. It now uses the model's stream; a grep of the core for other literal stream 0 on the twin's paths found only state
   copies (stream 0 by design) and MTP (refused at batch > 1).
4. **Single sampled row.** `forward_batch` with one row is the solo step, which has no top-k head; it now maps `LastTopK` to `LastLogits`
   (same token) instead of failing the engine.
5. **Loader free memory (PR #119) shows in the gates.** The Qwen3-8B G-C test failed at the `kv_cap_tokens` assert with 2.9 GB reported free
   after a warm page cache and passed with 37.5 GB after eviction; `serve --max-batch 16` on main got a context of 498 tokens in that state,
   with #119 it gets 4096 (default max-seq). Field evidence for #119.
6. New `INTARSIA_*` knobs have rows in `docs/tuning.md`: `INTARSIA_ROW_TABLES`, `INTARSIA_BATCH_GRAPHS` (default on), `INTARSIA_DEVICE_SAMPLING`
   (default on), `INTARSIA_OVERLAP` (opt-in).

## 7. Decisions and open points

Rows in `docs/decisions/LOG.md` (2026-10-23, agent rb3). Open for the user: (1) device sampling is default on because parity holds; say if
you want it opt-in; (2) overlap stays opt-in because it lowers aggregate throughput by 8 to 15%; a latency-oriented recipe could enable it;
(3) the 4-stream result is 141.2 mean, 139.3 worst burst: the remaining gap to a clear margin is prefill (0.9 s of serialized 1k prompts) and
weight traffic in the step; the next lever is prefill speed (lane pf), not the batched step.

## 8. Reproduce

```
# G-B / G-C (quiet slot)
cargo test -p intarsia-backend-cuda --release --test row_tables --test rows_exact --test rows_registry -- --test-threads=1
INTARSIA_BATCH_TEST_BACKEND=cuda INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-models --release --test batch_recurrent -- --test-threads=1
INTARSIA_QWEN36_DIR=<35B> INTARSIA_QWEN3_DIR=<8B> INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-models --release --test batch_cuda_real -- --test-threads=1
#   long prompts: ... real_35b_batched_engine_long_prompts -- --ignored ; sampled timing: ... real_35b_batch_step_sampled_timing -- --ignored
# speed: deploy/localeval/serve-and-bench.sh --served-model-name intarsia --max-seq 8192 --max-batch 16 -- \
#   uv run --project /opt/code/deepeval localeval speed --url http://spark-4e77:8290/v1 --model intarsia --thinking off --sizes 1024 --gen 128 --concurrency 4
# A/B switches: INTARSIA_ROW_TABLES=0  INTARSIA_BATCH_GRAPHS=0  INTARSIA_DEVICE_SAMPLING=0  INTARSIA_OVERLAP=1
```
