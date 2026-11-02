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
`dspark_rounds_match_the_official_drafter` (#7794, merged head, fixtures `layers_*.safetensors` copied from the tp2 lane's endor cache into `~/intarsia-dsv41-draftdev/fixtures`, md5 equal to the ref0 copy): cpu fp32 15/15 drafts equal (head.x 1.7e-3, logits 5.4e-4), cpu bf16 15/15 (3.0e-2 / 1.0e-2), the numbers of the tp2 doc;
cuda bf16 6/15 with worst head.x 4.593e-1, logits 1.988e-1, confidence 1.938: identical to six earlier runs in the tp2 lane's `dspark-parity-cuda.log` / `dspark-var.log` (4.593e-1 / 1.988e-1), which the tp2 doc traced to bf16 rounding flipping a near-tied expert choice (not a kernel bug). So no change from the draft-round work; no baseline-binary run needed.
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

## Task 4: TensorFold (BLACKFOLD) nsys profile, one 48-token request

Window #7792 (both Sparks, vLLM restored). TensorFold head spark1 + worker spark2 in `ds41-prof-dd` containers (the recipe's image and files, mounted read only; host nsys 2025.3.2
mounted in), `nsys launch --trace=cuda,nvtx --cuda-graph-trace=node`, `start/stop` around one request: 1222-token prompt (this lane's results prose), 48 greedy tokens, 16K pool,
`TENSORFOLD_MEMORY_RESERVE_GIB=10.9` (11.6 failed the budget check once: weights 98.9 GiB vs 98.4 = 110.0 free - 11.6; coordinator approved 10.9). Scripts, analysis and logs: `docs/results/2026-11-02-dsv41-draftdev/tfprof/`
(`run-tfprof.sh`, `bf-node-prof.sh`, `analyze.py`, `analysis-r{0,1}.txt`; the 4 MB CSVs are in `.scratch/tfprof/out`). The request ran 22 rounds, 25/44 drafts accepted (2.1 tokens a round).
`grid_breakdown.py` did not fit: it splits passes at `AllGather` and TensorFold has only 83 of them (one per pass at the head); its per-layer collectives are its own RDMA kernels
(`dsv41_rdma_stage` + `dsv41_rdma_collect`, 83 each a verify pass). `analyze.py` splits at `_embed_init` instead.

Rank 0 (rank 1 is within 0.4 ms everywhere), median per pass; "kernel ms" is the sum of kernel durations on all streams, so it exceeds the span where side streams overlap:

| | draft pass (5 rows, 3 stages + head) | verify pass (2-4 rows; 3 rows 32.2 ms, 2 rows 27.7, 4 rows 37.2) |
|---|---:|---:|
| span (first to last kernel) | 3.79 ms | 32.35 ms |
| GPU idle inside the span | 0.01 ms | 0.05 ms |
| kernels | 115 | 1250 |
| kernel ms: MoE experts (`grouped_cp`, rows, `down_combine`) | 1.29 | 20.19 (80 launches, 40 layers x 2) |
| kernel ms: dense linears (EXL3 `glinear`, `_gemm`, unpack, rotations) | 2.14 | 11.54 |
| kernel ms: L2 prefetch (side stream, overlapped) | 0 | 9.37 |
| kernel ms: hyper-connection | 0.29 | 4.81 |
| kernel ms: collectives (RDMA stage + collect; 83 pairs) | 0.32 | 4.08 |
| kernel ms: attention (sparse attn, KV, indexer, rope) | 0.20 | 3.53 |
| kernel ms: router / norm / engram / embed+head | 0.58 | 1.0 |

Round: draft start to next draft start 36.4 ms median (draft 3.8 + verify ~32.3 + two gaps of ~0.1 ms) for 2.1-2.2 tokens = about 16.7 ms/token, 60 tok/s on a request this short.
Ours (window 1, binary `f4ae35c84`, spec on; different weights: our Mia EXL3 against TensorFold's BLACKFOLD 2.9 bpw, and a different prompt mix): 51.5 ms a round (23.6 ms/token x 2.18 tokens).
The 4-row verify pass was 52.5 ms in the graphs lane's step-3 measurement (before split-K, so now lower; not re-measured here). So the verify pass is the gap; the draft pass is small in both.
What to take from it: (1) TensorFold's verify is **MoE 20 ms in 80 grouped_cp launches with the L2 prefetch overlapped on a side stream** (9.4 ms of kernel time hidden), and 11.5 ms of dense linears for 274 launches;
(2) its collectives cost 4 ms of kernel time a pass for 83 stage/collect pairs and no NCCL in the layers, against our 80 NCCL all-reduces (about 36-50 us each, 3-4 ms): similar in total,
so the collective cost is not the gap; (3) no idle between kernels in either pass (graphs), so TensorFold's advantage is kernel time, not launch overhead.

Memory (the rules' 2.8 GiB watcher): MemAvailable fell from 110.9 to 3.5 GiB on spark1 (6.1 on spark2) while loading (42.5 s, `loaded in 42.5s`, a warm page cache) and stayed at 3.2-3.5 on spark1 through the capture.
**The watcher killed the container on spark1 at 03:43:37 (2.4 GiB)** while the host ran `nsys stats` on the two reports, after the stop and the CSV export had completed: the traced request, both reports and both CSVs are complete,
the planned untraced 128-token request did not run (no plain tok/s from TensorFold in this window). vLLM restored on both (03:47, `Up 3 minutes`, `fleetq mark-up`). Not retried, as agreed.

### Task 4b: is the gap overlap? (TF vs ours, 3-row verify passes; analysis of the existing CSVs, no new window)

> **Partly withdrawn, see 4c and 4d.** The reading "kernel time is the same, the gap is 12.4 ms of concurrent work", the "70 % of the round gap" and the "-6 to -10 ms" estimate below counted waiting inside early-launched kernels as work. Kept for the record; do not use those three numbers.

Scripts: `tfprof/overlap.py` (TF: streams, class overlap, concurrency), `tfprof/pairs.py` (kernel pairs, same vs cross stream), `tfprof/ours.py` (ours). TF = this lane's capture (12 verify passes with 3 rows, rank 0). Ours = graphs lane's window 9 nsys
capture (`dsv41-graphs/.scratch/full-w9/ab1/nsys-r0-033722_cuda_gpu_trace.csv`, binary `6551ae509`, 21 verify passes with 3 rows, found by the `argmax` grid of the next segment). Classes are by kernel name on both sides and approximate (TF `other` holds router/rmsnorm; our `moe` holds `moe_route`).

**Kernel time is the same; TF overlaps it, ours does not.** ms per 3-row verify pass:

| | TF | ours |
|---|---:|---:|
| span (first kernel start to last kernel end) | 32.00 | 45.22 (nsys); 42.9 without nsys (engine `pass ms` for 3 rows, from the graphs lane's log) |
| sum of kernel durations, prefetch excluded | 44.15 | 43.12 |
| union of busy time (time with at least one kernel) | 31.96 (31.79 without prefetch) | 43.10 |
| sum minus union = work running concurrently | **12.4** (21.65 with prefetch) | **0.02** |
| MoE (`grouped_cp` + combine) | 20.05 | 18.63 |
| dense linears (TF `glinear/_gemm`; ours `linear/split/dense_f16w/rot`) | 11.59 | 15.70 |
| hyper-connection | 4.60 | 2.39 |
| attention | 3.51 | 1.24 |
| collectives (TF RDMA stage+collect; ours NCCL all-reduce + round) | 3.30 | 4.82 |
| router, norms, other | 1.10 | 0.34 |
| L2 prefetch (TF side stream, hidden) | 9.45 | none |
| kernels per pass | 1250 | 2375 |
| idle between kernels | 0.05 | 2.1 (nsys) |

So the coordinator's hypothesis holds: with TF's overlap removed, TF's verify pass would be about 44 ms against our 43; with ours overlapped like TF's, 43.1 - 12.4 = 30.7 ms (+ little idle) against TF's 32.0.
Concurrency histogram (ms a pass with n kernels running): TF without prefetch {1: 20.4, 2: 10.5, 3: 0.9}; ours {1: 43.1, 2: 0.02}.

**1. Streams.** TF: 10 lanes in the node view. Streams 45, 46, 48 carry the layer chain in rotation (about 270-300 kernels a pass each, every class), 13 holds the main chain's remainder plus 78 prefetch launches and 40/37 the other 42 (120 prefetch launches a pass of about 85 us), and 42/39/43/36 hold the rest, with hc kernels on every lane (the node view does not say which lane is the HC side lane).
Ours: one lane per layer segment (about 190 lanes, each a graph node group) and nothing runs on two at once.

**2. Overlap, ms a pass (TF, prefetch excluded; pairs, so a 3-way overlap counts per pair; total 13.3).**
Same lane (consecutive kernels of one chain that start before the previous ends, i.e. programmatic dependent launch): 9.1 ms. `grouped_cp` x `grouped_cp` 2.75 (the down-projection starts about 70 us before the gate/up ends, 40 times a pass), attention x dense 1.68, dense x dense 1.39 (`glinear` pairs 0.98),
attention x attention 0.85, collective x dense 0.67, hc x dense 0.54.
Other lane (the side stream): 4.2 ms. hc x collective 1.63 (`_hc_post_only` runs under `dsv41_rdma_collect`, 74 times a pass), hc x dense 1.34, hc x other 0.50, hc x MoE 0.41, hc x attention 0.33.
Prefetch (not in the 13.3): under dense 5.3, hc 3.5, collective 1.8, attention 1.7, MoE only 0.56.
TensorFold's own source says why: `zig/src/cuda/launch.zig` sets `programmatic_stream_serialization`, `qmm_frag.cuh` (the EXL3 GEMM) and `dsv41/rdma_gather.cu` carry `griddepcontrol.wait` / `launch_dependents`; `bf-node.sh` sets `TF_DS_HC_SIDE=1` and `TF_DS_L2_PREFETCH=1`. Ours has neither (no `griddepcontrol` or programmatic launch anywhere in `kernels/` or `crates/`).

**3. Dense against MoE.** They do not overlap in either direction: 0.00 ms a pass (MoE x collective 0.01, MoE x attention 0.00). The chain at a layer boundary, from the timeline: second `grouped_cp` ends -> `rdma_stage` starts at the same microsecond -> `rdma_collect` -> `_hc_post_only` -> `_hc_finish_s` -> next layer's first `glinear` (about 10 us after) while `_hc_mix_part` / `_hc_sinkhorn` run on the side lane.
So the next layer's dense waits for the collective and the hc post; the overlap sits inside the attention half of a layer (attention x dense 1.68, dense x dense, attention x attention) and in the MoE kernel pair (2.75), not across the MoE.

**4. Slower when overlapped?** Not shown, not excluded. Per-kernel durations of the same `glinear` template overlapped by prefetch vs not differ in both directions (-17 %, -26 %, +100 %, +36 % by size; position in the layer confounds it). For the MoE pair, 2nd-kernel end minus 1st-kernel end against their overlap has slope +0.45 (0 would be pure waiting, -1 pure concurrent work), so part of the overlap is the second kernel's launch and wait hidden under the first's tail
(the kernel sum overstates the work a little: the real saving in span is what the table shows, 12.4). TF's MoE kernels total 20.05 ms against our 18.63, so the prefetch gives no visible MoE speed-up.

**What the overlap explains (WITHDRAWN, see 4c).** Verify gap at 3 rows: 42.9 (ours, no nsys) - 32.2 (TF, under nsys) = 10.7 ms, and the overlap is 12.4 ms of TF's kernel sum. A round is 51.5 ms ours against 36.4 TF = 15.1 ms: about 10.7 in the verify pass and about 4.4 in the draft pass plus round overhead (ours: draft 6.2 ms under nsys, engine overhead 9 ms; TF: 3.8 + 0.3).
Kernel time and the rest of the pass explain the other 0-2 ms. So overlap accounts for roughly 70 % of the round gap (10.7 of 15.1).

**Smallest change that lets ours overlap the same way (estimates, not measured; not implemented; the ms figures here are WITHDRAWN, see 4c).**
1. PDL on the MoE pair: launch the two `grouped_cp` kernels of a layer with the programmatic-stream-serialization attribute and add `griddepcontrol.wait` before the first dependent read and `launch_dependents` after the last independent work. TF gets 2.75 ms a pass from exactly this pair (same kernel name; ours is a port).
   Estimate -2 to -3 ms a pass, about 5 % of a round, about +1 to 2.5 tok/s. Needs `cudaLaunchKernelEx` in the launch path of that kernel and graph capture of programmatic edges (CUDA >= 12.3; TF replays them under graphs).
   File owner: kernels/cuda/exl3_moe_cp.cu is next to the graphs lane's `exl3.cu`, so hand it to that lane.
2. PDL on the dense chain (`linear_kernel`, `split_kernel`, `split_finish_kernel`, `rot_in_kernel`, 90-390 launches a pass of 5-80 us): TF's same-lane overlap outside the MoE is 6.3 ms. Estimate -3 to -5 ms (ours has twice the kernels, 2375 against 1250, so the early-launch hide matters more; fusing `split_kernel` + `split_finish_kernel` + `rot_in_kernel` would also cut launches).
3. hc on a side lane: fork/join in the capture with events (`ik_rt_event_record` / `ik_rt_stream_wait_event` exist). Ours hc is 2.39 ms a pass; TF's cross-lane overlap is 4.2 including collectives. Estimate -1 to -2 ms.
The three together are an estimated -6 to -10 ms of the 10.7 ms verify gap (about +14 to 25 % tok/s at the same acceptance); step 1 alone is the cheapest test and tells whether programmatic launch works in our captured graphs at all. Not part of this estimate: our NCCL all-reduces (80 a pass, 4.8 ms) cannot take part in PDL; TF's collectives are its own RDMA kernels.
Caveat: one TF request (12 three-row passes), and the TF prompt is 1222 tokens against a few tokens in ours (attention differs: TF 3.5 ms against our 1.2).

### Task 4c: correction. Kernel-duration sums overstate TensorFold's overlap; the critical path says where the gap is

Task 4b read "sum of kernel durations minus union = 12.4 ms of concurrent work" as TF doing the same work as ours in less time. That is not established. TF launches the next kernel of a chain early (programmatic dependent launch) and the early kernel
sits in `griddepcontrol.wait`, so its recorded duration includes waiting: the sum is inflated by an unknown part of the 12.4 ms. What a trace can give without that ambiguity is the **critical path**: each instant of the pass charged to the class of the
kernel that started last among those running (prefetch excluded), idle charged to idle (`crit.py`). 3-row verify pass, ms:

| class | TF | ours (nsys) | ours - TF |
|---|---:|---:|---:|
| dense linears | 7.60 | 15.70 | **+8.1** |
| collectives | 1.32 | 4.81 | **+3.5** |
| idle between kernels | 0.21 | 2.12 | +1.9 |
| MoE | 17.29 | 18.61 | +1.3 |
| hyper-connection | 2.94 | 2.39 | -0.6 |
| attention | 1.90 | 1.24 | -0.7 |
| other | 0.75 | 0.34 | -0.4 |
| total (= span) | 32.00 | 45.22 | +13.2 (10.9 against our 42.9 ms without nsys) |

Per block, collective to collective (`blocks.py`, nsys both): attention block 237 us TF against 334 us ours, MoE block 409 against 544, a layer 731 against 968 us; a collective 14 against 30 us.

So the gap is **not mostly concurrency**: about 8 ms of the 13 sit in the dense linears (TF: about 274 launches of one `glinear` kernel; ours about 1100 launches of `linear_kernel`, `split_kernel`, `split_finish_kernel`, `rot_in_kernel`, transposes, 5-80 us each, of which 400-1000 are 1-5 us kernels),
3.5 ms in the collectives (TF hides its RDMA exchange under the hyper-connection post; ours is 80 serial NCCL all-reduces of about 30 us plus a bf16 round), 1.9 ms is idle time between our kernels (launch gaps; programmatic launch or fewer launches removes it).
What PDL and a side stream can win from this trace: the idle 1.9 ms, part of the 400-1000 tiny kernels' ramp, and part of the collective (an HC kernel beside the exchange). Estimate: 2-4 ms, not the 6-10 ms of Task 4b, which counted waiting as work. That estimate is smaller and still an estimate.

### Task 4d: dense bytes against the bandwidth bound

Per rank, one verify pass reads every dense weight once. Sizes from the rank-0 cache (`~/intarsia-dsv41-tp2/cache/r0of2`, safetensors headers summed over the 43 layers; the BLACKFOLD checkpoint's TP shards have the same byte counts per tensor):
EXL3 trellis `wo_b` 500 MiB, `wq_b` 500, `wo_a` 4 x 100, `wq_a` 157, `wkv` 62.8, indexer `wq_b` 25, compressor 6.2, shared experts `w1/w2/w3` 127.3 + 128.0 + 127.3, `head` 236.7, Engram `wkv` 168.8 = **2439 MiB = 2.56 GB**; plus the router gate (f16) 150 MiB and the hyper-connection mixing weights 150 MiB.
At the graphs lane's 254 GB/s: EXL3 trellis 10.1 ms, with the gate 10.7, with the hc weights 11.3 ms (at the 273 GB/s peak 9.4 ms for the trellis).
Against that: ours, linear kernels only (`linear_kernel` 7.57 + `split_kernel` 4.66 = 12.2 ms) is 210 GB/s, 83 % of the floor, and the whole dense class 15.7 ms; TF's dense class takes **7.6 ms on the critical path, below the 10.1 ms DRAM time of its own bytes (even without the Engram `wkv` and compressor, 9.3 ms)**.
That is not possible if TF's dense linears read all their weights from DRAM inside their own window. TF's L2 prefetch kernels run 120 launches, 9.45 ms of kernel time, which at 254 GB/s is 2.4 GB, the size of the dense trellis bytes. So the data fit prefetch moving most of the dense weights' DRAM time to where other kernels run (attention, hyper-connection, collectives),
and the 8 ms dense difference in 4c may be prefetch showing up as faster linears. That reading predicts what the knockouts measure: `TF_DS_L2_PREFETCH=0` should cost TF about 8 ms a verify pass. Whether prefetch can help ours depends on a second stream running inside a captured TP=2 pass (the fork experiment).
TF's runtime has no switch for PDL: `TF_EXL3_PDL`, `TF_RDMA_PDL` and `TF_DS_TRITON_PDL` appear only in comments of the Zig build, the settings are compiled defaults (`rdma.zig` `pdl: bool = true`, `tri.zig`), so the "PDL off" knockout is not available without patching TF, which was ruled out.

## Plan diff against main (e72ce1964) of the shared-code changes

Run #7795 (`plandiff.sh`, `BASE_REF=e72ce1964 GREEDY=1 BASE_MUST_LACK=INTARSIA_NCCL_GATHER`). **By accident `SPARK_DIR` was the default, the graphs lane's `~/intarsia-dsv41-graphs`**, so the "branch binary" was that lane's `bin/intarsia`, stamp `6551ae509 dirty=no`, which is the graphs lane's merge of this lane's `6a50029e0`
(it contains the `add_f32`, `ik_rt_f32_sum_round`, `nccl.rs` and verify-argmax changes). That makes the result a valid plan diff of the merged code against main, closer to what ships than this branch alone. Result: 35B NVFP4 plan show unchanged (76 lines), Qwen3-8B plan show unchanged (25 lines), 64 greedy tokens identical on both.
Could the wrong directory have made either build wrong? The base was rebuilt from nothing: `src-base` removed and recreated from `git archive e72ce1964`, 206 crates compiled, the run's own stamp found in the binary, and the guard (`INTARSIA_NCCL_GATHER`, absent from the base sources and the base binary, present in the branch binary) passed; so the base is main and the head is a build containing my changes. What was not pinned: the head binary is read at run time from the graphs dir, so a rebuild there during the run would change it; the stamp read after the plan/greedy steps was `6551ae509`, the same as the one the guard saw, and fleetq showed no other job on spark2 during the run (I did not check for an un-slotted rebuild). The graphs lane's `src-base` was deleted and rebuilt (it is the throw-away base tree that every plandiff run clears).
A rerun with this lane's own binary (#7798) was cancelled at the coordinator's request: the coordinator runs a plan diff on the final plan head before main. `plandiff.sh` now exits 2 without `SPARK_DIR` set (no default).

### Task 4e: TensorFold knockouts, full model, no nsys (window #7797, vLLM restored)

> **Measured with a custom streaming client (`knock_client.py`), not localeval, against the fleet rule.** The within-config comparison stands as evidence that the switches did not matter. Do not quote these tok/s as TensorFold's speed; the like-for-like figure is the localeval section below. Also: the switches taking effect was inferred from the env handling, not seen in a trace, and that limits this result: if the worker ignored them, or the head's flags did not reach the kernels, this table shows nothing about them.

`TF_DS_L2_PREFETCH` and `TF_DS_HC_SIDE` (both on in the recipe; the head's `native.zig` reads them as "on unless `0`") switched per config, head + worker restarted each time (reserve 10.9, 16K pool), same 4 prompts of ~950-990 tokens x 2 repetitions, 160 tokens each, temperature 0
(`tfprof/run-tfknock.sh`; 2 warm-up requests first; decode time = client stream time first token to last token, rounds and accepts from the head's `done` log; ms/round = decode time / rounds). Every config produced identical tokens (same `sha` per prompt) and the same 346 rounds, accepted 926/1190 = 3.68 tokens a round.

| config | ms/round | tok/s (decode) |
|---|---:|---:|
| prefetch on, hc side on (recipe) | 43.2 | 85.1 |
| prefetch **off**, hc side on | 42.8 | 85.9 |
| prefetch on, hc side **off** | 41.8 | 88.0 |
| both off | 43.5 | 84.4 |

Per-request spread inside one config is 39-55 ms/round (8 requests of 1.6-2.0 s each), so the four means (41.8-43.5) are the same within noise. **Neither knob matters for this workload.** The prefetch hypothesis of 4d (prefetch hides about 8 ms of dense DRAM time) is not supported: switching it off cost nothing.
Caveats: (1) no nsys on these runs, so that the switches really removed the prefetch kernels / the side stream is inferred from the head's env handling (the worker gets the same `-e` flags via the shared `common` string), not shown; (2) 4d's byte count came from this lane's Mia cache, not from BLACKFOLD's rank cache (different quantisation, so TF may read fewer bytes than I computed and sit above its own floor);
(3) 160-token requests at ~1K prompt, 3.68 tokens a round here against 2.1 in the nsys request (different prompts, more repetitive text), so absolute round times differ from 4. So the 4c critical-path gap (dense 7.6 vs 15.7 ms) is not explained by prefetch or side-stream; what remains is TF's kernels themselves (one `glinear` template, 274 launches against our ~1100) and its RDMA collectives.

### Task 4f: plan code against TensorFold, same prompts, same client (window 2, vLLM restored)

Ours: plan code at `f4ae35c84`, `win2/ab0` (speed, dataset, knock prompts) and `win2/ab1` (nsys). TF: BLACKFOLD recipe, reserve 10.9, 16K pool, side and prefetch on (`win2/run-tfll.sh`). Scripts and per-request logs are in `2026-11-02-dsv41-draftdev/win2/`; `final.py` prints the tables below (run it beside TF's `out/tf-r0_cuda_gpu_trace.csv`, with `OURS_CSV` set to `nsys-r0-051236_cuda_gpu_trace.csv`; the CSVs are not committed).

**localeval** (`localeval compare 20261102-051128-adhoc-dsv41-intarsia-tp2-speed-ab0 20261102-051817-adhoc-dsv41-blackfold-tf-speed-ll`; dataset runs `...050952...ldata-ab0` and `...051857...ldata-ll`, 8 rows, 4 identical prompts twice, 160 tokens):

| workload | decode tok/s ours | TF | TF/ours | prefill tok/s ours | TF |
|---|---:|---:|---:|---:|---:|
| 136 prompt | 44.9 | 61.8 | 1.38 | 264 | 396 |
| 1027 prompt | 38.7 | 60.0 | 1.55 | 545 | 1459 |
| 4095 prompt | 40.5 | 57.5 | 1.42 | 577 | 2074 |
| dataset, ~1K prompt | 50.0 | 71.2 | 1.42 | | |

localeval warned on the TF speed run: decode spread 14.8 % at 128 tokens, and no in-flight counters to prove the server idle (the slot was exclusive). The gap is far outside that spread. Compare within this window only: the Task 1 figure (42.4 tok/s) is a different measurement, not re-run here.

**The gap is ms/round on chat prompts.** Dataset: 238 rounds against 229 for the 640 tokens of one pass over the prompts (2.69 against 2.79 tokens a round), so about 54 against 39 ms/round (decode time from the localeval mean, rounds from the logs). Speed sweep at 1K: 2.10 against 2.18 tokens a round (183 and 176 rounds for 3 x 128 tokens), so about 54 against 36 ms/round.

**Raw-completion prompts also differ in tokens a round** (custom client `knock_client.py`, so not for quoting as engine speed; comparison only): 54.3 ms/round and 2.92 tokens a round for us, 43.2 and 3.70 for TF, on the same 4 prompt files. Greedy text is not the same: the first 40 characters match on prompts 0 and 1, and differ on 2 and 3 (prompt 2 differs from the first token). So these tokens a round are not measured on the same text. Not investigated: our prompt token count is not in our log, so a tokenisation difference is not excluded. The drafter finding above (bf16 leading drafts equal 6/15 on CUDA, 15/15 on CPU) is a candidate for the lower acceptance, not shown to be the cause.

**nsys, 3-row verify pass, median span without our draft part: 44.78 ms (13 passes) against TF 32.19 ms (12 passes), gap 12.6 ms.** Mean span with our draft part is 47.58 ms (it was 45.22 in Task 4 on another request and binary). Critical-path partition, ms/pass: moe 19.2 / 17.29, dense 16.11 / 7.6, coll 3.87 / 1.32, idle 3.48 / 0.21, hc 2.44 / 2.94, attn 2.13 / 1.9, other 0.35 / 0.75 (ours / TF). Our graph has no overlap (concurrency saving 0.02 ms). Same conclusion as 4c: dense first, then collectives and idle.

Dense linears per shape (kernel sum, ms/pass). Our graph is one stream, so ours is real time. TF kernels may wait inside PDL (`griddepcontrol.wait`), so **TF's times here are upper bounds and its GB/s lower bounds**:

| shape | MiB TF / ours | TF calls | TF ms | TF GB/s | ours calls | ours ms | ours GB/s | ours-TF ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_b | 518.7 / 518.7 | 41 | 3.12 | 174 | 41 | 3.21 | 169 | +0.09 |
| wq_b | 500 / 500 | 40 | 2.32 | 226 | 40 | 2.64 | 199 | +0.32 |
| wo_a | 400 / 400 | 40 | 1.86 | 226 | 160 | 2.81 | 149 | +0.95 |
| wq_a + wkv (+compr) | 230.7 / 230.7 | 41 | 1.29 | 188 | 90 | 1.81 | 133 | +0.52 |
| head | 236.7 / 236.7 | 1 | 1.02 | 243 | 1 | 1.08 | 231 | +0.05 |
| engram wkv | 84.4 / 168.8 | 2 | 0.39 | 229 | 2 | 0.83 | 214 | +0.44 |
| gate (f16) | 150 / 150 | 40 | 0.62 | 252 | 40 | 1.05 | 150 | +0.42 |
| indexer (wq_b, wk, proj) | 50.2 / 50.2 | 20 | 0.49 | | 20 | 0.23 | | -0.26 |
| helpers (TF rot_many; ours rot_in, split_finish, transposes) | - | 49 | 0.49 | | 679 | 1.02 | | +0.54 |

The per-shape excess sums to 3.07 ms of the 8.5 ms critical-path dense gap. The other ~5.4 ms is not explained by this table: TF's dense kernel sum (11.6) is 4 ms above its critical-path dense (7.6), which is either overlap with other work or PDL wait, and these numbers cannot separate the two (see the 4c correction). Where our per-shape excess is largest, we also make more calls: wo_a 160 against 40, wq_a + wkv 90 against 41, helpers 679 against 49.

Idle before collectives, median us/pass (ours total 2.3 ms median, max 6.87; TF 0.20 ms): Engram gather to all-reduce 1066, linear to all-reduce 355, argmax to embedding 283, combine to all-reduce 154. Engram bf16 all-reduce itself: median 128 us, mean 190, max 719 (26 calls); f32: median 29 us. TF's verify exchanges through its own RDMA stage (4122 `dsv41_rdma` kernel rows in the trace against 83 NCCL rows; which pass the 83 belong to was not checked), and its largest gap is 95 us.

Memory: min MemAvailable in the TF localeval window was 3.1 GiB, no kill. The brief's limit is 5 GiB except 2.8 for the TF profile; this was the same TF recipe but not the profile, and I used 2.8. Reserve 10.9 leaves too little margin for a 5 GiB limit.

## Idle gaps: Engram segments and rank skew (windows 3-5)

**Result, measured with localeval on the dataset, each A/B inside one window:**

| change | decode tok/s, off to on (means) | ms/token | 3-row pass | default |
|---|---|---:|---:|---|
| one graph a pass | 50.97 to 51.7 | -0.28 | about -0.9 ms | on |
| `INTARSIA_TP_SPIN_MS=100` | 51.85 to 52.9 | -0.38 | about -0.8 ms (draft pass about -0.45 ms) | off |

In both cases the worst run with the change is above the best run without it. Only the first ships as committed; the second needs the env var in the launch. The two were not run together against a config with neither. Golden 640/640 in every config that ran it.

### Different text on raw prompts: different weights

Tokens are the same: `tokenizer.json` and `tokenizer_config.json` have the same sha1 in our rank cache and in TF's model folder, and HF tokenizers give 988/990/990/947 tokens, TF's logged counts (counts, not ids: TF's Zig tokenizer was not run). Both raw paths encode with add_special=true and the post-processor is ByteLevel, so no BOS on either side. The weights differ: ours come from `Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw` (45ee5d0, EXL3 1.6.1), TF loads `/mnt/models/dsv41-1017/blackfold/exl3-2.9bpw-heretic` (the BLACKFOLD overlay, 1.6.2). 46961 of 47001 `tensor_storage` entries are identical; the rest are 40 `ffn.gate.bias` (f16 for TF, f32 for ours), plus our `hc_collapse`. So tonight's TF comparison holds for compute cost per shape, not for text, acceptance or tokens a round. That corrects the open point in 4f.

Our first-token margins (`idle/margin-*.json`): prompt 2 " I" -2.28, next " " -2.55 (0.26 nats); prompt 3 " the" -1.64, next " run" -2.63 (0.99 nats). TF's " [" on prompt 2 is not in our top 5. A margin across different weights does not say more than that.

### 1. One CUDA graph a pass (default on; `INTARSIA_DSV41_ENGRAM_ONEGRAPH=0` goes back)

Cause: the first collective of each CUDA graph starts late. In window 2's trace the gaps sit only at all-reduce indices 0, 2 and 29 (the first of each of 3 segments), median 208 / 704 / 313 us on rank 0, 16 / 273 / 380 us on rank 1; every other all-reduce has a median gap of about 3 us. Why the first one is late was not found. `NCCL_GRAPH_MIXING_SUPPORT=0`, `NCCL_GRAPH_STREAM_ORDERING=0` and `NCCL_GRAPH_HELPER_DISABLE=1` did not move it, so these are not the mechanism.

The change: the whole pass is one recording, all Engram records are waited for and uploaded before it starts. Engram wait goes 0.13 to 0.22 ms, because nothing overlaps it now.

| whole pass ms, median, rank 0 | rows=2 | rows=3 | rows=4 | Engram wait, rows=3 |
|---|---:|---:|---:|---:|
| 3 segments, window 3 | 38.74 / 38.27 | 43.40 / 43.15 | 48.80 / 48.73 | 0.13 / 0.16 |
| 1 graph, window 3 | 37.58 / 37.41 | 42.49 / 42.40 | 47.62 / 47.74 | 0.24 / 0.22 |
| 3 segments, window 4 | 38.17 | 43.38 | 49.06 | 0.13 |
| 1 graph, window 4 | 37.98 / 37.42 | 42.54 / 42.02 | 47.82 / 47.69 | 0.21 / 0.23 |

Trace on the same prompt, rank 0: idle before NCCL kernels 1.55 to 0.28 ms/pass, Engram all-reduce median 128 to 24 us (max 719 to 35), verify span 44.78 to 43.04 ms.

localeval, dataset decode tok/s: 51.1 / 51.2 / 50.6 to 52.1 / 51.9 / 51.4 / 51.4, about +1.4 %. The speed sweep does not resolve it (1K prompt 42.4 / 40.3 / 40.5 to 41.2 / 42.0 / 41.7 / 40.2).

### 2. Rank skew (`INTARSIA_TP_SPIN_MS`, default off)

Rank 1 started a pass a median 521 us after rank 0 (trace, aligned on the ends of later all-reduces), so rank 0 sat 813 us in the first all-reduce against about 30 us for the others. Rank 0 sends the command before its own pass, so this is the command's way across: ping over the link is 0.6 ms, cpuidle exit latency goes up to 433 us, `net.core.busy_poll` is 0.

The change, in `tp_remote.rs`, no sudo: rank 1's reader polls its socket with `MSG_PEEK | MSG_DONTWAIT` for `N` ms after a command, and a second thread spins on the core `SO_INCOMING_CPU` reports, keeping it out of idle. Polling alone shows no clear gain (3-layer 5.13 / 5.36 ms against base 5.20 / 5.37); with the second thread 5.00 / 4.92 ms. Cost: two busy cores on rank 1 while decoding and for `N` ms after. Not measured: power, or any effect on GPU clocks beyond the pass times below.

| full model, window 5, rank 0, medians | rows=2 | rows=3 | rows=4 | draft (rows=5) | dataset tok/s | 1K sweep tok/s |
|---|---:|---:|---:|---:|---:|---:|
| off | 37.39 / 37.37 | 42.47 / 42.48 | 48.03 / 48.09 | 5.41 / 5.50 | 51.9 / 51.8 | 42.2 / 41.7 |
| `INTARSIA_TP_SPIN_MS=100` | 36.61 / 36.68 | 41.68 / 41.68 | 47.12 / 47.02 | 4.99 / 5.01 | 53.0 / 52.8 | 43.0 / 43.2 |

Trace: skew 521 to 9 us, first all-reduce on rank 0 813 to 133 us. Golden 640/640 on both configs with it on. The sweep at 4K and 16K also moved up (off 42.4, 43.7 and 42.1, 43.7; on 44.7, 42.7 and 44.3, 43.9), with the spread as large as the gain at 4K.

It stays off by default because `tp_remote.rs` serves every TP model and it spends two cores. Turning it on for DSv4.1 is one env var in the launch.

### Gates

- Golden: 640/640 in every config that ran it (window 3: all 4; window 4: both with one graph; window 5: both with spin), against `tok-intarsia-tp2-w12-specoff.st` or a config of the same window. The window 4 off config and the window 5 off configs did not run it.
- 3-layer TP=2 over NCCL with the real Engram tier (`idle/run3.sh`): text of 8 requests and logprob hashes of 2 requests identical across every arm of three runs, same 758 rounds.
- CUDA `device_position_passes_equal_host_passes`: no bitwise difference over 52 passes, 6 runs. Engram is off in that test, so it covers the capture path but not the Engram change; the TP=2 arms cover that.
- MMLU was not run: the output is bitwise the same on every check above.
- Plan diff, #7822, `8b1e955c7` against main `e72ce1964`, `INTARSIA_TP_SPIN_MS` unset, `SPARK_DIR=~/intarsia-dsv41-draftdev`, vLLM up: 35B (76 lines) and Qwen3-8B (25 lines) `plan show` unchanged, 64 greedy tokens identical on both. Base rebuilt from nothing (206 crates, run's own stamp); guards `INTARSIA_TP_SPIN_MS`, `INTARSIA_DSV41_ENGRAM_ONEGRAPH`, `INTARSIA_NCCL_GATHER` absent from the base binary, present in the branch binary; branch stamp `8b1e955c7 dirty=no`. Scope: this shows the build with the `tp_remote.rs` change and the `libc` dependency leaves those two models' single-rank plans and tokens unchanged. `generate` is one rank, so `tp_remote::serve` itself did not run for them, with or without the switch. With the switch unset that path ran only on DSv4.1 (TP=2 off configs of window 5). No run covers 35B or Qwen3 over TP with this code, and nothing here covers those models with the switch set.

An earlier gate run of mine failed 4/4 with "the backend cannot capture": I had left out `INTARSIA_DSV41_BACKEND=cuda`, so it ran on CPU. Not a regression; rerun with the variable.

`scripts/dsv41-tp2/memfine.sh` from plan hard-coded the graphs lane's process pattern, so a window from another Spark dir had no 1 Hz kill line, only the slower ssh watcher. Fixed in `5b771cd25`. Minimum MemAvailable in windows 3, 4 and 5: 6.5, 6.3 and 6.4 GiB, no kill.

### What is left before the pass

The trace now shows about 0.5 ms before launch: two host gaps of 267 and 208 us between the uploads, which is Engram issue plus wait. Argmax to embedding was the third target; this is what remains of it.

### Next, ranked (all estimates, none measured)

1. Dense call counts, graphs lane: `wo_a` 160 calls vs TF 40, `wq_a` + `wkv` 90 vs 41, helpers 679 vs 49. Up to about 2 ms/pass from the 4f table, about 0.9 ms/token, if per-call overhead is the cause.
2. PDL and the side stream (out of scope tonight): 5.4 ms/pass of the dense gap is unexplained and could be overlap. Largest possible, least known.
3. Engram `wkv` K-split per rank (3072 each, as TF): 0.44 ms/pass, about 0.2 ms/token. Numerics risk: partials rounded to bf16 per rank and summed are not bitwise the replicated result, so it needs MMLU, and it changes the cache format. Described only.
4. Find why the first collective of a graph starts late: the remaining first all-reduce gap is about 0.2 ms/pass on one rank, about 0.1 ms/token.
5. Issue the Engram rows for all drafts earlier, overlapping the host confidence step: at most 0.3 ms/pass, about 0.1 ms/token, and it touches the k path.

## Final head with the shipped defaults (2026-11-02 10:04-10:13 EST)

User decisions of 09:59: speculation on by default, `INTARSIA_TP_SPIN_MS=100` in `run-full.sh`. Window #7827 ran binary 7509282be with neither set by hand; both logs show them in force (`speculative decoding on`, `tp: remote rank reader spins 100 ms`). One run, no A/B. Raw: `2026-11-02-dsv41-draftdev/final/`.

| check | result |
|---|---|
| Golden vs `tok-intarsia-tp2-w12-specoff.st` | 640/640 |
| Decode by kind, localeval (catspeed.py), tok/s | all 49.1; chat 45.2, code 64.3, prose 41.2, recall 51.4 |
| Tokens a round, this window | 2.33; drafts accepted 62.7 % |
| Lowest MemAvailable | spark1 9.0, spark2 9.3 GiB; no kill |
| Metal no-op vs origin/main 985eced1f (#7828) | 35B and Qwen3-8B: `plan show` unchanged, 256 ids identical at n128 and n4096, `--spec off` |

By kind was 46.9 on merged 6551ae509 (window 9), before one graph a pass and spin. Different windows, so read it as an order of magnitude, not as the size of those two changes.

A rank cache without its `mtp` part stays at spec off with a log line. If only one rank has it, the handshake fingerprints differ and the load fails.
