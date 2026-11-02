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
Could the wrong directory have made either build wrong? The base was rebuilt from nothing: `src-base` removed and recreated from `git archive e72ce1964`, 206 crates compiled, the run's own stamp found in the binary, and the guard (`INTARSIA_NCCL_GATHER`, absent from the base sources and the base binary, present in the branch binary) passed; so the base is main and the head is a build containing my changes. What was not pinned: the head binary is read at run time from the graphs dir, so a rebuild there during the run would change it; the stamp read after the plan/greedy steps was `6551ae509`, the same as the one the guard saw, and nothing else was running on spark2 at that time. The graphs lane's `src-base` was deleted and rebuilt (it is the throw-away base tree that every plandiff run clears).
A rerun with this lane's own binary (#7798) was cancelled at the coordinator's request: the coordinator runs a plan diff on the final plan head before main. `plandiff.sh` now exits 2 without `SPARK_DIR` set (no default).
