# CUDA graphs for tp=2 decode (lane G1c, 2026-10-21)

Qwen3.6-35B-A3B NVFP4 at tp=2 across the two Sparks (rank 0 on spark1, rank 1 on spark2, NCCL 2.30.7 over ConnectX-7),
following G1b ([2026-10-21-35b-tp2-cuda.md](2026-10-21-35b-tp2-cuda.md)). Branch `feat/tp2-cuda-graphs`, based on
origin/main `c4ea00b` (PR #44 merged). Raw logs: `2026-10-21-tp2-cuda-graphs/raw/`; scripts `scripts/tp2-graphs/`.

## Headline

**With the decode step captured as a CUDA graph on both ranks (NCCL all-reduces included), tp=2 decode beats tp=1:**
+15 % / +16 % / +20 % at 1k / 4k / 16k, and +24 % over tp=2 eager. Graph and eager logits are bitwise identical, so
graphs are on by default at tp=2 when the communicator can be captured (NCCL). Against vLLM (tp=1 per Spark, same day):
decode **+21 % / +21 % / +17 %**, prefill +2 % / +8 % / +5 %.

The all-reduce does not dominate the step. Measured per call it is 16.7 µs of transfer (median), the same as the
design's 17 µs link figure, plus 6 µs mean wait: 1.9 ms of a 10.9 ms step. The rest of the step is compute that is
less than half of tp=1's, because fusion is off under TP: 784 launches per step against 325. Row-local fusion under
TP is therefore the next lever. It is implemented here as an opt-in backend capability (section 3).

## 1. Numbers

### Speed: interleaved A/B ×3, `intarsia bench speed --at-most`

Window 1 (fleetq #3101, both vLLMs down). Binary `2363660` (the change before the rebases; section 4 ties it to the
shipped tree). Each arm is a fresh `serve`; sizes 1024 / 4096 / 16384, gen 128, 2 reps + 1 warmup. tp1g and tp2g ran
in all three rounds, with the order reversed in round 2; tp2e ran in rounds 1 and 3. Cells are medians of the round
medians (`raw/window1/ab-table.txt`, `scripts/tp2-graphs/ab_table.py`).

| arm | prefill 1k\* | 4k | 16k | decode 1k | 4k | 16k |
|---|---|---|---|---|---|---|
| tp1g: tp=1, CUDA graphs (shipped) | 4516 | 5842 | 5525 | 81.8 | 79.5 | 72.7 |
| tp2e: tp=2 eager (main today) | 5132 | 7086 | 6868 | 75.7 | 74.4 | 70.6 |
| **tp2g: tp=2, CUDA graphs (this PR)** | **5382** | **7148** | **6890** | **93.7** | **92.5** | **87.5** |
| tp2g vs tp1g | +19 % | +22 % | +25 % | **+14.6 %** | **+16.4 %** | **+20.3 %** |
| tp2g vs tp2e | +5 % | +1 % | 0 % | **+23.8 %** | **+24.2 %** | **+23.9 %** |

\* The 1k prompts have fewer than 1024 tokens, so the 1k prefill column is mostly TTFT.

- Decode per round (tok/s):
  - tp1g: 82.1 / 81.8 / 81.7 at 1k, and 72.8 / 72.7 / 72.2 at 16k;
  - tp2g: 94.3 / 93.7 / 91.0 at 1k, and 87.5 / 88.0 / 86.4 at 16k.
  - The rounds do not overlap.
- Every tp2g arm passed `--require-graphs`, and rank 1's counters show replays in every round (`REPLAY-CHECK ok` in
  `raw/window1/steps.log`).
- G1b's numbers repeat within 1 %: tp1g 82.3 / 80.0 / 72.2 and tp2e 77.4 / 75.6 / 70.3.

### Against vLLM (localeval)

| | prefill 1k\* | 4k | 16k | decode 1k | 4k | 16k |
|---|---|---|---|---|---|---|
| intarsia tp=2 graphs (localeval, 3 reps, window 1) | 5274 | 6863 | 6580 | 94.6 | 92.8 | 86.4 |
| vLLM tp=1 on spark1 (localeval, CUDA round 7, same day) | 5171 | 6353 | 6275 | 77.9 | 76.6 | 73.6 |
| tp=2 graphs vs vLLM | +2 % | +8 % | +5 % | **+21 %** | **+21 %** | **+17 %** |

- Run: `/opt/code/deepeval/results/20261021-133100-adhoc-intarsia-35b-tp2-graphs-xhost/result.json`
  (`raw/window1/localeval-tp2g.log`). Its unanimis capture failed (ssh session refused), as in G1b; the result file is
  saved.
- vLLM is G1b's reference: `/opt/code/deepeval/results/20261021-105608-spark-guidance-vllm-cuda-round7/result.json`,
  with the same flags. It was not re-measured; the window time went to the A/B. The fleet runs no vLLM tp=2 for this
  model (G1b section 4).

### Where a tp=2 decode step goes (nsys, both ranks)

`nsys profile --cuda-graph-trace=node` on both ranks, 1k prompt, 64 tokens, graphs and eager, in window 1
(`raw/window1/split-*.txt`, `kernel-table.txt`, `scripts/tp2-graphs/nsys_split.py` and `kernel_table.py`). Traces stay
on the Sparks (`~/intarsia-tp2/xout/window1/`, about 100 MB each).

- **Splitting wait from transfer.** Both ranks issue the same collectives in the same order. Pairing the i-th NCCL
  kernel of each rank, transfer = min(d0, d1), because the rank that arrives last waits for nothing; the excess on the
  other rank is wait. No cross-host clock is needed.

| tp=2 graphs, per all-reduce (80 per step) | mean | median | p90 |
|---|---|---|---|
| rank-0 kernel | 24.1 µs | 16.8 | 17.8 |
| rank-1 kernel | 22.0 µs | 19.4 | 25.2 |
| **transfer** (min of the two) | **18.2 µs** | **16.7** | 17.6 |
| wait on rank 0 | 5.8 µs | 0.0 | 0.4 |
| wait on rank 1 | 3.7 µs | 3.0 | 8.1 |

- **Rank-0 step:**
  - graphs: 10.93 ms wall = 10.30 ms of kernels + 0.63 ms idle. Of the kernels, NCCL is 1.92 ms (transfer 1.46 +
    wait 0.47) and compute 8.37 ms.
  - eager (same trace setup): 13.02 ms wall = 10.63 ms of kernels + **2.39 ms idle**.
  - Graphs remove about 2 ms of launch, host and command time per step. This is the "per-step host cost" G1b
    suspected.
- **Against tp=1 graphs (12.49 ms wall, 12.15 ms of kernels):** tp=2's compute is 8.4 ms, not the ~6 ms that halving
  the weights would give. Fusion is off under TP (`Model::fused_ok`), so tp=2 runs:
  - `fp8_gemv` x130 where tp=1 runs `fp8_gemv_pair` x40 + `fp8_gemv_gated` x30;
  - unfused rmsnorm (131 launches), `silu_glu` and bf16 `gemv`;
  - 784 kernels per step against 325.

  The expert GEMVs did halve (`moe_dec_gate_up` 1830 → 1004 µs, `moe_dec_down` 1062 → 687 µs).
- **The all-reduce branch of the brief did not fire.** The brief said to try LL/LL128 or a one-shot all-reduce only
  if the all-reduce dominated.
  - It takes 18 % of the step, at the link's per-call floor.
  - Even a perfect zero-wait all-reduce saves at most 0.5 ms per step.
  - A custom one-shot over RoCE would need RDMA work in-engine.
  - So none of these was tried; a decision is recorded in section 6.
- The eager rank-1 trace of window 1 was not written (the worker was stopped before nsys finalised). The eager split
  uses the vLLM-up smoke instead (`raw/smoke-prof/split-e.txt`): the median transfer is the same 17 µs, and rank 1 waits
  41 µs per call. That wait is rank 0's eager launch skew, and graphs remove it.

## 2. Gates

Final build `871f912` (`code-tree=fdd801a4a397 dirty=no`) on origin/main `c4ea00b`; base `c4ea00b`
(`code-tree=88f485a37154 dirty=no`). vLLM was up. Raw logs: `raw/gates3`. Earlier builds: `raw/gates`, `raw/gates2`
(`a0a8189` on the pre-merge branch) and `raw/mut`.

GATES3_TABLE

**Mutants** of the capture and replay logic (`scripts/tp2-graphs/mutate.py`, one GPU build each, judged by greedy ids
against tp=2 eager, the floor self-golden and the rank-1 replay check; `raw/mut`, `raw/mut-ids`):

| mutant | result |
|---|---|
| rank 1 skips its token and position upload before replay | **killed**: ids differ from token 0; decode KL21 4.35, 60 % flips |
| one all-reduce dropped from rank 1's captured graph | **killed**: hang, timeout (rc 124) on both runs |
| rank 1 ignores rank 0's graph flag and runs eagerly | **killed only by the replay check** (rank 1 replays = 0). The bits are identical: NCCL accepts a captured collective on one rank and an eager one on the other. The flag enforces lockstep; correctness does not depend on it |
| a recording is reused across sequence states | **survives, equivalent today**: every engine path holds one `SequenceState` for its lifetime (`LocalEngine::new`), so the recording never meets a second state. The re-key is tp=1's rule, unchanged |
| the NCCL f32 landing buffer is not reserved at load | **survives** (ids and floor self-golden identical): the buffer grows to the full chunk on the first 2048-row prefill, before any capture. A later move would be a use-after-free that does not change bits until the freed memory is reused. Not caught by any bit gate; `compute-sanitizer` across hosts was not run |
| (Mac) the GDN out-projection fold allowed under TP fusion | **killed** by `tp_hybrid::hybrid_moe_tp2_row_local_fusion_matches_tp1_f32` (rel L2 1.33 at the first decode row) |

**Mac / feature checks:**
- `cargo test --workspace` on frozen copies of the tree:
  - pre-rebase `48a9715`: 111 suites, 0 failures (`fleetq #3102`);
  - final tree: CARGO_FINAL.
- `tp_hybrid` (7 tests) includes the new fusion test and the TCP remote rank, which carries the new `Forward` frame.
- `cargo check --workspace --tests`, and `cargo check -p intarsia-cli` with `--features cuda`, `--features metal` and
  `--features hip`: pass.

## 3. Row-local fusion under TP (opt-in)

FUSION_SECTION

## 4. What changed

- **Graphs at world > 1 on a capturable communicator.**
  - `GraphState::new(opts.graphs)` replaces `opts.graphs && world == 1`.
  - `GraphState::usable` adds `world == 1 || (comm.capture_safe() && no INTARSIA_TP_TEST_BUG)`.
  - New `Comm::capture_safe` (default false): `NcclComm` says true. The host comm (one-box CUDA TP, cpu tests) and
    HIP's `P2pComm` say false, so they keep eager decode.
  - `ncclAllReduce` and the f32 → bf16 rounding kernel go on stream 0, the stream the model captures in thread-local
    mode. Each rank's graph therefore records its 80 all-reduces as nodes.
- **Fixed landing buffer (`Comm::reserve`).**
  - The f32 partial sum lands in a comm-owned scratch that used to grow on the first larger call (sync + free + alloc).
  - `Model::load_shard` now reserves `prefill_chunk × hidden` f32 at load. After that a larger call is an error, never
    a move, so a captured graph's address stays valid.
- **Lockstep between ranks.**
  - Rank 0 decides per step whether it replays (`Model::decode_graph_now`: one token, a head mode, graphs usable, not a
    profiled step). The remote rank gets that decision in the new flags word of `Forward` (`FWD_GRAPH`); in-process
    peers get it as an argument.
  - Rank 1 replays its own graph of the layers only (rank 0 owns the head), or runs eagerly, as told.
  - If rank 0 replays and rank 1 cannot, the step fails and the group goes down. Under TP a failed capture also fails
    the step, instead of tp=1's "decode eagerly from here on" on one rank only.
  - `--no-graphs` must match on both ranks: `LoadOptions` (with `graphs`) is part of the handshake fingerprint.
- **Invalidation** follows tp=1's rules on each rank's own model.
  - A recording is bound to one `SequenceState::id`.
  - It is dropped on plan re-resolve (`configure_plan` also invalidates in-process peers) and on kernel scratch growth.
  - Batched decode refuses world > 1 already (`enable_batch`), so only the single-sequence step is in scope.
- **Telemetry and logs.**
  - All-reduces recorded into a graph are now counted per replay; they were counted once, at capture.
  - `tp-worker` prints its graph counters when rank 0 disconnects, so every gate shows that rank 1 replayed.
- **Wire format.** `Forward` is now `sid | flags | ids`. Both ranks must run the same build, which the fingerprint
  already requires in practice.
- **Window binary vs shipped tree.**
  - Window 1 ran `2363660`. The shipped tree adds:
    - the rebases onto PR #42 (Metal FfnBlock) and #44 / #49: no CUDA code differs from `2363660` except the import
      fix in #44;
    - rustfmt;
    - the opt-in fusion.
  - The tp2g greedy ids at 1.5k and 16k from `2363660`'s predecessor (`raw/smoke1`, `raw/pre-gates`) and from
    `871f912` (`raw/gates3`) are compared in section 2.

## 5. Files and effect per backend

`git diff --name-only origin/main...HEAD` (shared files, so not self-merged):

| files | CUDA | HIP | Metal | CPU |
|---|---|---|---|---|
| core `comm.rs` (`capture_safe`, `reserve`), `backend.rs` (`fusion_under_tp`) | used | defaults (false / no-op): unchanged | unchanged | `fusion_under_tp` true (opt-in only) |
| core `model.rs`, `model/graph_step.rs`, `tp_remote.rs` | graphs at tp=2 (default on, bitwise to eager); opt-in fusion | tp=2 still eager (`P2pComm` not capture-safe); fusion still off | world 1 only: unchanged | TP tests only |
| models `load.rs` (worker log), tests `tp_hybrid.rs` | | | | new test |
| CLI `common.rs` (`INTARSIA_TP_FUSION`) | opt-in | env refused with an error at world > 1 (no capability) | none | |
| CUDA `nccl.rs`, `device.rs` | capture-safe NCCL, reserved scratch, fusion capability | | | |

- tp=1 is bitwise unchanged on CUDA (section 2).
- The coordinator's Metal no-op on endor and the HIP no-op remain the merge checks.
- **For G2c (HIP).**
  - To turn on graphs under TP, make the HIP communicator's collectives stream-ordered and capturable, then return
    `true` from `Comm::capture_safe` (and implement `Comm::reserve` if it has a growing buffer).
  - To turn on fusion under TP, return `true` from `Backend::fusion_under_tp` once the fused kernels take every dim from
    the call.
  - No core change is needed.

## 6. Downtime

| job | roles | down | serving after |
|---|---|---|---|
| #3101 window 1 (prof + A/B ×3 + localeval) | spark1-vllm, spark2-vllm | 758 s / 783 s (13:26:47-13:39:25 / 13:39:50) | yes, both |
WINDOW2_ROW

All other jobs ran with vLLM up. Total used: DOWNTIME_TOTAL of the 60 min allocation (`raw/downtime.log`).

## 7. Decisions for the user

DECISIONS
