# CUDA graphs for tp=2 decode (lane G1c, 2026-10-21)

Qwen3.6-35B-A3B NVFP4 at tp=2 across the two Sparks (rank 0 on spark1, rank 1 on spark2, NCCL 2.30.7 over ConnectX-7),
following G1b ([2026-10-21-35b-tp2-cuda.md](2026-10-21-35b-tp2-cuda.md)). Branch `feat/tp2-cuda-graphs`, based on
origin/main `c4ea00b` (PR #44 merged). Raw logs: `2026-10-21-tp2-cuda-graphs/raw/`; scripts `scripts/tp2-graphs/`.

## Headline

**tp=2 decode now beats tp=1** once the decode step is a CUDA graph on both ranks, NCCL all-reduces included.

| decode tok/s, final build `871f912`, window 2, A/B ×3 | 1k | 4k | 16k |
|---|---|---|---|
| tp=1 + graphs (shipped) | 81.7 | 79.3 | 72.9 |
| tp=2 eager (main today) | 72.4 | 71.6 | 68.7 |
| **tp=2 + graphs (default in this PR)** | **91.6** (+12 %) | **90.0** (+13 %) | **85.0** (+17 %) |
| **tp=2 + graphs + row-local fusion (opt-in `INTARSIA_TP_FUSION=1`)** | **96.5** (+18 %) | **94.5** (+19 %) | **88.2** (+21 %) |
| vLLM tp=1 (localeval, G1b, same day) | 77.9 | 76.6 | 73.6 |

- Percentages are against tp=1 + graphs.
- Against vLLM (localeval on both sides): tp=2 graphs + fusion decodes +22 % / +23 % / +18 % faster.
- **Graphs and fusion change no bits.** Greedy ids are identical, and the self-golden against the tp=2 eager dump
  matches eager's own run-to-run digits exactly.
- **The all-reduce is not the bottleneck.** Measured on both ranks:
  - 16.7 µs median transfer per call, which is the link's floor, plus about 6 µs mean wait;
  - 80 calls per step, about 2 ms of an 11 ms step.
  - LL/LL128 and a custom one-shot were therefore not tried (section 1).
- **Graphs win by removing per-step launch and host time:** 2.4 ms of idle per step eager, 0.6 ms with graphs.
- **Fusion wins on launch count:** 130 → 50 fp8 GEMVs, the 140 bf16 GEMVs gone, 131 → 32 rmsnorms.

## 1. Numbers

### Speed: interleaved A/B ×3, `intarsia bench speed --at-most`

**Window 2** (fleetq #3155, both vLLMs down, final build `871f912`; `raw/window2/ab-table.txt`).
- Arms tp1g / tp2g / tp2gf ran in all three rounds, with the order reversed in round 2; tp2e ran in rounds 1 and 3.
- Cells are medians of the round medians, in tok/s.

| arm | prefill 1k\* | 4k | 16k | decode 1k | 4k | 16k |
|---|---|---|---|---|---|---|
| tp1g: tp=1, graphs (shipped) | 4510 | 5858 | 5505 | 81.7 | 79.3 | 72.9 |
| tp2e: tp=2 eager | 4761 | 7120 | 6855 | 72.4 | 71.6 | 68.7 |
| **tp2g: tp=2, graphs (default)** | 5272 | 7132 | 6857 | **91.6** | **90.0** | **85.0** |
| **tp2gf: tp=2, graphs + fusion (opt-in)** | 5415 | 7130 | 6907 | **96.5** | **94.5** | **88.2** |
| tp2g vs tp1g | +17 % | +22 % | +25 % | +12.1 % | +13.4 % | +16.6 % |
| tp2gf vs tp1g | +20 % | +22 % | +25 % | **+18.1 %** | **+19.2 %** | **+21.0 %** |
| tp2gf vs tp2g | | | | +5.3 % | +5.0 % | +3.8 % |

- Decode per round, 1k / 16k, in tok/s:
  - tp1g: 81.7, 82.1, 81.6 / 72.9, 73.3, 72.2;
  - tp2g: 90.5, 92.1, 91.6 / 84.1, 86.4, 85.0;
  - tp2gf: 96.5, 97.6, 95.6 / 88.4, 88.2, 87.6.
- The tp2gf rounds do not overlap with tp2g or tp1g at any size.
- Every tp=2 graphs arm passed `--require-graphs` plus the rank-1 replay check.

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

- Window 1 (pre-rebase `2363660`) and window 2 agree:
  - tp1g within 0.3 %;
  - tp2g 2-3 % lower in window 2. That is the run-to-run spread of the cross-host arm (tp2e also moved, 75.7 → 72.4 at
    1k). No CUDA code changed between the two binaries apart from the opt-in fusion, which is off in tp2g.

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
| intarsia tp=2 graphs + fusion (localeval, 3 reps, window 2) | 5312 | 6896 | 6659 | 95.2 | 94.3 | 86.6 |
| vLLM tp=1 on spark1 (localeval, CUDA round 7, same day) | 5171 | 6353 | 6275 | 77.9 | 76.6 | 73.6 |
| tp=2 graphs vs vLLM | +2 % | +8 % | +5 % | **+21 %** | **+21 %** | **+17 %** |
| tp=2 graphs + fusion vs vLLM | +3 % | +9 % | +6 % | **+22 %** | **+23 %** | **+18 %** |

- Runs:
  - `/opt/code/deepeval/results/20261021-133100-adhoc-intarsia-35b-tp2-graphs-xhost/result.json`
    (`raw/window1/localeval-tp2g.log`);
  - `.../20261021-152359-adhoc-intarsia-35b-tp2-graphs-xhost/result.json` (`raw/window2/localeval-tp2gf.log`).
  - Localeval's single-stream decode (95.2) is 1-2 % below `bench speed`'s and matches it within run-to-run spread.
- Both runs' unanimis capture failed (ssh session refused), as in G1b; the result files are saved.
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

| gate | result |
|---|---|
| **tp=1 bitwise unchanged**: `plan show --json`, 35B / 8B, base vs dev | **identical** (13961 / 8630 bytes) |
| tp=1 256 greedy ids, 1.5k and 16k, base vs dev | **identical** |
| tp=1 self-golden, dev against the base dump (35b-short) | mean KL21 2.723472e-13 / 3.728103e-11 (prefill / decode), 0 flips: the bitwise floor, the same digits as G1b's base against its own dump. PASS |
| **tp=2 graphs vs tp=2 eager**: 256 greedy ids, 1.5k and 16k | **identical** (both builds `a0a8189` and `871f912`, and the pre-rebase build) |
| tp=2 graphs against the tp=2 eager dump, 35b-short | prefill 1.377398e-12, decode 6.618585e-13, max 1.4e-9, **0 flips**. PASS at floor budgets |
| tp=2 eager against its own dump (the floor) | **the same digits** in every SELF-STATS field: graphs give the bits eager gives |
| tp=2 graphs against the tp=2 eager dump, 35b-long | prefill 2.972350e-13, decode 1.094389e-11, 0 flips. PASS at floor budgets |
| vLLM floor (PR #40 judge), tp=2 eager run that wrote the dumps | short PASS, long PASS. Graphs are bitwise eager, so G1b's KL against vLLM's run-to-run (1.06x / 0.99x, excl. rows) holds unchanged |
| prefix reuse, tp=2 graphs | **PASS**: continue / restore / aligned 32/32 on every prompt, aligned restore **logits bit-identical to the cold run**; rank 1 replayed (1184 replays) |
| `kernels test --backend cuda` | 52 kernels, 746 cases, **0 FAIL** |
| rank-1 replay check (every graphs run) | ok in every gate and A/B run |

Floor budgets: `--self-max-kl 1e-9 --self-max-pos-kl 1e-6 --self-max-flip-pct 0 --self-max-tail 5e-8`.
- The first run set the tail limit at 1e-9, below the identical-logits floor (tail_abs 3e-9 to 1e-8, as in tp=1's
  bitwise self-check). It failed on that limit alone (`raw/gates`).
- `raw/gates2` and `raw/gates3` use the corrected budget.

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
  - final tree `871f912`: 114 suites, 0 failures (`fleetq #3147`).
- `tp_hybrid` (7 tests) includes the new fusion test and the TCP remote rank, which carries the new `Forward` frame.
- `cargo check --workspace --tests`, and `cargo check -p intarsia-cli` with `--features cuda`, `--features metal` and
  `--features hip`: pass.

## 3. Row-local fusion under TP (opt-in)

- **What it is.** Under TP, `Model::fused_ok` used to refuse every fused decode op. Fusion under TP is now a backend
  capability:
  - `Backend::fusion_under_tp`: CUDA and the CPU reference say yes; HIP and Metal keep the default, no.
  - It is opt-in through `LoadOptions::tp_fusion` (`INTARSIA_TP_FUSION=1`).
  - `INTARSIA_TP_FUSION=1` on a backend without the capability is a load error at world > 1, not a silent no-op.
- **Which ops (`Model::fused_under_tp`).** Every fused op whose replaced chain stays row-local on the shard: projection
  pairs, rmsnorm+add+pair, the GDN conv+delta, gated rmsnorm, q/k norm+rope+kv, the NVFP4 GLU and the MoE block.
  - **`GatedRmsnormMatmulFp8` stays off.** It folds the GatedDeltaNet out-projection, a row-parallel matmul whose output
    is a partial that `proj_reduce` must land in the f32 buffer and all-reduce. Fused, it writes the block output
    directly and swallows the reduce.
  - `SigmoidMulMatmulFp8` (attention o_proj) was already tried only with bf16 partials, where the reduce follows it.
- **Gates (`raw/gates3`, both ranks under `INTARSIA_TP_FUSION=1`, graphs on):**
  - 256 greedy ids at 1.5k and 16k: **identical** to tp=2 graphs unfused and to tp=2 eager.
  - Self-golden against the tp=2 eager dump, short and long: **the same digits as eager against itself** (short:
    prefill 1.377398e-12, decode 6.618585e-13, 0 flips). On these gates fusion is bitwise to unfused.
  - vLLM floor (PR #40 judge): short PASS, long PASS.
  - Prefix reuse: PASS, aligned restore bit-identical; rank 1 replayed 1184 times.
  - Mac: `tp_hybrid::hybrid_moe_tp2_row_local_fusion_matches_tp1_f32`. Rank 0 ran 8 fused ops (listed in the test
    output) and stayed within rel L2 3.4e-6 of tp=1. Allowing the out-projection fold breaks it (rel L2 1.33,
    mutant killed).
- **Speed (window 2):**
  - decode +5.3 % / +5.0 % / +3.8 % over tp=2 graphs unfused, and +18 % / +19 % / +21 % over tp=1;
  - rank-0 step 11.08 → 10.47 ms (nsys, `raw/window2/kernel-table-fusion.txt`).
  - Launch counts: `fp8_gemv` 130 → 50 (+40 pairs), the 140 bf16 `gemv` gone (into `nvfp4_glu` and pairs),
    `rmsnorm` 131 → 32.
  - The NCCL share is unchanged (2.0 ms, transfer 1.39 ms).
- **What is left.**
  - Compute per step is 7.8 ms against tp=1's 12.15 / 2 ≈ 6.1. The rest is ops that do not halve (GDN, attention,
    routing) and per-launch overhead.
  - The all-reduce is 2 ms at the link's floor. Fewer of them (one per block instead of two) would need a
    sequence-parallel layout, which is not in scope.

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
- **Binaries.**
  - Window 2, the final gates and the mutants' base all ran `871f912`. Commits after it change only docs and scripts.
  - Window 1 ran the pre-rebase `2363660`: the same graph code without the opt-in fusion. It is kept as the repeat
    measurement.

## 5. Files and effect per backend

`git diff --name-only origin/main...HEAD` (shared files, so not self-merged):

| files | CUDA | HIP | Metal | CPU |
|---|---|---|---|---|
| core `comm.rs` (`capture_safe`, `reserve`), `backend.rs` (`fusion_under_tp`) | used | defaults (false / no-op): unchanged | unchanged | `fusion_under_tp` true (opt-in only) |
| core `model.rs`, `model/graph_step.rs`, `tp_remote.rs` | graphs at tp=2 (default on, bitwise to eager); opt-in fusion | tp=2 still eager (`P2pComm` not capture-safe); fusion still off | world 1 only: unchanged | TP tests only |
| models `load.rs` (worker log), tests `tp_hybrid.rs` | | | | new test |
| CLI `common.rs` (`INTARSIA_TP_FUSION`) | opt-in | load error if set at world > 1 (no capability); unset: unchanged | no TP: unchanged | |
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
| #3155 window 2 (fusion A/B ×3 + prof + localeval) | spark1-vllm, spark2-vllm | 876 s / 876 s (15:19:03-15:33:39) | yes, both |

Total used: **27.7 min of both-down time** (13.1 + 14.6), each window under 30 min, 99 min of serving between them, of the 60 min allocation (`raw/downtime.log`). Every other job (builds, gates, mutants) ran with vLLM up.

## 7. Decisions for the user

1. **Make row-local fusion the tp=2 default on CUDA?**
   - For: +4-5 % decode; bitwise to unfused on every gate here; the out-projection fold is excluded and
     mutant-tested.
   - Against: one model, one day.
   - Recommendation: yes, after a short soak. It is a one-line default flip (`LoadOptions::tp_fusion`) in a follow-up
     PR. This PR keeps it opt-in.
2. **Serve the 35B at tp=2?**
   - Single-stream decode is +12-21 % over tp=1 and +17-23 % over vLLM.
   - But it holds both Sparks for one engine, and batched decode refuses world > 1. Aggregate throughput is therefore
     below two tp=1 servers.
   - Worth it only for a latency-first role. That is a fleet decision (which role gives up spark2).
3. **The `no-reserve` mutant survives the bit gates** (section 2). The reservation is there and is load-bearing by
   construction, but nothing tests it.
   - Option: a cross-host `compute-sanitizer` run, about 10 min with vLLM up.
   - Option: accept it as is.
4. **No all-reduce work.** The data says the all-reduce runs at the link's floor (section 1). LL/LL128 and a custom
   one-shot are not worth a lane unless a sequence-parallel layout is on the table.
5. **Merge.** This PR touches shared core files (`model.rs`, `comm.rs`, `backend.rs`), so it needs the Metal and HIP
   no-op checks before merge. HIP keeps eager decode and unfused ops at tp=2 until its communicator says
   `capture_safe` (G2c).
