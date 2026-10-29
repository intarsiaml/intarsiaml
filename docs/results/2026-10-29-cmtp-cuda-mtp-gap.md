# CMTP: how long does the GPU sit idle per `--spec mtp` pass on CUDA? (spark1, GB10, Qwen3.6-35B-A3B NVFP4, 2026-10-29)

Lane `cmtp` (second window), branch `lane-cmtp/cuda-gpu-accept`, base `origin/main` `0d02d3f36`. spark1 (spark-4e77), `spark1-vllm` (34 GB) left up and idle.
Build: `cargo build --release --features cuda -p intarsia-cli`, tree `~/intarsiaml-cmtp/src`. Raw: `2026-10-29-cmtp-cuda-mtp-gap/raw/`. Scripts: `scripts/cmtp/nsys-w1.sh` (capture),
`scripts/cmtp/nsys_gap.py` / `nsys_inner.py` (analysis), `scripts/cmtp/gen.py` (client), `scripts/cmtp/ab.sh`, `tools/cmtp/ab_table.py` (served A/B).
The `.sqlite` exports (3 to 9 MB each) are not committed; the `.nsys-rep` files were on spark1 and are deleted with the scratch tree.

## 0. Result

**Step 1 stop rule: the GPU is idle under 3 % of the pass on CUDA. Nothing was built for Step 2, no PR.**
The Metal finding (2.2 ms idle in a 16.4 ms pass, host-side acceptance and rollback) does not carry over: on CUDA the whole step (catch-up, draft, chain, verify) is one
captured graph (`Model::spec_step`, `crates/intarsia-core/src/model/spec_graph.rs`), the host does one `cudaGraphLaunch` and one 76-byte download per pass.

| 35B tp=1, 4k prompt, 1 stream, MTP k=2 (default) | pass period | graph (GPU busy) | gap between graphs | of which GPU idle (nothing running) | of which rollback copies |
| --- | --- | --- | --- | --- | --- |
| nsys, pageable download (main as is) | 22.13 ms | 21.17 ms | 0.957 ms (4.33 %) | 0.093 ms (**0.42 %**) | 0.21 ms (0.97 %) |
| nsys, same binary + pinned staging for the 76 B download | 21.04 ms | 20.75 ms | 0.292 ms (1.39 %) | 0.087 ms (**0.42 %**) | 0.20 ms (0.97 %) |
| unprofiled estimate (A/B `base`, 114.9 tok/s, 2.43 tokens/pass) | about 21.1 ms | | | | |
| `--max-batch 16`, 16 streams (plain batched decode, see section 3) | 41.35 ms (median) | 41.18 ms | 0.102 ms median (0.25 %) | 0.099 ms (**0.24 %**) | none |

Read the first row with care, it is the one a reviewer would question. Counting everything between two graph replays as lost time gives **4.33 %**, over the 3 % line. But 2.9
points of it are 1 to 4 ms stalls of the 76-byte token download (`memcpy kind 2`, section 2) that exist only while nsys/CUPTI traces the pageable copy path: with the download staged
through a pinned buffer the same trace shows 1.39 %, and unprofiled serving never ran slower than the pinned trace (section 2). What is left (1.39 %) is 0.42 % real idle
plus 0.97 % GPU copy work (the rollback restore). Moving acceptance and rollback into the graph can win at most the idle plus whatever part of the copies it can avoid:
**at most 1.4 % single stream, below the +3 % PR bar and below the 3 % stop line.**

## 1. Where the time goes per pass (single stream, 4k, 495 passes, 1200 tokens, 10.9 s window)

`raw/single-pageable-gap.txt`, `raw/single-pinned-gap.txt` (`nsys profile`-equivalent `nsys launch` + `start/stop` around one request, `-t cuda,nvtx,osrt --cuda-graph-trace=graph`).
- 1 `cudaGraphLaunch` per pass (545 replays, 4 captures over the run; graph keys by catch-up rows n = 1, 2, 3), 0 eager model kernels in the decode window. Host cost per launch 72 to 96 us.
- Acceptance 72.3 % of drafts (782 / 1081), 2.45 tokens per pass, 541 steps in the run.
- **Gap without rollback (306 of 494 passes, 62 %): median 96 us** (0.44 % of the period). It holds the 76 B D2H (about 1 us on the GPU, the host waits for the graph inside that call), two
  H2D uploads (4 B next token, 384 B position/id table), the acceptance compare and `st.commit_verify` (a few us of host work), and the `cudaGraphLaunch` call (72 to 96 us, the largest item: the launch
  of a 1200-kernel graph is not free). GPU idle is the part of that 96 us in which no copy runs: 87 to 93 us.
- **Gap with rollback (187 of 494 passes, 38 %): median 632 us.** 30 eager `memcpy` D2D of 2.146 MB (one per GDN layer: the per-row recurrent copy of the accepted row restored into the live
  state), 19 us each on the GPU (about 230 GB/s read plus write, the copy is bandwidth bound), issued by the host at 3 to 5 us each so the GPU queue never runs dry. The GPU is busy, not idle: the cost
  is the 0.57 ms of copy traffic, 0.97 % of the mean period. Moving the restore into the graph would not shorten it; only not copying (selecting the row's buffer by pointer in the next graph) would.
- Host functions in the gap: the codebase has no NVTX ranges and no CPU sampling was collected (`nsys --sample` produced no table on this box), so the evidence is the CUDA API trace
  (`cudaMemcpyAsync` x 13.7/pass, `cudaStreamSynchronize` x 1, `cudaGraphLaunch` x 1, essentially no `cudaLaunchKernel`).

Inside the graph (`--cuda-graph-trace=node`, 248 passes, `raw/single-node-inner.txt`): span 20.68 ms, kernel-busy 20.12 ms, **0.55 ms (2.7 %) of inter-kernel gaps**, of which 0.34 ms is 41 gaps of about 8 us
per pass in front of `gdn_conv_kernel`. That is the largest GPU-idle item on the step and it is not host work (a lead for a kernel lane, not for this one).

## 2. The 1 to 4 ms download stall is a tracing artifact

With the pageable `cudaMemcpyAsync` D2H of the original binary, 152 of 494 gaps (31 %) exceed 1 ms, 376 ms in all (3.4 % of the window): the 76 B copy starts 2 to 8 us after the graph ends and is
recorded as running 1 to 4.6 ms (`raw/single-pageable-tail.txt`), with no OS call on the engine thread (`raw/single-pageable-osrt.txt`), at random pass positions (gaps between stalls 1 to 12 passes). The same binary with
the download staged through a pinned buffer (`cudaHostAlloc`, same bytes; experiment only, reverted, it is not in the branch) has none of them: gap 0.292 ms mean, period 21.04 ms, **115.4 tok/s traced against 109.7 for the pageable trace**.

Unprofiled the difference is not there: served A/B, same binary, fresh server per arm, order rotated, 2 rounds x 2 reps, 4k prompt, 384 tokens (`raw/ab-quick/`):

| arm | decode tok/s (median of rounds) | rounds |
| --- | --- | --- |
| `base` pageable (pinned staging off, in the reverted experiment build) | 114.9 | 115.8, 114.1 |
| `dev` pinned | 112.4 | 111.6, 113.3 |

0.978x with a run-to-run spread of 107 to 119 tok/s inside either arm: no gain, so the stall costs nothing when nothing is tracing, and the 114.9 tok/s of the unprofiled `base` (2.43 tokens/pass => 21.1 ms)
matches the pinned trace (21.04 ms), not the pageable one (22.13 ms). I therefore did not ship the pinned buffer (it would touch every small download on the backend for no served gain).

## 3. `--max-batch 16`: there is no batched MTP on CUDA

`serve --max-batch N` is refused with `--spec mtp` (`crates/intarsia-cli/src/cmd/serve.rs:23`, `docs/tuning.md` batch row), so the 16-stream capture is **plain batched decode**
(`raw/metrics-b16.txt`: 0 spec steps), not verify passes. It is still the right cell to ask whether the host leaves the GPU idle between steps: median step 41.35 ms, graph 41.18 ms, gap 0.102 ms, idle 0.099 ms (**0.24 %**),
no rollback, no stalls (p90 gap 0.110 ms). The mean gap of 3.28 ms (7.5 %) in `raw/b16-gap.txt` is prefill admission at the start of the window (16 prompts, eager `moe_w4a16` / `nvjet` kernels between the first decode
graphs), not steady state. 16 streams x 23.2 tok/s = 370 tok/s under nsys. The "batched MTP verify cells" gate does not exist for CUDA.

## 4. Prior work not redone

`2026-10-26-cmtp-cuda-mtp-default.md` (MTP default, gates, k sweep), `2026-10-27-mpf-macro-prefill-mtp.md`, `2026-10-22-mtp-spec-decode.md`, `2026-10-29-score29-spark35b/`. The captured spec-step graph (HIP first, CUDA through
`verify_attn_per_row`) is why the CUDA host gap is already small.

## 5. What would still be worth doing, and what is not

- Rollback restore without copies (point the next graph at the accepted row's buffer, or restore only layers whose state differs): at most 0.97 % single stream. Below the bar, large change to a bitwise-critical path; not started.
- Moving acceptance / next-draft setup into the graph: the host work it would remove is a few us per pass plus part of the 87 to 93 us idle; the graph launch itself stays. Not started.
- The `gdn_conv_kernel` gaps inside the graph (0.34 ms, 1.6 %): a kernel-lane lead.

## 6. Gates and runs

No engine code changed (scripts, tools and docs only), so the bitwise gates (`equal.py`, `batch_cuda_real`, `plan show`, `cargo test --workspace`) were not rerun; main's are unchanged.
fleetq on spark1: builds #7150, #7201 (`--load cpu`); quiet GPU jobs #7178 (single), #7188 (b16), #7203 (A/B quick), #7208 (single pinned), #7210 (node trace); exports #7184, #7199, #7209, #7211 (`--load cpu`).
Jobs #7154 and #7165 failed on script errors before touching the GPU. Five quiet jobs of 3 to 5 minutes each, about 20 minutes of quiet GPU in all; the 3-window budget was counted in time, not jobs.
