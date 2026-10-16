# Phase 1b: 35B decode step profile, 2026-10-13

Status: PARTIAL. The profile below comes from a native replica of the decode step, not from the engine, and
from spark2, not spark1. spark1 was offline for the whole session (ssh timeout, `fleetq status`: host offline,
checked 02:36 to 02:53). No engine-level number (kernels test, accuracy, `intarsia bench`, eager vs graph
tokens) exists yet. See "What is not done".

**Correction (Phase 1b review finding 9, applied 2026-10-13 without a GPU):** every figure below was measured with the
replica's `ik_moe_experts_bf16` (four launches per MoE block). The engine selects `moe_experts_decode_bf16` for T = 1
(two launches, priority 10), a kernel the replica never launched, and a graph capture of it has not been exercised
anywhere. So "graph replay reaches vLLM's level" is a statement about the replica's kernel set; the replica has to be
rerun with `ik_moe_experts_decode_bf16` (the acceptance run measures the engine itself: `scripts/phase1b-acceptance.sh`).
The 13.15 ms / 76 tok/s graph figure and the 0.57 ms side-stream figure are NOT engine-kernel-set numbers.

Order note: the task asked for the profile before any code change. The device-position and graph code was written
first (it is needed to build the graph variant of the replica); this document was written after it.

## Method

`kernels/cuda_checks/decode_replica.cu` launches the kernel sequence of `Model::run_chunk` (with `ik_moe_experts_bf16`,
not the engine's `ik_moe_experts_decode_bf16`, see the correction above) for one
Qwen3.6-35B-A3B token (T = 1): 40 layers (30 GatedDeltaNet, 10 gated attention), every layer followed by the MoE
block (router, top-8 of 256, routed experts, shared expert, combine), then final norm, NVFP4 `lm_head`, argmax.
Shapes are the real ones (hidden 2048, 16/2 heads of 256, GDN 16 key / 32 value heads of 128, conv width 8192,
experts 512 wide). Weights are random bytes and the routed experts of all 40 layers share one set; per-layer
dense weights are separate, so L2 does not hide the 2.2 GB read per token. It links the real `kernels/cuda/*.cu`.

- Machine: spark2 (GB10 sm_121, CUDA 13.0.88, nsys 2025.3.2), fleetq jobs #1435 to #1438, `--mem 4..8`.
- Context 1024, 200 to 300 timed steps after 5 warmup, token id copied in and token copied out every step.
- nsys: `nsys profile -t cuda` (eager) and `--cuda-graph-trace=node` (graph), 30 steps each, `cuda_gpu_kern_sum`.

## Step time

| Variant | median ms/step | tok/s |
|---|---:|---:|
| eager: one launch per kernel, host position | 15.27 to 15.30 | 65.3 to 65.5 |
| CUDA graph, device-side position | 13.14 to 13.17 | 75.9 to 76.1 |
| graph + second stream for independent branches (shared expert, GDN z/b/a projections) | 12.58 | 79.5 |

The eager replica matches the engine (63 to 65 tok/s in `2026-10-12-phase1a.md`), so the replica is a fair model
of the launch pattern. GPU busy time per step is 13.5 ms eager and 13.3 ms graph (nsys kernel sums). So the
eager step wastes about 1.8 ms (12%) on host launch gaps, which graph replay removes. After that the step is
bound by kernel time, not launches.

## Where the GPU time goes (graph, per step)

| Group | Kernel | Calls | us/step | Avg us | Share |
|---|---|---:|---:|---:|---:|
| FP8 projections | `fp8_gemv_kernel` | 130 | 6123 | 47.1 | 46% |
| Routed experts | `moe_gate_up_kernel` | 40 | 1813 | 45.3 | 14% |
| | `moe_down_kernel` | 40 | 1110 | 27.7 | 8% |
| | topk + group + reduce + combine | 160 | 323 | 1 to 4 | 2% |
| lm_head | `nvfp4_gemv_kernel<float>` | 1 | 1292 | 1292 | 10% |
| Shared experts | `nvfp4_gemv_kernel<bf16>` | 120 | 512 | 4.3 | 4% |
| GDN core | `gdn_delta_rows_kernel` | 30 | 720 | 24.0 | 5% |
| | `gdn_conv` + `gdn_conv_state` | 60 | 143 | 2.4 | 1% |
| Router / gates | `gemv_kernel` bf16 | 140 | 530 | 3.8 | 4% |
| Norms | `rmsnorm` (with and without residual add) | 131 | 252 | 1.9 | 2% |
| Activation | `silu_glu` | 70 | 124 | 1.8 | 1% |
| Attention | splitkv + combine + rope + kv_append | 70 | 322 | 4.6 | 2% |
| Argmax | | 1 | 16 | 16 | 0.1% |
| Total | the 22 largest kernels | 1003 | 13279 | | |

(`gate_split`, `sigmoid_mul`, `qk_norm`, `embedding` are below the top-22 cut and add under 0.1 ms. The engine
counts 844 ops per step; one op can be several kernels, for example `moe_experts` is four.)

## What this says about the 10.6 ms floor

- The FP8 GEMVs read 1.27 GB per token (30 x 33.5 MB GDN, 10 x 27 MB attention). At 221 GB/s that is 5.7 ms; they
  take 6.1 ms, so they run at about 93% of the copy bandwidth. Little to win there.
- The routed experts read 0.53 GB (2.4 ms at 221 GB/s) and take 2.9 ms in gate_up + down: about 82%.
- The `lm_head` reads 0.29 GB (1.3 ms) and takes 1.29 ms: at the floor.
- Together these three are 8.6 ms of the 13.3 ms and sit near their byte floors. The floor is 10.6 ms because it
  adds the other bytes; the step cannot reach it while the 4.7 ms of small kernels stay serial.
- The small kernels (bf16 gemv 0.53, shared-expert gemv 0.51, norms 0.25, silu 0.12, GDN core 0.86, attention
  0.32, MoE routing 0.32) are 3.0 ms, latency bound at 1 to 4 us each, with a dependent chain between them.
- Running independent branches concurrently (second stream inside the capture) took the replica from 13.15 to
  12.58 ms: 0.57 ms, 4%. A fusion that removes launches inside a graph saves the kernel's ramp and tail
  (about 1 to 2 us each), not a host gap, so the ceiling for fusing norms, silu and GDN glue is roughly 0.3 to
  0.5 ms.
- `gdn_delta_rows_kernel` is the largest single non-GEMV kernel (24 us x 30). It reads and writes the f32
  recurrent state (32 x 128 x 128 x 4 B = 2.1 MB read plus written per layer, 63 MB per token, 0.29 ms at 221
  GB/s), so it takes 2.5x the state traffic. That is the best fusion target after graphs (conv + gating +
  recurrence + gated norm in one kernel).

Projection: graph replay alone is worth about 2.1 ms (65 to 76 tok/s in the replica); streams add about 0.6 ms.
vLLM measures 76 to 78 tok/s on the same checkpoint, so graph replay by itself could reach vLLM's level in the
engine, if the engine's decode MoE kernel (not the replica's) costs the same inside a captured graph: unmeasured. The remaining gap to 94 tok/s (10.6 ms) is mostly kernel efficiency, not launch overhead.

## What changed in the code (checks per item in the table below; the CUDA Rust path has never linked or run)

- Device-side position: `OpCall::pos` (an `i32 [1]` tensor) for `Rope`, `KvAppend`, `AttnDecode`. The launchers
  `ik_rope_bf16`, `ik_kv_append_bf16`, `ik_attn_decode_bf16`, `ik_attn_decode_bf16_splitkv` take an optional
  `const ik_tensor* pos` before `stream` (ABI header updated). The split-KV decode launches its grid at the
  length-independent maximum and the blocks past the active split count exit; the partition is unchanged.
- `Backend::{supports_graphs, graph_capture, graph_replay}`; CUDA implementation over
  `ik_rt_graph_begin/end/launch/destroy` (thread-local capture mode). cpu backend resolves the device position
  at launch.
- `Model` captures the whole T = 1 step once per `SequenceState` (token and position are device scalars) and
  replays it; profiled steps, a failed capture, `--no-graphs` and backends without graphs run the eager path.
  A new state (new id) gets its own recording; `reserve_scratch` and re-resolution drop it.
  `ResolvedPlan::decode_position_stable` is true only when the plan selects the same kernels at the first and
  last context position. Telemetry: `graph_replays`, `graph_captures`.
- KV layout is unchanged (contiguous `[Nkv, cap, D]`); no paged KV.

## Checks that ran

| Check | Where | Result |
|---|---|---|
| `cargo test --workspace --no-fail-fast` | Mac | all pass except `intarsia-backend-cuda --lib kernels::tests::every_launcher_is_declared_in_the_abi_header`: `ik_moe_experts_decode_bf16` is missing from `ik_abi.h`. That is the MoE agent's launcher (`kernels/cuda/moe_decode.cu`), not part of this change |
| `crates/intarsia-models/tests/graphs.rs` (6 tests) | Mac, cpu backend wrapped in a recording "graph" backend that forbids upload / download / sync / alloc while capturing | graph decode is bit-identical to eager for 40 greedy steps (tokens and full logits), argmax head identical for 30 steps, profiled steps interleave, second state recaptures, reset reuses, failed capture and `--no-graphs` fall back. Mutation check: baking position 0 into the replay makes 3 of the 6 fail |
| `RUSTFLAGS='--cfg ik_cuda' cargo check -p intarsia-backend-cuda --tests` | Mac | compiles (checks `device.rs`, `ffi.rs`, `tests/graph_pos.rs`; no link, no GPU) |
| `kernels/cuda_checks/graph_pos_check.cu` (nvcc, real GPU) | spark2 #1435 | 5717 checks, 0 failed: device vs host position bitwise equal for rope, kv_append, attn_decode and split-KV at 16 positions (0 to 4090) and 3 head configs; 5440 split-KV launches (T = 1, 2, 5, 16, positions 0 to 4090) bitwise equal to the pre-change kernel; split-KV vs SIMT within 2e-2; graph replay at 16 positions and backwards equals eager; a failed capture leaves the stream usable |

## What is not done

1. spark1 acceptance: `intarsia kernels test` admitted, accuracy vs the harness baselines (35B and 8B),
   `intarsia bench speed` before/after table (sizes 128,1024,4096, gen 128, reps 3), and the 256-step greedy
   eager vs graph comparison on the real model. The Rust workspace has not been built with CUDA at all in this
   session: there is no cargo on spark2 and installing one means downloading a toolchain, which was not
   authorised. The `cuda`-feature build, `tests/graph_pos.rs` and the engine on a real GPU are untested.
2. Fused decode ops (GDN chain, rmsnorm + residual into neighbours) are not written. The profile above says
   they are worth 0.3 to 0.5 ms (about 3%); the GDN fusion is the largest piece. Not started.
3. Multi-stream branches inside the captured graph (0.57 ms in the replica) are not in the engine.
4. The replica uses random weights and shared routed experts, so expert-cache effects and routing skew are not
   modelled. Treat the absolute numbers as +/- 5%.
5. The replica profile is at context 1024 only.

## Open risks (read from code, not run)

- Ordering and parity of the graph path were checked by reading `Model::forward_inner`: the `st.len + ids.len() <= st.cap`
  check runs before the graph branch (a device-position append cannot bounds-check), and the branch does the same
  `st.commit` and `telemetry.step` as the eager loop. Split-KV scratch is already a kv_len-independent bound in
  `kernels.rs::splitkv_scratch` (`splitkv_parts(usize::MAX/2, ..)`), equal to the maximum the new decode launcher uses.
- `decode_position_stable` compares plan probes at position 0 and cap-1. If a decode kernel's `when` looked at
  `kv_len`, graphs would switch off without a message. Not seen in the registrations, never run on the real plan.
- The first capture happens inside the first graph decode step. `StreamId 0` is a stream from `cudaStreamCreate`
  (not the legacy default stream), so capture should be legal; unverified on a GPU through Rust.
- The MoE block under capture: `moe.rs` has no per-step upload/download/sync by reading, but it never ran under
  a capture guard (the cpu fixture has no MoE). A violation shows as one stderr line and an eager fallback.
- ABI: `ik_rope_bf16`, `ik_kv_append_bf16`, `ik_attn_decode_bf16(_splitkv)` gained a `pos` argument. The HIP
  worktree (`phase2a/hip`) must take the same signature; not touched here.

## Reproduce

```sh
# on a GB10 host, from a tree with kernels/ (inside a fleetq slot)
nvcc -c -O3 --std=c++17 --expt-relaxed-constexpr -arch=sm_121 -Xcompiler -fPIC -I kernels -o obj/X.o kernels/cuda/X.cu   # each .cu
nvcc -O2 --std=c++17 --expt-relaxed-constexpr -arch=sm_121 -I kernels kernels/cuda_checks/decode_replica.cu obj/*.o -lcublas -o decode_replica
./decode_replica --ctx 1024 --steps 300 --mode both
nsys profile -t cuda --cuda-graph-trace=node -o rep ./decode_replica --mode graph --steps 30
nsys stats -r cuda_gpu_kern_sum rep.nsys-rep
```
