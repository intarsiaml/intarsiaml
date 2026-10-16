# HIP decode speed: Qwen3.6-35B-A3B NVFP4 on one R9700 (branch perf/hip-decode)

Tree: `df7cd32` (git tree `aba3ae4d018892ad2ee1cc810f22ac450b3191fe`) built and measured on tatooine (R9700 #0, gfx1201, ROCm 7.2.4 dev
image, `HIP_VISIBLE_DEVICES=0`), base `63b0f72` (arch/plan). Later commits on the branch only add ABI header declarations and this
page. The measured binary had no `INTARSIA_BUILD_STAMP` (the worktree is rsynced without `.git`), so reports say `unstamped`; the
tree hash above is the stamp. No radiance code was read or imported; `kernels/hip/NOTICE-radiance` is unchanged.

Scripts: `scripts/hip-decode/{burst,remote,build}.sh` (GPU bursts, the downtime rule is in `burst.sh`), `analyze.py` (per-step table
from a rocprofv3 kernel trace), `mutate.py`. Profiling image: `deploy/tatooine/Dockerfile.hip-prof` (the dev image plus `libdw1`,
`libelf1`; rocprofv3 itself already ships in the dev image, it only failed to load for lack of `libdw.so.1`). Base is digest-pinned
like `Dockerfile.hip-dev`.

## 1. Result

`bench speed --sizes 128,1024,4096 --gen 128 --reps 5` through `intarsia serve` (decode tok/s; prefill tok/s in brackets).

| | 128 | 1024 | 4096 |
|---|---|---|---|
| 35B before, server default (profile every 16th step) | 97.3 (1134) | 96.9 (2073) | 95.0 (2040) |
| 35B before, `--profile-every 0` (eager) | 106.2 (1210) | 105.8 (2059) | 103.3 (2070) |
| 35B + graphs only (v1 kernels) | 110.2 (1201) | 109.7 (2079) | 107.5 (2035) |
| **35B after** (graphs + v2 kernels + fusions), `--profile-every 0` | **125.8 (1226)** | **125.1 (2084)** | **122.2 (2086)** |
| 35B after, `intarsia qualify --quick-only` (server default profiling) | 125.8 (1198) | 125.1 (2115) | 122.0 (2059) |
| 8B before (server default) | 36.9 (2087) | 36.3 (3396) | 35.2 (3033) |
| 8B after (`--profile-every 0`) | 38.0 (2042) | 37.4 (3436) | 36.2 (3088) |
| 8B after, `qualify --quick-only` | 38.0 (2031) | 37.4 (3460) | 36.1 (3041) |

35B: +29 % decode at 128 against the default configuration (+18 % against the same unprofiled eager run). Prefill is unchanged
within noise (not touched). The 8B is bandwidth bound (92 % of the byte floor at 128, 87 % at 4096) and gains 3 %. The qualify
35B run reports 125.8 tok/s = 45 % of the 2.25 GB/token byte floor.

vLLM-radiance (141 tok/s, a different model) and llama.cpp Vulkan (112 tok/s, Q4_K_XL) stay the reference points: we are above
llama.cpp and 11 % below radiance.

## 2. Profile (rocprofv3 kernel trace, one 35B decode step, 157 steady-state steps of a 160-token greedy run)

The "dispatch bound" hypothesis of the Phase 2a page (sections 9 to 11) is **mostly refuted**:

* rocprofv3 shows a fixed ~3.0 us gap (p10 2.9 us, p99 3.4 us) between consecutive kernels of a step, identical eager and in a
  graph replay. That is 2.8 to 3.1 ms of 9.4 to 11 ms *under the profiler*, and the profiler itself costs: the 160-token run
  decodes at 104 to 106 tok/s under rocprofv3 against 123 to 124 tok/s without it.
* Un-profiled, the same counts say: before, 9.42 ms per token (eager, 106.2 tok/s) of which 7.97 ms is kernel time, so 1.45 ms
  (15 %) is idle; after, 8.0 ms per token of which 6.62 ms is kernel time, so 1.4 ms (17 %) is idle. Replaying the step as a
  HIP graph removed only ~4 % of the period (106.2 to 110.2 tok/s with unchanged kernels). The per-kernel boundary cost on this
  stack (~1.5 us un-profiled) is paid by the GPU's command processor, not by host launches; it scales with the launch count.
* Before: 946 launches per step, kernel time 7.969 ms. After: 897 launches, kernel time 6.621 ms (-17 %).
* CLR knobs tried on the graph and eager decode (`HIP_FORCE_DEV_KERNARG` 0/1, `ROC_USE_FGS_KERNARG=0`,
  `DEBUG_CLR_MAX_BATCH_SIZE=1`, `GPU_STREAMOPS_CP_WAIT=1`): all within +-1 % of the default (graph 123.1 to 124.4 tok/s, eager
  119.0 to 120.5; `DEBUG_CLR_MAX_BATCH_SIZE=1` eager is 18 % slower, 98.3). No knob worth shipping.

Per-kernel table, before (arch/plan `63b0f72`, eager, 946 launches, kernel 7.969 ms/step):

| kernel | n/step | us/step | avg us | % kernel |
|---|---|---|---|---|
| fp8_gemv<1,1> (qkv/z/o_proj, N < 6144) | 90 | 1845.8 | 20.5 | 23.2 |
| fp8_gemv<1,2> (qkv, q: N >= 6144) | 40 | 1432.9 | 35.8 | 18.0 |
| moe_gate_up_p1 | 40 | 877.3 | 21.9 | 11.0 |
| nvfp4_gemv<bf16> (shared expert) | 120 | 780.9 | 6.5 | 9.8 |
| moe_down_reduce_p1 | 40 | 780.3 | 19.5 | 9.8 |
| nvfp4_gemv<f32> (lm_head) | 1 | 472.9 | 472.9 | 5.9 |
| gemv bf16 (router, gates, GDN a/b) | 140 | 319.4 | 2.3 | 4.0 |
| gdn_delta_rows | 30 | 267.2 | 8.9 | 3.4 |
| moe_topk | 40 | 239.1 | 6.0 | 3.0 |
| rmsnorm (residual) / rmsnorm | 80 / 51 | 192.7 / 105.4 | 2.4 / 2.1 | 2.4 / 1.3 |
| silu_glu | 70 | 120.5 | 1.7 | 1.5 |
| splitkv attention | 10 | 113.6 | 11.4 | 1.4 |
| rope / rope_pass | 20 / 20 | 82.5 / 26.8 | 4.1 / 1.3 | 1.0 / 0.3 |
| moe_combine | 40 | 71.1 | 1.8 | 0.9 |
| gdn_conv / gdn_conv_state | 30 / 30 | 67.9 / 43.8 | 2.3 / 1.5 | 0.9 / 0.5 |
| rest (argmax, kv_append, combine, gate_split, sigmoid_mul, embedding, copies) | 46 | 126.7 | | 1.6 |

After (this branch, graph replay, 897 launches, kernel 6.621 ms/step):

| kernel | n/step | us/step | avg us | % kernel |
|---|---|---|---|---|
| fp8_gemv v2 (all projections) | 130 | 2483.4 | 19.1 | 37.5 |
| moe_gate_up_p1 v2 | 40 | 838.7 | 21.0 | 12.7 |
| nvfp4_gemv v2 <bf16> | 120 | 692.1 | 5.8 | 10.5 |
| moe_down_reduce_p1 v2 | 40 | 629.1 | 15.7 | 9.5 |
| nvfp4_gemv v2 <f32> (lm_head) | 1 | 460.1 | 460.1 | 6.9 |
| gemv bf16 | 140 | 275.6 | 2.0 | 4.2 |
| gdn_delta_rows | 30 | 257.3 | 8.6 | 3.9 |
| rmsnorm (residual) / rmsnorm | 80 / 51 | 166.2 / 87.6 | 2.1 / 1.7 | 2.5 / 1.3 |
| moe_topk v2 | 40 | 159.2 | 4.0 | 2.4 |
| splitkv attention | 10 | 123.0 | 12.3 | 1.9 |
| rope (now with the tail copy) | 20 | 112.5 | 5.6 | 1.7 |
| silu_glu | 70 | 100.7 | 1.4 | 1.5 |
| gdn_conv (now with the state write) | 30 | 60.8 | 2.0 | 0.9 |
| moe_combine | 40 | 57.8 | 1.4 | 0.9 |
| rest | 38 | 117.4 | | 1.8 |

Achieved bandwidth on the big consumers (DRAM-cold microbenchmarks, `quantmoe.rs::bench_decode_variants`; peak 627 GB/s):

| shape | v1 us / GB/s | v2 us / GB/s |
|---|---|---|
| fp8 8192x2048 (qkv, q) | 35.3 / 475 | 31.0 / 541 |
| fp8 4096x2048 (z) | 22.1 / 380 | 18.1 / 463 |
| fp8 2048x4096 (o_proj) | 21.3 / 394 | 17.2 / 489 |
| fp8 512x2048 (k, v) | 7.2 / 146 | 5.7 / 185 |
| nvfp4 lm_head 248320x2048 | 469.3 / 610 | 457.1 / 626 |
| nvfp4 shared gate/up 512x2048 | 5.5 / 107 | 4.9 / 120 |
| moe experts T=1, 8 of 256 | 43.9 / 323 | 37.6 / 377 |
| moe topk T=1 E=256 (hot) | 7.6 | 5.8 |

## 3. What was built

1. **HIP graph capture and replay of the decode step** (`crates/intarsia-backend-hip/src/device.rs`, `kernels/hip/ik_runtime.{h,hip}`):
   `supports_graphs()`, `graph_capture` (`hipStreamBeginCapture` thread-local mode, `hipStreamEndCapture`, `hipGraphInstantiate`)
   and `graph_replay` (`hipGraphLaunch`), driven by the existing `crates/intarsia-core/src/model/graph_step.rs` and the device `pos` tensor
   the HIP launchers already took. `INTARSIA_HIP_TRACE` stream events are skipped while a capture runs.
2. **v2 decode kernels**, registered next to their v1 kernels (`matmul_nvfp4_gemv_v2`, `matmul_fp8_gemv_v2`, `moe_topk_bf16_v2`,
   `moe_experts_bf16_v2`; the registry prefers them by priority; all four are ADMITTED, errors 0 against the cpu reference, see
   `intarsia kernels test`). They compute the same arithmetic in the same order as v1 and are **bitwise equal** to it:
   * nvfp4 gemv and the MoE p1 gate_up/down kernels request the first lane-iterations' weights, scales and activations *before* the
     shared-table barrier (the barrier used to precede every load), so the table fill overlaps the memory latency.
   * fp8 gemv stages the activations eight per thread, converts to e4m3 with the gfx12 hardware instruction
     (`v_cvt_pk_fp8_f32`, after an explicit clamp, scale a power of two), and uses two rows per warp at every N
     (`INTARSIA_HIP_FP8_Q=1|2|3`, `INTARSIA_HIP_FP8_ROWS`; default Q=2).
   * moe top-k packs each candidate into a 32-bit key so a warp argmax round is one shuffle and one max.
3. **Fusions inside existing launchers** (no new op, 50 fewer launches per step): the single-tile `gdn_conv_bf16` writes its own new
   state (`gdn_conv_state` only runs for T > 32), and the partial-rotary `rope_bf16` copies the un-rotated tail itself
   (`rope_pass` only runs for rotary_dim 0).
4. Knobs (all in the admission fingerprint, `kernels.rs::ENV_KNOBS`): `INTARSIA_HIP_FP8_Q`, `INTARSIA_HIP_MOE_PRE`
   (lane-iterations in flight before the barrier, default 4), `INTARSIA_HIP_XSTAGE` (activations through shared memory,
   default off: measured slower for the MoE kernels, 42.8 us against 37.6 us).

### Tried, no gain

* Shared-memory staging of the activations for nvfp4 gemv and the MoE p1 kernels: no change for nvfp4, 14 % slower for MoE.
* Warps per block for the MoE v2 kernels (2, 4, 8): 37.7 us in every setting. Prefetch depth 1 / 2 / 4: 45.7 / 43.0 / 37.8 us;
  4 already covers all lane-iterations of H = 2048, so more cannot help. The MoE kernels are limited by their dependent phases
  (ids -> weights -> reduce -> second kernel), not by occupancy or by request depth.
* The CLR environment knobs listed in section 2.
* Reusing a stream after an *invalidated* graph capture: two attempts (an empty failing body, and a synchronous copy inside the
  capture) both left the device returning `hipErrorStreamCaptureInvalidated` (901) for later calls; the wrong assumption was that
  `hipStreamEndCapture` + `hipGetLastError` + `hipStreamSynchronize` clears the capture error. Stopped. Consequence: a capture
  that fails would leave the HIP backend unusable although `graph_step.rs` would switch graphs off. Neither the 35B nor the 8B
  fails one (1 capture, 1920 and 1905 replays), and the test for it was dropped (documented in `tests/device_pos.rs`).

### Scope notes

* **Tensor parallel is not affected**: `GraphState::new(opts.graphs && tp.world == 1)` (`crates/intarsia-core/src/model.rs`) keeps graphs
  off for TP models, so the TP=2 path never reaches `graph_capture` (the unrecoverable invalidated-capture case above cannot be
  hit there). No TP run was made on this branch.
* Radiance port plan (`docs/research/radiance-port-plan.md`) decode techniques *not* tried, and why: the fused GDN decode block op
  (conv update + recurrent update + gated rmsnorm in one kernel, section 3.5) would remove ~3 launches per GDN layer (~90 per
  step) but is a new block-scope op across the CPU / CUDA backends; gate/up weight fusion at load (section 3.3, halves the shared
  expert GEMV launches and feeds `silu_mul` one row) needs a loader / recipe change; `qk_norm + rope(+gate)` fusion touches 10
  layers only; per-shape tune tables (section 7.4) were replaced by the knob sweeps above; the fused all-reduce kernels are TP.
  These are the next candidates: with a ~1.5 us boundary per launch and ~900 launches, every 100 launches removed is ~0.15 ms
  (~2 %).

### Not done

* Fusions that need a new op or a graph change: gate_split + sigmoid_mul (not adjacent in the dataflow), topk + combine (the
  routed experts sit between them), rmsnorm + residual (already one kernel, `rmsnorm_kernel<true,true>`), the GDN small GEMVs
  (a, b, router and shared-expert gate: 140 launches of ~2 us plus their boundaries; a two-weight GEMV or running them on a forked
  capture stream would remove ~0.2 to 0.4 ms), `shared gate/up + silu_glu`. Each is an op-level change across the CPU and CUDA
  backends; with a ~1.5 us boundary per launch they are worth ~1 to 2 % each.
* fp8 and MoE are at 77 % and 62 % of DRAM bandwidth in the step (cold microbenchmarks: 86 % for the big fp8 shapes); the
  remaining headroom is ~0.35 ms (fp8) and ~0.5 ms (MoE) per step if the dependent-phase latency of the MoE pair could be hidden.
* The full (non-quick) `intarsia kernels test --model <35B>` panics in the cpu reference (`reference.rs:120`, `chunk_size must not
  be zero`, a Matmul case with n = 0). **Pre-existing**: it reproduces on `63b0f72` built in `~/intarsiaml-base` (0 `_v2`
  kernels registered, same panic). It makes the `kernels` stage of `qualify` ERROR for the 35B (the other six quick stages
  pass). Not touched; `kernels test --quick` on the 35B passes.

## 4. Graph replay proof (real 35B, 256 greedy tokens)

`scripts/hip-decode/remote.sh graphproof`: the same prompt, `--no-graphs` against graphs, token ids compared with `cmp`.

* `GRAPH == EAGER: the 256 generated token ids are identical`, on the final tree's binary (job 1665: eager 121.2 tok/s, graph
  125.9 tok/s over the 256 tokens) and, earlier, on the graph + v2 kernels build without the fusions (job 1650). The first two
  attempts (jobs 1646, 1648) printed a mismatch because the script read the ids from the wrong stream; fixed in `remote.sh`.
* eager: `graph replays=0 (captures 0) eager decode steps=256`; graph: `graph replays=256 (captures 1) eager decode steps=0`.
* `/metrics` of the serving runs: `intarsia_graph_replays_total 1946`, `intarsia_graph_captures_total 1`,
  `intarsia_decode_eager_steps_total 0` (35B); 1926 / 1 / 0 (8B). The bench JSON carries it: `hipdecode-35b-after2.json` has
  `graph: {replays: 1920, eager_steps: 0, profiled_steps: 0}` for 15 cells x 128 tokens, `hipdecode-8b-after2.json` 1905 / 0 / 0.
  The `before` JSONs have `replays 0, eager_steps 1920`.
* Kernel-level: `device_pos.rs::graph_replay_follows_the_device_position` replays a captured kv_append + split-KV attention at
  moving positions (and back to position 3) and compares each replay bitwise to an eager launch with the host position.

## 5. Accuracy (final tree; the checks do not move: every new kernel is bitwise its v1 kernel and the fusions are copies)

| golden | result |
|---|---|
| `qwen36-35b-a3b-nvfp4-vllm-top20.safetensors`, `intarsia accuracy --backend hip --mode both` | PASS. prefill top-1 98.54 %, KL(21-way) 6.35e-3; **decode top-1 99.38 %, KL(supp) 9.962e-3, same as the baseline**; non-finite 0 |
| `qwen3-8b-fp32.safetensors` | PASS. prefill top-1 100 %, KL 2.34e-4; decode top-1 100 %, KL 3.63e-4 |
| `qwen3-8b-long-fp32.safetensors` | PASS. prefill top-1 99.58 %, KL 3.97e-4; decode top-1 100 %, KL 3.57e-4 |

Graph decode is token-identical to eager over 256 greedy steps on the real 35B, and the kernel-level replay test is bitwise
(section 4). There is no logits or logprobs dump on the serving path, so the model-level claim is argmax identity, not a logits
diff; graphs replay the same kernels with the same arguments, so they cannot change these numbers except through that path.

Kernel-level tests (`crates/intarsia-backend-hip/tests/decode_v2.rs`, GPU): nvfp4 / fp8 / topk / experts v2 are compared bitwise
to v1 and, within the admission budget, to the cpu reference on shapes the sweep does not draw (K past the prefetch window,
row tails, ties, duplicate and invalid expert ids, saturating and underflowing fp8 activations, all three fp8 staging variants);
the fused conv state is compared exactly and the rope tail exactly. **Mutation check** (`remote.sh mutate`, each mutation applied
to a copy of the tree, `cargo test --test decode_v2` must fail): conv state written doubled, nvfp4 iteration skipped, top-k tie
break flipped, rope tail copy strided wrong, MoE down kernel reading the wrong half row, fp8 output scale halved, fp8 hardware
conversion selecting the wrong byte pair: **7 of 7 fail** (job 1667, on the review-fix tree) (the passing tree: 5 of 5 pass under every `INTARSIA_HIP_FP8_Q` setting
and with `INTARSIA_HIP_XSTAGE=1`).

`intarsia kernels test --backend hip --model <8B>`: 34 kernels, 583 cases, all ADMITTED (the `qualify` run reports PASS). 35B:
see "Not done" for the pre-existing full-sweep panic; `--quick` ADMITTED.

`cargo test --workspace` and `cargo check -p intarsia-cli --features hip` pass on the Mac (the ABI header test caught the missing
declarations of the four launchers, now added to `kernels/abi/ik_abi.h`; declarations only, CUDA is unaffected), and the tree
builds and runs in the dev image.

## 6. Radiance downtime log

Bursts under `fleetq acquire --takes-down radiance --mem 30 --wait` (`FLEETQ_JOB` on every ssh), `systemctl --user stop radiance`,
work, `systemctl --user start radiance`, poll `/v1/models`, `fleetq release`. Times are tatooine's clock; "answered" is the first
`/v1/models` reply (the poll interval is 5 s; every burst answered at the 3rd poll, ~15 s after the start).

| job | what | stopped | answered | down |
|---|---|---|---|---|
| 1642 | baseline rocprofv3 attempt (needs libdw) | 16:27:22 | 16:27:56 | 0:34 |
| 1644 | baseline profile + bench 35B + 8B | 16:30:09 | 16:33:17 | 3:08 |
| 1644 | baseline profile retry | 16:33:42 | 16:34:09 | 0:27 |
| 1646 | graphs: tests, graph proof, bench | 16:40:31 | 16:47:23 | 6:52 |
| 1648 | v2 kernels: tests, sweep, graph proof | 17:02:36 | 17:03:26 | 0:50 |
| 1650 | xstage sweep, graph proof | 17:07:43 | 17:08:24 | 0:41 |
| 1652 | profile after, bench after | 17:09:31 | 17:12:39 | 3:08 |
| 1654 | CLR env sweep, accuracy 35B | 17:14:00 | 17:15:21 | 1:21 |
| 1657 | tests, accuracy 35B + 8B, kernels, profile + bench | 17:18:52 | 17:23:46 | 4:54 |
| 1658 | mutation tests, qualify 8B | 17:25:53 | 17:31:34 | 5:41 |
| 1659 | qualify 35B | 17:31:50 | 17:33:13 | 1:23 |
| 1661 | baseline-tree kernels test (panic check) | 17:33:50 | 17:34:21 (released), answered 15 s after the start | 0:31 |
| 1663 | MoE warps / prefetch sweep | 17:35:15 | 17:35:56 | 0:41 |
| 1665 | final-tree graph proof | 17:38:47 | 17:39:11 | 0:24 |
| 1667 | review fixes: tests, graph proof, accuracy 35B, 7 mutations | 17:44:13 | 17:47:06 | 2:53 |

Total ~34 min against the 90 min cap, longest burst 6:52 against 30 min. The last burst (1667) answered `/v1/models`
(`Qwen3.8`, 200000 context) at 17:47:06, before `fleetq release`, and a final check from the Mac answered again; builds and rsyncs ran in CPU-only slots (`--mem 0`) without
touching radiance. spark1 and spark2 were not used.

Process slips: the leases of jobs 1644, 1648, 1650 and 1654 stayed held for a few minutes while code was edited after their burst
had finished (radiance was already back up each time; the queue was blocked, not the engine). Job 1661 and the baseline panic
check ran `~/intarsiaml-base`, which overwrote the shared release binary in the target volume; it was rebuilt before any later run.
