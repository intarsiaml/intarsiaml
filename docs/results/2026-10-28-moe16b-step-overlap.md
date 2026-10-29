# moe16b: shorter Spark 35B decode step at 8 to 16 streams: side-stream overlap (shared expert, k / v, z / b / a); fp8 rows kernel diagnosis

Lane moe16b, 2026-10-28, branch `lane-moe16b/step-overlap` from `origin/main` 318d508fb (#224). spark1 (spark-4e77, GB10 sm_121),
Qwen3.6-35B-A3B NVFP4, `serve --backend cuda --max-batch 16 --max-seq 8192 --mem-budget 40`.

**Status: merged, both knobs default on, by coordinator decision (see Decision below). The lane's own speed gates were not met.**
- 16 streams: +1.3 % (gate +5 %).
- 8 streams: +3.3 % (gate +3 %, met).
- 1 stream: +2.6 % (gate: within 1 %, met).
- The fallback for a ready PR is not met either: it needs ≥ +2 % at 16 streams.

Every exactness gate passes:
- Served tokens identical to main: 21/21 cells, 9408 ids.
- `batch_cuda_real` on the 35B: 4290 pairs bitwise.
- `rows_exact` passes.
- `plan show` is byte-identical for 8B, 35B and Gemma-4.
- Mac `cargo test --workspace --release` passes.
- `batch_cuda_real` on the 8B with both knobs on: 4290 pairs bitwise.

The change is exact: it runs the same kernels on the same inputs in the same issue order. Only the stream of some launches changes.

**Both knobs are default on at merge** (lane moe16c; `INTARSIA_MOE_SHARED_STREAM=0` / `INTARSIA_SIDE_PROJ=0` opt out to
main's single stream). During the lane they were opt-in (`=1`), because the speed gates failed.
- W1 (identity, `batch_cuda_real`, the serve A/B, nsys) and W2 ran the build where both were on by default (1e2db288c). That is the
  same code path as setting both to `1` now.
- The step micro-timings of section 3 ran a temporary build with extra variant knobs, in which the shared expert forked before the
  router by default. Its "dev, both knobs on" row sets `INTARSIA_MOE_SHARED_FORK=late`, which is W1's code path.

GPU windows used: **2 of 2** (#6628 speed + exactness, #6647 8B batched-equals-solo). An earlier attempt, #6627, failed at once because the script was not executable. It loaded nothing and is not counted.
Raw: `2026-10-28-moe16b-step-overlap/raw/`. Scripts: `scripts/moe16b/`, which are copies of `scripts/msd/` retargeted to this lane's folders, plus `micro*.sh` and `w1.sh`.

## 1. What changed

### Side stream inside the captured decode graph (item 1, extended)

**Where it applies.** The code forks a side stream (`self.stream + 2`) with an event and joins it back with an event. Both work
inside a CUDA graph capture: the wait pulls the side stream into the capture, and the replayed graph has two branches. It applies
only to a decode step:
- one row, or a row-exact batched / verify pass of up to 16 rows;
- single rank;
- not profiled;
- not a plan dry run.

A 5+-row prefill chunk is excluded: it picks the cuBLASLt GEMMs, which take the plan scratch. The first micro run caught this. It
failed in prefill on the guard error described below. The fork condition was then fixed (commit 1e2db288c).

**What runs on the side stream:**
- **MoE block** (`INTARSIA_MOE_SHARED_STREAM=1`, opt-in): the shared expert (gate, up, silu_glu, down and, in a batched
  step, the gate logit). It forks after the router and runs beside the top-k and the routed experts, then joins before
  `MoeCombine`. The fold into `SharedDownCombine` is not tried while forked, because it reads the routed result. CUDA has no such
  kernel, so nothing changes there.
- **Attention** (`INTARSIA_SIDE_PROJ=1`, opt-in): k / v (+ bias) beside q and its gate split.
- **GatedDeltaNet** (`INTARSIA_SIDE_PROJ`): z, b, a beside qkv. This covers the solo block and the batched block. It joins before
  the conv reads them.

**Plan and bitwise identity.** The ops are issued in the same order as on main. Only `exec_call_one`'s launch stream differs (the
`Model::launch_on` override). The recorder therefore sees the same calls, so `plan show` is unchanged, and every kernel reads the
same inputs, so the bits are unchanged.

**Guard.** A side-stream launch whose kernel would take the plan scratch returns an error. The routed experts own that scratch.

**Backend changes:**
- `Backend::stream_fork()` is a new capability, default `false`.
- CUDA implements `stream_record` / `stream_wait_for` with a pool of 64 `cudaEventDisableTiming` events. New FFI:
  `ik_rt_event_create_sync`, `ik_rt_event_record`, `ik_rt_stream_wait_event`.
- `stream_overlap()` (the TPO lane) is untouched, so HIP's paths and the CUDA TP2 overlap behave as on main.

### Other files
- `docs/tuning.md`: rows for both knobs.
- `crates/intarsia-models/tests/batch_cuda_real.rs`: a new ignored micro-test, `timing_35b_batched_step`. It measures the wall time
  of the batched step at B = 8 / 16 with graphs on and device argmax.
- The #226 diagnostic `INTARSIA_MOE_ROUTE_STATS` is carried over, with its tuning row and its results doc
  (`2026-10-28-moe16-routed-experts.md`). Its scripts are not carried.

## 2. Serve A/B (W1, `scripts/moe16b/w1.sh`, `raw/w1/`)

Setup: localeval `--sizes 1024 --gen 256 --reps 3 --concurrency 1,4,8,16 --thinking off`, `ignore_eos`, 3 interleaved rounds (even
rounds reversed), a fresh server per arm.

The arms:
- **base**: `origin/main` 318d508fb.
- **dev**: this branch with both knobs on.
- **moeonly**: dev with `INTARSIA_SIDE_PROJ=0`, so only the shared expert is on the side stream.

Aggregate tok/s:

| streams | base (r1, r2, r3) | dev (r1, r2, r3) | dev vs base (per round) | moeonly | moeonly vs base | gate |
|---|---|---|---|---|---|---|
| 1 | 78.7, 78.7, 77.9 (78.43) | 80.1, 80.8, 80.4 (80.43) | **+2.6 %** (+1.8 / +2.7 / +3.2) | 79.67 | +1.6 % | within 1 %: **pass** |
| 4 | 165.4, 166.2, 165.8 (165.80) | 170.9, 168.4, 173.8 (171.03) | +3.2 % (+3.3 / +1.3 / +4.8) | 166.77 | +0.6 % | (none) |
| 8 | 221.0, 217.3, 217.4 (218.57) | 228.9, 225.1, 223.6 (225.87) | **+3.3 %** (+3.6 / +3.6 / +2.9) | 221.63 | +1.4 % | +3 %: **pass** |
| 16 | 269.7, 269.9, 268.4 (269.33) | 272.6, 274.0, 271.8 (272.80) | **+1.3 %** (+1.1 / +1.5 / +1.3) | 271.00 | +0.6 % | +5 %: **fail** |

Other single-stream figures:
- 1k-token decode: base 83.67 tok/s, dev 85.47 tok/s (+2.2 %).
- Prefill: unchanged (5.2 to 5.3k tok/s).
- 16-stream TTFT: 1.67 s in both arms.

Against vLLM at 16 streams (340.1 tok/s): 0.79x on base, 0.80x on dev.

## 3. Step time, untraced (micro-test `timing_35b_batched_step`, `raw/micro4/step.txt`)

Setup: wall time per batched step, graphs on, device argmax. Three runs of 60 steps each, run in order, as the context grows from 64
to 264 tokens. One process per variant, because the knobs are read once per process. "off" is the same binary with both knobs at
`0`, which runs main's launch sequence.

| variant | B=8, mean ms | B=16, mean ms | B=16 vs off |
|---|---:|---:|---:|
| off (three processes) | 29.05 / 29.06 / 28.99 | 46.72 / 46.80 / 46.85 | |
| **dev, both knobs on** (two processes) | 28.30 / 28.29 | 45.71 / 45.78 | **-1.04 ms (-2.2 %)**; B=8 -0.73 ms (-2.5 %) |
| shared expert forked before the router (`early`) | 28.63 | 45.75 | same as dev |
| GDN z kept on the main stream (only b / a on the side) | 28.79 | 45.93 | -0.8 ms |
| shared expert only (`INTARSIA_SIDE_PROJ=0`) | 28.67 | 46.15 | -0.6 ms |
| side streams at the default priority instead of the highest | 28.25 | 45.77 | same as dev |

The `early`, z-on-main and priority variants used temporary knobs. They were removed again, so the branch carries the W1-measured
form. Spread between processes is about ±0.1 ms.

**Why 16 streams gains less in serve than here.** Two reasons, both estimates from the A/B figures:
- *Prefill in the burst.* 16 streams x 256 tokens / 272.8 tok/s is about 15.0 s of wall time. Decode at 19.3 tok/s per stream is
  256 / 19.3 = 13.3 s. That leaves about 1.7 s (11 %, close to the 1.67 s mean TTFT) for prefill and start-up, which the change
  does not touch. So a 2.2 % shorter step gives at most about +2.0 % aggregate.
- *Context length.* This micro decodes at 64 to 264 tokens of context. Serve decodes at about 1.0 to 1.25k, where attention is a
  larger share of each step, so the same absolute saving is a smaller relative one. This covers the rest, from +2.0 % down to the
  measured +1.3 %.

## 4. nsys at 16 streams, before and after (W1, `prof.sh`, `--cuda-graph-trace=node`, `raw/base-b16`, `raw/dev-b16`)

Per-step kernel durations, about 150 steps in the decode window:

| bucket (ms per step, B=16) | base | dev | dev - base |
|---|---:|---:|---:|
| routed MoE (gate_up + down_v2) | 21.72 | 22.73 | +1.01 |
| GDN (delta, conv) | 8.86 | 8.85 | -0.01 |
| fp8 K=2048 (GDN qkv / z, attn q) | 5.20 | 5.48 | +0.28 |
| fp8 K=4096 (GDN out, o_proj) | 2.65 | 2.69 | +0.04 |
| attention | 2.14 | 2.15 | +0.01 |
| lm_head nvfp4 | 1.86 | 1.86 | +0.00 |
| shared expert nvfp4 (gate, up, down) | 1.33 | 2.26 | +0.93 |
| norms, top-k, glue | 0.95 | 1.12 | +0.17 |
| bf16 small GEMVs (router, GDN b / a, shared gate) | 0.69 | 0.69 | +0.00 |
| fp8 k / v 512x2048 | 0.36 | 0.41 | +0.05 |
| **sum of kernel durations** | **45.76** | **48.24** | **+2.48** |
| step span (`gaps.py`) | 46.39 | 46.97 | +0.58 |

**The overlap works.** In dev, the kernel durations add up to 48.24 ms inside a 46.97 ms step span (0.43 ms of it idle). So at
least 1.7 ms of kernel time runs concurrently.

**What the overlap costs.** The side kernels are latency-bound and take about twice as long while they share the SMs: the shared
expert's split GEMV goes from 10.7 to 22.4 us. The bandwidth-bound main-stream kernels beside them slow down: the routed experts
take +1.0 ms and the fp8 K=2048 projections +0.3 ms. In a decode step that is mostly bandwidth-bound, most of what the overlap hides
is paid back as contention.

**Why the traced span grew.** Node-level graph tracing adds a per-node cost, and that cost is higher with two branches: the traced
dev span is 0.6 ms longer than base. The untraced micro (section 3) and the serve A/B (section 2) are the measurements. The nsys
table is the attribution.

## 5. fp8 dense projections at M = 16 (item 2): why they run over the floor, and what did not help

Shapes at B=16, from the #224 profile and this one:

| projection | weights | calls per step | in-graph time per call | effective rate |
|---|---|---:|---:|---:|
| GDN qkv, attn q | 8192x2048 | 30 + 10 | 87 us | 193 GB/s |
| GDN z | 4096x2048 | 30 | 51 us | 165 GB/s |
| GDN out, o_proj | 2048x4096 | 40 | 66 us | 127 GB/s |
| k, v | 512x2048 | 20 | 18 us | 58 GB/s |

Together they take 8.1 ms per step, against a 5.6 ms floor at 230 GB/s. The one-row kernel streams the same weights at 200 to
219 GB/s.

**Re-reads.** None. Every fp8 projection of the 35B has a static `input_scale`, read from the checkpoint headers on spark1. So every
call takes the W8A8 path of `fp8_gemv_rows_q_kernel`. Each block stages all M rows once, as e4m3 codes (M x K bytes), and reads
every weight once. The re-reading path (bf16 row groups) is only the W8A16 fallback, and no 35B projection takes it.

**Occupancy:**
- Every M=16 instance uses 128 registers, with no spills (`cuobjdump -res-usage`).
- K=2048: 2 blocks of 8 warps per SM (96 blocks; 32 KB of staged codes per block).
- K=4096: 1 block of 8 warps per SM (48 blocks; 64 KB per block).
- At K=4096, the 1024 tiles of 2 outputs fall unevenly over 384 warps: 32 SMs get 24 tiles and 16 SMs get 16.

**What the time tracks: instruction count, not DRAM latency.** Evicted micro-timing of 2048x4096 (`raw/micro1/evict.txt`):
- M = 2 / 4 / 8 / 12 / 16 take 39 / 40 / 44 / 51 / 56 us.
- That is about 1.2 us per row past 4 rows, against 42 us for the one-row kernel.

The SASS of the M=16 / NT=2 instance (`cuobjdump -sass`) has, per 16 weight bytes per lane:
- 512 FFMA;
- 288 HADD2 and 144 F2FP: the e4m3 to f16 to f32 unpack of the 16 staged activation rows, redone for every 2-output tile;
- 32 FADD and 16 LDS.

So about 40 % of the issued instructions are activation conversions, and their count grows with M. The prologue (W8A8
re-quantisation of x with `__fdiv_rn`, M x K per block) adds an estimated 4 us per call at M = 16 (from its instruction count; not
measured separately). ncu could not confirm the bound:
`ERR_NVGPUCTRPERM`, and this lane may not change the counter permission.

**Exact variants tried.** Each keeps every output's arithmetic: its lane-strided blocks, its fmaf chain and its warp tree. Each
passed `fp8_rows_q_are_bitwise_the_one_row_gemv`.

| variant | result, M=16 (evicted) | file |
|---|---|---|
| L2 prefetch of each warp's next 1 to 3 tiles (and the first ones during staging) | slower everywhere: 2048x4096 56 -> 58 to 61 us, 8192x2048 87 -> 94 to 105 us | `raw/micro1/` |
| balanced split: every warp a contiguous run of N / warps outputs, as 4-, 2- and 1-output tiles; 8, 12 or 16 warps per block | 2048x4096: 54.5 -> 51.8 us (12 warps), within noise elsewhere; 8192x2048: 85.6 -> 90 to 97 us (slower) | `raw/micro2/` |

Neither variant is worth shipping. The prefetch and balanced-split code was reverted, and the diffs are kept in raw.

**The tensor-core (mma) path was not attempted.** It would not be bitwise equal to main. To be batch-invariant, it would have to
replace the one-row GEMV at M = 1 as well. Greedy ids would then have to match main over 9408 tokens, which is unlikely for a
numerics change. And this lane has no KL tool run budgeted for it.

## 6. Gates

| gate | result |
|---|---|
| +5 % at 16 streams, +3 % at 8, 1 stream within 1 % (3 rounds) | **16: fail (+1.3 %)**; 8: pass (+3.3 %); 1: pass (+2.6 %) |
| served greedy tokens identical to main (21-cell set) | **pass**: 21/21 cells, 9408 ids (`raw/ident/compare.log`) |
| `batch_cuda_real` (batched equals solo) on the 35B | **pass**: 4290 (row, step) pairs bitwise, plus the engine test (`raw/batch-real.txt`, in W1); the decode test also as micros (`raw/batch-real-decode-micro2.txt`); 8B with both knobs on: 4290 pairs bitwise (`raw/batch-real-8b.txt`, W2) |
| `rows_exact` | **pass**: 9 tests (`raw/rows-exact.txt`) |
| `plan show` byte-identical, 8B / 35B / Gemma-4 | **pass**: json and txt (`raw/build-3.log` first build, `raw/build-final.log` final build) |
| Mac `cargo test --workspace --release` | **pass**: exit 0, 0 failed (`raw/mac-cargo-test.log`; final tree `raw/mac-cargo-test-final.log`) |
| CI | see the PR |

## 7. Process

GPU windows: **2 of 2**.
- **#6628, W1**, `--quiet gpu,cpu --mem 50`, about 24 min. It ran the identity check, `batch_cuda_real`, the 3-round 3-arm A/B,
  and nsys at B=16 for base and dev.
- **#6627** started the same script and failed on exec. It loaded no GPU work.
- **#6647, W2**, `--quiet gpu,cpu --mem 30`: `batch_cuda_real` on the 8B (`real_qwen3_8b_batched_decode_is_bitwise_the_solo_step`)
  with both knobs on, because it covers the attention k / v fork on a dense model (`raw/batch-real-8b.txt`). The micro attempt
  (#6646) ran out of its 170 s limit. No speed run was left that could reach +5 % at 16 streams, or even +2 %: the variants in
  sections 3 and 5 were all measured as micros first.

Micro jobs under 3 min on spark1, not windows:
(durations from `fleetq log`; each is under 3 min):
- #6603 (23 s): fp8 prefetch.
- #6604 (2 s): ncu probe.
- #6615 (33 s), #6621 (1 min 48 s): batch_cuda_real decode. #6615 failed in prefill, which led to the fork fix.
- #6624 (18 s): fp8 balanced split.
- #6634 (2 min 24 s), #6636 (2 min 33 s), #6639 (2 min 37 s): step-timing variants.
- #6644 (17 s): rows_exact.
- #6646 (2 min 51 s): 8B batch_cuda_real, cut off by its 170 s limit before a result.

Windows: #6628 ran 23 min 23 s, #6647 ran 6 min 14 s. Window 2 went to the 8B batched-equals-solo check, which is not a merge gate
(the gate names the 35B). It was the remaining exactness coverage for the attention k / v fork on a dense model.

`--load cpu` / `--load none` builds and checks: #6599, #6602, #6605, #6606, #6608, #6617, #6626, #6632, #6638, #6641.

spark1-vllm stayed up throughout, with no `--takes-down`. During #6636, fleetq reported the GPU busy with untracked work before the
job started. That job's off variant matched the other two off runs to 0.1 ms.

## 8. Next steps

1. **Per-step kernel time at B=16, not the overlap.** The overlap trades latency for contention (section 4), so the remaining gap
   to vLLM (about 7 ms per step) has to come out of the kernels themselves. The routed experts (21.7 ms) and GDN (8.9 ms) dominate.
   Burst prefill is only about 11 % of the 16-stream burst (section 3 estimate). It is scheduler work and a separate lane.
2. **The fp8 / nvfp4 rows kernels at M = 16 are instruction-bound on activation unpacking.**
   - An exact fix would have to cut the per-tile e4m3 -> f16 -> f32 unpack of the staged rows. For example, stage x as bf16 where
     M x K x 2 fits. That trades the F2FP + HADD2 for integer shifts, but costs occupancy. Or find a cheaper exact unpack.
   - Otherwise, take a batch-invariant mma path with a KL check, opt-in.
   - An ncu run with counter access would settle which pipe is the limit.
3. **lm_head nvfp4 at B=16** (1.86 ms against a 1.24 ms floor) has the same structure as point 2.

## Decision

Merged with both knobs default on (`INTARSIA_MOE_SHARED_STREAM`, `INTARSIA_SIDE_PROJ`; `=0` opts out) by coordinator
decision, 2026-10-28, lane moe16c. Reason: the change is bitwise exact (served greedy 21/21 cells, `batch_cuda_real` on the 35B
and the 8B, `rows_exact`, `plan show` identical) and positive at every stream count (+2.6 / +3.2 / +3.3 / +1.3 % at
1 / 4 / 8 / 16 streams, 3 rounds). It missed only the +5 % gate at 16 streams.

Lane moe16c re-checked the merged tree on spark1 with no knob variables set, so the defaults engage: `rows_exact` and
`batch_cuda_real` on the 35B (`raw/moe16c-defaults.txt`).

