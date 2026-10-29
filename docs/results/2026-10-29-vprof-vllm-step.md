# vprof: where vLLM's 35B NVFP4 B = 16 decode step beats ours on spark1: GDN, fp8 projections, attention, lm_head; the routed experts are not it

Lane vprof, 2026-10-29, branch `lane-vprof/vllm-step` from `main` ce2bfa6ba. spark1 (spark-4e77, GB10 sm_121),
`nvidia/Qwen3.6-35B-A3B-NVFP4`, vLLM `missioncntrl/vllm-openai:65b7662d` (0.26.1rc1.dev602) with the vr16 flags
(`--max-num-seqs 16`). Investigation only: docs and scripts, no engine code.

**Status: PR (docs + scripts). Not merged.**

## Result

Per-kernel nsys profile of vLLM's own 16-stream decode step, bucketed as `scripts/step16/buckets.py` buckets ours, steady
decode, same burst shape as the step16 profile (16 streams, ~1k-token prompts, 256 tokens, CUDA graphs on, kernels inside the
graphs traced with `--cuda-graph-trace=node`).

- **vLLM's decode step is 38.3 ms in this burst (255 steps), ours is 41.7 ms (252 steps): 3.5 ms, 8 %.** The 40.6 ms of
  vr16/srv16 is the same quantity from the served tokens (24.6 tok/s per stream); the profile reproduces it (below).
- **The routed experts are not where vLLM is faster.** Marlin W4A16 MoE takes 19.9 ms per step in the same window where ours
  takes 18.9 ms. It reads the same NVFP4 bytes (each distinct expert once) and streams them a little slower than our
  kernels (about 212 against 223 GB/s at equal expert counts). Its down projection is faster than ours (163 against 188 us per
  layer), its gate_up slower (335 against 283 us).
- **The 3.5 ms is spread over four places**: GDN 1.2 ms (the delta-rule kernel moves the same f32 state 12 % faster per call,
  250 against 279 us), fp8 projections 1.4 ms (merged CUTLASS GEMMs), attention 1.2 ms (fp8 KV, half the bytes) and the lm_head
  0.3 ms; the routed experts and glue go the other way (+1.0 and +0.3 ms).
- **Kernel sums are almost equal (42.9 ms ours, 42.7 vLLM); the critical path is not.** The union of kernel intervals (what the
  GPU is busy for) is 41.5 ms for ours and 38.75 for vLLM, a measured 2.7 ms. vLLM hides 3.9 ms of aux-stream work (its shared
  expert is bulky: 3.9 ms of kernels against our 1.6) behind the routed experts; ours hides about 1.8 ms, although by my
  reading of the buckets about 3.5 to 4 ms of our work sits on side streams (shared expert 1.6, GDN z about 1.5, k/v 0.4, part of
  the 0.7 bf16 GEMVs). **How that splits into "kernel is slower" and "overlap is worse" depends on which of our kernels are the
  hidden ones; step16's trace was not kept, so the per-bucket attribution below assumes ours hides the shared expert and
  that the rest is exposed.** If instead our hidden 1.8 ms is z / k / v, the exposed shared expert makes the main-stream rows
  smaller and the overlap item in section 4 larger.
- **The delta-rule hardware floor is higher than step16 assumed.** step16 showed that no exact change got under 279 us; vLLM's Triton
  kernel moves the same f32 state in 250 us, about 259 GB/s in step16's units (232 x 279/250), 95 % of the 273 GB/s peak. The
  state is f32 in this run (section 1).

What did not work: the ncu pass that would have measured DRAM bytes per MoE kernel hung (section 6), so routed-expert bytes are
derived, not measured.

## 1. Method

Window #6853 (`scripts/vprof/window.sh`, `--quiet gpu,cpu --mem 45 --takes-down flight,guidance,capcom`, 04:27 to 04:37 EDT,
10 min after a 16 min wait for spark1-vllm's own client to go idle). `spark1-vllm` (container `missioncntrl-guidance`) stopped,
container `vprof-vllm` started with the vr16 launcher's flags (`raw/vprof-cmd.json`; ONLY changes: `--host 127.0.0.1 --port
8011`, run under `nsys profile`), service restored and answering at 04:36:58.

- vr16's launcher is spark2's, so this run has **no** `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass` (spark1's production launcher
  sets it; it would change only dense NVFP4 linears, here the lm_head and the shared expert). The point is the setup behind the
  40.6 ms.
- `nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --trace-fork-before-exec=true` with `VLLM_WORKER_MULTIPROC_METHOD=spawn`
  (vLLM runs its GPU work in a child EngineCore process). The host's nsys 2025.3.2 is mounted into the container.
- vLLM picked (startup log, `raw/vllm-startup.log`): NVFP4 MoE `MARLIN`, NVFP4 linears `MarlinNvFp4LinearKernel`, FlashInfer
  attention (`decode_backend=flashinfer-native`, KV `float8_e4m3fn`), the Triton/FLA GDN kernels, CUDA graphs `FULL` for decode
  (sizes 1, 2, 4, 8, 16) and `PIECEWISE` for prefill.
- The GDN state is f32 in this run, checked from the log: it sets the attention block to 2096 tokens "to ensure the attention
  page size is >= the mamba page size"; 2096 x 1024 B (2 KV heads x 256 x K and V, fp8) = 2,146,304 B, which is exactly an f32
  SSM state (32 x 128 x 128 x 4 = 2,097,152 B) plus the 8192 x 3 bf16 conv state (49,152 B). A bf16 state would give about 1072 tokens.
- Driven by `scripts/vprof/burst.py` (same prompt construction as `scripts/step16/burst.py`: first 3700 chars of
  `tools/prompts/corpus_docs.txt` behind a per-stream tag, `DISTINCT=1` gives each stream its own slice; thinking off,
  `ignore_eos`): a 2-stream warm-up, burst **a** (16 streams, same body, comparable to step16's `base-b16`), burst **b**
  (16 streams, distinct bodies), burst **c** (1 stream). Prefix caching on; the tag at the start of every prompt means no
  stream shares a cached prefix. All 16 streams generated 256 tokens (b: two streams 254), no preemptions (`raw/bursts.txt`,
  `raw/metrics.txt`).
- Analysis: `nsys stats -r cuda_gpu_trace`, then `scripts/vprof/tail.py` (last four kernel clusters) and
  `scripts/vprof/vsteps.py` (steps, buckets, kernel table). A decode step is delimited by its embedding kernel; steps are
  found from the 80 `marlin_moe_wna16` launches (40 layers x gate_up + down) per step, and every step of both bursts matched
  (255 steps each, `raw/step-periods.txt`).
- Our side comes from step16's `raw/base-b16/` (`rank.txt`, `gaps.txt`, `timeline.txt`, `raw/buckets-b16.txt`): same machine,
  same burst, `origin/main` ebf8192a0 (main since is docs only). Its bucket window is 167 steps at 4 to 11 s of the burst; the
  vLLM window is steps 30 to 196 (167 steps), after the prefill steps.

### vLLM's own step time, cross-checked

| source | step |
|---|---:|
| trace, burst a, all 255 decode steps (mean; thirds 34.8 / 40.0 / 40.0) | 38.3 ms |
| trace, burst a, steps 30..196 (the bucket window, wall mean) | 39.0 ms |
| trace, burst b (distinct bodies), all steps (thirds 39.1 / 42.1 / 42.7) | 41.3 ms |
| server `inter_token_latency` over the whole run (includes the prefill-stalled tokens of every burst) | 41.0 ms |
| vr16 / srv16 (24.6 tok/s per stream, localeval) | 40.6 ms |
| ours, same burst, 252 decode passes (thirds 38.2 / 42.5 / 44.5), step16 `timeline.txt` | 41.7 ms |
| ours, 167-step bucket window | 42.0 ms |
| ours, `timing_35b_batched_step` (64..264 context, the figure srv16 used) | 44.5 ms |

The step time grows through a burst for both engines (more distinct experts per layer as the streams' continuations
diverge: 26 to 40 ms for vLLM's burst a). A comparison is only fair at the same point of the same burst, which is why the
tables use the same burst and the thirds; **44.5 against 40.6 mixes two different loads** and overstates the gap by about 0.5 ms.
On the same burst the gap is 3.5 ms (thirds: 3.4, 2.5, 4.5).

## 2. Buckets, ours against vLLM (ms per step)

Kernel durations summed per bucket (as step16's `buckets-b16.txt`), vLLM steps 30..196 of burst a (`raw/vsteps-burst-a-30-197.txt`),
ours the 167-step window of step16.

| bucket | ours | vLLM | delta | note |
|---|---:|---:|---:|---|
| routed MoE (gate_up, down) | 18.94 | 19.92 | **+0.98** | vLLM: `marlin_moe_wna16`, 80 calls |
| GDN (delta, conv) | 8.96 | 7.74 | **-1.22** | delta 279 -> 250 us, conv 18.3 -> 7.6 us |
| fp8 K=2048 (GDN qkv + z, attn q; ours + k, v 0.37) | 5.52 | 4.36 | **-1.16** | vLLM merges qkv+z (12288 rows) and q+k+v (9216 rows) |
| fp8 K=4096 (GDN out, o_proj) | 2.35 | 2.15 | -0.20 | 58.6 -> 53.8 us |
| attention (ours: kernel + kv append + rope) | 2.24 | 1.04 | **-1.20** | fp8 KV, FlashInfer: 96 us against our 212 us per layer |
| lm_head nvfp4 | 1.52 | 1.23 | -0.29 | Marlin, 233 GB/s against 188 |
| shared expert nvfp4 (ours) / shared expert + shared gate (vLLM, **aux stream**) | 1.59 | 3.89 | (hidden) | see below |
| bf16 GEMVs (router, GDN b/a, ours also shared gate) | 0.71 | 0.94 | +0.23 | |
| norms, top-k, glue | 1.08 | 1.40 | +0.32 | vLLM: 537 launches per step |
| **sum of kernel durations** | **42.91** | **42.66** | -0.25 | |
| hidden by overlap (kernel sum - union of intervals) | about 1.8 | 3.89 | | ours from `rank.txt` (8244 vs 7909 ms over the 8 s window) |
| GPU idle inside the window | 0.5 | 0.28 | | ours 1.2 %, `rank.txt` |
| **step wall** | **42.01** | **39.03** | **-2.98** | |

Reading it (the attribution assumes our hidden 1.8 ms is the shared expert; see the Result section): the step wall is the main-stream chain. vLLM's main stream carries 38.8 ms of kernels (everything except the 3.89 ms
on the aux stream), ours about 41.5 (42.9 minus the 1.8 hidden). The 2.7 ms difference of the chain is the sum of the main-stream
rows (-1.22 - 1.16 - 0.20 - 1.20 - 0.29 + 0.98 + 0.23 + 0.32 = -2.54, plus rounding of the overlap estimate) and 0.24 ms less
idle. Burst b (distinct bodies) has the same non-MoE rows and 22.7 ms of routed MoE (`raw/vsteps-burst-b-30-197.txt`).

### Top 10 vLLM kernels by time per step (burst a, steps 30..196, `raw/vsteps-burst-a-30-197.txt`)

| # | kernel | grid | block | stream | calls/step | us/call | ms/step |
|---|---|---|---|---|---:|---:|---:|
| 1 | `marlin_moe_wna16::Marlin<...>` (routed experts, gate_up and down) | 144x1x1 | 128 | main | 80 | 249.0 | 19.92 |
| 2 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` (Triton) | 4x512x1 | 32 | main | 30 | 250.3 | 7.51 |
| 3 | CUTLASS sm120 GEMM (GDN qkvz, fp8, 12288 x 2048) | 1x192x1 | 384 | main | 30 | 116.7 | 3.50 |
| 4 | CUTLASS sm120 GEMM (GDN out / o_proj, fp8, K = 4096) | 1x32x1 | 384 | main | 40 | 53.8 | 2.15 |
| 5 | `internal::gemvx::kernel` (cuBLAS, shared-expert gate logit) | 4x1x1 | 32x4 | aux | 40 | 31.9 | 1.28 |
| 6 | `marlin::Marlin<...>` (lm_head, NVFP4 W4A16) | 48x1x1 | 256 | main | 1 | 1228.3 | 1.23 |
| 7 | `flashinfer::BatchPrefillWithPagedKVCacheKernel` (decode, fp8 KV) | 48x1x2 | 32 | main | 10 | 96.0 | 0.96 |
| 8 | CUTLASS sm120 GEMM (attention qkv, fp8) | 1x144x1 | 384 | main | 10 | 85.7 | 0.86 |
| 9 | `marlin::Marlin<...>` (shared expert gate_up) | 48x1x1 | 128 | aux | 40 | 19.7 | 0.79 |
| 10 | `at::native::elementwise_kernel` (shared-expert gate multiply) | 64x1x1 | 128 | aux | 40 | 16.8 | 0.67 |

One 3-step slice of the whole kernel sequence: `raw/decode-steps-100-102-burst-a.csv.gz` (time, duration, stream, grid, block,
name; 3504 launches). A step is about 1170 launches (ours: about 870, 165042 kernels over the 8 s window of step16's profile), all inside one CUDA graph; the aux stream is a second
branch of that graph.

## 3. Routed MoE: which kernel, bytes, bandwidth

- **Kernel**: `marlin_moe_wna16::Marlin` (Marlin W4A16 MoE: NVFP4 weights dequantised in registers, bf16 activations, mma),
  `moe_align_block_size` with block size 8 so each distinct expert is read once per layer, then gate_up, a separate
  `act_and_mul` (silu), and the down GEMM, then `moe_sum`. Grid 144 blocks of 128 threads for both GEMMs. Not CUTLASS or FlashInfer:
  `VLLM_USE_FLASHINFER_MOE_FP4=0` on this image selects `MARLIN` (startup log). Our per-pair row-exact kernels are a different
  design; both read each distinct expert's NVFP4 payload and block scales once.
- **Bytes** (derived, not measured): 1.13 MB gate_up + 0.59 MB down per expert (moe16 doc, payload + e4m3 scales). Measured
  distinct experts per layer for this load on our side: 61.45 (burst a's load, moe16 `snap-p1`) and 66.3 (distinct bodies,
  `snap-p4`). So about **4.2 GB per step for burst a, 4.6 GB for b**. vLLM's `D` was not measured (the ncu attempt failed); the
  same model and prompts route alike, but its outputs differ in the last digits from ours, so treat the bandwidths as +-5 %.
- **Achieved bandwidth** at those bytes:

| | gate_up us/layer | down us/layer | per layer | per step | bytes/step | GB/s |
|---|---:|---:|---:|---:|---:|---:|
| ours (burst a load; step16 `base-b16`, 283 / 188 us) | 283 | 188 | 471 | 18.9 ms | 4.23 GB | 223 |
| vLLM, burst a, steps 30..196 | 335 | 163 | 498 | 19.9 ms | 4.23 GB | 212 |
| vLLM, burst b (distinct), steps 30..196 | 381 | 186 | 567 | 22.7 ms | 4.56 GB | 201 |
| per kernel at D = 61.45: gate_up 69.4 MB, down 36.3 MB | ours 245 GB/s, vLLM 207 GB/s | ours 193, vLLM 223 | | | | |

- **Does vLLM read fewer bytes?** No evidence of it: same weight format, each distinct expert once, and the time is not shorter.
  **Does it reach higher bandwidth?** Not overall: 5 % lower than ours at equal `D`. Per kernel it is mixed: its down projection
  streams 15 % faster than ours, its gate_up 15 % slower. The 40 layers are not alike (layers 0 to 2 take 460 us gate_up,
  later ones 280 to 350 us, tracking each layer's distinct-expert count, `raw/moe-per-layer.txt`).
- The one routed-expert item that is real: **our down projection** (188 us/layer, 193 GB/s) is below vLLM's (163 us, 223 GB/s)
  and below our own gate_up (245 GB/s). Matching 163 us would save 1.0 ms per step.

## 4. What to change to close the gap

Ordered by estimated gain. "Exact" means bitwise the same tokens as today's row-exact engine, which is the bar for default-on
changes (`docs/tuning.md`).

1. **GDN delta-rule kernel and conv: about -1.0 to -1.2 ms (delta 279 -> 250 us x 30, conv 18.3 -> 7.6 us x 30). The conv
   part is exact (a 4-tap elementwise op); the delta part is exact only if the left-to-right key sum is kept, which is not
   shown.** vLLM's Triton kernel (`fused_recurrent_gated_delta_rule_packed_decode_kernel`, read in the image's
   `third_party/flash_linear_attention/ops/fused_recurrent.py`) streams the same f32 state (about 32 MB read + 32 MB written per layer
   at B = 16) in 250 us against our 279 us, 11.6 % faster per call. Its shape: grid (4 value tiles of 32 rows, 16 sequences x 32 heads), one 32-thread block each,
   the whole `[32 x 128]` state tile in registers, and the key-dimension sums done with `tl.sum(..., 1)`, a Triton tree
   reduction. That tree is **not** our summation order: `gdn_delta_split_kernel<128, LPR=2>` (`kernels/cuda/gdn.cu`) sums
   left to right with a carry handed between the two lanes of a row, and its header records that other trees change the 35B's
   greedy tokens and KL. So vLLM's kernel cannot be dropped in. What is open is whether a launch shape closer to vLLM's (more,
   smaller blocks per head, state loads issued earlier; `INTARSIA_GDN_SPLIT_LPR` accepts only 2, 4, 8) can reach 250 us while
   keeping our left-to-right carry. step16's claim stands for exact changes it tried, but "at the bandwidth floor" does not:
   the hardware allows at least 11.6 % more on this kernel.
2. **Fewer, wider fp8 projection launches and a higher streaming rate: about -0.5 to -1.5 ms (bucket difference 1.36 ms, plus any exposed side-stream work) (K=2048
   5.52 -> 4.36, K=4096 2.35 -> 2.15). Exact.** vLLM runs GDN qkv + z as one 12288-row GEMM (116.7 us, 216 GB/s) and the
   attention q + k + v as one 9216-row GEMM (85.7 us); ours runs 8192-row, 4096-row, 8192-row and two 512-row GEMVs at
   180 to 200 GB/s. Concatenating the weight rows at load and launching the existing row-exact kernel once over all rows keeps
   every output row's dot-product order, so it is bitwise. Caution: INTARSIA_SIDE_PROJ already overlaps z, k, v with qkv, q,
   so the saving is the part the overlap does not already hide (launch tails, the 512-row kernels' low occupancy). The range is
   wide because it depends on how much of our side-stream work is really hidden: qkv and z are both DRAM-bound GEMVs, so
   running them beside each other may hide little, and vLLM's aux kernels are small latency-bound ones next to a DRAM-bound
   Marlin. If z, k and v are mostly exposed today, one merged launch recovers more than the low end. K = 4096 (-0.2 ms) is a bandwidth-only item.
3. **Routed-expert down projection at vLLM's rate: about -0.5 to -1.0 ms (188 -> 163 us per layer). Exactness depends on
   keeping the per-pair FMA chain, so only a streaming/tiling change qualifies; the moe16 lane's expert-major attempt did
   not pay.** This is the only MoE lever the profile supports; gate_up is already faster than vLLM's.

Not exact, already known, opt-in only: **fp8 KV cache for decode attention, -1.2 ms** (vLLM's FlashInfer decode on an fp8
cache reads 19.7 MB per layer at 206 GB/s; ours reads 39 MB of bf16 at 184 GB/s). The exact part of the attention gap is only
what a bf16-KV kernel can still gain by reaching 230 GB/s: about 0.3 ms. **lm_head**: -0.3 ms if our NVFP4 GEMV reached
vLLM's 233 GB/s (188 today); exact if only the streaming changes.

Together the items that can be exact are 2.0 to 3.0 ms of the 3.5 ms gap, i.e. B = 16 step 42.0 -> about 39.5 ms in the burst window, or
44.5 -> about 42 ms on the micro-timing basis. Nothing here needs a routed-expert redesign.

## 5. Caveats

- Ours is step16's profile (a different run, `origin/main` ebf8192a0), not re-profiled next to vLLM in the same window.
  Per-bucket deltas under 0.3 ms are inside run-to-run noise.
- nsys with node-level graph tracing adds a little per-kernel time on both sides (same flags as step16).
- The step time depends on the burst's expert diversity (vLLM burst a 38.3 ms, burst b 41.3 ms); compare at the same point of
  the same burst only.
- vLLM's kernel sums include 3.9 ms on the aux stream that overlaps the main stream; sums are not step times.
- Identical-prompt streams (burst a) route alike at the start; the MoE rows would be larger with fully distinct prompts
  (burst b: 22.7 ms).
- Only the decode graph at B = 16 was profiled; prefill, mixed steps and other batch sizes were not.

## 6. What the ncu pass would have added (not obtained)

A second window (#6862, `scripts/vprof/window2.sh`) ran the same server under `ncu --graph-profiling node` with
`--profile-from-start off` and vLLM's `/start_profile` to read `dram__bytes_read.sum` of the Marlin MoE, GDN, CUTLASS and
FlashInfer kernels for a few B = 16 steps. It hung: ncu connected to the EngineCore process, the engine stalled with 16 running
requests, no kernel was reported in the 10 minutes before I killed the container (`docker kill`; `raw/ncu/`). A first attempt
(#6859) aborted within 2 minutes on a shell quoting error in `window2.sh` (no GPU work), fixed in the retry. Not tried again: the
GPU-window cap. The next attempt should try `ncu --replay-mode application` or `--graph-profiling graph` with a small
`--launch-skip`/`--launch-count` on the MoE kernels only, or the CUPTI-free `nsys --gpu-metrics-devices` (GB10 support
unchecked). Until then routed-expert bytes are derived.

## 7. Process

Three fleetq jobs ran against the brief's cap of two GPU windows (#6859 did no GPU work but took the service down for 3 minutes). GPU jobs on spark1, all with `--takes-down flight,guidance,capcom --quiet gpu,cpu --mem 45` and the service restored each time:

| job | what | result |
|---|---|---|
| #6853 | nsys profile of vLLM, bursts warm / a / b / c, 04:27 to 04:37 | good, all numbers above |
| #6859 | ncu attempt | aborted at once on a quoting bug in `window2.sh`, no GPU work |
| #6862 | ncu retry, 04:46 to 05:05 | ncu hung, container killed at 05:02, no data |

`spark1-vllm` was down 10 + 3 + 19 minutes in total and answered `/v1/models` again after each (04:36:58, 04:45:49, 05:05:34). The profile cost: 16 minutes
of waiting at the start of #6853 for another client's requests on the standing service to go idle; no
`--force`. `~/intarsiaml-vprof` on spark1 holds the nsys report and 500 MB of csv; deleted at the end (the 705k-kernel
tail extract, 12 MB, is not committed; the 3-step slice is).

Scripts: `scripts/vprof/window.sh` (nsys window), `window2.sh` (ncu, did not work), `burst.py`, `tail.py`, `vsteps.py`
(buckets, steps, kernel table), `slice.py` (decode-step dump). Raw: `docs/results/2026-10-29-vprof-vllm-step/raw/`.
