# Metal continuous batching: M4 and M5 go criteria, measured (B = 8 decode step breakdown)

2026-10-16. Qwen3-8B BF16, endor (Apple M5 Max, macOS 27.0 build 26A428), branch `metal-cb-m45-profile` off `spike/metal` 66feda6. Measurement only: no engine code changed. New code is one example (`crates/intarsia-models/examples/batch_prof.rs`, public model API only) and analysis scripts. Raw data: `docs/results/raw/2026-10-16-metal-cb-m45-profile/` (`run.sh`, `run-control.sh`, `run-ce1.sh`, `analyze*.py`; `run1/`, `run2/`, `control-ce1/`; large traces are gzipped, the scripts read them as they are).

Per the coordinator's instruction the go / no-go call is left to the user. Below, each criterion is stated with the measured number and whether the number meets it as worded.

## 1. Results

| milestone | criterion (design doc section 7 / 9) | measured | as worded |
|---|---|---|---|
| **M4** (R = 8 alignment remedy) | N = 8 aggregate improves >= 3% | time on the 8-mod-16 weight path at B = 8 is **0.98-1.07 ms of a 39-47 ms step**; measured against the normal-stream step (38.6 ms at 1k) removing all of it gives **+2.6-2.8% aggregate at 1k** (profiler-timeline figure 2.6-2.7%) and **+2.3-2.5% at 4k** (using an estimated 44.5 ms step); ceiling, assumes the 16-byte path then runs as fast as the layers that already take it. The margin below the 3% gate is about 0.2 points at 1k, so a 1-2% measurement error would not change the outcome but it is not a wide miss | **not met** (ceiling below 3% at both contexts, by about 0.2 points at 1k) |
| **M5** (ragged attention, arena KV) | M2 profiling shows per-row attention launches >= 8% of a B = 8 step | per-row attention (`qk_norm_rope_kv` + `attn_decode`, 576 of 868 launches) is **16-21% of the B = 8 step at 1k and 27-30% at 4k** (GPU time = wall minus ~0.35 ms host lead / tail). Part of that is real KV bandwidth a ragged kernel still has to pay; the part a fused kernel can remove (launch / tail latency, bandwidth floor excluded) is **at most about 9-15% at 1k and 4-9% at 4k** (range from the profiler-overhead bound in section 3) | **met** as worded (16-30% >= 8%); the removable part has a ceiling of 9-15% at 1k and 4-9% at 4k against the M5 gate "B = 8 time >= 5%": the low end at 4k is below that gate |

M5 caveat for whoever decides: the 8% criterion was derived as a bound on the saving (design doc 3.2 / 4.3: 6-8%), not as the total attention share. Read as the total share it is met by a wide margin; read as the saving, the ceiling above is the relevant number and a real kernel will land below it. At 4k the per-row kernel already streams KV at 390 GB/s (81% of the 481 GB/s the GEMVs reach), so little is left to fuse there; at 1k it streams at 181 GB/s and the rest is latency / tail.

## 2. Where a B = 8 decode step goes

Primary table: `INTARSIA_METAL_COMMIT_EVERY=1` with `INTARSIA_METAL_PROFILE=cb:` (every dispatch its own command buffer, committed back to back and not waited on; see 3). 40 steps per cell, medians. GPU 1610 MHz. `control-ce1/analysis_ce1.txt`.

| class (per step) | launches | 1k: ms | 1k: share | 4k: ms | 4k: share | B = 1, 1k: ms |
|---|---:|---:|---:|---:|---:|---:|
| gate + up GEMV (matmul_pair, 24576 x 4096) | 36 | 15.05 | 36.7% | 15.04 | 32.1% | 12.26 |
| down GEMV (4096 x 12288) | 36 | 7.80 | 19.0% | 7.82 | 16.7% | 6.22 |
| q / o GEMV (4096 x 4096) | 72 | 5.13 | 12.5% | 5.13 | 10.9% | 4.30 |
| k + v GEMV (matmul_pair, 2048 x 4096) | 36 | 1.49 | 3.6% | 1.49 | 3.2% | 1.13 |
| **attention, per row: `attn_decode` (split-KV)** | 288 | 6.68 | 16.3% | 12.39 | 26.4% | 1.03 |
| **attention, per row: `qk_norm_rope_kv`** | 288 | 1.87 | 4.6% | 1.89 | 4.0% | 0.29 |
| head (lm_head GEMV + argmax) | 1 | 2.54 | 6.2% | 2.53 | 5.4% | 2.09 |
| norms, silu, embedding, argmax | 111 | 0.48 | 1.2% | 0.48 | 1.1% | 0.49 |
| GPU idle between launches | | 0.03 | 0.1% | 0.03 | 0.1% | 0.02 |
| **GPU span of the step** | 868 | **41.06** | | **46.86** | | 27.85 |
| host wall per step | | 41.64 | | 47.44 | | 28.20 |

* GEMM (all projections + FFN + head): 73% at 1k, 68% at 4k. Attention: 16-21% / 27-30%. Everything else: 2% / 1.5%.
* Host / dispatch gaps: none visible. GPU idle between launches is 0.03 ms per step; in plain `cb` mode GPU busy equals GPU span (868 dispatches in 7 command buffers), and the step wall exceeds the GPU span only by the 0.2 ms lead and 0.1 ms tail of step-to-step host work (argmax download, next step). The B = 8 step is GPU-bound; host encode is not a factor.
* Normal (unprofiled) stream, same state, 1k: `batch_step` B = 8 batched step **38.6 ms** (3 runs 38.51 / 38.61 / 38.65), B = 1 27.5 ms. The `commit-every-1` timeline is 2.4 ms longer (6%) because every launch pays command-buffer start / end cost (about 1 us per launch at B = 1, more on the very short per-row kernels); that is the correction for the profiler, see 3. A normal-stream 4k step in the fast GPU state was not measured (see 4); `commit-every-1` gives 46.9 and the same correction puts it near 44-45 ms.
* Increase from B = 1 to B = 8 at 1k (13.2 ms in total): attention +7.2 ms (55%), gate+up +2.8, down +1.6, q/o +0.8, head +0.45, k+v +0.4, rest ~0.

## 3. Method and how the serialisation problem is handled

* Kernel mode (`INTARSIA_METAL_PROFILE=kernel:`) commits and waits after every launch. It does serialise, but the Metal encoder is `MTLDispatchTypeSerial` in normal operation too (lib.rs: every dispatch starts after the previous finished), so no kernel overlap is lost; what kernel mode changes is (a) a fixed per-command-buffer cost in every record and (b) the GPU clock: the host wait makes the GPU run at 1617 MHz, which is faster than some normal-stream states (see 4). `run2/analysis.txt` removes an assumed fixed cost c = 0 / 1.8 / 3.1 us per record (smallest records are 3.1 us; the design doc prices an 8-row Add at about 1.2 us). Attention share at B = 8 is 18.7-19.3% at 1k and 28-30% at 4k for c = 1.8, and 17.9% / 28.3% for c = 3.1: the share is not sensitive to c.
* `INTARSIA_METAL_COMMIT_EVERY=1` with `cb:` mode gives per-launch GPU start / end timestamps from the real, continuously loaded stream (no host wait; GPU clock 1610 MHz; zero idle between buffers). Each record's contribution is taken as `end - max(start, previous end)` so overlap is not double counted (`analyze_ce1.py`). Its step total (41.06 ms) versus the normal stream (38.6 ms) bounds the profiler overhead at 6%. If all of that 2.4 ms sat in the 576 per-row launches, per-row attention would be 6.1 ms (16%) instead of 8.55 ms (21%) at 1k; hence the 16-21% range quoted for 1k (the same bound at 4k gives 27-30%) (kernel-mode c-corrected and `commit-every-1` agree: 7.5-8.55 ms before the in-stream bound).
* GPU timestamps are the command buffer `GPUStartTime` / `GPUEndTime`; step boundaries are marks written by `batch_prof` on the same `CLOCK_UPTIME_RAW` clock (`*.marks`).
* Not done: no ablation of the per-row attention launches in the live engine (needs a hook in `batch.rs`; the coordinator ruled out engine changes).

### M5: what part of the attention time a fused kernel could remove

Achieved GEMV rate: gate + up, 201.3 MB in 418 us (`commit-every-1`) is 481 GB/s. KV bytes per row and layer: 4.19 MB at 1k, 16.8 MB at 4k (8 kv heads x 128 x 2 bytes x K and V).

| | 1k | 4k |
|---|---:|---:|
| `attn_decode` per row (ms / 288 launches) | 23.2 us | 43.0 us |
| achieved KV rate | 181 GB/s | 390 GB/s |
| floor at GEMV rate | 8.7 us | 34.9 us |
| excess over floor x 288 | 4.2 ms (10.2% of the 41.06 ms timeline) | 2.3 ms (5.0% of 46.86) |
| `qk_norm_rope_kv`: 288 launches -> one per layer (about 1.6 ms saved, almost all launch cost) | 3.9% | 3.4% |
| **ceiling of any fused / ragged attention, timeline as measured** | **about 14%** (5.8 ms) | **about 8%** (3.9 ms) |
| same, if all of the 2.4 ms profiler overhead (4.2 us per launch) sits in the 576 per-row launches; denominators 38.6 ms (1k) and about 44.5 ms (4k) | 3.5 ms = **9%** | 1.7 ms = **4%** |

The design doc's own model (fused k = 8 launch costs 3.63 / 4.01 x one row at 1k / 4k, from G0b) would save more than the 4k floor allows (it implies 920 GB/s, which needs the L2 sharing of rows of one sequence; with distinct KV that is not available), so the floor is the bound used here. Single-row attention is also slower in this measurement than the G0b chain figures (23 us vs 14.5 us at 1k; B = 1 is 28 us) which the design doc flagged as cache-warm.

## 4. M4: the misaligned path

Design doc 4.6 and the shelved branch `metal-weight-realign` (8b83805): Qwen3-8B shard 1 data starts at byte 9,336 (8 mod 16), so the first layers' weights take the 8-byte-load multi-row GEMV at R = 8; the remedy copies them to 16-byte aligned buffers. The per-launch traces show it directly: at B = 8 the gate+up launches of layers 0-6 take 480 us against 398 us for layers 7-35, down 237 vs 211, q/o (layers 0-7) 80 vs 68, k+v 48 vs 39 (`run2/align-*.txt`, `control-ce1/align-ce1-*.txt`).

| trace | step | excess of the slow launches over the aligned median | share | aggregate gain if all removed |
|---|---:|---:|---:|---:|
| kernel mode run1, 1k | 39.4 ms | 1.024 ms | 2.60% | 2.67% |
| kernel mode run2, 1k | 39.4 ms | 0.983 ms | 2.50% | 2.56% |
| commit-every-1, 1k | 41.06 ms | 1.064 ms | 2.59% | 2.66% |
| kernel mode run2, 4k | 44.9 ms | 0.979 ms | 2.18% | 2.23% |
| commit-every-1, 4k | 46.86 ms | 1.070 ms | 2.28% | 2.34% |

Against the normal-stream 38.6 ms step (1k) the three 1k excesses are 2.55%, 2.65%, 2.76% (aggregate +2.6%, +2.7%, +2.8%).

Per class at 1k (commit-every-1): gate+up 0.58 ms, down 0.20, q/o 0.21, k+v 0.08. This matches the design doc's 5.03 -> 3.99 ms (-1.04 ms) and the branch's measured T(8)/T(1) 1.197 -> 1.152. Share at lower GPU-clock states was not measured (no per-launch trace exists there; the R = 8 cliff is an instruction-count effect, so it need not scale like bandwidth-bound time). Against the normal-stream step the same excesses are 2.55-2.76% of 38.6 ms at 1k (aggregate +2.6-2.8%) and, with an estimated 44.5 ms step, 2.2-2.4% at 4k (aggregate +2.25-2.5%). The numbers are ceilings: the realigned kernel is 0.7-2.5% slower per launch than the a8 kernel on the same shape.

Facts about the existing prototype from its own report (`metal-weight-realign:docs/results/2026-10-15-metal-weight-realign.md`), not re-measured here: +3.82 GB non-evictable memory (not 2.75 GB), load +0.55 s, and the aligned kernels are not bit-identical to the 8-byte ones (one bf16 ulp on 2 of 512 random outputs in its unit test), so enabling it changes batch-1 decode bits; the exact-only policy (decision Q1) is affected.

## 5. Caveat: B = 8 step time depends on the GPU clock state

The same binary gave very different B = 8 step times in different slots, with B = 1 unchanged (28 ms):

| run | B = 8, 1k step | GPU clock during the step (macmon, mean) | GPU power |
|---|---:|---:|---:|
| `control-ce1`, `batch_step` (3 runs, new and old binary alternating) | 38.6 ms | median 1362 MHz over the whole slot | |
| `commit-every-1` | 41.6 ms | 1610 MHz | 58 W |
| `run2` plain, run 1 / 2 / 3 | 43.4 / 48.5 / 48.6 ms | 1498 / 1195 / 1223 MHz | 51 / 28 / 29 W |
| `run1` plain (first slot) | 56-59 ms | 970-1050 MHz | 18-21 W |
| kernel mode | (160+ ms, host waits) | 1617 MHz | 21 W |

`batch_step` inside the `run2` slot gave B = 8 57.5 ms (1k) and 65.1 ms (4k) with B = 1 at 28 ms. The control slot (`run-control.sh`) re-ran today's `batch_step` against an earlier scratchpad binary (`batch_step-v3`, Oct 15 16:58, provenance not verified): 38.51 / 38.61 / 38.65 vs 38.59 / 38.59 / 38.58 ms, B = 1 27.5 ms for both, and matching the M1 step 1 report (38.8 ms), so there is no engine regression. The slow B = 8 steps coincide with a lower GPU clock (1000-1500 MHz instead of 1610) and lower GPU power, and the GPU timeline is continuous in those runs (GPU span equals busy, no host gaps), but the cause is not known: low clock may be a symptom of something else (memory pressure: endor showed about 10.9 GB of swap in use in a macmon sample; other agents' work on the machine, whose processes were not serialised against mine outside fleetq). The observation is kept; the shares in section 2 come from the traces taken at 1610-1617 MHz, and how the shares move in a slow state was not measured. The `run1` / `run2` plain and `cb` step times (43-63 ms) should not be compared with the 38.6 ms figure.

## 6. Commands and slots (all endor, `--exclusive --mem 24`)

* Build: `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer cargo build --release -p intarsia-models --example batch_prof --example batch_step` (binaries copied to the scratchpad `bin-m45/`). The first slot (before this doc's data) used a build with `DEVELOPER_DIR` set to the Command Line Tools, which compiles an empty metallib; every run failed at once and that output was deleted (the failed fleetq job produced no data).
* `run.sh` (plain B = 1, 8 x 3; `cb:` and `kernel:` traces B = 1, 8; `batch_step`; 1k and 4k): fleetq #2204 (build without metallib, no data, output deleted), #2206 (`run1/`), #2225 (`run2/`). `run-control.sh`: #2213. `run-ce1.sh`: #2215. macmon logged at 0.25-0.5 s (`macmon*.jsonl`).
* Analysis: `analyze.py run2`, `analyze_ce1.py control-ce1`, `analyze_align.py TRACE MARKS STEP_MS` (outputs `analysis*.txt`, `align-*.txt` next to the data).

## 7. Limits

* Per-launch kernel durations carry a fixed command-buffer cost (1-3 us) that depends on the launch; the ranges in section 3 bound it. No in-stream ablation was done.
* The M5 ceiling assumes a fused kernel reaches the GEMV bandwidth; it will not. No fused kernel was built or measured.
* The M4 figures assume the 16-byte kernel is as fast as it is on layers that already use it; the existing branch's end-to-end serve numbers were not rerun.
* One machine, one model, random prompts, greedy decode; B = 8 only (B = 1 for contrast).
* While working, a scratchpad binary of the shared session directory (`.../scratchpad/bin/batch_step`) was overwritten by this task's build of the same example (66feda6); it was replaced with a working build of the same source, `batch_prof` was removed from there.
