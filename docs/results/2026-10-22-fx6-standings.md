# Standings 2026-10-22 (lane fx6): intarsiaml against the other engines, restated honestly

Supersedes `docs/results/2026-10-21-standings.md` (kept for the record, with a pointer added). Lane fx6 answers findings 3 and 6 of the
2026-10-22 independent review (requirement 4, "aim for the top 3 on the charts for each supported platform"). Measurement only: no
engine code changed. Base `origin/main` cd617de2 (PR #70 merged). Raw data, scripts and result JSON: `docs/results/2026-10-22-fx6-standings/raw/`,
scripts `scripts/standings-fx6/`.

## 0. Read this first

Every number in this document is **single stream** (one request at a time, concurrency 1) unless a row says otherwise. No
concurrency or continuous-batching comparison was measured, so no rank here says anything about serving load. Where the GPU count
matters it is in the column or caption. Ranks are by median; two engines are called **tied** when the min-max bands of their visit
medians overlap. "Top 3" is only a test where the field has 3 or more engines, so every platform table states the number of
engines `N`.

| platform | model | GPUs | N engines measured live | what survives | what does not |
|---|---|---:|---:|---|---|
| R9700 (tatooine, HIP) | Qwen3.6-35B-A3B | 1 | **4** (intarsia, llama.cpp Vulkan, llama.cpp HIP, hipfire) | intarsia **#1 of 4** on prefill (1.9-6.0x) and on decode (1.38x llama.cpp Vulkan, 1.91x llama.cpp HIP). Single stream, 1 GPU, four different 4-bit formats | The decode lead over hipfire (1.10x) **depends on hipfire running 27 % below its own 2026-09-26 result** (same container, config and script, cause unknown): against its 09-26 figure (202-204 tok/s) hipfire would be ahead. "Second behind hipfire 204" is withdrawn only in the sense that 204 does not reproduce today |
| Metal (endor, M5 Max) | Qwen3-8B BF16 | 1 | **3** (intarsia, mlx-lm, llama.cpp) | prefill: ahead of llama.cpp 1.4-1.7x, ahead of mlx-lm at 1k and tied at 4k and 16k. Decode: ahead of mlx-lm on raw medians (1.12-1.14x); clock-normalised the lead is a tie at 128 and 1k and 1.03-1.10x at 4k and 16k. Single stream, 1 GPU | "#1 of 3 on decode at every length": against llama.cpp the margin is 1.04-1.08x raw, 1.08-1.12x reference-normalised and 0.99-1.02x clock-matched by MHz, so the rank against llama.cpp is method-dependent: a tie |
| Spark (GB10, CUDA) | Qwen3.6-35B-A3B NVFP4 | 1 | **2** (intarsia, vLLM) | 4k and 16k default prefill is behind vLLM (-9.0 % / -15.3 %); decode is a tie. Single stream, 1 GPU. Carried forward, not re-measured here | "#1 of 2 on decode at 1k and 4k": +5.7 % and +2.1 % are inside the ~5 % decode noise |

Counts of engines: with `N = 2` on Spark a rank is just "ahead" or "behind". Only the R9700 has a field of 3 or more besides intarsia
and only Metal has exactly 3 engines in total.

## 1. Method

`localeval` (`/opt/code/deepeval`, 0.3.0), `speed` sweep `--sizes 128,1024,4096,16384 --gen 128 --reps 3 --thinking off`, nonce-first
prompts (no prefix-cache hit). Prefill is localeval's per-request "prefill tok/s" (uncached prompt tokens / TTFT); the 1k prompt has
990 tokens, under localeval's 1024 threshold, so its per-request prefill includes request overhead. The marginal column (delta
tokens / delta TTFT) is the cleanest prefill figure and is shown too. Decode is tok/s from server-reported token counts.

**Interleaving and noise band (R9700).** One visit = one server start + 3 reps (first rep discarded as warmup). Each visit contributes
its median; a cell is the **median of the visit medians** and the band is **min-max over visits**, n = number of visits. Seven
measurement passes (p1-p7) were used, each in one radiance window, plus two aborted windows (section 5). Passes p1 to p4 and p6 visited all arms in rotated order (p1 intarsia, llama-vk,
hipfire, llama-hip; p2 reversed; p3 llama-vk, intarsia, llama-hip, hipfire, llama-vk4; p4 intarsia-old, intarsia, llama-vk;
p6 llama-vk4, hipfire, llama-hip, intarsia, llama-vk); p5 and p7 were diagnostic passes with a 128-token (p7: 128 and 1k) sweep, so
n differs by column. Radiance was down on both cards for the whole of each window; all arms ran on HIP/Vulkan device 0, which is sysfs `card1` in every arm (the GPU samples in `raw/hip/<pass>/<arm>-<pass>-gpu.txt`
show the VRAM on card1 and card0 idle), with the previous engine's server stopped before the next one started.

**Clock-normalised method (Metal).** `scripts/metal-speed-gate/sameslot.sh` (docs/benchmarking.md "Metal speed gate") runs
`refbw.py` (a bf16 matrix-vector stream, independent of any engine) before and after every arm. Machine-state factor = arm
reference GB/s / 554.2 GB/s (`ref-baseline.json`); normalised decode = raw decode / factor; factors outside 0.70-1.30 are
STATE-INVALID. **It is applied to every arm here, not only intarsia**; the earlier standings used raw numbers for
mlx-lm and llama.cpp. Prefill is reported raw only: the reference is a memory-bound GEMV and dividing a compute-bound prefill by it
would be unfounded. A second, MHz-based sensitivity is given because the reference cannot see a per-process clock difference.

## 2. Platform tables

### 2.1 R9700 (tatooine, HIP device 0, **1 GPU**, single stream): Qwen3.6-35B-A3B, N = 4 engines

Engines and builds, all on the same card in the same session (`raw/hip/`):

| engine | build | quant / checkpoint | note |
|---|---|---|---|
| intarsia HIP | origin/main cd617de2, `--max-seq 17408`, container `intarsia-hip-dev` | NVFP4 (`Qwen3.6-35B-A3B-NVFP4`) | defaults |
| llama.cpp Vulkan | v0.5.0 (`0.5.0-dev`, build 1, commit 7fe450e), RADV, `-ngl 999 -fa on -c 20480 --cache-ram 0 -np 1` | MXFP4_MOE GGUF | the v0.5.0 build that exists on the box; `llama.cpp-v0.5.0-vk2604` needs glibc 2.43 and does not run on the host |
| llama.cpp HIP | the same v0.5.0 source built for HIP (ROCm 7.2.1), same flags | MXFP4_MOE GGUF | loads: see below |
| hipfire | `ad10b3d`, `hipfire:ad10b3d` container, MQ4R, `hipfire serve` sequential mode, reasoning and speculation off | MQ4R (`qwen3.6-35b-a3b.mq4r`) | live, not second-hand |
| llama.cpp Vulkan (old) | `0.4.0-dev` build 10864, the comparator of 2026-10-21 | MXFP4_MOE | 2 visits, continuity only |

The four engines use four different 4-bit formats (NVFP4+fp8, MXFP4, MQ4R); no common quant exists. That is a property of what each
engine can load, not a fairness fix.

**Decode, tok/s** (median of visit medians, min-max over visits)

| engine | n visits | 128 | 1k | 4k | 16k |
|---|---:|---|---|---|---|
| **intarsia HIP** | 5-6 | 159.3 (159.2-159.7) | 159.1 (158.9-159.5) | 155.5 (155.4-155.9) | 141.6 (141.4-141.6) |
| hipfire MQ4R | 4-6 | 146.4 (145.5-146.6) | 145.2 (145.1-146.2) | 141.9 (141.2-142.6) | 127.6 (127.1-127.9) |
| llama.cpp v0.5.0 Vulkan | 5-6 | 115.4 (114.7-115.6) | 115.3 (114.7-115.5) | 112.0 (111.6-112.5) | 106.7 (106.0-106.8) |
| llama.cpp v0.5.0 HIP | 4 | 83.3 (83.0-83.4) | 83.3 (83.0-83.3) | 81.6 (81.5-81.7) | 77.7 (77.7-77.8) |
| llama.cpp 0.4.0-dev Vulkan (old) | 2 | 113.1 (113.1-113.1) | 112.7 (112.6-112.9) | 110.3 (110.3-110.4) | 105.3 (105.1-105.4) |

**Prefill, tok/s per request** (1k prompt is 990 tokens: overhead-influenced) and marginal prefill

| engine | 1k | 4k | 16k | marginal 1k / 4k / 16k | TTFT 1k / 4k / 16k (s) |
|---|---|---|---|---|---|
| **intarsia HIP** | 5753 (5657-5824) | 9106 (9068-9122) | 7739 (7725-7783) | 9816 / 11214 / 7379 | 0.17 / 0.45 / 2.15 |
| llama.cpp v0.5.0 HIP | 2989 (2983-3027) | 3382 (3373-3394) | 3235 (3227-3238) | 3864 / 3527 / 3188 | 0.33 / 1.21 / 5.15 |
| llama.cpp v0.5.0 Vulkan | 2796 (2784-2817) | 3198 (3183-3200) | 2967 (2963-2972) | 4079 / 3338 / 2902 | 0.35 / 1.28 / 5.61 |
| hipfire MQ4R | 2762 (2754-2800) | 2343 (2341-2346) | 1283 (1283-1284) | 3195 / 2236 / 1118 | 0.36 / 1.75 / 12.98 |
| llama.cpp 0.4.0-dev Vulkan (old) | 2540 (2536-2544) | 2865 (2861-2869) | 2685 (2680-2691) | 3722 / 2986 / 2632 | 0.39 / 1.43 / 6.21 |

**Ratios of intarsia to each engine** (medians; no band overlaps anywhere in this table, so no cell is a tie):

| engine | decode 1k / 4k / 16k | prefill 1k / 4k / 16k |
|---|---|---|
| hipfire MQ4R | 1.10x / 1.10x / 1.11x | 2.08x / 3.89x / 6.03x |
| llama.cpp v0.5.0 Vulkan | 1.38x / 1.39x / 1.33x | 2.06x / 2.85x / 2.61x |
| llama.cpp v0.5.0 HIP | 1.91x / 1.91x / 1.82x | 1.92x / 2.69x / 2.39x |
| llama.cpp 0.4.0-dev Vulkan (old, 2026-10-21 comparator) | 1.41x / 1.41x / 1.35x | 2.26x / 3.18x / 2.88x |

**Rank: intarsia #1 of 4 on decode and on prefill at 1k, 4k and 16k**, single stream, one GPU. The noise band is under 2 % for decode and
under 1.5 % for prefill across 4-6 visits, far smaller than any gap above. What this does not show: quality (the comparison was
speed-only; the 2026-10-21 MMLU smoke was run on the pre-J1 binary and is not repeated), concurrency, or an equal-quant comparison.

What changed against the 2026-10-21 standings, and why each open point is closed:

1. **llama.cpp v0.5.0 Vulkan** was measured: the old `0.4.0-dev` comparator was not handicapping the result. intarsia's decode margin
   is 1.38x against v0.5.0 versus 1.41x against 0.4.0-dev, prefill 2.1-2.9x versus 2.3-3.2x.
2. **llama.cpp HIP loads**, and is the slowest decode of the four (83 tok/s, 0.52x intarsia). Cause of the earlier "unloadable": the
   binary links `libhipblas.so.3` and `librocblas.so.5`, which neither `intarsia-hip-dev` nor `rocm/dev-ubuntu-24.04:7.2.4` contains,
   and the image it was built in (`r9700-rocm-build`) no longer exists on the box. They are in AMD's apt repo that the ROCm image already
   points at; `scripts/standings-fx6/Dockerfile.rocm-llama` adds `rocblas hipblas` (7.2.4 runtime for a 7.2.1 build; the sonames match) on top
   of the stock image as `fx6-rocm-llama:7.2.4`, and the binary runs unchanged with `LD_LIBRARY_PATH=/lcpp:/opt/rocm/lib`.
3. **hipfire was measured live** (image, config and model files were still on the box; recipe from the 2026-09-26 session): **146 tok/s
   decode**, not the 204 of the second-hand figure, and prefill 2343 tok/s at 4k and 1283 at 16k (the 09-26 figures were 2143 at 6k and 1235
   at 18k, so prefill reproduces and decode does not). To rule out the harness, the 09-26 `stream_bench.py` (same
   `(tokens-1)/(last-first)` decode formula as localeval) was run against the same hipfire server: 147.5 tok/s at 830 prompt tokens,
   141.5 at 6k, 127.7 at 18k (`raw/hip/p7/`). The 09-26 figures were 202 / 198 / 173 with that script, so hipfire is about 27 % slower
   today with the same script and the same container. Cause not found. Ruled out: a cold kernel cache (pass p8, `raw/hip/p8/hipfire-p8-server.log`: pre-compiled kernels loaded from `/var/cache/hipfire/gfx1201`, no fallback message) and a localeval artefact (two harnesses agree). The server log's own pre-warm line reports 134.9 tok/s decode on this boot, and logs `[redline] enabling fail-closed retained default on gfx1201 ... transport=pm4`; whether that mode was on in the 09-26 run is not known. Other candidates: GPU or driver state, other lanes' use of the box. **The "intarsia is second behind hipfire" line is withdrawn.**
4. **Pre-J1 binary of 2026-10-21 against the new one:** decode 159.6 / 158.6 / 155.3 / 141.5 versus 159.3 / 159.1 / 155.5 / 141.6, prefill
   within 1 %. Structured output (J1) cost no throughput.

### 2.2 Metal (endor, M5 Max, **1 GPU**, single stream): Qwen3-8B BF16, N = 3 engines

intarsia (md5 `bf72ec9ecb5e7e381d797f90d24c5479`, built from this tree, same Metal code as cd617de2), mlx-lm 0.31.3, llama.cpp
(BF16 GGUF, `-ngl 999 -fa on -c 20480`). `sameslot.sh` in one exclusive endor slot per run, gen 256, 3 reps, `--max-seq 20000`.
Visit orders: run 1 intarsia, mlx-lm, llama.cpp, intarsia; run 2 llama.cpp, mlx-lm, intarsia, mlx-lm, llama.cpp
(fleetq #3821). Cells are the median of the visit medians over n = 3 visits per engine, with the min-max band. The tables are pasted from `raw/metal/aggregate.md`, which `scripts/standings-fx6/aggregate-metal.py raw/metal` regenerates from the committed run directories.

**Decode tok/s, raw** - median over visits (min-max), n visits per cell

| engine | n | 128 | 1024 | 4096 | 16384 |
|---|---:|---:|---:|---:|---:|
| intarsia | 3 | 35.3 (33.8-36.3) | 34.0 (33.2-36.4) | 32.8 (31.0-33.7) | 29.2 (28.3-30.1) |
| mlx-lm | 3 | 31.1 (28.2-31.6) | 30.3 (29.4-31.3) | 29.0 (28.3-30.5) | 25.6 (24.6-26.2) |
| llama.cpp | 3 | 32.8 (29.3-33.1) | 32.1 (30.6-32.7) | 30.6 (28.5-31.4) | 28.0 (27.9-28.0) |

Order (~ = bands overlap, a tie): 128: intarsia > llama.cpp ~ mlx-lm; 1024: intarsia > llama.cpp ~ mlx-lm; 4096: intarsia ~ llama.cpp ~ mlx-lm; 16384: intarsia > llama.cpp > mlx-lm

**Decode tok/s, clock-normalised (raw / state factor, every engine)** - median over visits (min-max), n visits per cell

| engine | n | 128 | 1024 | 4096 | 16384 |
|---|---:|---:|---:|---:|---:|
| intarsia | 3 | 37.6 (34.8-38.8) | 36.5 (35.1-37.6) | 34.1 (33.9-34.9) | 31.1 (30.2-31.1) |
| mlx-lm | 3 | 34.3 (29.3-37.0) | 34.6 (30.5-35.5) | 33.2 (30.1-33.6) | 28.2 (27.2-28.8) |
| llama.cpp | 3 | 33.6 (30.6-34.1) | 32.9 (31.9-33.7) | 31.6 (29.7-32.2) | 28.8 (28.6-29.3) |

Order (~ = bands overlap, a tie): 128: intarsia ~ mlx-lm ~ llama.cpp; 1024: intarsia ~ mlx-lm ~ llama.cpp; 4096: intarsia > mlx-lm ~ llama.cpp; 16384: intarsia > llama.cpp ~ mlx-lm

**Sensitivity: clock-matched by MHz** (decode raw median + 0.014 tok/s per MHz x (intarsia median busy MHz - engine median busy MHz); the slope was measured on intarsia only, docs/results/2026-10-21-g3-metal.md section 2, so for the other engines it is an assumption)

| engine | median busy GPU MHz | decode 128 | decode 1024 | decode 4096 | decode 16384 |
|---|---:|---:|---:|---:|---:|
| intarsia | 1184 | 35.3 | 34.0 | 32.8 | 29.2 |
| mlx-lm | 1180 | 31.1 | 30.4 | 29.1 | 25.6 |
| llama.cpp | 1064 | 34.4 | 33.8 | 32.3 | 29.6 |

**Prefill tok/s, raw (per request)** - median over visits (min-max), n visits per cell

| engine | n | 1024 | 4096 | 16384 |
|---|---:|---:|---:|---:|
| intarsia | 3 | 3124 (2965-3978) | 2165 (2043-3899) | 1991 (1834-2100) |
| mlx-lm | 3 | 1895 (1760-1940) | 1799 (1594-2256) | 1692 (1571-1858) |
| llama.cpp | 3 | 1999 (1448-2158) | 1535 (1480-1555) | 1206 (1199-1280) |

Order (~ = bands overlap, a tie): 1024: intarsia > llama.cpp ~ mlx-lm; 4096: intarsia ~ mlx-lm > llama.cpp; 16384: intarsia ~ mlx-lm > llama.cpp

**Marginal prefill tok/s, raw** - median over visits (min-max), n visits per cell

| engine | n | 1024 | 4096 | 16384 |
|---|---:|---:|---:|---:|
| intarsia | 3 | 3382 (3178-4457) | 1972 (1861-3876) | 1827 (1775-1940) |
| mlx-lm | 3 | 3399 (3344-3440) | 1770 (1547-2382) | 1727 (1509-1758) |
| llama.cpp | 3 | 2222 (1533-2400) | 1429 (1427-1489) | 1137 (1118-1211) |

Order (~ = bands overlap, a tie): 1024: mlx-lm ~ intarsia > llama.cpp; 4096: intarsia ~ mlx-lm > llama.cpp; 16384: intarsia > mlx-lm > llama.cpp

**Ratios intarsia / engine** (medians; `tie` = the min-max bands of the visit medians overlap)

| engine | basis | 128 | 1024 | 4096 | 16384 |
|---|---|---:|---:|---:|---:|
| mlx-lm | decode raw | 1.14x | 1.12x | 1.13x | 1.14x |
| mlx-lm | decode clock-normalised (reference) | 1.10x tie | 1.05x tie | 1.03x | 1.10x |
| mlx-lm | decode MHz-matched (sensitivity) | 1.13x | 1.12x | 1.13x | 1.14x |
| mlx-lm | prefill raw | - | 1.65x | 1.20x tie | 1.18x tie |
| mlx-lm | marginal prefill raw | - | 0.99x tie | 1.11x tie | 1.06x |
| llama.cpp | decode raw | 1.08x | 1.06x | 1.07x tie | 1.04x |
| llama.cpp | decode clock-normalised (reference) | 1.12x | 1.11x | 1.08x | 1.08x |
| llama.cpp | decode MHz-matched (sensitivity) | 1.02x | 1.01x | 1.02x | 0.99x |
| llama.cpp | prefill raw | - | 1.56x | 1.41x | 1.65x |
| llama.cpp | marginal prefill raw | - | 1.52x | 1.38x | 1.61x |

**Reading the Metal tables**

1. **Prefill.** intarsia is ahead of llama.cpp at every length with non-overlapping bands (1.4-1.7x raw). Against mlx-lm it is ahead at 1k
   (1.65x per request, but the marginal figures tie: the per-request 1k figure carries mlx-lm's larger request overhead) and **tied at
   4k and 16k** (1.2x and 1.18x with overlapping bands; marginal 1.11x and 1.06x). The 2026-10-21 "+27 % at 4k" is not reproduced as a
   robust lead: in this session intarsia's 4k prefill ranged 2043-3899 tok/s across its visits (about 90 %), the run-1 visit at the highest busy clock (1276 MHz)
   matches yesterday (3899 against 3917), the others sit near 2000-2200. Prefill is compute-bound and clock- and contention-sensitive and
   the reference cannot normalise it, so read Metal prefill cells with a wide band. Lead over llama.cpp: robust. Lead over mlx-lm: not
   established at 4k and 16k.
2. **Decode against mlx-lm.** Ahead on raw medians by 12-14 % with separated bands at every length; after the reference normalisation
   1.03-1.10x (tie at 128 and 1k, a 3 % separation at 4k that is below the ~5 % decode noise, 10 % at 16k). Ahead, with the size of the lead
   depending on the method.
3. **Decode against llama.cpp: not established.** Raw medians: 1.04-1.08x (4k tie). Reference-normalised: 1.08-1.12x. But the reference runs between
   arms and cannot see a per-process clock: llama.cpp's busy-GPU clock is about 120 MHz below intarsia's in every run (median 1064 against 1184
   MHz), the same difference as on 2026-10-21. Matching the clocks with the documented slope (0.014 tok/s per MHz, an intarsia-only
   measurement used as an assumption for llama.cpp) gives **0.99-1.02x, a tie**. The two corrections disagree and neither is a measurement of
   llama.cpp at intarsia's clock. Only the reference-normalised figure (1.08-1.12x) is clearly above the roughly 5 % decode noise; raw (1.04-1.08x) is at it and MHz-matched (0.99-1.02x) is inside it. **"#1 of 3 on decode
   at every length" is withdrawn; the defensible statement is "ahead of mlx-lm, level with llama.cpp".**
4. **Same-slot gate.** `gate.py` verdicts: run 1 FAIL (4k: intarsia-1 / llama.cpp 0.988; that visit ran at 1122 MHz and CPU 0.54), run 2 PASS.
   The floor gate (normalised decode against the 85 % context-aware floor) passed in both runs on the best valid intarsia arm.
5. Machine-state factors were 0.85-0.98 (all within 0.70-1.30, no arm STATE-INVALID); one mlx-lm visit ran at 0.854, which is why the raw
   mlx-lm figure for that visit differs from the normalised one by 17 %.

Per visit (the data behind the cells):

**Per visit**

| visit | ref GB/s | factor | busy GPU MHz | decode 128 raw / norm | decode 1024 raw / norm | decode 4096 raw / norm | decode 16384 raw / norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| run1/intarsia-1 | 504.5 | 0.910 | 1122 | 35.3 / 38.8 | 33.2 / 36.5 | 31.0 / 34.1 | 28.3 / 31.1 |
| run1/intarsia-2 | 535.5 | 0.966 | 1276 | 36.3 / 37.6 | 36.4 / 37.6 | 33.7 / 34.9 | 30.1 / 31.1 |
| run1/llamacpp | 541.3 | 0.977 | 1064 | 32.8 / 33.6 | 32.1 / 32.9 | 31.4 / 32.2 | 27.9 / 28.6 |
| run1/mlx-lm | 534.2 | 0.964 | 1180 | 28.2 / 29.3 | 29.4 / 30.5 | 29.0 / 30.1 | 26.2 / 27.2 |
| run2/intarsia-1 | 537.0 | 0.969 | 1184 | 33.8 / 34.8 | 34.0 / 35.1 | 32.8 / 33.9 | 29.2 / 30.2 |
| run2/llamacpp-2 | 531.0 | 0.958 | 1014 | 29.3 / 30.6 | 30.6 / 31.9 | 28.5 / 29.7 | 28.0 / 29.3 |
| run2/llamacpp | 537.7 | 0.970 | 1070 | 33.1 / 34.1 | 32.7 / 33.7 | 30.6 / 31.6 | 28.0 / 28.8 |
| run2/mlx-lm-2 | 473.1 | 0.854 | 1177 | 31.6 / 37.0 | 30.3 / 35.5 | 28.3 / 33.2 | 24.6 / 28.8 |
| run2/mlx-lm | 501.8 | 0.905 | 1206 | 31.1 / 34.3 | 31.3 / 34.6 | 30.5 / 33.6 | 25.6 / 28.2 |

**Per visit, prefill tok/s raw (marginal), unnormalised**

| visit | busy GPU MHz | prefill 1024 | prefill 4096 | prefill 16384 |
|---|---:|---:|---:|---:|
| run1/intarsia-1 | 1122 | 2965 (3178) | 2043 (1861) | 1834 (1775) |
| run1/intarsia-2 | 1276 | 3978 (4457) | 3899 (3876) | 2100 (1827) |
| run1/llamacpp | 1064 | 1448 (1533) | 1480 (1489) | 1206 (1137) |
| run1/mlx-lm | 1180 | 1760 (3344) | 1594 (1547) | 1692 (1727) |
| run2/intarsia-1 | 1184 | 3124 (3382) | 2165 (1972) | 1991 (1940) |
| run2/llamacpp-2 | 1014 | 1999 (2222) | 1535 (1429) | 1199 (1118) |
| run2/llamacpp | 1070 | 2158 (2400) | 1555 (1427) | 1280 (1211) |
| run2/mlx-lm-2 | 1177 | 1895 (3440) | 1799 (1770) | 1571 (1509) |
| run2/mlx-lm | 1206 | 1940 (3399) | 2256 (2382) | 1858 (1758) |

### 2.3 Spark (GB10, CUDA, **1 GPU**, single stream): Qwen3.6-35B-A3B NVFP4, N = 2 engines (carried forward, not re-measured)

No Spark measurement was taken in this lane (vLLM serves the box and a second engine does not fit next to it). The figures are the round 6
and round 7 pairs of `2026-10-21-standings.md` section 2.1; the interpretation is restated against the noise. Noise used: decode about
±5 % and long-prompt prefill about ±3 % (unanimis 6a7efe40, localeval's own 10 % rule for speed deltas), vLLM's own day-to-day drift between
10-19 and 10-21 was +10 % at 1k (overhead-dominated) and within 2 % at 4k and 16k, and a same-GPU run on 10-21 (Q1) read decode 75.7 against 76.4
at 1k (behind) and 81.9 against 76.7 at 4k (ahead).

| metric (intarsia default round 6 against same-evening vLLM) | delta | verdict at the stated noise |
|---|---|---|
| prefill 1k (990 tokens) | -8.3 % | inside the 11 % drift of that column: not resolved |
| prefill 4k | -9.0 % | outside noise: behind |
| prefill 16k | -15.3 % | outside noise: behind (the robust Spark finding) |
| decode 1k / 4k / 16k | +5.7 % / +2.1 % / -4.2 % | all within ±5 %: **tie at every length** |

Opt-in rows (round 7 `INTARSIA_CUDA_FP8_GEMM_LT=1`, `--prefill-chunk 4096`) are not the shipped default, fail the prefix-reuse gate
(`chat_reuse_4k` continue 3/32), and the lt + chunk 4096 pairing tunes only our side of a comparison against vLLM at its defaults. They
are not standings. Quality on Spark was later measured (`2026-10-21-q1-spark-quality.md`, MMLU n=1710: 0.8386 against 0.8351, tie; GSM8K n=1319:
0.2403 against 0.2813, with the harness caveat of LOG O10).

**Rank: with N = 2, intarsia is *behind* vLLM on 4k and 16k prefill and *tied* on decode.** There is no "#1 of 2 on decode".

**tp=2 (two GPUs).** `2026-10-21-tp2-cuda-graphs.md` and `2026-10-21-35b-tp2-cuda.md` compare intarsia on **two** GB10s (spark1 + spark2 over
RoCE) with vLLM on **one**; no vLLM tp=2 arm exists. The tp=2-over-tp=1 numbers are sound, the "faster than vLLM" rows are 2 GPUs against 1
and are not part of any rank here.

## 2.4 Rank summary, every cell with its mode

Mode of every cell below: **single stream (concurrency 1)**, GPU count as in the column header. "tie" = the min-max bands of the visit
medians overlap, or the lead is below the ~5 % decode noise. N is the number of engines compared.

| metric | R9700, 1 GPU, N = 4 | Metal raw, 1 GPU, N = 3 | Metal clock-normalised (decode), 1 GPU, N = 3 | Spark, 1 GPU, N = 2 (carried forward) |
|---|---|---|---|---|
| prefill 1k (990 tokens, overhead-influenced) | 1 of 4 (1.9x the next engine) | 1 of 3 per request; marginal tie with mlx-lm | not normalised | not resolved (-8.3 %, inside the 1k drift) |
| prefill 4k | 1 of 4 (2.7x the next) | 1 of 3 vs llama.cpp (1.41x); tie with mlx-lm | not normalised | 2 of 2 (-9.0 %) |
| prefill 16k | 1 of 4 (2.4x the next) | 1 of 3 vs llama.cpp (1.65x); tie with mlx-lm | not normalised | 2 of 2 (-15.3 %) |
| decode 1k | 1 of 4 (1.10x hipfire, which ran 27 % below its 09-26 result) | 1 of 3 (1.06x llama.cpp; 1.12x mlx-lm) | 1.05x mlx-lm (tie), 1.11x llama.cpp (MHz-matched 1.01x: tie) | tie (+5.7 %) |
| decode 4k | 1 of 4 (1.10x hipfire) | tie with llama.cpp (1.07x), ahead of mlx-lm (1.13x) | 1.03x mlx-lm (below noise), 1.08x llama.cpp (MHz-matched 1.02x: tie) | tie (+2.1 %) |
| decode 16k | 1 of 4 (1.11x hipfire) | 1 of 3 on the median (1.04x llama.cpp, inside the noise; 1.14x mlx-lm) | 1.10x mlx-lm, 1.08x llama.cpp (MHz-matched 0.99x: tie) | tie (-4.2 %) |
| quality | not re-measured | not re-measured | n/a | tie MMLU, -0.04 GSM8K (Q1, harness caveat) |

Read the Metal columns together: the same visits give "#1" on raw medians and a tie once the machine state is accounted for, and the engines'
different clocks during their own runs (llama.cpp about 120 MHz lower) are exactly what the reference cannot see.

## 3. The 8x MMLU latency (finding 3)

Observation of 2026-10-21 on the R9700: intarsia 2.8-3.0 s per MMLU call against llama.cpp's 0.37 s. Cause established in about 25
minutes, with three controls on the same card in one window (p4, p5; `raw/hip/p4/`, `raw/hip/p5/`). 8 MMLU questions
(`high_school_physics`, `virology`, 4 each), localeval defaults (`--structured json_schema`: deepeval sends
`response_format: json_schema` `MultipleChoiceSchema`, `{"answer": "A".."D"}`), `--thinking off`, greedy:

| server | `--structured` | calls for 8 questions | completion tokens median (max) | s per call | MMLU score |
|---|---|---:|---|---:|---|
| intarsia, 2026-10-21 standings binary (pre-J1) | json_schema | **16** | 120 (1024) | **1.92** | 0.75 |
| intarsia, cd617de2 (J1 merged) | json_schema | 8 | 68 (209) | 0.71 | 0.75 |
| llama.cpp v0.5.0 Vulkan | json_schema | 8 | 12 (19) | 0.41 | 0.75 |
| all three | none (plain prompt) | 8 | 2 (2) | 0.14-0.28 | 0.75 |

1. **Cause of the 8x: the 2026-10-21 binary ignored `response_format`.** Direct probe (`raw/hip/p5/*-probe.jsonl`) with the
   localeval request shape: the old build answers with 160-240 tokens of step-by-step prose; deepeval cannot parse it, retries
   (hence 16 calls for 8 questions) and sometimes runs to the 1024-token cap. At 160 tok/s that is the observed 2-3 s per call, against 12 tokens on
   llama.cpp. With `--structured none` (2-token letter answers) there is no gap, which also rules out TTFT, the chat template and thinking
   mode. J1 (PR #50, 04c4d405) merged on 10-21 afternoon (first commit 14:27), hours after the HIP run (10:21); 789511c2 does not contain it. It is the same mechanism as
   the Metal GSM8K 0.0 of the old section 10, which was a server feature gap and not only a harness fault.
2. **A smaller gap remains on the new build (0.71 against 0.41 s/call, 1.7x).** `response_format` is now honoured, but the schema has
   `required: ["answer"]` and no `additionalProperties: false`, which JSON Schema reads as "extra keys allowed". 2 of 3 direct probes returned
   `{"answer": "C", "reasoning": "To determine ..."}` with 165-234 tokens; llama.cpp's grammar converter emits only the declared property
   (12-30 tokens). The answer key comes before the reasoning, so the answer is committed first (one probe answered "D" and its own reasoning ended on "C").
   Not changed here (engine owner's call; see LOG): an opt-in or default `additionalProperties:false` for unspecified objects would close it.
3. The 2026-10-21 R9700 quality line ("MMLU tie, 87.6 % on 9 subjects") was measured on the pre-J1 binary; it is a smoke result, not a rank.

## 4. Gaps

- Quality was not re-measured on any platform in this lane; the quality columns of 2026-10-21 stand as the caveat they were. The Metal quality
  re-run (`resume-quality-diag.sh`) is unblocked now that J1 is merged and was not done here.
- Concurrency: not measured anywhere. intarsia's 35B serves one sequence at a time on CUDA and HIP (continuous batching is Metal-only
  and excludes MoE), llama.cpp ran with `-np 1`, hipfire in sequential mode (its 8-lane beta mode is a separate 09-26 measurement).
  A concurrency leaderboard would need those paths first.
- Metal has n = 3 visits per engine and the 4k and 16k prefill bands are wide (intarsia 4k: 2043-3899 tok/s); the third run was not obtained (section 5).
- The R9700 comparison spans four quant formats. An equal-quant comparison does not exist on this box.
- Spark was not re-measured; vLLM on that box is the only other engine and a tp=2 vLLM arm does not exist.
- No Metal 35B row (the 35B checkpoint on Metal is lane m35).

## 5. Downtime and fleet

radiance on tatooine (both cards) was down for 10 short windows, about 31 minutes in total, all between 01:26 and 03:10 EDT (inside the
00:00-07:00 free window, each window at most 5 minutes, role cap 3 h per day): fleetq #3814 (4 s, script bug), #3815 (147 s, aborted: localeval
re-queued the hipfire URL on endor; fixed by pointing it at the host name and by bash-3.2 array handling), #3817 (p1), #3820 (p2),
#3825 (p3), #3831 (p4), #3836 (p5), #3842 (p6), #3846 (p7), #3894 (p8, 45 s: hipfire server log). The EXIT trap restored radiance each time; it answered `/v1/models` 15-30 s after start.
endor: #3821 (exclusive, build + two `sameslot.sh` runs, 01:46-02:11). A third run (#3854, rotated order, 5 more visits for the noisy prefill cells) was queued at 02:12 behind another lane's long endor job (#3824) and released unrun at 03:10, because an exclusive request at the head of the queue was holding back other lanes' non-exclusive jobs. `scripts/standings-fx6/metal-run3.sh` is kept for a later re-run. Read-only and build jobs on tatooine: #3798-#3809, #3811-#3813 (#3810 was an endor build job, cancelled and replaced by #3821).

## 6. Files

- `scripts/standings-fx6/`: `hip-window.sh` (one `fleetq run` window, ARMS order, preflight, restore trap), `remote/` (per-engine start scripts),
  `Dockerfile.rocm-llama`, `build-hip.sh`, `aggregate-hip.py`, `metal-job.sh`, `aggregate-metal.py`, `probe-mmlu.py`, `queue-passes.sh`.
  `scripts/metal-speed-gate/sameslot.sh` gained an optional `ORDER`.
- `raw/hip/<pass>/`: per-visit localeval text, result JSON and summary, GPU clock samples, MMLU runs, probes, stream_bench; `raw/hip/aggregate.md`.
- `raw/metal/run<n>/`: per-arm localeval JSON, `refbw` samples, macmon, `gate.txt`; `raw/metal/aggregate.md`.
