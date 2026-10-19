# GPU idle gap after >= ~1 ms kernels (gfx1201, ROCm 7.2.4)

> **Correction (HIP round 9, `./gap_repro pat`, see "Where the stall sits" below).** The stall is not created by the profiler: unprofiled it
> exists too, one dispatch later, whenever a SHORT dispatch (under ~260 us) follows a kernel of >= 1.05 ms. The sweep below measures
> `start[i+1] - end[i]` with the next dispatch always as long as 0.5 ms or more, which overlaps the unprofiled window, so there the
> unprofiled run really loses nothing and `rocprofv3`, which moves the window in front of the next dispatch, really adds ~265 us per long
> kernel. The numbers below are right; the reading "only a profiler artifact" was wrong.

Standalone HIP reproducer, no dependencies. `gap_repro.hip` queues 100 back-to-back launches of a spin kernel on one stream (no host
sync, so the next dispatch was enqueued long before) for 0.5, 0.9, 1.0, 1.1, 1.5, 2, 3 ms and a mixed 2 ms / 0.5 ms sequence, in three
modes (`alu` FMA chains, `mem` 128 MB streaming read, `wmma` WMMA + 32 KB LDS) and three grids (1 block, 32 = one per CU, 64 = two per
CU). Each launch writes its own start/end with `wall_clock64()`; gap = start[i+1] - end[i].

```bash
hipcc -O3 --offload-arch=gfx1201 gap_repro.hip -o gap_repro && ./gap_repro      # device timestamps, no profiler (~10 s)
./gap_repro pat                                                                  # long/short sequences, gap by position (below, ~10 s)
./run.sh [sweep|pat|all]                                                         # everything, into results/ (~2 min per half)
```

`run.sh` runs, for the base config and for `HSA_ENABLE_INTERRUPT=0`, `ROC_ACTIVE_WAIT_TIMEOUT=5000`, `GPU_MAX_HW_QUEUES=4`, `GAP_STREAM=1` (a created
stream instead of the null stream; the `pat` half only for now: the sweep results below were taken before that knob existed):
`dev-<n>.txt` (plain run), `dev-<n>-under-rocprofv3.txt` (the same in-kernel timestamps while `rocprofv3 --kernel-trace` is attached) and
`rocprofv3-<n>.txt` (gap from the profiler's own Start/End of consecutive dispatches). `env.txt` has kernel, vbios, firmware, ROCm.
Needs `hipcc`, `rocprofv3` and `python3` (the stock `rocm/dev-ubuntu-24.04:7.2.4` image lacks `libdw1`, which rocprofv3 needs).

Output shape (`mode grid dur_ms n kdur_ms gap_mean gap_p50 gap_p90 gap_max >100us`, gaps in us):

```
alu      32 1.0         99    1.001      11.8      11.8      11.8      18.1      0%
alu      32 1.1         99    1.101      14.3      16.6      16.9      17.8      0%
alu      32 mix:2.0     50    2.001      16.8      16.8      16.9      17.9      0%
```

## Result on tatooine (Radeon AI PRO R9700, kernel 7.0.0-38, in-tree amdgpu, vbios 113-APM107573-101, ROCm 7.2.4, radiance up)

Median gap after a kernel, us (alu, 32 blocks; mem and wmma, other grids and the three knobs are the same to within 2 us, see `results/`):

| kernel | plain (in-kernel timestamps) | same, rocprofv3 attached | rocprofv3 trace |
|---|---|---|---|
| 0.5 ms | 3.0 | 5.6 | 4.9 |
| 0.9 ms | 11.7 | 14.3 | 13.6 |
| 1.0 ms | 11.7 | 14.3 | 13.6 |
| 1.1 ms | 15.9 | **280.2** | **279.5** |
| 1.5 / 2 / 3 ms | 16.8 | **281.0 / 280.2 / 280.3** | **280.3 / 279.6 / 279.7** |
| mixed 2 ms, 0.5 ms | 16.8, 3.0 | **281.9**, 5.5 | **281.6**, 4.9 |

* With `rocprofv3 --kernel-trace` attached the gap reproduces: ~280 us (max ~300) after every kernel that runs longer than ~1.05 ms
  (measured duration; the cut-off follows kernel duration, not the nominal label: grid 64 at 0.9 ms runs 1.3 ms and has it), none
  after shorter ones, in all modes and grids. The knobs do not change it (279 us in each).
* The kernels' own timestamps show the same gap, so it is real GPU idle while the profiler is attached, not a trace artifact.
* Without the profiler the gap `start[i+1] - end[i]` after long kernels is 16-17 us (11-12 us for 0.9-1.0 ms, 2-3 us for 0.5 ms; there the next
  dispatch is a long one, so it overlaps the unprofiled window and nothing is lost; with a short next dispatch the stall shows, see "Where
  the stall sits"): two small steps, 0.5 -> 0.9 ms
  (+9 us) and 1.0 -> 1.1 ms (+5 us), the second at the same duration as the 265 us step under the profiler. Only 5-6 of 7128 gaps
  per run exceed 100 us (single 276-308 us events, not tied to duration).
* The runs used GPU 1 of two (`HIP_VISIBLE_DEVICES=1`) inside the dev image, whose launcher sets `GPU_MAX_HW_QUEUES=1` (listed in `env.txt`),
  so "base" is one hardware queue and the `GPU_MAX_HW_QUEUES=4` variant is the other value; the gap is the same.

## Where the stall sits (`./gap_repro pat`; results in `results/pat-*`, same machine, GPU 1)

The sweep above times one thing, the gap `start[i+1] - end[i]`. `pat` runs sequences of kernels on one stream (no host sync, 5 runs each;
alu, long = 32 blocks) and stamps start/end of EVERY kernel, including tiny ones. It reports, by position after the long kernel (`+1` = the
dispatch right after it): `gap` = `start[i] - end[i-1]`, `dur` = the dispatch's own `end - start`, `since` = `start[i] - end[long]`,
`s2s` = `start[i] - start[i-1]`. Kinds: `L` long spin, `T` 1-thread 2 us spin, `P` 1-thread pure stamp (start and end back to back), `S` 50 us,
`M` 300 us. "first" = after a sequence's first long kernel (it follows a sync), "steady" = later groups. Medians, us. Long = 2 ms:

| `LPP` (long, stamp, stamp) | no profiler | `rocprofv3 --kernel-trace` (same in-kernel stamps) | rocprofv3's own trace |
|---|---|---|---|
| gap before +1 | 17.0 | **280.8** | **280.2** |
| gap before +2 | 3.1 | 5.4 | 4.8 |
| gap before +3 (the next long) | 2.9 | 5.4 | 4.8 |
| `dur` of the +1 stamp kernel (nominal ~1) | **259.7** | 1.0 | 1.5 |
| `s2s`, +1 start -> +2 start | **262.7** | 6.3 | 6.4 |
| `since`: long's end -> start of +2 | 279.7 | 287.1 | 286.5 |

From the long kernel's end to the start of +2 it is ~280 us either way; what moves is where it is spent. Unprofiled, +1 starts promptly
(17 us), but its end stamp comes ~260 us later and +2 starts 3 us after that: `gap` shows nothing at +2, `s2s` shows 263 us. Under the
profiler +1 does not start until ~280 us after the long kernel. This matches the engine's 1-thread stamp kernels (260 us stamp to stamp
unprofiled, 4-6 us under rocprofv3).

* **Positions** (`LTTT`, `LPPP`, `LSSS`): only +1 is affected; +2 and +3 are 2.8-3.1 us unprofiled. Under rocprofv3 only the gap before +1.
* **A window, not an additive delay**: the +1 dispatch cannot finish before ~277 us after the long kernel's end. A 2 us `T`, a 0 us `P` and a
  50 us `S` all end up at `dur` ~260 us (260.8, 259.7, 259.7; so 50 us of work cost 210 us extra); a 300 us `M` stays at 300.7 (+2 starts 17 + 301 + 3 us after
  the long kernel, nothing added). So the stall is free when the next dispatch is longer than ~260 us (that is why the old mixed 2 ms / 0.5 ms
  sequence showed nothing unprofiled) and costs ~260 us minus that dispatch's length when it is shorter.
* **Threshold** (`LPP`, measured long duration): 0.90, 0.95, 1.00 ms: no stall (+1 `dur` 1.0-1.1, gap 11.9 us); 1.05, 1.10, 2, 3 ms: stall (+1 `dur`
  260.7, 259.2, 259.4, 259.0). Same ~1.05 ms as under the profiler (rocprofv3's own trace, gap before +1: 13.7 us up to 1.00 ms, 279.8 at 1.05).
* **First long after a sync, long after long** (the engine's layer 0 had no gap): neither explains it. "first" is the same as "steady" (+1
  `dur` 258.2 vs 259.7). In `LLPP` the dispatch after the first long is the second long (`dur` 2000.6, no stall visible, as above) and the stall
  follows the second one (`dur` 260.5 first, 259.8 steady). A following dispatch longer than ~260 us (above) is a candidate for layer 0;
  not tested in the engine.
* The knobs `HSA_ENABLE_INTERRUPT=0`, `ROC_ACTIVE_WAIT_TIMEOUT=5000`, `GPU_MAX_HW_QUEUES=4` and a created stream (`GAP_STREAM=1`) change nothing:
  `LPP` 2 ms, unprofiled +1 `dur` 260.0 / 260.2 / 259.4 / 260.0 (base 259.7); rocprofv3's own trace, gap before +1, 280.0 / 279.9 / 279.6 (base 280.2),
  in-kernel stamps with a created stream under rocprofv3 280.5.

Total wall time of one sequence, median of 5 runs, device first start -> last end (`results/pat-base.txt`, `pat-base-under-rocprofv3.txt`).
`nominal` = sum of the kernels' requested durations, `exc/grp` = (span - nominal) per group in us; the control without the stall (`LTT`, 0.9 ms
long) has 18 us unprofiled, 26 us under the profiler:

| sequence (groups) | span ms, no profiler | exc/grp, no profiler | span ms, rocprofv3 | exc/grp, rocprofv3 |
|---|---|---|---|---|
| `LTT` 2 ms (33) | 75.56 | 286 | 75.67 | 289 |
| `LPP` 2 ms (33) | 75.44 | 286 | 75.78 | 296 |
| `LSS` 2 ms (33) | 77.07 | 235 | 79.04 | 295 |
| `LMT` 2 ms (33) | 76.77 | **24** | 85.65 | **293** |
| `LTTT` 2 ms (25) | 57.31 | 286 | 57.63 | 299 |
| `LLPP` 2 ms (25) | 107.59 | 304 | 114.50 | 580 |
| `LTT` 0.9 ms, control (33) | 30.43 | 18 | 30.68 | 26 |

So the stall costs real time unprofiled: 9.4 ms of 75.4 ms for `LPP` (12.5%), ~285 us per long kernel followed by short work. The profiler
does not create it. It turns the window into a hard gap before +1, so sequences whose +1 dispatch is long (`LMT`, `LL..`) pay ~290 us per
long kernel under it and nothing, or one stall instead of two, without it.

Caveats: one machine, one GPU (GPU 1 of two, `GPU_MAX_HW_QUEUES=1` in the dev image), alu long kernels with 32 blocks (the sweep above found no
difference for mem/wmma or other grids, but `pat` ran only alu/32). The stamps are in-kernel `wall_clock64()` reads, so "+1 stalls after its first
read" and "+1's second read is slow" cannot be told apart here; either way nothing useful executes for ~260 us. The first sequence of a process
(`LTT`) was a slow outlier in one earlier run of the same code (79.4 ms, not in the final one), so read the `LPP` rows first.
