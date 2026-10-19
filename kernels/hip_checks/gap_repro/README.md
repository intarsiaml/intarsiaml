# GPU idle gap after >= ~1 ms kernels (gfx1201, ROCm 7.2.4)

Standalone HIP reproducer, no dependencies. `gap_repro.hip` queues 100 back-to-back launches of a spin kernel on one stream (no host
sync, so the next dispatch was enqueued long before) for 0.5, 0.9, 1.0, 1.1, 1.5, 2, 3 ms and a mixed 2 ms / 0.5 ms sequence, in three
modes (`alu` FMA chains, `mem` 128 MB streaming read, `wmma` WMMA + 32 KB LDS) and three grids (1 block, 32 = one per CU, 64 = two per
CU). Each launch writes its own start/end with `wall_clock64()`; gap = start[i+1] - end[i].

```bash
hipcc -O3 --offload-arch=gfx1201 gap_repro.hip -o gap_repro && ./gap_repro      # device timestamps, no profiler (~10 s)
./run.sh                                                                         # everything below, into results/ (~2 min)
```

`run.sh` runs, for the base config and for `HSA_ENABLE_INTERRUPT=0`, `ROC_ACTIVE_WAIT_TIMEOUT=5000`, `GPU_MAX_HW_QUEUES=4`:
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
* Without the profiler the gap after long kernels is 16-17 us (11-12 us for 0.9-1.0 ms, 2-3 us for 0.5 ms): two small steps, 0.5 -> 0.9 ms
  (+9 us) and 1.0 -> 1.1 ms (+5 us), the second at the same duration as the 265 us step under the profiler. Only 5-6 of 7128 gaps
  per run exceed 100 us (single 276-308 us events, not tied to duration).
* The runs used GPU 1 of two (`HIP_VISIBLE_DEVICES=1`) inside the dev image, whose launcher sets `GPU_MAX_HW_QUEUES=1` (listed in `env.txt`),
  so "base" is one hardware queue and the `GPU_MAX_HW_QUEUES=4` variant is the other value; the gap is the same.
