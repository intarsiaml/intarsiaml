<!-- DRAFT correction for https://github.com/ROCm/rocm-systems/issues/13775. NOT POSTED. Review, then post. -->

**Proposed new title:** gfx1201: ~260 µs idle after dispatches ≥ 1 ms (opens one dispatch later without a profiler; rocprofv3 shifts it)

## Correction: the profiler does not create this stall

My original report said the ~280 µs gap only exists while `rocprofv3 --kernel-trace` is attached, i.e. that the profiler adds it. The
measurements in it are right, the conclusion was too broad. The stall exists without a profiler too, whenever a short dispatch follows a
kernel of 1.05 ms or longer. In my first reproducer the dispatch after each long kernel was itself 0.5 ms or longer, so unprofiled it
overlapped the stall window and lost nothing, while under `rocprofv3` the window sits before that dispatch and adds ~265 µs per long kernel.
I also only measured `start[i+1] - end[i]` (kernel i's end stamp to kernel i+1's start stamp). With a short dispatch right after the long
one, that metric shows nothing unprofiled: the stall is inside that dispatch's own start..end interval, not between the two stamps.

## What happens

One stream, no host sync, every kernel writes its own start and end with `wall_clock64()`. Sequence: a 2 ms spin kernel (32 blocks), then
three 1-thread kernels that only take two timestamps (`LPPP`), repeated 25 times. Medians, µs, gfx1201, ROCm 7.2.4:

| position after the long kernel | gap before it, no profiler | gap before it, `rocprofv3 --kernel-trace` | its own duration (start..end), no profiler | its own duration, rocprofv3 |
|---|---|---|---|---|
| +1 | 17.0 | **281.7** | **259.4** (nominal ~1) | 1.0 |
| +2 | 3.1 | 5.3 | 1.0 | 1.0 |
| +3 | 2.8 | 5.4 | 1.0 | 0.9 |
| next long kernel | 2.8 | 5.4 | | |

- Without a profiler dispatch +1 starts 17 µs after the long kernel, but its end stamp comes ~260 µs later. Dispatch +2 then starts 3 µs
  after that. So the literal gap before +2 is small (3.1 µs); what shows the stall is +1's own duration (~260 µs) and start to start,
  +1 to +2 (262.5 µs in this table, 262.7 µs in `LPP`). That start-to-start time is also what two single-stamp kernels around a long dispatch
  see in our inference engine (260 µs unprofiled, 4-6 µs under `rocprofv3`).
- Under `rocprofv3` the same time (~280 µs from the long kernel's end to the start of +2) shows up as a gap before +1 instead. The profiler's own
  Start/End timestamps agree with the in-kernel stamps (gap before +1 281.6 µs in `LPPP`, 280.2 µs in `LPP`).
- It is a window, not an added delay: a 2 µs, a 0 µs and a 50 µs kernel at +1 all end ~260 µs after they start (the 50 µs one: 259.7); a
  300 µs kernel at +1 takes 300.7 µs and nothing is added. Only +1 is affected (`LTTT`, `LPPP`, `LSSS` checked).
- Threshold: no stall for a 1.00 ms (measured) long kernel, stall for 1.05 ms and above (`LPP`: +1 duration 1.0 µs up to 1.00 ms, 260.7 at
  1.05, 259.2 at 1.10, 259.4 at 2 ms, 259.0 at 3 ms). The same ~1.05 ms as under the profiler.
- It costs real wall time: 9.4 ms of 75.4 ms (12.5%) for the 33-group `LPP` sequence, ~285 µs per long kernel. The 0.9 ms control without the
  stall: 18 µs per group.
- The profiler does change the cost: it turns the window into a hard gap before +1. When the +1 dispatch is longer than ~260 µs (`LMT`:
  2 ms, 300 µs, 2 µs) it overlaps the window unprofiled and costs nothing (24 µs per group), under the profiler it costs the full stall
  (293 µs per group; `LLPP`: 304 vs 580 µs per group). That is the case my first reproducer measured.
- Not changed by `HSA_ENABLE_INTERRUPT=0`, `ROC_ACTIVE_WAIT_TIMEOUT=5000`, `GPU_MAX_HW_QUEUES=4`, or a created stream instead of the null stream.

## Corrected minimal repro

```bash
git clone -b tools/rocm-gap-repro https://github.com/intarsiaml/intarsiaml && cd intarsiaml/kernels/hip_checks/gap_repro      # gap_repro.hip: 160 lines, HIP only
hipcc -O3 --offload-arch=gfx1201 gap_repro.hip -o gap_repro
./gap_repro pat | grep -E '^(seq|LPPP +2.00 steady)'                                    # no profiler
rocprofv3 --kernel-trace --output-format csv -d prof -o run -- ./gap_repro pat | grep -E '^(seq|LPPP +2.00 steady)'   # same in-kernel stamps under the profiler
```

Columns: `seq L_ms long pos n Lms gap_p50 gap>100 dur_p50 since_p50 s2s_p50` (µs; `gap` = start[i] - end[i-1], `dur` = end[i] - start[i],
`since` = start[i] - end[long], `s2s` = start[i] - start[i-1]). Expected on our R9700: unprofiled `P+1` has `gap_p50` ~17 and `dur_p50` ~260,
`P+2` has `gap_p50` ~3 and `s2s_p50` ~263; under `rocprofv3` `P+1` has `gap_p50` ~281 and `dur_p50` ~1. `./run.sh pat` runs everything including the
profiler's own trace and the knobs, raw output in `results/pat-*`.

Setup: Radeon AI PRO R9700 (gfx1201), kernel 7.0.0-38 (in-tree amdgpu), vbios 113-APM107573-101, ROCm 7.2.4, `rocm/dev-ubuntu-24.04:7.2.4` plus
`libdw1`, second GPU of two, `GPU_MAX_HW_QUEUES=1`. alu spin kernels only in `pat` (the earlier sweep saw no difference for mem/wmma or grids).

## Still open

What holds the dispatch after a ≥ 1.05 ms dispatch for ~260-280 µs? The timestamps cannot tell whether the +1 wave is stalled after its first
clock read or the second read is slow; either way nothing useful runs for ~260 µs. Is this firmware/CP work after a long dispatch
(cache writeback, a barrier/signal sequence, power state)? And why does `rocprofv3` move it to before the +1 dispatch?
