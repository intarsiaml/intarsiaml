# Inkling-2-Mini Metal: what puts a process into the "slow mode"? (lane ink-mode, 2026-11-25/26)

Status (2026-11-26 07:00): **cause found, outside the engine.** endor is a MacBook Pro (M5 Max) on a 140 W adapter. A process is fast when
its 16k prefill gets the GPU's short high-power state (1617 MHz, 80-110 W GPU, system power at or above the 140 W adapter rating) and slow when
that state is already used up (about 1100-1300 MHz, 64-80 % busy, 30-50 W GPU, about 98 W system). Whether it is available depends on the box's
recent power history (the previous process, idle gaps), not on anything the process itself does. No code change can put every process in the
fast mode; see Decision.
Branch `ink/mode` (base `072dd0863`). Not pushed.

## Question

Lane ink-spec (`2026-11-21-inkling2-mtp.md` on `ink/spec-flip-merge`, "Fast / slow process mode", "Brief L8 step 1/2") found that fresh
`--spec off` processes with identical code and flags land in one of two prefill speeds at 16k: "fast" 5.6-5.9 s or "slow" >= 6.15 s
(about +10 %), 55-63 % slow over about 110 `generate` processes, steady within one process, with the slow arm changing from slot to slot.
A per-op table at 4k showed no op-specific difference; 1.1 GB of ballast and the keep-warm residency set did not change it.
This lane looks for the per-process state (or external box state) that decides the mode.

## Prior evidence worth keeping in mind

- 2026-10-15 overnight report (Qwen3-8B, macmon logged): the GPU sometimes dropped from about 1600 MHz / 90 W to 900-1300 MHz / 20-50 W in
  the middle of a prefill with no thermal warning; separately the chunk-2048 4k prefill varied about 20 % at full clock.
- `2026-11-25-inkling2-pfgap2.md`: a 30-70 ms GPU-side stall on the first compute buffer after a few seconds of GPU idle; and inside one
  `serve` process 16k requests took 5.86 / 5.98 / 6.41 / 7.66 s of GPU busy time, so the speed is not always steady within a process.
- Re-split of the L7 4k op tables (`ink-spec-flip/.scratch/l7/op`, 20 profiled processes) by per-op total: the slowest 7 vs fastest 7
  differ by 2.5-8 % in every op (`moe_experts` +2.6 %, `conv_mix` +6.6 %, `matmul` +8.3 %, attention +3.9 %), a uniform slow-down,
  not one op. The profiled totals (1671-1798 ms excluding `embedding`, which absorbs the queue wait) rise with run order, so in that slot
  the profiled spread looks like slow drift, not two modes.

## Box state seen while preparing (2026-11-25 ~21:30, no job of this lane running)

- Memory: 137 GB, wired 43 GB, compressor holding 32 GB in 18 GB, swap 20.8 of 21.5 GB used; an `rsync` of the Inkling checkpoint to
  tatooine (another lane) was reading the same 19.5 GB file.
- Foreign GPU time (per-process `accumulatedGPUTime` from `ioreg -c AGXDeviceUserClient`, 6 s window while another lane's engine ran):
  Chrome helper 314 ms, WindowServer 181 ms, BambuStudio 133 ms, WebKit 20 ms, about 650 ms of foreign GPU time per 6 s. The GUI
  session's apps share the GPU with every benchmark; an exclusive fleetq slot does not stop them.

## Probe design

Code (diagnostic only, off unless the variables are set; numerics untouched):

- `crates/intarsia-backend-metal/src/modeprobe.rs` + hooks in `host.rs`, `INTARSIA_METAL_MODE_PROBE`:
  - one `modeprobe file` line per whole-file weight buffer when it is created: host VA and `gpuAddress`, each mod 16 KiB / 2 MiB / 32 MiB /
    1 GiB, length, creation order;
  - `MetalBackend::mode_probe(tag)`: device allocated size and recommended working set, `mincore` residency of each wrapped file, and GPU
    timestamps (median / min over reps) of `alu` (register-only FMA chain: clock), `seq_file` (stream every file buffer: DRAM bandwidth
    through the no-copy mapping), `gather_file` / `gather_file2g` (one 16-byte read per random 16 KiB page over the 19.5 GB buffer / its first
    2 GiB: GPU page-walk cost of the mapping), `seq_anon` / `gather_anon` (the same over a fresh 2 GiB `newBufferWithLength` buffer: control),
    and the backend's copy-bandwidth probe.
- `INTARSIA_METAL_NO_WRAP=1` (forcing arm): `wrap_host` refuses every import, so the loader copies the weights into
  `newBufferWithLength` buffers (alloc + upload) instead of mapping the file.
- `intarsia generate` calls the probe at the sites named in the variable (`load`: after the model load, before the first prefill;
  `end`: after the last prompt).

Slot script `.scratch/mode/mode.sh` (fresh `generate` per process, `longp.json` 1k / 4k / 16k prompts, 32 greedy tokens each, `--spec off`):
arms interleaved `A N A L` (A default + probe at end; N no-wrap + probe at end; L default + probe after load and at end), time-boxed to 23 min.
Around each process: foreign GPU ms (ioreg diff, the engine's own pid excluded), `vm_stat` deltas (compressions, swap, page-ins), swap use,
`pmset -g therm`, and a `macmon` stream for the slot (GPU MHz / W / temperature per process window). `.scratch/mode/analyze.py` builds the
per-process table, splits fast / slow at the largest gap of the 16k times, and gives medians per mode and Spearman rho against the 16k prefill
for every feature. The greedy ids of every process are compared (N must match A bitwise).

## Slot 1: fleetq #11922, endor exclusive, 2026-11-26 06:25-06:48 (binary md5 82497434119880675dafce00b3962370, code 125af63f9)

77 processes (2 warm + 75 in `A N A L` cycles: 38 A, 19 N, 18 L), all rc 0. Per-process table and correlations:
`docs/results/2026-11-25-inkling2-mode/slot1-analysis.txt` (`analyze.py run-slot1 6150`, the L7 cut).

**Bitwise:** one greedy id set over all 77 processes (1k / 4k / 16k, 32 tokens each), so the no-wrap arm is identical to the mapped arms.

**Everything per-process that the probe logs is the same in every process** and does not separate fast from slow:

| feature | fast (16k < 6150 ms, n 24) | slow (n 51) | Spearman rho vs 16k |
|---|---|---|---|
| weight buffer host VA | 0x7000000000 (1 GiB aligned) in every process | same | - |
| `gpuAddress` | 0x10000088000 (0x88000 past 2 MiB / 32 MiB / 1 GiB) in every process | same | - |
| file residency (`mincore`) | 100 % | 100 % | - |
| `alu` (ms, median) | 2.59 | 2.63 | +0.04 |
| `seq_file` / `seq_anon` | 558-560 / 561 GB/s | same | -0.20 |
| `gather_file` / `gather_file2g` / `gather_anon` (ms) | 0.986 / 0.865 / 0.865 | same | +0.02 |
| copy probe | 548 GB/s | 548 GB/s | +0.10 |
| swap used, free, compressor (at start) | 20.8 GB, 28 GB, 34 GB | same | ~0 |
| compressions / swap-outs during the run | 0 / 0 | 0 / 0 | - |
| foreign GPU time during the run (ioreg) | 15 ms | 21 ms (longer L runs) | +0.48 |
| thermal (`pmset -g therm`) | no warning | no warning | - |
| GPU temperature at start | 74.7 C | 78.0 C | -0.06 |
| **GPU power, busy samples (median W)** | **80.4** | **50.9** | **-0.83** |
| **time above 65 W GPU during the run** | 2.9-4.2 s | 0.3-0.6 s | **-0.91** |
| GPU peak temperature | 88.7 C | 79.8 C | -0.77 |

Placement and memory are ruled out: the VA, GPU address and residency never change, the microbenchmarks (clock, streaming, page gather on the
mapping and on anonymous memory) are flat to 1-2 %, and copying the weights into `newBufferWithLength` buffers (arm N) does not remove the mode
(N is slow in 7 of 19 processes). The probe kernels never show the mode because they draw about 38 W, below the point where the GPU is limited.

**The mode follows the previous process, not the process itself** (median ms by position in the `A N A L` cycle):

| position | n | 1k | 4k | slow 4k (> 1450 ms) | 16k (range) |
|---|---|---|---|---|---|
| A after L | 19 | 627 | 1666 | 11 of 19 (10 of 10 from #33) | 6549 (5620-6803) |
| N | 19 | 446 | 1284 | 2 of 19 | 6043 (5667-6527) |
| A after N | 19 | 669 | 1286 | 0 of 19 | 6219 (5904-6650) |
| L | 18 | 604 | 1622 | 14 of 18 | 6713 (6211-7022) |

L ends with about 10 s more GPU work than A (the probe's 4 s sustained test, then a 2 s idle and more kernels) and starts with the same probe before
its first prefill. Every process after an L is slow at 4k from #33 on; no process after an N is. Same code, same weights, same flags: the difference
is what the GPU did in the previous ~10-60 s.

**What fast and slow look like on the GPU** (macmon, 250 ms samples, the 16k prefill windows):
- Fast process (e.g. #35 A after N): 1617 MHz, 100 % busy, 76-110 W for about 4 s, then 1050-1400 MHz, 65-88 % busy, 30-60 W.
- Slow process (e.g. #33 A after L): one sample at 1617 MHz, then straight to 940-1500 MHz, 58-94 % busy, 25-54 W.
- Over all samples of the slot: in the high state (busy > 97 %, GPU > 65 W, n 474) the GPU runs 1617 MHz at 92 W and the **system draws 140 W
  (median), up to 173 W: the 140 W adapter's rating and above it**. In the limited state (busy 50-90 %, n 1435) the GPU runs 1269 MHz at 43 W and
  the system about 98 W. The battery reported 100 %, "finishing charge", charging at +2.1 A (about 28 W) at idle.
- Spearman rho of the 16k time against the mean GPU power over the 60 s before the process is +0.60 (4k: +0.68); over the 10 s before, -0.06.

So the GPU's high-power state is a budget: it lasts a few seconds and refills during GPU idle. A process gets it if the budget refilled since the
last heavy GPU work. The 16k prefill (6 s) outlasts it in every process; what differs is whether it starts with 0 or about 4 s of it. That also
explains the other observations: the 2x / 25 % slowdown of ink-moepf2's bench within 0.4-4 s of back-to-back dispatch, with a new process starting
fast again (the process start-up is 4-5 s of GPU idle); the 10-15 "1600 MHz / 90 W drops to 900-1300 MHz / 20-50 W mid-prefill"; the steady speed within
a `serve` process with back-to-back requests (the budget never refills); and the arms swapping sign from slot to slot (each arm's share of fast
processes depends on what ran just before it). The system power at the adapter rating suggests the adapter / battery power budget of the
MacBook Pro as the limit (a short supplement from the battery, then a clamp), not GPU temperature (slow processes start no hotter and peak cooler,
no thermal warning); the battery current during the high state was not sampled in slot 1 (see slot 2).

**Side findings:**
- `alu_idle2s`: the same ALU buffer takes 4.17 ms after 2 s of GPU idle against 2.35 ms back-to-back (+77 %): the idle-clock start latency of
  pfgap2, at kernel scale.
- The sustained ALU-only and memory-only series (4 s, about 38 W) do not drift (last / first 1.00 and 1.003), and a fresh command queue changes
  nothing (2.351 vs 2.35 ms). The slowdown needs the high-power workload; it is not a queue or device state.
- Arm N (weights copied) has a 1k prefill of 446 ms against 604-669 ms with the mapped file: the first GPU use of the 19.5 GB no-copy buffer costs
  about 160-220 ms per process (wiring the file for the GPU). 4k / 16k do not differ in the same power state (A after N 1286 / 6219 vs N 1284 / 6043).
- 16k decode is the same in both modes (64.3 vs 64.1 tok/s): decode draws less power and is not limited.
