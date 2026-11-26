# Inkling-2-Mini Metal: what puts a process into the "slow mode"? (lane ink-mode, 2026-11-25/26)

Status (2026-11-25 22:00): probe built, slot reserved (fleetq #11922, endor, exclusive, Thu 06:25, 30 min). No measurement yet.
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
