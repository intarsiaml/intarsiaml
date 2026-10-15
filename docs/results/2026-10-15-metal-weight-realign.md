# Metal: 16-byte weight re-alignment at load (R = 8 multi-row remedy)

Qwen3-8B BF16 on endor (Apple M5 Max, macOS 27.0). 2026-10-15. Branch `metal-weight-realign` (worktree `/opt/code/intarsiaml-weight-realign`), off `spike/metal` c556008 plus `git merge metal-g0b-attn` (multi-row kernels, `gemv_g0` / `g0b_bench` harness). Raw data: `docs/results/raw/2026-10-15-metal-weight-realign/` (`slot*.sh`, per-slot output dirs).

**Verdict.** The switch works and fixes the R = 8 cliff, but it costs **+3.8 GB of non-evictable memory** (not 2.75 GB) and 0.55 s of load time, and it is **not bit-identical** to the 8-byte-load kernels in decode. It ships **default OFF** (`INTARSIA_METAL_REALIGN=1` turns it on). Nothing needs it for k <= 6.

| item | result |
|---|---|
| T(8)/T(1), census-weighted layer (GEMV time), before (8-mod-16 mix as served) | **1.197** this run (1.208 in the G0 report); gate 1.20 |
| T(8)/T(1), after (all 36 layers 16-byte aligned) | **1.152** |
| memory, steady state (`footprint`, serve, after the speed sweep) | **+3.82 GB** (1669 -> 5490 MB); the copy is anonymous/IOAccelerator memory |
| load time (generate, page cache warm) | 0.3 -> 0.9 s (+0.55 s); first token 240 -> ~225 ms |
| batch-1 decode ids vs c556008 | identical in 12/12 comparisons (also with the switch off) |
| batch-1 decode bits | **not guaranteed identical**; prefill identical to the printed digit, teacher-forced decode KL moves by noise-level amounts (section 4) |
| decode tok/s | 36.6 / 36.3 / 35.3 (128 / 1k / 4k) ON, same as c556008 within 0.3 |
| tests | `cargo test -p intarsia-backend-metal --release`: 103 passed; `cargo test --workspace --release`: 373 passed |

## 1. Change

`crates/intarsia-backend-metal/src/host.rs`, `lib.rs`; tests `tests/realign.rs`, small fixups in `tests/wrap_host.rs`, `tests/multirow.rs`. No shared code touched (the hook is `Backend::wrap_host`, which the Metal backend implements itself; `intarsia-core`'s loader is unchanged).

* When enabled, `wrap_host` checks the address of the requested range. A range that is not 16-byte aligned (every tensor of a shard whose JSON header length is 8 mod 16: Qwen3-8B shard 1, header 9,328 bytes) is copied into a fresh shared-storage `MTLBuffer` (16-byte aligned, `base` 0, padded to a 16-byte multiple, read-only like any wrapped buffer). The GEMV planners then choose the 16-byte-load and multi-row-16-byte kernels from the buffer address as before; nothing in the kernel selection changed. Aligned ranges keep the zero-copy window.
* After the copy the whole pages inside the source range are released with `madvise(MADV_DONTNEED)`.
* Switch: env `INTARSIA_METAL_REALIGN` (set and not `0`) read at backend construction, or `MetalBackend::set_realign(bool)`. Default off. `MetalBackend::realigned()` returns (tensors, bytes) copied; `INTARSIA_METAL_DEBUG=1` prints one line per copy.
* `wrap_host` carries no tensor name, so the backend cannot tell GEMV weights from the gathered `embed_tokens` table. **81 ranges, 3,996,241,408 bytes** are copied (the whole of shard 1), not the 2.75 GB of the G0b estimate: that figure excluded `embed_tokens` (1.245 GB, gathered, gains nothing). Skipping it would need a role hint in the core `wrap_host` / `load_tensor` API, which this task excluded; with it the cost would be 2.75 GB.

## 2. Memory: duplicate or replace

Measured in `intarsia serve` (same flags as the final same-slot runs), `footprint` / `vmmap -summary` / `vm_stat` after load and after the localeval sweep, order base, on, on, base (`slot2/serve/*-footprint.txt`, `*-vmmap.txt`, `*-vmstat.txt`):

| | c556008 | realign ON |
|---|---|---|
| footprint after load | 127 / 124 MB | 3948 / 3945 MB |
| footprint after sweep | 1669 / 1666 MB | 5490 / 5487 MB (IOAccelerator 1413 -> 5228 MB) |
| `ps` RSS after sweep | 0.20 GB | 4.11 GB |
| `vm_stat` anonymous pages (16 KiB) after sweep | 2,298,282 / 2,310,069 | 2,549,346 / 2,553,028 (+3.8 GB) |
| `vm_stat` pages free after sweep | 237,373 / 283,011 (3.6 / 4.3 GB) | 44,710 / 47,590 (0.7 / 0.8 GB) |
| `vmmap` mapped file (virtual) | 15.4 GB | 11.6 GB |

* The copy is **additional**: +3.82 GB footprint, non-evictable, and free memory shrinks by 3 to 3.6 GB (the rest was already in the noise of other processes). It does not replace the weights in the way that matters for memory pressure.
* The mmap side is released at the process level: no tensor of shard 1 remains windowed, so the shard-1 mapping is dropped (virtual mapped file -3.8 GB) and the process never counts its pages. The baseline does not count those pages either: no-copy `MTLBuffer` pages are system page cache, outside the process footprint (127 MB after load).
* The source pages stay in the system page cache as clean file-backed pages (`File-backed pages` did not fall: 2.38 / 2.31 M base, 2.32 / 2.31 M ON). They are cold after the copy and evictable first, so under pressure the hot set matches the baseline and the extra cost is the 3.8 GB anonymous copy. Until then both exist.
* Peak at load: a 1-token `generate` peaks at 10.3 GB RSS / 6.45 GB peak footprint (base 2.3 GB / 2.45 GB) because the copy reads the mapped pages while it writes the new buffers (`slot1b/memory/load-on-*.err`, `/usr/bin/time -l`). The steady-state numbers above are the ones that matter.

## 3. Load time

`generate --max-tokens 1`, 5 repetitions each, page cache warm (`slot1b/memory/`):

| | "loaded in" | wall (`time`) | prefill of the first token |
|---|---|---|---|
| c556008 | 0.3 - 0.4 s | 0.60 - 0.62 s | 239 - 248 ms |
| this branch, switch off | 0.3 - 0.4 s | 0.59 - 0.67 s | 238 - 291 ms |
| switch on | 0.9 s (all five) | 1.14 - 1.21 s | 210 - 237 ms |

+0.55 s load (the G0b estimate was 0.15 s of copy; the extra is the 81 buffer allocations at 4 GB and the madvise). The first token is about 15 to 30 ms faster (the baseline pages weights in lazily during the first forward). `intarsia serve` startup is unchanged (43.4 / 37.9 s base, 43.7 / 38.4 s ON; dominated by the server's own warm-up). A cold page cache would add the file read to both.

## 4. Bitwise status

* **Greedy ids** (`generate --temperature 0 --ignore-eos --show-ids --max-seq 8192`, prompts `short`, `4k` (4062 tokens), `w4107` (4107 tokens), 48 and 256 tokens): c556008 vs switch-off and c556008 vs switch-on are **identical in all 12 + 12 comparisons** (`slot1b/identity/summary.txt`).
* **Kernel level: not bit-identical in general.** `gemv_bf16_rt_u2s8` adds `acc += dot8(...)` (an 8-element fma chain per 16-byte load) while the 8-byte variant adds two `dot4`s, so the f32 grouping differs and the bf16 outputs can differ in the last bit. In `tests/realign.rs` (K 4096, N 512, random bf16) 0 of 512 outputs differ, and the test accepts one bf16 ulp. Do not read the task's "aligned and a8 should be bitwise equal" as a property; it holds on that data, not by construction.
* **Accuracy path** (`intarsia accuracy`, no baseline gate: the scratch binaries are unstamped, so `--baseline` / `--noise-envelope` cannot be used; `slot5/`, repeat in `slot7/`). The prefill ("forward") sections are identical to every printed digit. Teacher-forced decode, where the GEMVs run, moves:

| golden | decode top-1 c556008 / ON | decode KL c556008 / ON | verdict |
|---|---|---|---|
| short (128 positions) | 100.00 / 100.00 % | 3.313e-4 / 2.106e-4 | PASS / PASS |
| long (500 positions) | 99.80 / 99.60 % | 3.538e-4 / 3.444e-4 | PASS / PASS |
| 4k top-20 (192 positions) | 98.96 / 99.48 % | 2.573e-4 / 3.959e-4 (21-way 2.548e-4 / 3.922e-4) | PASS / PASS |

  The repeat reproduces both columns exactly, so the runs are deterministic and the difference is the kernel, not run-to-run noise. The movement is in both directions and inside the 1e-3 KL gate (floor class bf16, same gates as the qualify stages), but it is a numerics change: **turning the switch on would need the Metal golden baselines and envelopes re-recorded** (acc-short / acc-long / acc-4k are baseline-gated). Not done; default stays off.

## 5. T(R)/T(1) with aligned weights

Harness: the existing `gemv_g0` (`G0_ROWS=2,4,6,8`, cases qo, down, kv, gu, 30 rounds, macmon 100 ms, rounds under 1450 MHz dropped by `analyze_g0.py`; run 1 and 2 aligned, 1 with `G0_A8=1`; `slot2/bench/`, `analysis.md`). Method note: the bench uses a rotating pool of allocated `MTLBuffer`s, at offset 0 (aligned) or 8 (a8). A realigned weight is exactly an allocated buffer at offset 0 read by the same kernel, so the bench's aligned column **is** the realigned configuration; there is no separate real-weight bench on this branch (the speculative verify path lives on another branch). All 264 in-bench bitwise checks (multi-row vs R serving launches, same alignment) passed.

T(8)/T(1) multi-row, per shape (T(1) is the serving kernel at the same alignment):

| shape | 8-mod-16 (a8 kernels, as served for shard 1) | 16-byte aligned (after realign) |
|---|---|---|
| q/o 4096x4096 | 1.312 | 1.097 |
| down K=12288 | 1.355 | 1.190 |
| k+v pair | 1.421 | 1.174 |
| gate+up pair | 1.423 | 1.149 |

Layer sum (2 x q/o + down + k/v + gate/up, GEMV time only):

| R | 8-byte shard | 16-byte aligned | census-weighted (7-8 of 36 layers per shape misaligned = before) |
|---|---|---|---|
| 2 | 1.002 | 1.003 | 1.003 |
| 4 | 1.009 | 1.009 | 1.009 |
| 6 | 1.066 | 1.021 | 1.030 |
| 8 | 1.383 | **1.152** | **1.197** |

So at R = 8 the layer goes from 1.197 (1.208 in the G0 report; the gap is run-to-run) to 1.152 against the 1.20 gate, i.e. it passes with 0.05 margin instead of 0.003 or a 0.008 miss. Against the unchanged current 1-row time (T_layer(1) = 663.1 us census-weighted) the realigned 8-row layer is 766.3 us = 1.156. Nothing changes at R <= 6 in a way that matters (1.030 -> 1.021 at 6), consistent with G0b's finding that k <= 6 does not need this.

Cost on batch-1: in the bench the 1-row aligned kernel is 0.7 to 2.5 % slower than the a8 one on the same shape (layer 665.5 vs 653.6 us, +1.8 %), affecting 7-8 of 36 layers. End to end it is not visible: localeval speed, `--sizes 128,1024,4096 --gen 256 --reps 3 --thinking off`, order base, on, on, base (server flags as the final same-slot run, `slot2/serve/`, result dirs in `slot2/localeval-results/`):

| run (localeval run id) | decode 128 | decode 1k | decode 4k |
|---|---|---|---|
| base-1 `20261015-114230-adhoc-base-1` | 36.4 | 36.1 | 35.3 |
| on-1 `20261015-114442-adhoc-on-1` | 36.7 | 36.1 | 35.3 |
| on-2 `20261015-114647-adhoc-on-2` | 36.6 | 36.5 | 35.3 |
| base-2 `20261015-114851-adhoc-base-2` | 36.6 | 36.5 | 35.3 |

Not slower within the +-5 % decode noise (here within 0.3 tok/s). Prefill (marginal 4k) 3595 / 3845 / 3567 / 3623 tok/s, same band. localeval's unanimis records were written by the harness (not captured separately).

## 6. Limits and not done

* Default OFF; the memory cost (+3.8 GB, 81 ranges including `embed_tokens`) and the need to re-record the accuracy baselines are the operator's call. Skipping `embed_tokens` (-1.245 GB) needs a role hint through the core `wrap_host` API (not done: shared code).
* The remedy only matters when a multi-row verify at R = 7..8 runs (speculative decoding, not on this branch); at R <= 6 it buys nothing. T(R)/T(1) is measured on synthetic aligned vs offset-8 buffers (equivalent by construction, argued above), not through a real multi-row model path.
* The pair (k+v, gate+up), qkv-fold and other alignment-dependent decode kernels also switch variants for realigned tensors; the decode accuracy shift in section 4 is the sum of all of them. Only the matmul kernel has a direct unit comparison.
* Binaries: the perf, identity and accuracy slots (2, 1b, 5, 7) ran the scratch binary md5 4eef61698ffbd4c3659a485e733e1893, built from a8846a9 + the debug-line commit, before the later commits (copy padded to a 16-byte multiple; test-only edits). The padding changes an allocation by < 16 bytes (the shard-1 total is already a multiple of 16), so it was not re-measured. The c556008 reference binary is a `git archive` build (md5 e6415c069e72d902c95260a08bd6b443), not this branch with the switch off; switch-off ids were compared separately and match.
* Merge fixup: `tests/multirow.rs::serving_selection_unchanged` (from G0) predates `NAX_MIN_M = 2` (tail fix in c556008) and failed on the merged tree (expected the tiled GEMM at M = 2..15, the registry picks the NAX GEMM). Updated to expect the NAX picks (and the multi-row pair only up to `MR_MAX_REGISTERED`); not a re-alignment issue. `tests/wrap_host.rs` pins the switch off (it tests the zero-copy window layout).
* No `qualify` run, no cold-cache load time, no TP.

## 7. Tests

* `tests/realign.rs`: the realigned buffer is 16-byte aligned (`host_ptr % 16 == 0`, `base` 0), read-only, contents equal to the source (byte compare with the mapped file and the reference vector), aligned tensors are not copied (counter unchanged), a rewrite-refault check after `madvise`, and the batch-1 GEMV on the copy agrees with the 8-byte-load kernel on the unaligned window within one bf16 ulp and with the f64 reference.
* slot 8 (#2035, final tree a06ac68): `cargo test -p intarsia-backend-metal --release` 103 passed, 0 failed; same with `INTARSIA_METAL_REALIGN=1`: 103 passed; `cargo test --workspace --release`: 373 passed, 0 failed. Slot 4 (#2019) ran `multirow` with `INTARSIA_METAL_MULTIROW=1`.

## 8. fleetq slots (endor, exclusive)

#2006 (slot 1: invalid, the scratch binary had been overwritten by a `cargo test --workspace` build without `--features metal`; its test logs show the stale multirow test failure, kept under `slot1/tests/`), #2007 (G0 bench + serve runs), #2008 (tests, slot 3), #2018 (slot 1b: ids, memory, load; its accuracy runs failed on the unstamped binary), #2019 (env-mode tests), #2032 (slot 5 accuracy), #2033 (slot 6), #2034 (slot 7 accuracy repeat), #2035 (slot 8 final tests). Every GPU run (including every `cargo test` execution) ran under `fleetq acquire --host endor --exclusive --wait` with `FLEETQ_JOB` exported and released by trap; builds and `cargo test --no-run` compiles ran outside slots and execute no GPU work. No slot was left held.
