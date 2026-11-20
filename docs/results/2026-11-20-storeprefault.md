# storeprefault: take new-host-page cost off the prefix store's critical path (opt-in `INTARSIA_PREFIX_STORE_PREFAULT`)

Lane `storeprefault`, branch `storeprefault` stacked on `storettft` (PR #466, not merged at the start), 2026-11-20 03:35 to 07:45 EST, fleetq owner `claude-overnight-storeprefault`, box spark2 (GB10, driver 580.178.04, vLLM `retro` resident and untouched), Qwen3.6-35B-A3B NVFP4 on CUDA. Raw: `docs/results/2026-11-20-storeprefault/raw/`. Scripts: `scripts/storeprefault/` (copies of storettft's, with arms added).

## Verdict

1. **Gate 1 met.** Fresh server, back-to-back cold prompts, `INTARSIA_PREFIX_STORE_MACRO=1` + **`INTARSIA_PREFIX_STORE_PREFAULT=1`** against store off, 4 ABBA visits per arm, 3 reps each: TTFT **+13.6 / +4.5 / +2.2 %** at 4k / 16k / 32k (16k and 32k within 5 %, 16k with a margin of 0.5 points: over the 16 pairings of a pf visit with an off visit the 16k ratio is 1.037 to 1.051; 4k recorded: cause B, the store's schedule, plus no macro step at 4k). #466 alone (smacro) in the same windows: +29.2 / +20.8 / +16.5 %. (Ratios from localeval's millisecond medians, `raw/ttft-precise-w1w4.md`; the 10 ms console values give the same to 0.5 points.)
2. **The cause, found by microbenchmark:** the store's host buffers are `vec![0u8; n]`, whose data starts 16 bytes into a page, and on the GB10 the FIRST device-to-host copy into such a new buffer runs at about 1.3 GB/s (into a page-aligned one: 54 GB/s, first copy or not). That, plus 4 KiB page faults, was storettft's cause C ("fresh host pages"). The knob gives every new host buffer its own page-aligned mapping, 2 MiB-aligned with `MADV_HUGEPAGE` when it is 2 MiB or more. The settle of a cell of 11 requests: **5.0 s to 0.54 s**. No helper thread, no pool, no pinned memory, no change to when bytes land (so no landing / restore ordering to get wrong; no concurrency test needed: nothing concurrent was added).
3. **Gate 2 met:** back-to-back (the w1 / w4 cell itself) pf is 12 / 14 / 12 % faster in TTFT than #466 alone (0.670 / 2.625 / 5.840 against 0.765 / 3.040 / 6.655 s). A fresh server now matches storettft's warm-host-tier numbers.
4. **Gate 3 met:** replies identical to store-on (smacro): ab35d replay seeds 0 and 1, 48 of 48 in all 8 comparisons; storettft's 12-size cold + restored greedy set 24 of 24, identical cached-token counts.
5. **Gate 4 met:** 2-stream replay TTFT sum, both seeds, pf 144.0 s against smacro 152.1 s (-5.3 %); decode and combined output not worse (combined 66.5 / 65.7 against 62.7 / 65.3 tok/s).
6. **Gate 5 met:** no pool and nothing pinned, so the allowance is 0; the MemAvailable floors are the same as #466's within the visit-to-visit spread (speed cells: pf lowest 41.09 GB, smacro lowest 41.16 GB, a 0.07 GB difference, visits 41.1 to 41.7 vs 41.2 to 41.6 GB; replay: pf 37.4 / 37.4 vs smacro 36.0 / 35.9 GB); never under 15 GB.
7. **Gate 6 met:** `cargo test -p intarsia-core -p intarsia-models` and `-p intarsia-cli --lib --tests` on the Mac: 1042 passed, 0 failed, 28 ignored; the prefix store tests on spark2 (Linux, where the `MADV_HUGEPAGE` branch compiles): 8 of 8. Three new tests (`host_buf_tests`).
8. Also measured: with a warm host tier pf is the same as smacro (not slower); `pf4k` (alignment without huge pages) gets most of the gain (+13.7 / +6.7 / +4.2 %). Without #466's knob (plain `--prefix-store on`), `PREFAULT=1` alone cuts TTFT by 10 / 15 / 13 % (w7).
9. Opt-in only, default off (lane rules); flipping it is the user's call. It pairs with #466's knob but does not need it.

## What the slow "fresh host page" really is (refines storettft's cause C)

storettft measured that a never-used host buffer cost about 3.6 GB/s to populate plus a "first-use copy at 7 GB/s", while a recycled one copied at 43 to 55 GB/s. Three microbenchmarks on spark2 split that up (every buffer new pages, kept alive while measured):

- `raw/d2h4.txt` (`d2h4.cu`): **the slow first copy is about where the buffer starts, not how old its pages are.** A device-to-host copy into a new mapping that starts on a page: 5 MiB in 0.10 ms, 32 MiB in 0.56 ms (54 GB/s), the same as the second copy, also when it follows the page touch immediately (mode `I`). The same copy starting 16 bytes into the page (modes `O`, `J`): 3.8 to 4.1 ms (5 MiB) and 3.8 to 4.7 ms (32 MiB), about 1.3 GB/s; the second copy is fast again. `calloc` (mode `C`, what Rust's `vec![0u8; n]` does for a large block: glibc puts a 16-byte header first, so the data starts 16 bytes into a page) is the misaligned case: 3.9 / 4.2 ms. `posix_memalign(4096)` (mode `A`) is fast.
- The store's `host_buf` is `vec![0u8; n]` plus a write per 4 KiB page: aligned to 16 bytes, not to the page. That matches the debug lines of storettft's fresh server (`demote 5120 KiB: alloc 0.85 ms, download 4.0 ms`, `demote 32160 KiB: alloc 8.0 ms, download 4.1 ms`), and why a recycled buffer (copied into before) is fast.
- Populating the pages (the `alloc` part): about 3.5 GB/s with 4 KiB pages (5 MiB 1.0 to 1.6 ms, 32 MiB 6.8 to 9.5 ms), 13 to 25 GB/s with `MADV_HUGEPAGE` (0.38 / 1.25 ms; `raw/d2h3.txt` mode `H`; THP is `madvise` on spark2).
- The brief's candidates, measured (`raw/d2h3.txt`, per 32 MiB buffer, prep = on a helper thread): priming with a D2H copy from a scratch buffer (`P`) 9 to 10 ms, priming with an H2D read (`R`) 10 ms, `cudaHostRegister` (`G`) 8.7 to 12.6 ms prep: each costs as much as today's whole demote, only moved to another thread, and a pinned pool is unreclaimable memory. Alignment plus huge pages removes the cost at the source (1.25 ms prep, 0.56 ms copy), on the scheduler thread, with no new thread, no pool, no pinning and no change to when bytes land (so no landing/restore ordering to get wrong). Helper-thread contention check (`d2h3.txt`, last lines): a memory-bound D2D stream ran 0.6 to 1.2 % slower while a helper primed buffers.
- `raw/d2h3-calloc.txt` is a first, flawed run of `d2h3.cu` (calloc / free reused the freed pages, so every mode after the first was not fresh); kept for the record, not used.

## Design: `INTARSIA_PREFIX_STORE_PREFAULT` (default off)

- `1`: every NEW host buffer of the prefix store (a demote at the settle, a capture past the 1 GiB deferral cap, `take_host_buf` when nothing recycled fits) is an anonymous mapping of its own, starting on a 2 MiB boundary with `MADV_HUGEPAGE` over its whole 2 MiB blocks when it is 2 MiB or larger (the tail past the last whole block stays 4 KiB pages, so no huge page is half used), page-aligned otherwise; every page touched as before. `4k`: page-aligned mappings, 4 KiB pages (the control that separates alignment from huge pages). Unset or anything else: `vec![0u8; n]` as before.
- Code: `HostBuf` in `crates/intarsia-core/src/prefix_store.rs` (an enum of `Vec<u8>` and an owned mapping; derefs to the object's bytes; unmaps on drop) replaces `Vec<u8>` as the host tier's buffer type (`Mem::Host`, `staged`, `host_free`, the disk writer's queue, disk loads). Recycling (`host_free`), budgets, eviction, the disk tier and restores are unchanged; a mapping that fails falls back to a `Vec`.
- Bitwise: the bytes are the same bytes (the buffer's address changes, nothing reads its alignment); replies checked below.
- Memory: the same bytes as before (mappings trimmed to the page; huge pages only over whole 2 MiB blocks); one VMA per object, two for an object of 2 MiB or more (the `madvise` over its whole 2 MiB blocks splits off its 4 KiB tail): about 4500 for the 8 GiB default host tier of the 35B, against `vm.max_map_count` 65530.
- First version (window `wd`) mapped only buffers of 1 MiB or more: the debug lines showed the first download into a new misaligned `Vec` slow at EVERY size the store makes (4 KiB 0.6 ms, 6 KiB 1.5 ms, 512 KiB 1.4 ms: the 406 head-page objects of 512 KiB were half of an 885 MiB settle's 300 ms), so the final build maps every non-empty buffer.

## Method

- Cell: storettft's and ab35d's. `localeval speed --structured prompt --thinking off --sizes 4096,16384,32768 --gen 256 --reps 3` (localeval 0.4.0), nonce first in every prompt (no prefix reuse), one request at a time, **back-to-back** (the next request goes out the moment the previous stream ends: the regime where the previous request's settle lands in the next TTFT). Every visit is a FRESH `intarsia serve` (this branch's binary, CUDA, `--max-seq 131072 --mem-budget 40 --profile-every 0 --max-batch 2` plus the arm's flags / env; MTP N=2 is the recipe default), the visit's value is the median of its 3 reps, the arm's the mean of its visits, visits in ABBA order. Serve line and env of every visit: `raw/recipes/*.txt`.
- Arms: `off` (store off: the control), `smacro` (`--prefix-store on` + `INTARSIA_PREFIX_STORE_MACRO=1`, i.e. #466 alone), `pf` (smacro + `INTARSIA_PREFIX_STORE_PREFAULT=1`, this lane), `pf4k` (smacro + `INTARSIA_PREFIX_STORE_PREFAULT=4k`), `on` / `onpf` (the shipping store without / with `PREFAULT=1`, no macro knob), `pfdbg` (pf + `INTARSIA_PREFIX_STORE_DEBUG=1`, timings only).
- Clock guard (lane rules, 2026-11-19 addendum) on every cell: SM clock / util / power every 2 s, SW power capping counter before and after, vLLM request counters before and after. No cell tripped: SM min 2353 MHz, 2372 MHz in all other cells, mean 2384 to 2391 MHz at > 90 % util, max power 83 to 92 W, SW power cap delta 0 us in every cell, vLLM request delta 0 in every cell (`raw/window-*.log`, `raw/marks-*.tsv`, `raw/clk-*.tsv`).
- Memory: `memwatch.sh` logs MemAvailable every 2 s per visit and kills the server under 15 GB (no KILLED line anywhere); builds ran under the Spark-side watchdog (lowest 73 GB available). THP counters (`thp_fault_alloc`, `thp_fault_fallback`, `compact_stall` from `/proc/vmstat`) logged at the start and end of every cell, and the server's `AnonHugePages` (window2.sh onwards; in w1 the line read the supervisor process, not the server).
- Two binaries: build 1 (mapped buffers only from 1 MiB; window `wd`, diagnostic only) and build 2 (this PR's code, every new buffer mapped): all other windows.

## Results

### Gate 1 and 2: cold TTFT, fresh server, back-to-back (windows w1 and w4, ABBA, build 2)

TTFT s (visit medians, then arm mean); ratio to `off` in brackets. w1: `off smacro pf pf4k pf4k pf smacro off`; w4: `off pf pf off`.

| arm | visits | 4096 | 16384 | 32768 |
|---|---|---|---|---|
| off | 4 | 0.593 | 2.510 | 5.715 |
| smacro (#466 alone) | 2 | 0.765 (1.291) | 3.040 (1.211) | 6.655 (1.164) |
| **pf (this lane)** | 4 | **0.670 (1.131)** | **2.625 (1.046)** | **5.840 (1.022)** |
| pf4k | 2 | 0.670 (1.131) | 2.685 (1.070) | 5.955 (1.042) |

Per visit (`raw/tables-w1w4.md`): pf 0.67 / 0.67 / 0.67 / 0.67, 2.63 / 2.63 / 2.62 / 2.62, 5.86 / 5.83 / 5.84 / 5.83; off 0.59 / 0.60 / 0.59 / 0.59, 2.52 / 2.50 / 2.51 / 2.51, 5.75 / 5.70 / 5.70 / 5.71. w4 alone: pf 1.136 / 1.044 / 1.023. From localeval's millisecond medians (`raw/ttft-precise-w1w4.md`): pf 1.1356 / 1.0447 / 1.0222, smacro 1.2924 / 1.2084 / 1.1647, pf4k 1.1373 / 1.0673 / 1.0420; pf / off over all 16 visit pairings 1.119 to 1.147, 1.037 to 1.051, 1.015 to 1.028. Against storettft's numbers for the same regime: smacro 1.31 / 1.19 / 1.19 (w9) and its best case, a warm host tier, 1.13 / 1.04 / 1.02 (w4 there): **a fresh server with `PREFAULT=1` is where a warm one was.** `pf4k` (alignment only) gets most of it; huge pages take the rest (16k 1.070 to 1.046, 32k 1.042 to 1.022).

Store counters per cell of 11 requests (`raw/store-counters-w1.md`, from the `/metrics` snapshots): the settle took **5.00 / 5.01 s with smacro, 0.54 / 0.54 s with pf**, 1.46 / 1.41 s with pf4k; the inserts themselves (scheduler-thread time minus the settle) 7.27 s either way; 33 inserts and 4.88 GB in every store cell.

Decode tok/s (single stream, arm mean of visit medians): off 119.6 / 101.3 / 81.6, smacro 120.1 / 100.3 / 84.3, pf 120.9 / 100.2 / 81.3, pf4k 121.0 / 98.8 / 83.2 (4k / 16k / 32k): no change.

Where the 4k residual is: cause B (the store's schedule, 0.05 s at 4k in storettft) and the macro step that a 4k prompt cannot use; storettft's warm-tier and idle-gap cells were 1.13 / 1.14 at 4k too. Not this lane's.

### Gate 3: replies identical to store-on (smacro)

- Cold and restored greedy replies (storettft's `ids_check.py`: 12 fixed prompts of 1.5k to 33k tokens, each sent cold then again, restored from the store; 64 tokens; window w2, `raw/compare-ids-w2.txt`): **pf vs smacro 24 of 24 identical**, cached-token counts of the restored sends identical at all 12 sizes (1441, 2920, ..., 34064); both also 24 of 24 identical to storettft's smacro run (`w11-v3-smacro-ids.json`, the #466 binary).
- 2-stream interleaved replay (ab35d's plan and traces, `scripts/interleave/replay2.py`, seeds 0 and 1, 48 greedy requests per run, window w3 ABBA `smacro pf pf smacro`; `raw/compare-replay-w3.txt`): **pf vs smacro 48 of 48 identical in all 8 comparisons**, and identical to storettft's smacro replay (`w10-v2-smacro`) in both seeds. Cached-token fraction 0.732 in every run. The replay restores from host buffers (uploads), so the mapped buffers' H2D path is exercised too.

### Gate 4: 2-stream replay TTFT sum and decode (window w3)

| visit | seed | errors | TTFT sum s | TTFT p50 / p90 s | decode tok/s per stream (median) | combined out tok/s | wall s |
|---|---|---|---|---|---|---|---|
| w3-v1-smacro | 0 | 0 | 81.1 | 1.56 / 2.30 | 45.9 | 62.8 | 120.9 |
| w3-v1-smacro | 1 | 0 | 73.1 | 1.50 / 2.07 | 47.4 | 65.3 | 112.0 |
| w3-v2-pf | 0 | 0 | 70.3 | 1.46 / 1.59 | 47.4 | 66.5 | 114.1 |
| w3-v2-pf | 1 | 0 | 74.9 | 1.42 / 2.53 | 48.9 | 65.7 | 111.4 |
| w3-v3-pf | 0 | 0 | 69.6 | 1.40 / 1.57 | 47.7 | 66.5 | 114.2 |
| w3-v3-pf | 1 | 0 | 73.2 | 1.48 / 2.09 | 47.3 | 65.6 | 111.5 |
| w3-v4-smacro | 0 | 0 | 74.9 | 1.51 / 1.79 | 44.9 | 62.5 | 121.5 |
| w3-v4-smacro | 1 | 0 | 75.0 | 1.41 / 2.56 | 48.1 | 65.2 | 112.3 |

Both seeds summed, mean of the 2 visits: **pf 144.0 s against smacro 152.1 s (-5.3 %)**; seed 0 70.0 vs 78.0 s, seed 1 74.1 vs 74.1 s. Decode median per stream pf 47.4 / 47.7 and 48.9 / 47.3 against 45.9 / 44.9 and 47.4 / 48.1; combined output 66.5 / 66.5 and 65.7 / 65.6 against 62.8 / 62.5 and 65.3 / 65.2 tok/s; wall 114.1 / 114.2 against 120.9 / 121.5 s on seed 0. Not worse anywhere; seed 0 better. (Full table with client ITL: `raw/replay-table-w3.md`.)

### Gate 5: memory

Lowest MemAvailable per visit (`raw/mem-*.txt`, 2 s samples): speed cells pf 41.09 / 41.72 / 41.67 / 41.74 GB, smacro 41.57 / 41.16 GB, pf4k 41.73 / 41.71 GB, off 46.2 to 46.6 GB; ids cells pf 41.57, smacro 41.63 GB; replay pf 37.36 / 37.35 GB, smacro 36.03 / 35.89 GB. **pf's floors are smacro's within the visit spread** (lowest to lowest 0.07 GB lower in the speed cells, 1.3 GB higher in the replay); no pool is pinned (nothing is pinned or locked: the buffers are ordinary anonymous memory, the same bytes the host tier held before), never under 15 GB, no KILLED line. Server RSS after a speed cell: pf 6.86 / 6.83 GB (of which `AnonHugePages` 3.7 GB), smacro 6.97 GB (w2: smacro 6.97 GB, 0 huge; pf 7.05 GB, 3.86 GB huge). THP: `thp_fault_fallback` 0 and `compact_stall` 0 in every cell (spark2, 70+ GB free; with a fragmented box the huge-page fault can stall on compaction, `defrag=madvise`: not seen here, not tested under fragmentation).

### Not slower with a warm host tier (window w6)

Steady state as in a long-running server: 8 requests of 32k before the measured cell, `--prefix-store on,host_mb=4096` (host tier at 3.46 GiB after the fill, then evicting and recycling), ABBA `smacro pf pf smacro`: TTFT smacro 0.675 / 2.630 / 5.850 s, **pf 0.675 / 2.630 / 5.820 s** (`raw/tables-w6.md`). With recycled buffers the knob has nothing left to do, and costs nothing. A fresh server with `pf` (w1 / w4: 0.670 / 2.625 / 5.840) now matches this warm state.

### Where the settle went (windows wd and w5, `INTARSIA_PREFIX_STORE_DEBUG=1`, timings only)

| | storettft fresh (`on`, w1 there) | build 1, mapped from 1 MiB (wd) | build 2, every buffer mapped (w5) |
|---|---|---|---|
| settle after a 16k request | 641 MiB in 600 to 618 ms | 452 MiB in 147 to 153 ms | 452 MiB in 48 to 52 ms |
| settle after a 32k request | - | 885 MiB in 296 to 299 ms | 885 MiB in 97 to 104 ms |
| demote 5120 KiB (KV page): alloc + download | 0.85 + 4.0 ms | 0.32 + 0.10 ms | 0.32 + 0.11 ms |
| demote 32160 KiB (recurrent blob) | 8.0 + 4.1 ms | 0.91 + 0.59 ms | 0.91 + 0.57 ms |
| demote 512 KiB (MTP head page) | - | 0.13 + 1.43 ms | 0.13 + 0.02 ms |

(storettft's cell had 2048-token captures, these have `INTARSIA_PREFIX_STORE_MACRO=1`, so the settle sizes differ; the per-object lines compare like for like.) With the macro knob no capture of these cells went past the 1 GiB deferral cap (all 33 inserts `deferred`, w5), so cause C1 does not occur in this regime; captures to host (`insert ... host`) take buffers from the same `take_host_buf`, so they get the same allocation when they happen. What remains of the settle (about 9 GB/s) is mostly the page touch (0.32 of each 5 MiB page's 0.43 ms: a 5 MiB page object is 2 huge pages plus a 1 MiB tail of 4 KiB pages), then the copies; it is now 0.54 s per cell of 11 requests against 5.0 s.

### Shipping store without the macro knob (window w7)

`--prefix-store on` alone (captures after every 2048-token chunk, no macro knob) without / with `INTARSIA_PREFIX_STORE_PREFAULT=1`, ABBA `on onpf onpf on` (`raw/tables-w7.md`): TTFT on 0.765 / 3.335 / 7.300 s, **onpf 0.685 / 2.840 / 6.345 s (-10 / -15 / -13 %)**; settle per cell 4.17 / 4.09 s against 0.51 / 0.51 s (88 inserts each). Against the store-off control of w1 / w4 (other windows, same box and binary): on 1.29 / 1.33 / 1.28, onpf 1.16 / 1.13 / 1.11. The knob helps the shipping store by itself; with #466's knob as well, both gaps (cause A and C) close and 16k / 32k reach the gate.

## Not verified / limits

- One model (35B NVFP4, MTP N=2), one box (spark2, THP `madvise`, `defrag=madvise`, 70+ GB free), CUDA only. HIP (`ik_rt_memcpy_d2h_pinned` staging there) and Metal were not run; the knob only changes how host memory is allocated, which they also use, but whether misaligned first copies are slow there is not measured.
- Huge pages under memory fragmentation: `compact_stall` and `thp_fault_fallback` stayed 0 in every cell here. On a fragmented box a `MADV_HUGEPAGE` fault with `defrag=madvise` can stall on direct compaction; `4k` avoids that and keeps most of the gain.
- Why the driver's first copy into a misaligned pageable buffer is slow (and later ones fast) is not known; measured, not explained. The 4 KiB / 6 KiB objects stay at 0.35 to 0.5 ms per first download even mapped (33 + 22 per cell, negligible).
- The disk tier (off by default) was not exercised; disk loads come back as plain `Vec` buffers (`HostBuf::Vec`), unchanged.
- Not verified with `--max-batch 1`, packed prefill, or another model's object sizes; `vm.max_map_count` headroom was reasoned (about 4500 mappings for the 8 GiB tier of this model), not measured at a full 8 GiB tier with small objects.
- Cause C1 (captures past the 1 GiB deferral cap, synchronous inside the prefill) did not occur in these cells with the macro knob; with the plain store (`on`, captures every 2048 tokens) it does at 32k, and the knob gives those buffers the same fast allocation (w7), but a larger deferral cap was not re-tested now that the settle is cheap.

## Revert

Unset `INTARSIA_PREFIX_STORE_PREFAULT` (default): `new_host_buf` then makes the same `vec![0u8; n]` + page touch as before (`HostBuf::Vec`). `git revert` of this branch's commits removes the knob, `HostBuf` and the tests; the host tier's type goes back to `Vec<u8>`.

## Raw

`raw/window-*.log` (run logs, clock guard and THP lines), `raw/recipes/*.txt`, `raw/w*-speed.out` and `raw/2026*/summary.md` (localeval cells), `raw/tables-w*.md` (`scripts/storeprefault/tables.py`), `raw/store-counters-w1.md`, `raw/w2-*-ids.json` and `raw/compare-ids-w2.txt`, `raw/w3-*` and `raw/replay-table-w3.md`, `raw/compare-replay-w3.txt`, `raw/serve-*.log` (wd and w5 with the debug lines), `raw/mem-*.txt`, `raw/clk-*.tsv`, `raw/snap-*` (metrics before / after each cell), `raw/d2h3.txt`, `raw/d2h3-calloc.txt` (flawed first run), `raw/d2h4.txt`, `raw/rtest-linux.txt`, `raw/run-*.txt` and `raw/build2.txt` (fleetq job output).
