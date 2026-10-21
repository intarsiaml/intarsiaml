# Metal prefill GEMM at M = 256 and mid-size M: where the gap is, and a tile table for 513..1344 rows (2026-10-15)

> **Update (2026-10-21, docs-accuracy pass, see `docs/decisions/LOG.md`):** Merged to main in `62bb22d1` (Merge metal-gemm-midm 9190772); "not pushed" / "nothing merged" wording in this page describes the state when written.

Branch `metal-gemm-midm` (from spike/metal 56ec2d5), Qwen3-8B BF16, endor (M5 Max, 40 GPU cores), `--backend metal`. Raw logs, scripts and the summaries below: `docs/results/raw/2026-10-15-metal-gemm-midm/` (per-run traces are in `eng*.tar.gz`, `chunk.tar.gz`, `cb*.tar.gz`; `tar xzf eng3.tar.gz` there recreates the directory the `enga.py` / `engdense.py` / `cbsum.py` / `cbchunk.py` / `chunksum.py` scripts read; the `summary*.txt` files stay unpacked). Code: `crates/intarsia-backend-metal/src/gemm_nax.rs` (tile pick), `examples/gemm_nax.rs` (bench modes `sweep`, `waves`, `pick3`), `tests/gemm_nax.rs`. Nothing outside the Metal GEMM files changed; not merged, not pushed.

## Outcome

1. **The M = 256 GEMM is not where the 256-chunk prefill loses time.** Per layer, the four Qwen3-8B GEMM classes run at 49 to 54 TFLOPS at M = 256 and 44 to 57 at M = 2048 (same bench, interleaved). Eight 256-row chunks cost 0.97x of one 2048-row chunk in GEMM time. In the engine at 4k, chunk 256 against 2048 costs **+4.6% GPU busy, +9.3% command-buffer span, +8.7% prefill wall** (cb mode, interleaved, 8 runs each). About half of the wall penalty is idle time between command buffers (host side), and the GPU-busy part is attention (see the breakdown). The "~7% slower at 256" estimate in `continuous-batching-design.md` 8.1 holds (measured 8.7% at 4k, 14.7% at 1k), but closing it is a pipeline/attention job, not a GEMM-tile job.
2. **Two honest attempts at a faster M = 256 GEMM found nothing.** Tile shapes (eight tried, 32x32 to 96x128) and threadgroup order (swizzle 0/1/2): no gain beyond noise. Split-K is excluded by the bitwise rule.
3. **Mid-size M had a real, measurable loss and is fixed.** 128x64 four-simdgroup tile (the P1 pick) is up to 11% slower per layer than a 64x64 tile at 513..1344 rows. A per-64-row-step table (`mid_tile`) replaces it. In-engine per-layer GEMM time: **-3.5% to -11% at 585..1000 rows, -4% at 1230..1330**, 0% where the old pick was already best; GPU busy of a whole prefill (cb mode, 6 runs per arm): **-2.1% (1k), -0.9% (2688), -3.0% (3000)**, and -0.1% at 4062 where the tile does not change (control). Bits unchanged.
4. **Wall-clock TTFT through `localeval speed` could not resolve this** (3% effect, 4% noise between servers started minutes apart). GPU busy is the number to trust; localeval is reported below as measured.

Merge-ready: **yes** (all gates pass on HEAD 20c0a93, output bitwise identical to the 56ec2d5 pick). Gain is small (2 to 3% of a 1k..3k prefill's GPU time, nothing at 4k+ or at chunk 256), so it is a judgement call whether it is worth carrying.

## Method

- Bench harness: the repo's `gemm_nax` example (new modes `sweep`, `waves`, `pick3`; all cells and arms interleaved, rotating 3 GiB weight pool so weights stream from DRAM as in a 36-layer forward). In-engine: `INTARSIA_METAL_PROFILE=kernel:` traces of a single-chunk prefill with `INTARSIA_NAX_TILE` forcing one tile, or `INTARSIA_NAX_PICK=p45` for the old pick, arms interleaved with rotated order (`engine-sweep*.sh`, `engine-ab*.sh`, `enga.py`, `engdense.py`).
- Whole-prefill GPU busy: `INTARSIA_METAL_PROFILE=cb:` (sum of command-buffer GPU times), ABBA/BAAB per round, GPU idle-gated (`waitidle.sh`), `cbsum.py`, `cbchunk.py`.
- GPU clock: `macmon pipe -i 200` next to every run (`clk.py`, `*.macmon.jsonl`). Median busy-sample clock: bench sweeps 1047 to 1355 MHz (run to run), localeval servers 1290 to 1362 MHz. Clock state moves TFLOPS by 10% between runs, so only cells measured in the same interleaved run are compared.
- All GPU commands in fleetq exclusive slots, each released: #2041, 2044, 2046, 2050, 2054, 2060, 2069, 2070, 2076, 2085, 2094, 2099, 2105, 2111, 2112, 2121, 2132, 2133, 2134. Cancelled by me after a bad idle gate (20-minute waits): #2110, #2119, #2098 (duplicate).

## TFLOPS and per-layer time against M (shipped pick at the time = the old pick, `sweep2.txt`)

TFLOPS and us per call for the engine's launches: q and o single GEMM 4096x4096, k+v pair 2x1024, gate+up pair 2x12288, down K = 12288, lm_head 151936x4096 (one call per prefill at M = 1 in the engine, shown for the shape).

| M | q / o | k+v pair | gate+up pair | down | lm_head |
|---|---|---|---|---|---|
| 64 | 18.3 TF, 117 us | 12.5, 86 | 22.8, 565 | 17.5, 367 | 19.5, 4088 |
| 128 | 36.3, 118 | 26.0, 82 | 42.7, 604 | 35.5, 363 | 38.2, 4165 |
| 256 | 54.0, 159 | 49.4, 87 | 53.1, 970 | 48.6, 530 | 52.9, 6024 |
| 384 | 50.2, 257 | 52.8, 122 | 50.3, 1538 | 45.3, 853 | 49.3, 9703 |
| 512 | 56.6, 304 | 53.1, 162 | 51.9, 1987 | 52.3, 986 | 46.3, 13754 |
| 1024 | 54.6, 630 | 51.6, 333 | 51.8, 3980 | 48.7, 2115 | 44.2, 28832 |
| 1366 | 55.1, 832 | 51.0, 449 | 49.9, 5509 | 46.1, 2985 | 42.2, 40280 |
| 1536 | 56.9, 906 | 53.7, 480 | 51.7, 5985 | 44.1, 3506 | 44.2, 43279 |
| 1850 | 54.5, 1140 | 53.3, 582 | 50.1, 7434 | 41.2, 4514 | 42.1, 54632 |
| 2048 | 57.4, 1197 | 52.4, 655 | 51.9, 7943 | 44.4, 4646 | 46.5, 54770 |

Per layer (2 x q/o + k+v + gate+up + down), `layer.py sweep2.txt`: M = 64 1.25 ms, 128 1.29, 256 1.91, 384 3.03, 512 3.74, 1024 7.69, 1366 10.61, 1536 11.78, 1850 14.81, 2048 15.64 ms. 4k prompt, 36 layers: 16 x M = 256 = 1097 ms, 2 x M = 2048 = 1126 ms (**ratio 0.974**).

(Microbench TFLOPS run at 1.0 to 1.4 GHz; in-engine GEMMs at 2048 rows measure 59 to 65 TFLOPS at higher clock in the P4/P5 traces. Ratios between M are the usable part.)

## What the gap is made of

Efficiency loss against the ~64 TFLOPS the tile reaches with all cores filled (`waves1.txt`, cache-resident operands):

- **Below M = 128: weight streaming, not compute.** Time is flat in M: q/o 117 us at M = 64 and 118 at M = 128 (32 MB of weights: ~280 GB/s; gate+up 333 to 356 GB/s, down 274, lm_head 304). The kernel reaches roughly half of the 614 GB/s DRAM peak, and below ~128 rows that is the floor. At M = 256 the weights stream at 190 to 210 GB/s (q/o 211, gate+up 207, down 190, lm_head 207), a third of peak, so M >= 256 is not bandwidth bound.
- **Wave/tail quantisation of the grid on 40 cores.** The `waves` probe (one tile over T threadgroups, K = 4096): 64x64 four-simdgroup tile 46 us for T = 8..32, 53 at 40, 56 at 64, 66 at 80..96, 84 at 120..160, then 138 at 200 and a steady 0.57 us per threadgroup from 320 up (= 59 TFLOPS). Consistent with about 160 resident threadgroups (four per core) per wave. q/o at M = 256 is 256 threadgroups on the 64x64 tile = one full wave plus 96: probe predicts ~150 us, measured 159 (ideal at the 64 TFLOPS ceiling: 134 us). That is the 16% under the ceiling at M = 256. At M = 2048 (1024 threadgroups of 128x64) the grid is many waves and the tail is a small fraction, but the 128x64 tile has its own loss and also measures 52 to 57 TFLOPS. Net gap between 256 and 2048 in GEMM: within the 10% between-run noise of the bench.
- **Weight re-read per M tile.** Each M tile row re-reads the full weight. Swizzling the threadgroup order (0, 1, 2) changes TFLOPS by at most 1.5% at M = 256 and 512 (`swz-pool3072.txt`, 21 rounds): the re-reads are served from cache, not DRAM. Not a cost at this M.
- **Padding.** A 64-row tile at M = 641 pads 63 rows. The 32-row tail tile (32x128) removes that for q/o and down at chunk tails, but it is 9% slower than 128x64 at 2000 rows (15029 vs 13790 us per layer), so padding-minimising everywhere is wrong (see "Tried").
- **Launch overhead / host.** GEMM kernels at 256 rows are 90 us (k+v) to 970 us (gate+up) each, so per-kernel launch cost is not the issue. The cost is per command buffer: at 4062 tokens chunk 256 has 48 command buffers against 6, and the GPU sits idle between them: span - busy = 54 ms against 1 ms (`cbchunk/summary.txt`).

Chunk 256 against 2048, 4062 tokens, cb mode (`cb-chunk.sh`, 8 interleaved runs each, idle-gated):

| prompt | chunk | GPU busy ms | span ms | prefill wall ms | cbs |
|---|---|---|---|---|---|
| 4062 | 256 | 1161 [1133..1462] | 1215 | 1352 | 48 |
| 4062 | 2048 | 1110 [1043..1375] | 1111 | 1244 | 6 |
| 1024 | 256 | 273 [264..284] | 321 | 433 | 12 |
| 1024 | 2048 | 254 [245..255] | 254 | 378 | 3 |

c256 / c2048 at 4062: busy 1.046, span 1.093, wall 1.087. At 1024: busy 1.073, span 1.265, wall 1.147. So at 4k: +51 ms GPU busy, +54 ms idle between command buffers, +~0 other. Attention is the busy part: per-kernel traces (`chunk/summary.txt`, kernel mode, 4 runs each) show attention 144 ms at c256 against 97 at c2048 (+48 ms, 1.49x) and the four GEMM classes at 46 to 52 TFLOPS against 57 to 64. Caveat: kernel mode serialises 5800 launches and sampled a lower clock (1414 MHz at chunk 256 against 1617 at 2048, few samples), so its GEMM column overstates the GEMM gap (1336 vs 1048 ms total kernel time against 1161 vs 1110 ms busy in cb mode, where the GEMM penalty is invisible). Reading of the data: the GEMM share of the 4k busy gap is small and cannot be separated from attention in cb mode; the 0.974 of the isolated bench is the cleanest GEMM number.

Where the 256-chunk time actually goes, in order: command-buffer gaps (~54 ms at 4k, ~48 at 1k), attention at short Q chunks (+48 ms, another session's area), GEMM (~0 to a few ms).

## Tried

**Attempt 1: tile shapes and order at M = 256 and 512** (`sweep3.txt`, `swz-*.txt`). Besides the shipped five tiles, built and measured 32x64/2, 64x32/2, 96x64/2, 64x96/2, 96x128/4 (simdgroups after the slash). Over the 48 cells of `sweep3.txt` (4 shapes, M = 256..2048) the best of the five new tiles is at most +2.7% over the best existing tile (median -4.5%): +0.8% at k+v M = 256, +2.4% at down M = 384, +2.7% at k+v M = 768, all inside the run-to-run noise. Swizzle 0/1/2: see above. Experimental tile kernels removed again. **No gain.**

**Attempt 2: split-K / wider N at small M.** Not built: split-K or K-parallel reduction changes the accumulation order along K and would make a row's bits depend on M (the invariant at `gemm_nax.rs:47`). Wider N tiles are the 32x128 and 64x96 shapes above. **No gain.**

**Attempt 3 (mid M): padding-minimising pick for every M > 512** (64x64/2 or 32x128/2 by `M mod 64`). Bench (`pick3-a.txt`, `pick3-b.txt`) predicted +4 to +40% at 513..1433; the engine disagreed at large M: cb-mode GPU busy of whole prefills with this rule was **+5.4% (1k), +6.6% (2688), +6.6% (3000), +5.5% (3328), +13.3% (4062)** slower than the old pick (`cb-padaware/`), because 32x128 loses 8% on a 2014-row call (`eng3`: 13790 vs 15029 us per layer at 2000 rows). Dropped.

**Attempt 4 (mid M): in-engine per-step table.** Dense in-engine sweep, all four candidate tiles forced for every call, 2 rounds, M = 640..2048 in 25 steps (`eng3/summary.txt`, `engdense.py`). Fastest tile changes from one 64-row step to the next (wave quantisation of the grid, not only padding), so no closed form fits: a wave model from the cache-resident probe picks within 3.6% of the oracle at some sizes and 11% off at others (`wavemodel.py`). The table: ceil(M/64) = 11..12 (641..768) 64x64/4, 13 (769..832) 64x64/2, 14..17 (833..1088) 64x64/4, 19 (1153..1216) 64x64/4, 20 (1217..1280) 64x64/2, 21 (1281..1344) 64x64/4; step 18 (1089..1152) and >= 1345 keep 128x64/4 (the old pick, measured best or within 1%). Chunk range limit raised 512 -> 640 (`NAX_CHUNK_MAX_M`). Table entries were taken from one dense sweep and checked on held-out sizes.

In-engine per-layer GEMM time, old pick -> table, interleaved kernel traces (`eng2/`, `eng4/`, `eng5/`; us per layer, [min..max] over rounds):

| M | old | new | change |
|---|---|---|---|
| 585 | 4369 | 4150 | -5.0% (6 rounds) |
| 619 | 4365 | 3919 | -10.2% (6) |
| 640 | 4481 | 4000 | -10.7% (6) |
| 663 | 4932 | 4727 | -4.1% |
| 745 | 5484 | 5157 | -6.0% |
| 830 | 5594 | 5222 | -6.6% |
| 930 | 6530 | 5852 | -10.4% |
| 1000 | 6570 | 6343 | -3.5% |
| 1075 | 7012 | 6850 | -2.3% |
| 1130 | 7021 | 7017 | 0.0% (old pick kept) |
| 1230 | 8200 | 7845 | -4.3% |
| 1330 | 8658 | 8285 | -4.3% |
| 1400 | 8685 | 8692 | +0.1% (old pick kept) |
| 1470..2048 | | | 0.0% (old pick kept; spread < 0.2%) |

(`eng4/` is the held-out set: sum over its 19 sizes -2.1% including the unchanged ones. At M <= 512 old and new pick the same tile; a 3-round run of M = 64..512 shows +-20% between identical-tile arms in kernel-mode traces, which is the noise floor of that method and why only the 585..640 edge got 6 rounds.)

Whole prefill, GPU busy (cb mode, 6 runs per arm, ABBA/BAAB, idle-gated; `cb-abba-c2048.txt`, `cbsum.py 2048`), chunk 2048:

| prompt (tokens) | last chunk M | old pick ms | new pick ms | change |
|---|---|---|---|---|
| 1024 | 1024 | 254.4 [253.7..257.0] | 249.0 [248.6..250.7] | **-2.1%** |
| 2688 | 640 | 685.2 [684.3..688.9] | 678.9 [668.9..681.3] | **-0.9%** |
| 3000 | 952 | 772.2 [767.0..790.1] | 749.0 [747.2..780.3] | **-3.0%** |
| 4062 | 2048 + 2014 | 1045.0 [1043.0..1068.6] | 1044.0 [1042.7..1060.8] | -0.1% (unchanged tiles: control) |

The kernel-trace deltas predict the busy deltas except at 2688: per-layer -10.7% at 640 rows x 36 layers = 17 ms = 2.5% of that prefill, measured -0.9% (6 ms; ranges overlap). 3000 predicts 3.2%, measures 3.0%; 1024 predicts 3.6%, measures 2.1%.

Chunk 256 is unchanged by construction (M <= 512 hits the same tile in both arms, so the `localeval` c256 pair below is a noise measurement).

## Bits and accuracy

Every tile accumulates along K in the same order, so the change is a pure kernel selection. Evidence (HEAD 20c0a93, gates in `gates/`):

- New test `nax_mid_size_chunks_keep_row_bits` (`tests/gemm_nax.rs`): the same rows computed in 2048-row calls on the 128x64 tile and in calls of 513, 600, 640, 700, 800, 952, 1100, 1200, 1280, 1344, 1366, 1433, 1536, 1850, 2047, 2048 rows at other row offsets through the registry (single GEMM and pair kernel, k+v-sized N), bitwise equal. `nax_tile_pick_is_by_shape` pins the pick. Existing `nax_*` row-independence tests (row bits independent of call M, row position, tile) unchanged and passing.
- `cargo test -p intarsia-backend-metal --release -- --test-threads=1`: 103 passed, 0 failed, including `serving_selection_unchanged` (`tests/multirow.rs`), the G0 bitwise multi-row GEMV tests and `rows_bitwise_*`. M = 1 and M = 2..15 paths are untouched (`NAX_MIN_M` and the GEMV selection unchanged).
- `intarsia kernels test`: 27 kernels, 321 cases, all admitted.
- Greedy ids, Qwen3-8B, 4k prompt, 48 tokens: chunk 256 vs 128 / 512 / 1024 / 2048 identical, for both the old pick and the new pick; old vs new identical at chunks 128, 256, 512, 1024, 2048; 256 tokens short and 4k prompts old vs new identical.
- `intarsia accuracy --gate-class bf16`: old and new give the same KL to every digit printed. Short golden prefill top-1 100.00% KL 2.832e-4, decode 100.00% KL 3.313e-4; long golden prefill 99.72% KL 3.699e-4, decode 100.00% KL 3.371e-4; prefill-mode runs at chunk 128 / 512 / 2048 identical KL (2.832e-4). **KL movement: none** (bits unchanged), so no golden or baseline needs updating.
- `intarsia qualify --backend metal`: Verdict PASS, 13 stages, 11 pass, 0 fail, 2 skipped (no `--reference`, `--tp 1`); acc-short, acc-long, acc-4k all PASS; envelope and baseline written and gated.

## End to end: `localeval speed`, chunk 2048 and 256, old vs new, same binary

`INTARSIA_NAX_PICK=p45` selects the old pick in the same binary. `serve --prefill-chunk N`, four servers per chunk size in order old, new, new, old (`localeval-abba.sh`, GPU idle-gated), `localeval speed --gen 128 --thinking off`, 5 reps at chunk 2048, 3 reps at chunk 256. TTFT seconds, median over all reps of both servers [min..max].

Chunk 2048 (`localeval-c2048/`, `leval.py 2048`):

| prompt tokens | old TTFT | new TTFT | new vs old | prefill tok/s old / new |
|---|---|---|---|---|
| 990 | 0.250 [0.250..0.330] | 0.275 [0.250..0.320] | +10.0% | 3897 / 3630 |
| 2654 | 0.715 [0.680..0.790] | 0.765 [0.670..1.250] | +7.0% | 3700 / 3510 |
| 2975 | 0.930 [0.770..1.280] | 0.965 [0.750..1.380] | +3.8% | 3238 / 3164 |
| 4088 (control, tiles identical) | 1.655 [1.070..2.140] | 1.430 [1.100..2.010] | -13.6% | 2528 / 2963 |

**These do not resolve the effect and are not evidence for or against it.** The control row (identical kernels) moves 13.6% the other way. Cause is visible per server: the first two servers ran at ~1.9 to 2.0 s for 4088 tokens, the last two at ~1.1 s (`leval.py` per-server table: old-1 2.010, new-1 1.880, new-2 1.110, old-2 1.080; at 2654: 0.750, 1.100, 0.670, 0.710). Something on the box slowed the first half and I did not find out what (the idle samples before the servers show 0 to 23% device utilisation in single samples; the gate accepts three consecutive samples of 12% or less; no other fleetq job held endor). Within the fast pair: new 1.110 vs old 1.080 at 4088 (control, identical kernels), 0.670 vs 0.710 at 2654 (new faster), 0.780 vs 0.780 at 2975. A first pass with the same script showed the same slow/fast phases (`localeval-c2048-noisy/`). The GPU-busy table above is the measurement to use for chunk 2048.

Chunk 256 (`localeval-c256/`): 1k 0.270 s old and new (3641 / 3632 tok/s); 4k 1.180 old [1.170..1.300], 1.220 new [1.170..1.330] (+3.4%, 3466 / 3344 tok/s). The kernels are identical in both arms (M <= 512), so +3.4% is this method's noise floor at 4k. Chunk 256 against 2048 on the fast servers (not same-slot, indicative): 4k 1.18 to 1.25 s at 256 against 1.08 to 1.11 s at 2048, 1k 0.27 against 0.25 s: 256-chunk prefill is 8 to 14% slower in TTFT, consistent with the 8.7% (4k) and 14.7% (1k) of the cb-mode wall column above.

## Recommendation

- Keep `mid_tile` (small, bitwise-safe, measured in the engine); it gains 2 to 3% GPU time on prompts that end in a 513..1344-row chunk and nothing elsewhere. Chunk 256 does not benefit from any GEMM change.
- To reduce the 256-chunk cost, work on (a) the ~54 ms (4k) of GPU idle between the 48 command buffers (encode chunk N+1 while N runs, or fewer command buffers per chunk), and (b) attention at 256-row Q chunks (+48 ms at 4k, flash prefill, outside this branch). The GEMM share is within the noise of the isolated bench (0.974).
- A tile change cannot recover the M = 128..256 region by much: it is already at 36 to 54 TFLOPS with a weight-streaming floor of ~120 us per q/o call; M = 64 to 128 costs the same as 128 (flat time), so 128-row chunks gain nothing per token over 256 except latency.

## Not done / uncertain

- The `localeval` chunk-2048 pair is inconclusive (box-state drift between servers); no clean TTFT-level statement of the 2 to 3% gain. GPU busy is clean.
- The chunk 256 vs 2048 GEMM-only comparison relies on the isolated bench (0.974) and cb-mode totals; in-engine kernel-mode traces suggest a 15 to 25% GEMM gap at 256 but that mode serialises launches and sampled a lower clock; I could not separate the two causes. Treat the true GEMM gap at 256 as 0 to ~10%.
- `lm_head` is one M = 1 call per prefill in the engine (GEMV), so its M-curve above only matters if prefill logits for all rows are ever needed.
- Table entries come from M-steps sampled every 64..130 rows on one box and clock state; entries within ~1% of the old pick were left on the old pick.
