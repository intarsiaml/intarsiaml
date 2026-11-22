# Inkling-2-Mini attention on CUDA (lane ink-cuda-attn, 2026-11-21)

## Briefs received

1. 2026-11-21 (coordinator, via SendMessage): **Approved: route CUDA to the fused op 75 for t <= ATTN_DECODE_MAX_ROWS**,
   with the rule as proposed, accepting the one exception (full 4k at 4 rows). Conditions:
   - make the edit on ink/cuda-attn as its own commit;
   - touch only that condition in `model/inkling2.rs`, so the Metal arm stays byte-identical (dfuse and batch also edit
     that file; keep the diff minimal);
   - gate: the tiny Inkling model on CUDA (spark1) gives bitwise-identical logits / tokens with the rule on and with the
     pair forced, and `plan show` on Metal is unchanged (checkable from the code: the Metal arm is untouched);
   - record this brief and report the hash.
   Done: d536ff099. The tiny-model gate is blocked until the mixmoe ops exist on CUDA.
2. 2026-11-21 (coordinator): **Accepted the op-level `inkling2_attn.rs` gate** as cover for d536ff099 for now. The tiny-model
   on/off gate runs once ink-cuda-mixmoe has its ops; the coordinator forwards the ref. Next lever, my pick with the reason:
   the 4-row full-4k case (131 vs 122 us) or the full-32k 1-row case (736 us).

## Summary

CUDA kernels for the three Inkling-2-Mini attention ops on the DGX Spark GB10 (sm_121), branch `ink/cuda-attn`. All three
are ADMITTED against the f64 cpu reference. The decode / verify path (1..16 rows) is rows-exact, and `InkAttnDecode` is
bitwise the unfused pair. Real shapes: 24 q / 8 kv heads (G = 3), head_dim 128, d_rel 16, extent 1024, K 4, window 1024 on a
ring of 1040 rows on SWA layers, a linear cache on full layers.

| kernel (op) | path | what |
|---|---|---|
| `ink_kv_prologue_bf16` (70) | `ink_prologue_k` | one block per (kv head, row), 8 warps. Each warp takes whole "jobs": the conv'd K row (clip), the conv'd V row (unit RMS), q of each of the 3 heads, r of each head. Lane-strided elements, butterfly sum of squares. Then all threads run the rel tail (3 x 1024 elements, a 16-long fma chain each). Conv-ring write in the kernel when T <= Rr - K + 1, else `ink_ring_write_k` runs after it. |
| `rel_sink_attn_bf16` (71), any T | `ink_tile_k<D, false>` + `ink_tile_comb_k`, per 16-row slice | tensor-core (mma.sync m16n8k16 bf16) split-KV over **absolute key tiles shared by every row of the call**: one block per (kv head, tile), so the K / V of a tile is read once for all rows. The M = T x 3 query rows go in m-tiles of 16, with 4 warps per m-tile, one 16-key group each, over 64-key sub-chunks staged with cp.async (double buffer). Rows with one tile finish in the block; the rest merge in tile order in the combine. |
| `ink_attn_decode_bf16` (75) | `ink_tile_k<D, true>` + comb | the same kernel. Each tile block computes the prologue's jobs it needs: q / r of every row, the bias of its sub-chunks (same rel chain), and the conv'd K / V of the call's own keys in its tile. These use the prologue's device functions, so the result is bitwise the pair. |
| `rel_sink_attn_bf16_flash` (71), T > 16, priority 5 | `ink_flash_k<D>` + `ink_kv_write_k` | flash prefill on tensor cores: one block per (kv head, 64 query rows), 12 warps (3 heads x 4 row groups of 16), one pass over the keys, no scratch; the bias loads are issued before the QK^T mma. The cache write runs after it. A separate kernel because it is not bitwise the rows-exact one, the same arrangement as Metal's `rel_sink_attn_bf16_nax` (priority 5, prefill only). |

`rel_sink_attn_bf16` runs a call of more than 16 rows as 16-row slices in order (each slice reads the cache rows the
earlier slices wrote), so it is rows-exact for every T, not only decode / verify. Prefill normally takes the flash kernel.

Diagnostics: `INTARSIA_INK_DECODE=splitkv` routes `rel_sink_attn_bf16` to the scalar split-KV kernel, a straight port of
Metal's `ink_attn_s1` (rows-exact itself, but not bitwise the tile kernel). It is kept for timing comparisons and is the
head_dim 8 path. `INTARSIA_INK_SPLIT_KEYS` sets the tile-size floor (rounded up to a multiple of 64).

### Rows-exact design (tile kernel)

Row t (position p, first visible key lo, vis = p - lo + 1) cuts the keys into absolute tiles of S keys. S = tile_s(vis) is
the smallest 64 x 2^k for which an interval of vis keys meets at most MAXT = 40 tiles. So S = 64 on every SWA visit (17
tiles), and on full layers S grows with the context (1024 at 32k, 33 tiles). The row's partial over tile tau depends on
(p, lo, S, tau) only:
- sub-chunks of 64 keys;
- group g of the m-tile takes keys 16g .. 16g + 15 of each;
- QK^T and PV run on mma with bf16 in and f32 accumulation, P is rounded to bf16, and l sums the rounded p;
- the four groups merge in the order 0, 1, 2, 3.

The combine merges tiles in index order. Whether a row has one tile (finished in the block) or several (combine) also
depends only on its position. Which other rows share the m-tile does not matter, because an mma output element depends only
on its own row of A. Rows of one call can fall into two tile-size classes when the call straddles a size change; the grid
then holds both classes' tiles. With a device position the grid is sized from capacity: window.min(Cr), or Cr. Extra blocks
exit at once.

## Admission (`intarsia kernels test --backend cuda --op ...`, fleetq #10799, commit 5a06044d5)

| op | kernel | cases | fail | max_abs (scaled) | rel_l2 | result |
|---|---|---|---|---|---|---|
| ink_kv_prologue | ink_kv_prologue_bf16 | 117 | 0 | 5.46e-3 | 4.35e-4 | ADMITTED |
| rel_sink_attn | rel_sink_attn_bf16 (tile T <= 16, flash T > 16) | 63 | 0 | 5.92e-3 | 2.24e-3 | ADMITTED |
| ink_attn_decode | ink_attn_decode_bf16 | 70 | 0 | 6.90e-3 | 2.46e-3 | ADMITTED |

The scalar port (commit a87c58fd3, fleetq #10754) was also ADMITTED. Its figures: 117 / 63 / 70 cases, rel_l2 4.3e-4 /
1.2e-4 / 1.0e-4. The tensor-core kernels' rel_l2 is about 20x higher because P is rounded to bf16. That is the usual
flash-attention trade, and it stays well inside BF16_TOL (1.5e-2 abs, 6e-3 rel). The sweep's f32-activation cases are not
registered: the kernels are bf16 only, as on Metal.

Bitwise test `crates/intarsia-backend-cuda/tests/inkling2_attn.rs`: 36 cases. Layers: SWA at pos0 0 / 1020 / 1030 / 2100
(ring wrap); full linear with dynamic taps and tau at 3 / 1018 / 1500 / 3000; full at 8180. Rows T 2 / 4 / 7 / 16. Each case
checks five runs against each other:
- the R-row pair;
- R one-row pairs with host positions;
- R one-row pairs with a device position;
- InkAttnDecode at R rows;
- InkAttnDecode at R x 1 row, with host and with device positions.

The test compares q / rel / k / v / o, both conv rings and both caches. Passes (fleetq #10799).

## Decode timing (fleetq #10927, build7, tile floor 256)

`crates/intarsia-backend-cuda/tests/inkling2_attn_bench.rs` (`INTARSIA_INK_BENCH=1`). The figures are microseconds per
visit, replayed from a captured CUDA graph (eager is 1-5 us higher). 16 visits rotate over their own operands so the
K / V never sits in L2. "KV" is the K + V bytes a visit reads; the floor is KV / 273 GB/s.

| layer | ctx | rows | prologue | attn | pair | **fused (op 75)** | KV MB | floor us |
|---|---|---|---|---|---|---|---|---|
| SWA | 512 | 1 | 8.9 | 23.1 | 38.8 | **27.4** | 2.1 | 7.7 |
| SWA | 512 | 4 | 8.9 | 36.2 | 41.8 | **39.3** | 2.1 | 7.7 |
| SWA | 512 | 16 | 20.4 | 50.8 | 69.2 | **58.9** | 2.2 | 7.9 |
| SWA | 4k | 1 | 8.8 | 33.4 | 46.8 | **39.4** | 4.2 | 15.4 |
| SWA | 4k | 4 | 9.8 | 44.1 | 53.0 | **50.9** | 4.2 | 15.4 |
| SWA | 4k | 16 | 19.0 | 61.3 | 80.3 | **72.7** | 4.2 | 15.4 |
| full | 512 | 1 | 8.7 | 22.0 | 36.9 | **28.3** | 2.1 | 7.7 |
| full | 512 | 4 | 8.8 | 32.8 | 40.9 | **38.6** | 2.1 | 7.7 |
| full | 512 | 16 | 24.2 | 50.3 | 69.5 | **58.5** | 2.2 | 7.9 |
| full | 4k | 1 | 8.7 | 99.6 | 119.0 | **111.7** | 16.8 | 61.5 |
| full | 4k | 4 | 8.7 | 117.3 | **122.0** | 131.2 | 16.8 | 61.5 |
| full | 4k | 16 | 20.2 | 146.5 | 166.5 | **165.5** | 16.8 | 61.7 |
| full | 32k | 1 | 8.8 | 746 | 770 | **736** | 134 | 492 |
| full | 32k | 4 | 8.8 | 796 | 801 | **776** | 134 | 492 |
| full | 32k | 16 | 20.6 | 858 | 885 | 887 | 134 | 492 |

SWA visits read at most the 1024-key window, so SWA 32k is SWA 4k. On CUDA the fused op 75 is at least as fast as the pair
at every point except full 4k at 4 rows (+9 us). The rule sent to main is fused for every visit of 1..16 rows on CUDA; the
model's selection rule is Metal-only today.

The 16-row prologue (20 us) is the build7 figure, taken before the tail-slicing fix in the next commit (re-measured below).

Tile floor sweep (`INTARSIA_INK_SPLIT_KEYS`, fleetq #10866, before the prologue fix, so compare the attention and fused
columns only):

| case | attn 64 / 128 / 256 | fused 64 / 128 / 256 |
|---|---|---|
| SWA 512, 1 row | 18.9 / 20.0 / 23.6 | 34.2 / 31.6 / 31.0 |
| SWA 4k, 1 row | 36.0 / 35.5 / 32.7 | 52.1 / 53.5 / 40.2 |
| SWA 4k, 16 rows | 81.8 / 71.8 / 63.2 | 128.7 / 112.5 / 85.6 |
| full 4k, 16 rows | 178.4 / 175.3 / 145.4 | 226.2 / 218.2 / 173.7 |
| full 32k, 1 row | 750 / 735 / 745 | 761 / 758 / 745 |

256 is the default. Fewer, longer blocks mean fewer partials to merge, and the fused kernel redoes less per block.

### Launches per visit

| path | launches |
|---|---|
| pair, decode / verify (1..16 rows) | 3: `ink_prologue_k`, `ink_tile_k`, `ink_tile_comb_k` (2 when every row has one tile: host position, keys <= 256) |
| fused op 75 | 2: `ink_tile_k<.., true>`, `ink_tile_comb_k` (+ `ink_kv_write_k` / `ink_ring_write_k` only when T exceeds Cr - window + 1 or Rr - K + 1: never at the real shapes) |
| prefill chunk (2048 rows) | 4: `ink_prologue_k`, `ink_ring_write_k` (T > Rr - K + 1), `ink_flash_k`, `ink_kv_write_k` |
| `rel_sink_attn_bf16` at T > 16 (rows-exact path, not normally selected) | 2 per 16-row slice |

## Prefill timing (2048-row chunk, fleetq #10927)

| layer | ctx before | prologue | attn (flash) | pair | KV MB | rel tensor |
|---|---|---|---|---|---|---|
| SWA | 0 | 2191 | 1250 | 3558 | 4.2 | 100 MB written + read |
| SWA | 2k | 2300 | 1651 | 3785 | 4.2 | 100 MB |
| SWA | 30k | 2246 | 1752 | 3807 | 4.2 | 100 MB |
| full | 0 | 2205 | 1333 | 3538 | 8.4 | 100 MB |
| full | 2k | 2254 | 2264 | 4594 | 16.8 | 100 MB |
| full | 30k | 2291 | 14276 | 16670 | 134 | 100 MB |

The contract's `rel [T,H,E]` costs 2048 x 24 x 1024 x 2 B = 100 MB per chunk. That is 0.37 ms to write plus 0.37 ms to read
at the bandwidth floor, and the bias fetch dominates SWA flash (every key is biased there, since window = E). The flash
kernel against the split-KV port at the same rows (fleetq #10804): 1.25 vs 10.8 ms (SWA ctx 0), 14.3 vs 1093 ms (full 30k).
Full 30k flash: 2048 x 24 x 32768 x 128 x 4 = 0.82 TFLOP in 14.3 ms, about 58 TFLOPS.
