# Inkling-2-Mini attention on CUDA (lane ink-cuda-attn, 2026-11-21)

## Briefs received

(none yet)

## Summary

CUDA kernels for the three Inkling-2-Mini attention ops on the DGX Spark GB10 (sm_121), branch `ink/cuda-attn`. All three
are ADMITTED against the f64 cpu reference. The decode / verify path (1..16 rows) is rows-exact, and `InkAttnDecode` is
bitwise the unfused pair. Real shapes: 24 q / 8 kv heads (G = 3), head_dim 128, d_rel 16, extent 1024, K 4, window 1024 on a
ring of 1040 rows on SWA layers, a linear cache on full layers.

| kernel (op) | path | what |
|---|---|---|
| `ink_kv_prologue_bf16` (70) | `ink_prologue_k` | one block per (kv head, row), 8 warps. Each warp takes whole "jobs": the conv'd K row (clip), the conv'd V row (unit RMS), q of each of the 3 heads, r of each head. Lane-strided elements, butterfly sum of squares. Then all threads run the rel tail (3 x 1024 elements, a 16-long fma chain each). Conv-ring write in the kernel when T <= Rr - K + 1, else `ink_ring_write_k` runs after it. |
| `rel_sink_attn_bf16` (71), T <= 16 | `ink_tile_k<D, false>` + `ink_tile_comb_k` | tensor-core (mma.sync m16n8k16 bf16) split-KV over **absolute key tiles shared by every row of the call**: one block per (kv head, tile), so the K / V of a tile is read once for all rows. The M = T x 3 query rows go in m-tiles of 16, with 4 warps per m-tile, one 16-key group each, over 64-key sub-chunks staged with cp.async (double buffer). Rows with one tile finish in the block; the rest merge in tile order in the combine. |
| `ink_attn_decode_bf16` (75) | `ink_tile_k<D, true>` + comb | the same kernel. Each tile block computes the prologue's jobs it needs: q / r of every row, the bias of its sub-chunks (same rel chain), and the conv'd K / V of the call's own keys in its tile. These use the prologue's device functions, so the result is bitwise the pair. |
| `rel_sink_attn_bf16` (71), T > 16 | `ink_flash_k<D>` + `ink_kv_write_k` | flash prefill on tensor cores: one block per (kv head, 64 query rows), 12 warps (3 heads x 4 row groups of 16), one pass over the keys, no scratch. The cache write runs after it. |

Diagnostics: `INTARSIA_INK_DECODE=splitkv` and `INTARSIA_INK_PREFILL=splitkv` route RelSinkAttn to the scalar split-KV
kernel, a straight port of Metal's `ink_attn_s1` (rows-exact itself, but not bitwise the tile kernel). It is kept for timing
comparisons and for head_dim 8. `INTARSIA_INK_SPLIT_KEYS` sets the tile-size floor (multiple of 64).

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

## Decode timing

(pending)

## Prefill timing

(pending)
