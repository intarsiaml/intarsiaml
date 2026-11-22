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
3. 2026-11-21 (coordinator), in order:
   1. The tiny-model gate moves to ink-cuda-mixmoe: they merge tip fd86495f2, run fused-rule-on vs pair-forced, and
      extend the refusal test. No ref needed here.
   2. Do idea 1 (64 keys per warp step on full-context decode). Target: <= 560 us at full 32k, 1 row. Admission and the
      rows-exact tests must still pass; no case at 512 or 4k may get more than 2 us slower.
   3. Then idea 3 (the combine folded into the tile kernel), if rows-exact stays bitwise.
   4. Idea 2 (contract change): not now; Metal shares the contract, revisit after decode. Idea 4 later.
   Scope: only this worktree; timing runs on a spark1 slot with --quiet gpu,cpu; no tool call or wait over 4 minutes.
4. 2026-11-21 (coordinator, rule for all Inkling lanes): this session and its worktrees run on endor, the Metal timing
   box. No local cargo build / check / test there outside a fleetq slot, since that skews endor's quiet timing jobs.
   CUDA builds and tests stay on spark1 / spark2 through fleetq. Reading files, git and local edits are fine. (Before
   this rule the lane ran `cargo check` / `cargo test --lib` for the CUDA crate and one tiny-model refusal test locally;
   from now on it runs none.)
5. 2026-11-21 (coordinator): **Approved: the staged body restored for window > 0 layers** (a per-layer constant; rows-exact
   and fused == pair stay bitwise). Then, in order:
   1. One cheap diagnostic, a single spark1 quiet job: the same 134 MB read three ways with the read-only kernel:
      (a) one linear stream; (b) today's 264 interleaved K/V row streams; (c) each block one contiguous range of a single
      head, >= 64 KB per block. If (a) and (c) reach ~240+ GB/s and (b) stays ~195, the cause is stream count / pattern:
      rework the decode grid so each block owns a contiguous key range per head, one attempt, measured against 668. If all
      stay ~195: footprint / TLB, record and accept 668.
   2. Then idea 3 (fold the combine into the tile kernel) under the same bitwise rules.

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

## Decode timing (fleetq #10963, final build, tile floor 256)

`crates/intarsia-backend-cuda/tests/inkling2_attn_bench.rs` (`INTARSIA_INK_BENCH=1`). The figures are microseconds per
visit, replayed from a captured CUDA graph (eager is 1-5 us higher). 16 visits rotate over their own operands so the
K / V never sits in L2. "KV" is the K + V bytes a visit reads; the floor is KV / 273 GB/s. The faster of pair and fused is
in bold.

| layer | ctx | rows | prologue | attn | pair | fused (op 75) | KV MB | floor us |
|---|---|---|---|---|---|---|---|---|
| SWA | 512 | 1 | 8.2 | 23.9 | 41.6 | **28.7** | 2.1 | 7.7 |
| SWA | 512 | 4 | 8.5 | 34.4 | 44.5 | **37.7** | 2.1 | 7.7 |
| SWA | 512 | 16 | 12.0 | 50.0 | 61.2 | **57.3** | 2.2 | 7.9 |
| SWA | 4k | 1 | 8.2 | 35.1 | 47.4 | **40.8** | 4.2 | 15.4 |
| SWA | 4k | 4 | 8.5 | 46.5 | 53.7 | **47.8** | 4.2 | 15.4 |
| SWA | 4k | 16 | 11.8 | 60.3 | 70.0 | **69.6** | 4.2 | 15.4 |
| full | 512 | 1 | 8.2 | 22.5 | 39.3 | **29.7** | 2.1 | 7.7 |
| full | 512 | 4 | 8.4 | 35.9 | 41.2 | **37.0** | 2.1 | 7.7 |
| full | 512 | 16 | 13.4 | 51.2 | 58.5 | **56.1** | 2.2 | 7.9 |
| full | 4k | 1 | 8.1 | 97.2 | 110.9 | **108.5** | 16.8 | 61.5 |
| full | 4k | 4 | 8.3 | 115.9 | **119.5** | 123.9 | 16.8 | 61.5 |
| full | 4k | 16 | 13.7 | 146.1 | **156.3** | 161.5 | 16.8 | 61.7 |
| full | 32k | 1 | 8.1 | 732.8 | 732.6 | **715.4** | 134.2 | 491.7 |
| full | 32k | 4 | 8.4 | 756.6 | 761.1 | **749.7** | 134.2 | 491.7 |
| full | 32k | 16 | 11.6 | 829.8 | **845.1** | 846.3 | 134.3 | 491.9 |

SWA visits read at most the 1024-key window, so SWA 32k is SWA 4k. On CUDA the fused op 75 is at least as fast as the pair
by 1-13 us at 1-4 rows (SWA 512, 1 row: 28.7 vs 41.6), and within 5 us elsewhere. The pair wins at full 4k with 4 / 16
rows (by 4-5 us, after the last prologue fix) and ties at full 32k with 16 rows. Approved rule, commit d536ff099: on CUDA,
fused for every visit of 1..16 rows.

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

## Prefill timing (2048-row chunk, fleetq #10963)

| layer | ctx before | prologue | attn (flash) | pair | KV MB |
|---|---|---|---|---|---|
| SWA | 0 | 1224.9 | 1221.4 | 2450.1 | 4.2 |
| SWA | 2k | 1307.2 | 1581.3 | 2803.6 | 4.2 |
| SWA | 30k | 1205.5 | 1600.0 | 2837.0 | 4.2 |
| full | 0 | 1171.1 | 1429.2 | 2484.9 | 8.4 |
| full | 2k | 1203.3 | 2293.6 | 3640.1 | 16.8 |
| full | 30k | 1215.8 | 14282.0 | 15631.1 | 134.2 |

The contract's `rel [T,H,E]` costs 2048 x 24 x 1024 x 2 B = 100 MB per chunk. That is 0.37 ms to write plus 0.37 ms to read
at the bandwidth floor, and the bias fetch dominates SWA flash (every key is biased there, since window = E). The flash
kernel against the split-KV port at the same rows (fleetq #10804): 1.25 vs 10.8 ms (SWA ctx 0), 14.3 vs 1093 ms (full 30k).
Full 30k flash: 2048 x 24 x 32768 x 128 x 4 = 0.82 TFLOP in 14.3 ms, about 58 TFLOPS.

## Full-32k decode: what was tried (brief 2)

Pick: full 32k at 1 row (736 us against a 492 us floor) over full 4k at 4 rows (+9 us, verify only). The 32k case costs about
240 us above the floor on every full layer of every long-context token.

Hypothesis tested: wave quantisation (33 tiles x 8 heads = 264 blocks on 48 SMs) or one 70 KB-smem block per SM with
1-deep prefetch. Results, attention microseconds (graph), fleetq #10948:

| setting | full 4k, 1 row | full 32k, 1 row | full 32k, 16 rows |
|---|---|---|---|
| default (at most 40 tiles, 2 stages) | 94.0 | 716 | 842 |
| at most 64 tiles | 98.6 | 746 | 861 |
| at most 96 tiles | 99.7 | 709 | 838 |
| 1 stage (2 blocks / SM) | 96.4 | 704 | 808 |
| 1 stage, at most 96 tiles | 97.9 | 723 | 836 |

All within noise, so neither the tile count nor occupancy is the limit. The defaults stay; the knobs remain as diagnostics.
The kernels sit at about 180-190 GB/s on both the SWA (40 blocks) and the full (264 blocks) visits: about 6-8 us per
32 KB sub-chunk per block, far above DRAM latency. That points at the per-sub-chunk serial path of one block: stage, wait,
three `__syncthreads`, 16 keys of mma per warp. 8B bf16 GEMV decode reaches about 257 GB/s on this box, so about 1.4x is
left. ncu cannot run here (RmProfilingAdminOnly = 1), so the next step needs either profiling rights or the restructure in
idea 1 below.

## Full-32k decode: read pattern (brief 5, commit a5e3ede16)

Diagnostic `.scratch/bw.cu` (fleetq #11016, #11019-#11021): the 134 MB K + V of one full-32k visit
(8 kv heads x 32832 keys x 128 dims, bf16), read and discarded by a read-only kernel. There are 16 rotating copies (the
bench's footprint) unless noted.

| read pattern | us per visit | GB/s |
|---|---|---|
| (a) one linear stream, 768 x 256 threads | 534 | 252 |
| (b) the tile kernel's pattern: 4 warps round-robin over 16-key blocks, 16 B per lane | 637 | 211 |
| (b) with 1 copy (134 MB footprint) | 647 | 208 |
| (c) each block reads its contiguous K then V range, 128 threads | 570-579 | 232-236 |
| b2: (b), but each warp owns a contiguous quarter of the tile | 605 | 222 |
| b3: (b) with 4 16-key blocks in flight | 673-690 | 195-200 |
| b4: (b), but each load instruction covers 2 whole rows (not fragment layout) | 577 | 233 |
| b5: (b) with 32 B loads (`ld.global.v8.b32`) | 607 | 221 |
| b6: 32 B loads + warp-contiguous quarters (fragment-compatible) | 577 | 233 |
| b7: 32 B loads + 64 contiguous keys per warp per 256-key slab | 595 | 226 |

1 copy and 16 copies run at the same speed, so TLB and footprint are not the limit. The access shape is the limit: in
fragment layout, a 16 B load instruction touches 8 rows x 64 B. 32 B loads (8 rows x 128 B) plus long per-warp runs recover
most of the gap to (c).

The one rework attempt is b6 in the direct body (full layers):
- Warp group g owns the tile quarter `tk0 + g S/4 ..`, in key order, so the key-to-warp map stays absolute and the kernel
  stays rows-exact.
- With head_dim >= 64, W = 16 dims per lane and chunk. That is one 32 B load from the cache when both bases are 32-byte
  aligned (`F_AL32`), else two 16 B loads with the same values.
- The bias is staged per step of 4 x 64 keys, one 64-key run per warp.

Gate on fleetq #11036: all ADMITTED, 0 fails, and the 45 rows-exact cases bitwise. Fused op 75, graph us, against the
staged-SWA tip (#11016), one run each:

| case | 1 row | 4 rows | 16 rows |
|---|---|---|---|
| full 32k | 657 -> **613** | 721 -> **649** | 787 -> 784 |
| full 4k | 89.4 -> 87.9 | 104.7 -> 99.6 | 157.8 -> 164.8 |
| full 512 | 27.7 -> 28.2 | 37.1 -> 37.5 | 53.2 -> 56.8 |
| SWA (staged body, unchanged) | 28.5-38.2 -> 30.6-40.6 | | |

Full 32k at 1 row is now 613 us against a read-only ceiling for this pattern of about 577 (b6) and a floor of 492. The
16-row full-layer cases did not gain and may have lost 3-7 us. In those cases ptxas reports a 72-88 B spill for
`ink_tile_k<128>` at 168 registers, and the partial last tile now falls on one warp.

## Next ideas (ranked)

1. **Decode tile kernel: more keys per warp per barrier.** Each warp takes 64 keys per sub-chunk (4 mma k-groups)
   instead of 16, giving 256-key sub-chunks with a quarter of the barriers. Or load K / V fragments straight from
   global, with no shared-memory staging, for M <= 48 rows. Target: full 32k from 716 to about 520 us.
2. **Contract idea for main: attention computes the bias from r and relproj.** It would drop the `rel [T,H,E]` tensor:
   100 MB per 2048-row chunk written and read, about 0.75 ms at the floor. The prologue's rel tail is most of its
   1.6 ms per chunk, and the bias fetch dominates SWA flash. Not changed here: it is a contract change.
3. **Fold the tile combine into the tile kernel** (last block of a row merges in tile order, counter in scratch zeroed by
   the last block): one launch fewer per visit, about 3-5 us at decode.
4. **Flash prefill:** stage the bias slice with cp.async per sub-chunk; 128-row query blocks for full layers (14.3 ms at
   30k is about 58 TFLOPS).
5. Full 4k at 4 rows (fused 9 us slower than the pair): the fused blocks each redo the call's q / r jobs; share them
   through a 1-block pre-pass if verify rows matter.
