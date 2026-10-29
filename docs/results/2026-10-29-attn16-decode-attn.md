# attn16: batched decode attention on the 35B at B = 16, exact: a fixed-grid pipelined kernel and an L2 discard

Lane attn16, 2026-10-29, branch `lane-attn16/decode-attn` from `main` 07f178dad (merged with 7c2dd03ae, HIP-only). spark2
(spark-fa00, GB10 sm_121), `nvidia/Qwen3.6-35B-A3B-NVFP4`. CUDA only.

**Status: draft PR. Not merged.** STATUS_LINE

## Result

RESULT_BLOCK

## 1. Why our kernel was slow

Qwen3.6-35B-A3B has 10 full-attention layers (the other 30 are GDN), 16 query heads, 2 KV heads and head_dim 256, so G = 8
query heads share a KV head. Batched decode runs `attn_decode_bf16_splitkv_rows` (`kernels/cuda/attention_splitkv.cu`):
`splitkv_kernel<GP = 8, TPK = 32, KB = 2, bf16, ROWS>` and then `combine_kernel<256, ROWS>`.

- **Grid.** The grid is (64 splits, 2 KV heads, 16 rows) = 2048 blocks of 256 threads. 64 is the length-independent
  maximum that the CUDA graph needs. At about 1.1k keys a row uses 18 splits of 64 keys, so 576 blocks work and 1472 load a
  position and exit.
- **Occupancy.** 242 registers (ptxas) means 1 block per SM, 8 warps.
- **Per block.** Each working block loads its position, a row-table entry and q (64 scalar 2-byte loads per thread). It then
  streams only 4 iterations (64 keys: 32 KB of K and 32 KB of V). The loads are issued at the top of each iteration with no
  prefetch, and the 8-head lane merge (16 barriers) follows.
- **Partials.** About 4.7 MB of f32 partials per layer at B = 16 (16 rows x 16 heads x 18 splits x 1 KB). They are written,
  pushed out of L2 by the 39 MB KV stream, and read back by the combine.
- **Measured, step16 profile.** 214.6 us per call at B = 16 in the served burst (39 MB of bf16 KV at about 182 GB/s), plus
  7.4 us of combine and 2 x 5.4 us of `kv_append`.
- **Measured, micro (`timing_attn_rows`, 10 layers' own caches, rows at 1100..1300 keys).** 227 us for stage 1 + combine.

What bounds it (short checks, `scripts/attn16/rdprobe.cu`):

| probe on spark2 | us per 39-40 MB | GB/s |
|---|---:|---:|
| flat read, grid-stride, 1 to 8 blocks per SM | 166.5 to 172 | 232 to 239 |
| the attention's own pattern (32 K and 32 V areas in separate allocations, 64-key items, warp per key, 2 keys per warp per iteration) | 165.9 | 235 |

The access pattern itself reaches the read floor, about 166 us for this footprint, so the 60 us above it is schedule and
traffic. Experiments, each bitwise exact (G-B), micro at M = 16 (stage 1 + combine):

| variant | us |
|---|---:|
| main (`INTARSIA_ATTN_ROWS_PERSIST=0`) | 227 |
| head passes (4 heads per block, 126 registers, 2 blocks per SM), full grid | 204 |
| fixed grid, items round-robin, K / V one iteration ahead, 8 / 4 / 2 heads per block | 206 / 204 / 202 |
| ... plus `__launch_bounds__` for 2 / 3 blocks per SM (4 or 2 heads) | 212 / 203; 942 / 333 (spills) |
| fixed grid, cp.async ring of 4 stages in shared memory (72 KB, 1 block per SM) | 252 / 241 / 246 |
| fixed grid, loads only (diagnostic: no scores, no merge, no q) | 197 |
| fixed grid, 8 heads, + streaming K / V loads (`__ldcs`) | 205 |
| fixed grid, 8 heads, + L2 discard of the partials in the combine | 192 |
| **fixed grid, 8 heads, + both (this PR)** | **187.6** |

Occupancy did not move the time, and the deeper shared-memory ring made it worse. The loads-only diagnostic was still at
197 us. What moved it was the partials: once the combine discards them from L2 (`discard.global.L2`: no write-back) and
K / V are streaming loads (evict first, so the partials survive until the combine), stage 1 fell from 196 to 178 us
(nsys, M = 16). Per layer that removes 4.7 MB of DRAM writes and 4.7 MB of re-reads at B = 16.

## 2. The change

`splitkv_rows_persist_kernel<TPK>` (default on; `INTARSIA_ATTN_ROWS_PERSIST=0` restores main) serves the row-table decode
launch for groups of 5 to 8 heads (KB = 2, the 35B). Other groups (the 8B's 4) keep main's grid.

- **Items.** It enumerates exactly the items whose blocks do work in main's grid: (row, KV head, split < the row's split
  count), row by row. They are dealt round-robin to a grid of SMs x resident blocks (48 x 1).
- **Pipelining.** Each block streams its items back to back. K / V of the next iteration are loaded one iteration ahead, and
  the last iteration of an item loads the next item's first.
- **Per-item setup.** The next item's row-table entry is read during the current one, positions are kept in shared memory,
  and q is read 16 B at a time (`ik_unpack8`, exact).
- **Loads.** K / V use `__ldcs`.
- **Math.** The per-iteration math and the lane merge are the same statements as `splitkv_kernel`'s. They are copied into
  `splitkv_step` / `splitkv_merge`, so `splitkv_kernel` itself is byte-for-byte main's code. Keys past an item's end are
  zero, as in main.
- **Combine.** `combine_kernel<D, true, true>` (DISCARD) drops the 128-byte lines of the partials it has read. This needs
  128-byte-aligned partials; the engine's scratch is a dedicated allocation at offset 0, and the launcher checks.
- **Exactness.** Split policy, partial layout, scratch size and the combine's arithmetic are unchanged, so every partial is
  bitwise the full-grid block's and every output row is bitwise main's.
- **SASS.** All 58 kernels of main's `attention_splitkv.cu` compile to identical SASS in this branch
  (`scripts/attn16/sass.sh`). The solo decode, main's row path, prefill, fp8-KV and Gemma-4 512 kernels are untouched.

Considered and not shipped:
- **Head passes.** Exact; superseded by the fixed grid.
- **cp.async ring.** Slower.
- **Fusing the combine into stage 1.** It needs counters in scratch, which changes the scratch declaration mirrored in
  `kernels.rs`.
- **Fusing the K and V appends** (2 x 5.4 us per layer). That is an engine-op change.

## 3. Numbers

PER_KERNEL_BLOCK

## 4. Gates

GATES_BLOCK

## 5. Caveats

- **The brief says vLLM's KV is bf16; it is not.** VPROF's startup log shows FlashInfer decode on an `float8_e4m3fn` KV
  cache: 19.7 MB per layer against our 39 MB. Most of the 1.2 ms "attention gap" is bytes. An exact bf16 kernel's floor at
  B = 16, about 1.1k keys, is about 166 us per layer (the pattern probe), so the exact headroom from main's 214.6 us is
  about 0.5 ms per step, of which this PR takes ANSWER.
- **Gemma-4 `plan show` was not run.** The checkpoint is not on spark2, and lanes may only use spark2. All pre-existing
  kernels have identical SASS, and plan show is host-side.
- **`discard.global.L2` makes the discarded lines' contents undefined.** They are this launch's dead partials only, inside
  its own scratch, aligned to the line.
