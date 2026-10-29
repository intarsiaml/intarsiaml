# attn16: batched decode attention on the 35B at B = 16, exact: a fixed-grid pipelined kernel and an L2 discard

Lane attn16, 2026-10-29, branch `lane-attn16/decode-attn` from `main` 07f178dad (merged with 7c2dd03ae, HIP-only). spark2
(spark-fa00, GB10 sm_121), `nvidia/Qwen3.6-35B-A3B-NVFP4`. CUDA only.

**Status: draft PR. Not merged.** Exact and default on. The attention bucket fell by 0.20 ms in the served burst against the 0.5 ms the speed bar asks for, so this is a draft.

## Result

- **Why main's kernel is slow.** The cost is the schedule and the f32 partials' DRAM traffic, not the math.
  - The row-table grid launches 64 x 2 x 16 = 2048 blocks; at about 1.1k keys, 576 of them work.
  - At 242 registers it runs 1 block per SM.
  - Each working block waits on its position, row-table entry and q, then streams only 4 iterations without prefetch.
  - It writes 4.7 MB of partials per layer, which the 39 MB KV stream evicts to DRAM before the combine reads them back.
- **The pattern can reach the floor.** The same K / V access pattern as a pure read probe reaches 235 GB/s, 166 us per
  layer at B = 16.
- **The change.** `splitkv_rows_persist_kernel` (default on, `INTARSIA_ATTN_ROWS_PERSIST=0` restores main):
  - It runs a fixed grid of SMs x 1 block that streams only the working items, prefetching across item boundaries.
  - K / V use streaming loads.
  - The combine drops the dead partials from L2 (`discard.global.L2`).
  - It is bitwise main for every row: same split policy, same statements, same partial layout.
  - All 58 of main's kernels in the file have identical SASS.
- **Per-kernel numbers.**

  | measurement | main | this PR |
  |---|---:|---:|
  | micro, M = 16, 1.1k to 1.3k keys, stage 1, nsys | 228 us | 178 us |
  | micro, stage 1 + combine | 227 us | 187.6 us |
  | served 16-stream burst, nsys (B = 16 decode) | 211.7 us | 191.1 us |

- **Step and serve numbers.**
  - Attention bucket in the served burst: 2.22 -> 2.02 ms, **-0.20 ms**.
  - Step in the served burst: 44.49 -> 44.30 ms.
  - `timing_35b_batched_step` at 1024-token prompts, B = 16: about -0.4 ms.
  - Serve: +1.1 % at 16 streams and +1.8 % at 4. The 1-stream cell is -0.5 %, but that cell runs the solo step, whose
    kernels have main's SASS.
- **Exact.** batch_cuda_real 35B 4290 pairs bitwise (pre- and post-merge), row_tables / rows_exact pass, served `equal.py`
  21/21 identical, `plan show` 8B / 35B byte-identical.
- **Short of the speed bar (-0.5 ms bucket).**
  - In the engine, the partial write-backs that the discard removes were partly hidden behind other kernels, so the micro's
    -50 us per layer became -21 us per layer.
  - The remaining exact headroom to the 166 us floor is about 0.25 ms per step.
  - The rest of the 1.2 ms gap to vLLM is fp8 KV (half the bytes), which is not exact.

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

**Micro** (`timing_attn_rows`, the row-table launch alone; 10 layers' own caches per row; row r at 1100 + 37 r % 200
keys). The rows below are us per call (stage 1 + combine), `INTARSIA_ATTN_ROWS_PERSIST` 0 / 1, on the pre-merge tree:

| M | 1100 keys: main | 1100 keys: PR | 64 keys: main | 64 keys: PR |
|---:|---:|---:|---:|---:|
| 1 | 19.3 | 16.6 | 14.6 | 14.6 |
| 4 | 64.7 | 57.9 | 18.7 | 14.7 |
| 8 | 117.8 | 97.4 | 28.7 | 16.6 |
| 16 | 227.4 | 187.6 | 47.3 | 33.0 |

**Served burst** (`scripts/attn16/prof.sh`: `serve --max-batch 16`, 16 streams, ~1k-token prompts, 256 tokens; nsys,
decode window 5 to 14 s; main = 781d21833, this PR merged with it; `raw/base-b16`, `raw/dev-b16`):

| kernel (B = 16 decode) | grid | main us/call | PR us/call |
|---|---|---:|---:|
| `splitkv_kernel<8,32,2,bf16,ROWS>` / `splitkv_rows_persist_kernel<32>` | 64x2x16 / 48x1x1 | 211.7 | 191.1 |
| `combine_kernel<256,ROWS>` (DISCARD in the PR) | 256 | 7.6 | 7.0 |
| `kv_append_rows_kernel` (x2) | | 5.5 | 5.7 |

| bucket (`buckets.py`, ms per step) | main | PR | delta |
|---|---:|---:|---:|
| attention (splitkv + kv_append) | 2.22 | 2.02 | -0.20 |
| sum of kernel durations | 44.57 | 44.32 | -0.25 |
| decode step, mean of 245 passes (`timeline.txt`) | 44.49 | 44.30 | -0.19 |

**Step micro-timing** (`timing_35b_batched_step`, B = 16, three runs of 60 steps; dev binary under
`INTARSIA_ATTN_ROWS_PERSIST=0`, which is main's kernels, against the default):

| tree, prompt | PERSIST=0 | default | delta |
|---|---|---|---:|
| pre-merge (07f178dad), 64 | 44.39 / 44.44 (two runs, middle third) | 44.37 / 44.15 | about -0.1 |
| pre-merge, 1024 | 46.53 / 46.56 | 46.20 / 46.20 | about -0.35 |
| merged (781d21833), 1024 | 45.52 (mean of thirds) | 45.10 | -0.42 |

(W3's base-vs-dev step run is not used: main's test binary has no `INTARSIA_STEP_PROMPT`, so its base rows ran 64-token
prompts.)

**Serve A/B** (localeval, `serve --max-batch 16`, 1024-token prompts, 256 tokens, 3 interleaved rounds, aggregate tok/s,
main 781d21833 vs this PR merged; `raw/w2`):

| streams | main | PR | delta | range main / PR |
|---:|---:|---:|---:|---|
| 1 | 79.9 | 79.5 | -0.5 % | 79.7-80.2 / 79.2-79.8 |
| 4 | 170.6 | 173.6 | +1.8 % | 170.2-171.3 / 172.7-174.1 |
| 8 | 226.1 | 226.6 | +0.2 % | 225.1-227.4 / 223.4-229.1 |
| 16 | 291.8 | 295.0 | +1.1 % | 291.3-292.5 / 293.7-296.5 |

One stream is the solo step (`batch.rs`: "one row is the solo step itself"). Its kernels have main's SASS, so the -0.5 % is
run-to-run noise. The ranges overlap.

## 4. Gates

| gate | result |
|---|---|
| `batch_cuda_real` 35B (`real_35b_batched*`) | PASS, 4290 (row, step) pairs bitwise, pre-merge (W1) and merged (W3) |
| `row_tables` (incl. G-B `attn_decode_bf16_splitkv_rows`, 3220480 bytes, M 1..16, 7 shapes incl. G = 5 / 8, D = 64 / 128 / 256) | PASS (6 tests), plus the G-B under `INTARSIA_ATTN_ROWS_PERSIST=0` |
| `rows_exact` | PASS (10 tests) |
| served `equal.py`, main vs PR | 21/21 cells identical, 9408 token ids |
| `plan show` 8B / 35B, json + txt | byte-identical (`raw/plan.txt`); 35B unchanged (scratch size and kernel name are unchanged) |
| `plan show` Gemma-4 | NOT RUN: no checkpoint on spark2 (see caveats) |
| SASS of main's 58 kernels in `attention_splitkv.cu` | identical (`scripts/attn16/sass.sh`) |
| Mac `cargo test --workspace --release` | PASS after the tuning.md rows |
| speed: attention bucket -0.5 ms at B = 16 | NOT MET: -0.20 ms served, -0.5 ms on the isolated micro |
| speed: no serve cell slower than -0.5 % | 1 stream -0.5 % (solo path, unchanged kernels), others +0.2 to +1.8 % |

GPU windows: W1 #6977 (14 min), W2 #6981 (12 min), W3 #6982 (12 min). Short checks, each < 3 min (ptxas, SASS, probes,
micro, nsys micro, plan show, one step check), about 25.

## 5. Caveats

- **The brief says vLLM's KV is bf16; it is not.** VPROF's startup log shows FlashInfer decode on an `float8_e4m3fn` KV
  cache: 19.7 MB per layer against our 39 MB. Most of the 1.2 ms "attention gap" is bytes. An exact bf16 kernel's floor at
  B = 16, about 1.1k keys, is about 166 us per layer (the pattern probe), so the exact headroom from main's 214.6 us is
  about 0.5 ms per step, of which this PR takes 0.2 ms served (0.4 ms on the micro-timing step, 0.5 ms on the isolated kernel).
- **Gemma-4 `plan show` was not run.** The checkpoint is not on spark2, and lanes may only use spark2. All pre-existing
  kernels have identical SASS, and plan show is host-side.
- **`discard.global.L2` makes the discarded lines' contents undefined.** They are this launch's dead partials only, inside
  its own scratch, aligned to the line.
