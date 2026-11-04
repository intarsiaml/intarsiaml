# DSv4.1 CUDA prefill: the 583 ms attention + hyper-connection gap to TensorFold

> Coordinator check, 2026-11-03 evening: `replayCut` and its call at `li == c.layers / 2` were read directly in the BLACKFOLD copy (`prompt.zig`, `model.zig:594`) and are as described here. The other numbers come from one read-only review of traces and source: nothing was run for this report, and no experiment below has been tried. In the token-agreement record (`2026-11-01-dsv41-tensorfold-token-agreement.txt`), the two golden prompts long enough to cut (`long_prose_300`, `long_prose_700`) matched our engine 64/64, so the replay did not change those two; two prompts say little about long documents.

Date 2026-11-03 (evening, read-only; no build, no ssh, no GPU). Inputs: the three attention docs and their raw traces/benches under
`docs/results/2026-11-03-dsv41-attn-fusion/`, `2026-11-02-dsv41-prefill.md`, `2026-11-02-dsv41-moegate.md`, our kernels in
`kernels/cuda/` (review-main worktree), and TensorFold's launch code and AOT metadata in the BLACKFOLD copy
`/opt/code/intarsiaml/.worktrees/blackfold/src/` (`tensorfold/zig/src/families/dsv41/*.zig`, `bertholomus-tp2/kit/aot/aot.json`).
TF's attention kernel is a Triton kernel shipped as cubin only; its shape comes from the launch code, the AOT record and the nsys trace
(`2026-11-02-dsv41-prefill/raw/tf/tf-r0-gpu_trace.csv.gz`). The path the brief named (`/opt/code/TensorFold/src/tensorfold/zig/...`)
does not exist; the BLACKFOLD copy is what every earlier lane used.

## 0. The headline: the comparison is not like for like

**TensorFold runs the decoder half of the model (layers 20-39) only on the last 128 prompt rows.** We run all 40 layers on every row.
Every class in the class table, attention and HC included, carries a factor of about 2 from that alone.

Evidence, rank 0 traces, same 4106-token prompt (2048 + 2048 + 10 rows):

| per request | ours (ab1, chunk 2048) | TF (job 7840) |
|---|---|---|
| `attn_sparse_mma` / `_sparse_attn_part` launches at 2048 rows | **80** (= 2 chunks x 40 layers) | **40** (= 2 chunks x 20 layers) |
| launches on the 10-row tail chunk | 40 | 40 |
| launches at 118 rows | none | 20 (`118 + 10 = 128` = window: the replay rows) |
| `_index_score_rows` at 2048 rows / at 118 rows | n/a | 3 per chunk (encoder index sources) / 5 (decoder index sources, replay rows only) |
| HC partial kernels at 2048 rows per chunk | 80 `hc_sub_partial` (2 per layer x 40) | 37-38 `_hc_mix_pf` (2 per layer x 19, first layer uses `_hc_partial`) |

The mechanism in TF's source: `model.zig:594-606` runs `prompt.replayCut` at `li == c.layers / 2` (= 20); `prompt.zig:1288-1313`
("the decoder's bounded replay at its first layer (`_forward_k`'s `replay`, CED's prefill)") runs layer 20's attention mixes and its
kv-source update (compress + index keys) over every row, then cuts the chunk to rows at positions `>= replay` where
`replay = prompt length - window` (`model.zig` doc comment, line 552), copying `h[first:]` and `pre[first:]` down; "an encoder-only chunk
(no row at or past replay) ends here". The decoder layers then run on at most 128 rows "with no window key before replay"
(`prompt.zig:720, 735, 749`). This works because layers 20-39 all read layer 20's compressed KV and index keys (our
`config.rs:280-283`, test `shared_sources_follow_layer_order_and_ratio`: `kv[20..40] == Some(20)`), so below the floor the decoder half
produces nothing another row consumes except its own sliding-window keys, which TF drops. Our copy of the official code
(`.worktrees/dsv41-official/model.py`, `generate.py`) has no `replay`; this is TensorFold's (or its source "CED"'s) approximation, not the
reference numerics. The decoder-layer windows of the replayed rows see fewer than 128 keys, so TF's last-token logits are not the full
model's; our tokens already disagree with TF's early on most prompts (`2026-11-01-dsv41-tensorfold-token-agreement.txt`, 52 % leading
match), so that comparison cannot separate this from the quantisation difference.

Two earlier statements are therefore wrong and should be corrected in the docs: `2026-11-02-dsv41-prefill.md` ("Ours rank 0 and TF rank 0
compute the same layers on the same tensor layout; the MoE gap is kernel quality, not work done") and `2026-11-02-dsv41-moegate.md:64`
("TF mean over the 20 matched layers") matched TF's 20 launches per chunk against 20 of our 40 without noticing the other 20.

### Like for like, per launch (same rows, same 640 keys = 128 window + 512 picks, same 20 tiles of 32 keys)

From the two traces (durations clustered; the small cluster is the two window-only layers 0-1):

| shape | ours `attn_sparse_mma` | TF `_sparse_attn_part` | ratio |
|---|---|---|---|
| 2048 rows, compressed layer (n = 228 / 108) | 4.995 ms | 3.031 ms | **1.65x** |
| 2048 rows, window-only layer (n = 12 / 12) | 1.156 ms | 0.672 ms | 1.72x |
| 1034 rows, compressed | 2.422 ms | 1.509 ms | 1.60x |
| 512 rows, compressed (ab0; TF has no 512 run) | 1.280 ms | - | |

So the kernel gap is 1.65x, not 3.4x. Per 4K request at chunk 2048, if our engine ran TF's layer schedule, attention would be about
390 / 2 = 195 ms against TF's 115; the kernel-quality part of the attention gap is **about 75-80 ms**, not 275.

HC per (layer, chunk, sublayer) at 2048 rows: ours `hc_sub_partial` 1.005 + `hc_post_kernel` 0.890 + `hc_sub_finish` 0.484 = **2.38 ms**;
TF `_hc_mix_pf` 1.739 + `_hc_finish` 0.536 = **2.27 ms**. **Parity within 5 %.** The "HC gap of 197 ms" is entirely the layer count.
(TF's HcPost fold is not faster than our separate HcPost; it is the same 2.3 ms either way, which is what the fold experiment also found.)

Rope per (layer, chunk) at 2048 rows: ours 2 x `rope_tail_kernel` (grid 131072, 0.62 ms each) + kv rope 0.015 = **1.26 ms**; TF 2 x
`_rope_heads` (0.10 + 0.13) + `_kv_norm_rope` 0.027 = **0.26 ms**. A real 1.0 ms per layer-chunk, and it has a bit-identical fix (section 3, E3).

Honest decomposition of the 562 ms (chunk 2048) gap: layer schedule about 455 ms (half of our 910), attention kernel about 65-80 ms,
rope about 40 ms (TF's 20 layers' worth), index about 10 ms, HC about 0. The 583 ms at chunk 512 adds our chunk-512 overheads
(more launches, `hc_post` 646 calls) on top of the same structure.

## 1. What TF's attention kernel does differently

TF launch for a prompt chunk (`tri_attn.zig:75-117`, AOT record hash `03b08f42...` in `aot.json`; trace: grid (2048, 2, 1), block 128,
255 registers, 50176 B dynamic shared):

| | ours `attn_sparse_mma` (`dsv41_attn.cu`) | TF `_sparse_attn_part` |
|---|---|---|
| block | 1 query x **32 heads**, 256 threads (8 warps) (`:19-21, :495`) | 1 query x **16 heads** (`hb = 16`, `tri_attn.zig:76`), 128 threads (`num_warps 4`) |
| grid at 2048 rows, 32 heads/rank | (1, 2048, 1) = 2048 blocks | (2048, 2, 1) = 4096 blocks (`:89, :107`) |
| key tile | 32 keys (`kMmaKeys`, `:234`) | 32 keys (`bn = 32`, `:77`) |
| tiles per query | `cdiv(128 + 512, 32)` = 20 (`:280-281`) | `WIN/32 + pow2(cdiv(512, 32))` = 4 + 16 = 20 (`:91-97`); pick list padded to 16-entry tiles with -1 |
| Q | staged in shared memory, 32 x 520 bf16 = 33 KB (`:261, :284-289`), read as `lds32` A fragments per k-step (`:338-342`) | **in registers** (16 heads x 512 bf16 / 128 threads = 128 regs of A fragments, loaded once; the only way 255 regs and 49 KB smem add up: K tile 32 KB + a 16 x 512 bf16 exchange buffer 16 KB + S/P) |
| shared memory per block | 73,344 B + 768 static -> **1 block per SM** (2blk doc, section 1) | 50,176 B -> **2 blocks per SM** (2 x (50,176 + 1,024) = 102,400 B, exactly the SM's shared memory) |
| warps per SM | 8, one barrier domain | 8, **two independent barrier domains** |
| compressed keys | bf16 rows of the `CompressedKv` slot, 1024 B per key, gathered with 16-byte `__ldg` (`:322-331`) | **packed FP4**: `COMP *u8 [N, HD/2]` codes + `CSC *u8 [N, HD/16]` e4m3 scales (`PACKED = 1`, `tri_attn.zig:29, :79-86`; written by `_fp4_store`, `:134-137`, `prompt.zig:785`): 256 + 32 = **288 B per key**, dequantised in the kernel |
| window keys | bf16 (fp8 quant-dequant stored as bf16) | bf16 (`WSRC *bf16`, fp8 qdq inside `_kv_norm_rope`) |
| idx | `int32 [T, 512]`, read by 32 threads at the top of every tile, then barrier (`:305-321`): a global round trip on the critical path | `int64 [rows, n_idx]`, -1 padded; Triton's pipeliner can hoist it (cubin only, not verified) |
| QK | per warp one (m-tile, n-tile) of 32 chained `mma.m16n8k16` over k = 512 (`:340-342`) | per warp one 16 x 8 n-tile over k = 512: the same 32-mma chain, 4 warps |
| softmax | each warp 4 rows **serially**, each row: smem read, warp max, exp, bf16 round, warp sum, smem write, `__syncwarp`, lane-0 scalar update (`:351-368`) | 16 rows x 32 keys over 4 warps, cross-warp max/sum through shared memory (layout conversion), one barrier |
| PV | each warp D/8 = 64 columns = 8 n-tiles x 2 k-steps x 2 m-tiles = 32 mma, `ldmatrix.trans` B (`:389-409`) | each warp D/4 = 128 columns = 16 n-tiles x 2 k-steps = 32 mma; O accumulator 64 regs |
| barriers per tile | 5 (`:304, :321, :332, :350, :369`) + 1 before the epilogue | unknown (cubin), likely 3-4 |
| output | bf16 directly (`nsplit == 1`, `:427-428`) | bf16 directly (`FINAL = 1`, no merge kernel for chunks) |
| launch | plain | PDL (`pdl: true`) |
| rope | separate `rope_tail_kernel` before (q) and after (o, inverse), each a **full copy of q/o** (`dsv41_misc.cu:211-234`: `if (e < base) st(y, i, ld(x, i))`, grid over T x H x 512) into `q_rope` / `o_derot` (`graph.rs:316-324, :894, :910`) | `_rope_heads` **in place, rope dims only** (`tri_norm.zig:78`: X is the only tensor, consts `RD`, `INV`); `_kv_norm_rope` fuses RMSNorm + rope + fp8 qdq + ring write (`:67-69`) |

Work per layer-chunk: identical mma count (TF's two 16-head blocks per row = our one 32-head block), identical tiles and keys. TF does
**more** gather work per row (each of the two head-group blocks gathers the same 20 tiles) but each gathered key is 288 B instead of
1024 B, so total gather bytes per row are 0.56x ours. Nothing is booked elsewhere: the merge kernel is not used for chunks (`FINAL`),
the indexer and compressor are separate in both engines and classified the same way. The only "work moved" is the replay (section 0).

## 2. Which differences explain the no-math floor

Our per-tile budget at T = 2048 (5.0 ms / 42.7 waves / 20 tiles = 5.85 us per tile, floor 2.95 us): the knockouts attribute gather 28 %
(1.4 ms = 1.6 us/tile), QK 22 %, softmax 19 % (1.0 us/tile), PV 10 %. The floor is two dependent global round trips per tile (idx, then
the gather it addresses) plus 5 barriers, executed by the only block on the SM, so nothing overlaps them. 2048 x 20 x 32 KB = 1.3 GB
per launch read in 2.5 ms would be 530 GB/s, above DRAM; the data is in L2 (compressed cache 1 MB + window 2 MB), so the gather cost
is latency and barrier serialisation, not bandwidth.

TF's number says the same thing from the other side: 3.03 ms / 42.7 waves / 20 tiles = 3.55 us of SM time per tile for **two** blocks.
Each TF block's own chain is therefore about as long as ours (about 6-7 us per tile); TF is faster because two independent chains share
the SM, and because its gather moves 3.6x fewer bytes per compressed key, which also makes a register prefetch of the next tile cheap
(16 registers of packed codes instead of 64 of bf16; our register prefetch cost 16 registers and 22 % and is consistent with the
compiler serialising it, not with "latency is not the problem").

Why the two-blocks experiment did not show this: it freed shared memory by reading the Q fragments from global memory on every tile
(128 `LDG.32` per thread per tile = 128 KB of L1 traffic per tile against the 32 KB gather), which lengthened each block's chain by about
the amount the second block gave back (T = 37: +40 %, T = 2048: -0.4 %). TF pays nothing per tile for Q: it lives in registers for the
whole kernel. The 2blk doc's reason for not doing that ("would take 128 registers per thread") is exactly TF's design at 255 registers.

Softmax: our four rows per warp are serialised by `__syncwarp` and lane-0 scalar updates inside the loop (`:361-367`); each row is two
dependent 5-step shuffle trees plus `expf` and two shared-memory round trips. Interleaving the four rows (same per-row arithmetic and
reduction tree, so bit-identical) is a cheap test; knockout 3 bounds it at 19 % of the kernel.

What does not explain the floor: HC (parity), the softmax arithmetic itself, the mma count (identical), the number of SMs busy (both
at 42.7 waves of full occupancy), the chunk size (per-launch ratio is 1.6-1.7x at 512, 1034 and 2048 rows).

## 3. Experiments, ranked

Expected gains are per 4K request at chunk 2048 against 4387 ms busy / about 4.4 s wall, and are reasoning, not measurements.

### E1. Bounded-replay prefill (TF's layer schedule). Changes numerics. Gain about 1.9 s (about 43 %).

Change: at layer 20 of a prompt chunk run the attention mixes and the kv-source update (compress, index keys) on all rows, then run
layers 20-39 only on rows at positions `>= prompt_len - 128` (plus the tail chunk). The graph builder already truncates layers
(`GraphTp.layers`, `graph.rs:139, :576`) and the row-cut is a `copyRows` of `h` and `pre` (`prompt.zig:1299-1310`).

Bit-identical: **no**, and it cannot be made so by construction: the replayed rows' decoder-layer windows lose the keys before the floor,
and golden prompts longer than 128 tokens cut (shorter ones have replay 0 and do not), so golden 640/640 will not hold against main. A variant with the *same* row floor but exact
numerics does not exist cheaply: exactness needs layer 20 + k over the last 128 x (20 - k) rows (a triangle, 33 % of the decoder work
instead of 6 %), and even that is only row-exact if our kernels are chunk-invariant, which `2026-11-02-dsv41-chunkdep.md` shows they are
not at 8K+. So this is a product decision with the golden and KL path, not a kernel lane.

Why first anyway: it is the only item of the size of the gap, it is what the "speed reference" actually does, and no kernel work closes
the remaining 2x per class without it. It also scales with prompt length (at 32K the decoder half is still only 128 rows for TF).

Cheapest falsifier (one engine run, about 15 min): run the 4K prefill with the model truncated to 20 layers (`--layers 20`, the test
knob) to measure the encoder-only cost; the replay implementation costs that plus about 5 % (layer 20's update on all rows, 20 layers on
128 rows: TF's 118-row launches total about 40 ms per request). If the 20-layer prefill is not at least 35 % faster than the 40-layer one,
the estimate is wrong.

Risk: output quality (TF's own numerics are unmeasured against the reference; our MMLU/golden gates would be the first measurement),
decode-side window state for the first 128 decode tokens (the replayed rows' window keys are the ring's content; TF's `TF_DS_REPLAY_FLOOR`
handles the slot's stale rows below the floor, `prompt.zig:714-736`), DSpark taps and Engram layers in the decoder half.

### E2. TF-shaped attention block: 16 heads, 128 threads, Q in registers, 2 blocks per SM. Bit-identical by construction. Gain 80-120 ms (2-3 %).

Change in `attn_sparse_mma`: `kHeads = 16`, 4 warps, `__launch_bounds__(128, 2)`; the warp's 32 A fragments of Q (128 registers) loaded
once from global before the tile loop (same bf16 bits as today's `Qs`); shared memory = K tile 33,280 + S 16 x 33 x 4 + P 16 x 40 x 2 +
static about 37 KB, so two blocks fit (2 x 38 KB); keep the 2blk patch's double-buffered `valid/srcrow` with the idx load one tile ahead
(4 barriers per tile). Grid (nsplit, Tn, H/16). The decode split path (T <= 16) uses the same kernel and must keep its hash.

Why bit-identical: every accumulator sees the same operands in the same order. QK: one (m-tile, n-tile) chain of 32 mma per warp over
k = 0..512, as now (`:340-342`), only the m-tile is fixed. Softmax: the same 32-lane shuffle trees per row, same bf16 rounding of p, same
`alpha`/`l`/`m` updates. PV: per (m-tile, n-tile) accumulator `ks = 0` then `ks = 1` as now (`:390-409`); a warp owning 16 instead of 8
n-tiles changes nothing per accumulator. Epilogue identical. Gate: `attn_ab_bench` hashes at the five shapes (`4ee9a847...`,
`aa254a57...`, `90ad1d7f...`, T = 37, T = 11) must match.

Expected gain: TF reaches 3.03 ms with this shape plus packed keys; with bf16 keys the two blocks gather 2x our bytes per row (2.7 GB
per launch from L2), so I expect 3.6-4.2 ms from 5.0 (15-30 % of the kernel), i.e. 60-110 ms per request. The 2blk result says the
second block is worth about what the Q tax cost (about 20 % of the kernel at T = 2048); removing the tax should leave that 20 % as gain.
If E2 alone is below +12 % at T = 2048, run E2 + E4 together before giving up: that is TF's exact design.

Falsifier: `ab-build.sh` / `ab-run.sh` with a new patch, 6 alternating pairs, T = 2048 and T = 512 (about 15 min of spark2 next to vLLM).
Stop if the gain at T = 2048 is under 12 % or any hash differs.

Risk: 128 Q + 64 O + 8 P + addresses at a 255 cap: spills (`cuobjdump -res-usage` `LOCAL` must stay 0; TF manages it in Triton, so it
is feasible). T = 37-style small chunks where no SM holds two blocks get no gain and may lose a little (the 2blk doc's 40 % loss there
was the Q tax, which E2 does not have). Also the window-only layers (4 tiles) gain less.

### E3. Rope in place on the 64 rope dims (or fused into attention). Bit-identical. Gain about 90 ms (2 %).

Today `rope_tail_kernel` is launched over every element of q and of o and copies the 448 non-rope dims (`dsv41_misc.cu:217-220`), 620 us
per launch at 2048 rows x 32 heads (128 MB moved at 206 GB/s, i.e. the DRAM floor of a pointless copy); 2 launches per compressed
layer-chunk = 1.24 ms, 99 ms per 4K request. TF rotates 64 of 512 dims in place (100-130 us).

Change, minimal form: the same kernel body and expression (`re * cs - im * sn`, same `sincosf`, same template), launched over
T x H x 64 elements with `y == x` (in place), the copy branch gone; `graph.rs:316-324` aliases `q_rope` to `q` and `o_derot` to `o`.
Arithmetic per rotated element is unchanged, so the bits are the same if nvcc keeps the same contraction; the failed row-vector rope
differed because it restructured the expression (sin/cos shared across heads), which this does not. Verify by hash in `hc_rope_bench`'s
rope section (same inputs, 64-dim in-place vs full) before touching the graph. Fused form (q rotated while loading A fragments in E2,
o counter-rotated in the epilogue from the bf16-rounded output) saves the two launches entirely but has a double-rounding subtlety on o
(today: bf16 out, re-read, rotate, bf16 again) that must be reproduced explicitly; do the in-place form first.

Expected: 2 x 620 -> 2 x about 80 us per layer-chunk, about 85-90 ms per request. Falsifier: bench hash + in-situ `kcls.py` on the
rope class. Risk: near zero for the in-place form; the in-place aliasing in the graph must not break the device-position (dpos) graphs
that record these ops (`ops.rs:125-127`).

### E4. Packed FP4 compressed keys in the `CompressedKv` slot, dequantised in the gather. Bit-identical by arithmetic. Gain 40-60 ms alone (1 %), more with E2.

Today the compressed latent is quantised FP4 e2m1 with a block-16 e4m3 scale and stored dequantised as bf16 (`graph.rs:1018`:
`qdq(lat, Fp4E2m1, 16, E4m3, cb)`; `ops.rs:21-38`). Every stored bf16 is exactly `code x scale` (2 x 4 significant bits <= 8), so
gathering `u8` codes + e4m3 scales and multiplying in fp32 reproduces the bf16 bits with no rounding at all. Store format = TF's
(`[N, D/2]` codes, `[N, D/16]` scales). Gather bytes per compressed key 1024 -> 288; per row 640 keys: 655 KB -> 278 KB.

Change: a packed writer for the slot (the `qdq` op gains a packed output form), the attention gather path decodes nibbles into the bf16
tile (same `Ks` layout, so QK/PV untouched), the decode path too (same kernel), and `index_score_mma` is unaffected (it reads `IndexK`,
a separate slot). Memory per source cache falls 3.6x (65K prompt: 64 MB -> 18 MB).

Expected: the knockout bounds the whole gather at 28 % (1.4 ms); bytes are perhaps half of that cost (the rest latency and barriers), so
about 0.4-0.6 ms per launch alone; with E2 it is what lets two blocks gather per row without doubling L2 traffic, and what makes a
one-tile register prefetch cheap (16 registers). Falsifier: unit test that packs then dequantises the slot and compares bitwise with the
bf16 slot; `attn_ab_bench` hash and time with a packed `cmp`. Risk: medium engineering (slot dtype, writer, decode kernel, Metal/HIP
parity as a no-op), and the gain alone is small; do it as E2's companion, not alone.

### E5. Interleaved four-row softmax. Bit-identical. Gain 0-10 % of the kernel (0-40 ms).

Change: in `:351-368` compute the four rows' `sv`, warp maxima, `expf`, bf16 rounding and warp sums as four independent chains written
side by side (no `__syncwarp` between rows, one at the end; the four lane-0 updates after), keeping each row's reduction tree and
rounding. Expected: the knockout puts softmax at 19 %; interleaving hides shuffle and `expf` latency across rows, maybe half of it.
Falsifier: `knock.sh`-style single-file variant in `attn_ab_bench`, hash must match. Risk: the compiler may already overlap them (then 0 %);
15 minutes to find out, so it rides along with E2's bench run.

### Not ranked (would need new evidence first)
- cp.async double-buffered K tile with 32 heads per block and Q in registers (1 block per SM, 73 KB): the alternative to E2 if E2 spills.
  Hides the gather latency but keeps one barrier domain; TF's evidence favours two blocks.

## 4. What to stop trying

- **Chasing a 3.4x attention kernel gap or a 2x HC gap.** They are 1.65x and 1.0x per launch. Any plan that promises "70 % of 583 ms"
  from kernels is not supported by the traces; the kernel-side ceiling in attention + HC + rope is about 200-250 ms per 4K request.
- **HC kernels.** Per layer-chunk ours is within 5 % of TF's `_hc_mix_pf` + `_hc_finish`, with HcPost separate. The fold, vector
  staging, and a numerics-changing 4-stream fold are all off the table; TF's fold is not faster either.
- **Per-element rope speedups** (`sincosf`, vectors). The kernel is at the memory floor of a copy that should not exist; E3 removes the
  copy instead.
- **Two blocks per SM with Q read from global per tile**, register prefetch of a bf16 tile, fewer barriers alone, graph capture, PDL
  on the MoE pair (`2026-11-02-dsv41-pdl.md`: no gain), 64-row grouping: all measured, none moved.
- **Fusing `rmsnorm + rope + qdq + window_build + ring_write` into one kv kernel** (TF's `_kv_norm_rope`): about 37 us per
  layer-chunk, 3 ms per request. Not worth a lane.
- **Comparing class totals against TF without correcting for the replay.** Either run TF with the replay disabled (there is no switch;
  `replay` is a parameter of `fillChunk`, so a patched TF build would be needed) or compare per launch, as above.

## 5. Suggested window plan (tomorrow morning)

1. 15 min, spark2 next to vLLM, `--quiet gpu,cpu`: E2 (+E5 as a second variant) in `attn_ab_bench`, 6 alternating pairs at the five
   shapes, hashes first. `cuobjdump -res-usage` on both objects (spills), and `-sass` of main's `rope_tail_kernel` for E3's contraction.
2. 10 min: E3 in-place rope in `hc_rope_bench` (hash + time).
3. If E2 >= 12 % at T = 2048: wire it (grid `H/16`, `attn_mma_smem`, `cudaFuncSetAttribute`), E3 into the graph, one `run-full.sh`
   window with `pfnsys` for the per-kernel table, golden 640/640, 32K/65K `longp` bit check against main, Metal/HIP no-op.
4. Separately, one engine run with `--layers 20` on the 4K prompt for E1's upper bound, and a note to the coordinator that TF's prefill
   speed includes a numerics-changing schedule; the decision on E1 is theirs.
5. Correct the two doc statements named in section 0 and add the layer-count check to `class_table.py` (count attention launches per
   chunk; refuse to tabulate if the two engines differ).

## Appendix: numbers used

- TF trace, 4K request (span 2025 ms): `_sparse_attn_part` grid 2048 x 39-40, grid 10 x 40, grid 118 x 20; `_hc_mix_pf` grid 10240 x 75;
  `_kv_norm_rope` 2048 x 39; `_rope_heads` 2048 x 84; `_index_score_rows` 2048 x 6, 118 x 5.
- Ours ab1 (chunk 2048), 4K request: `attn_sparse_mma` grid (1, 2048) x 80, (20, 10) x 38, (4, 10) x 2, merges x 40; `hc_sub_partial`
  (12, 128) x 160; `rope_tail` 131072 x 160; `hc_post` 40960 x 160.
- Totals over the whole ab1 trace (3 x 4K + 3 x 1K): rope ours 388 ms vs TF 68 ms; HC ours 1459 ms vs TF 758 ms (the HC ratio is the
  layer count; per launch they match).
- TF AOT record for the chunk variant: `num_warps 4`, `shared 50176`, `pdl true`, consts `H 32, HD 512, HB 16, WIN 128, BN 32,
  HAS_COMP 1, PACKED 1, SPLITS 1, NBLK 20, FINAL 1`; trace shows 255 registers.
- Device (2blk doc): 48 SMs, 102,400 B shared per SM, 1,024 B reserved per block, 65,536 registers, 1,536 threads.
