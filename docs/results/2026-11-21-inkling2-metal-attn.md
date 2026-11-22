# Inkling-2-Mini attention on Metal (lane ink-m-attn, 2026-11-21)

## Briefs received

1. 2026-11-21 (coordinator, via SendMessage): **Approved `InkAttnDecode`**, appended as discriminant 75 (reserved for this
   lane; no other lane appends ops now). Conditions: (1) semantics exactly RelSinkAttn(InkKvPrologue(...)) with the
   intermediates rounded to A; the CPU reference is that composition of the existing two refs, plus a test that the CPU op
   equals the pair bitwise; (2) rows 1..16 only, the op refuses larger T in its shape check so the graph can fall back to
   the pair; (3) admission cases at the real shapes plus rows-exact; (4) measure the unfused pair first and put both
   numbers, per visit, in this doc. Build the fused kernel regardless; the coordinator picks at integration. Graph wiring
   (emit 75 on decode when the backend admits it, else the pair) belongs to the integration lane: leave the model code
   alone and write the selection rule into this doc.

2. 2026-11-21, end of lane (coordinator, via SendMessage): **Confirmed: discriminant 75 (`InkAttnDecode`) is this lane's.**
   The coordinator checked the ops.rs diffs on ink/skel, ink/m-mix and ink/m-moe against 3feffee6e. None appends an op at
   75 or above. Lane done, no further work.

## Summary

Four Metal kernels for the Inkling-2-Mini attention ops. All of them are ADMITTED against the f64 cpu reference, and the
decode path is rows-exact (bitwise). Shapes are the real model's: 24 q / 8 kv heads (G = 3), head_dim 128, d_rel 16,
extent 1024, K 4, window 1024 on a ring of 1040 rows on SWA layers, a linear cache on full layers.

| kernel (op) | file | what |
|---|---|---|
| `ink_kv_prologue_bf16` (70) | kernels/metal/inkling2_attn.metal | one threadgroup per (kv head, row): K/V conv (static or dynamic taps), k clip, v RMS, q/r clip + tau, rel = r . relproj (4 columns per thread, 16 relproj rows in registers), conv-ring write in place (or a separate `ink_ring_write_bf16` pass when T > Rr - K + 1) |
| `rel_sink_attn_bf16` (71) | same | split-KV, (kv head, split, row) threadgroups, 4 simdgroups; G = 3 heads per K/V read; rel bias staged per split in dynamic threadgroup memory; two-segment ring read; sink in the final normalisation; sigmoid gate; combine pass when a row has > 1 split; cache write by split 0 (or `ink_kv_write_bf16` after the reads when a ring chunk would overwrite rows still read) |
| `ink_attn_decode_bf16` (75, `InkAttnDecode`) | same | the two above in one threadgroup for 1..16 rows: each (kv head, split, row) threadgroup recomputes its row's q / r, its split's bias and the call's own K/V rows inside its split with the prologue's arithmetic, so it is bitwise the pair |
| `rel_sink_attn_bf16_nax` (71, prefill) | kernels/metal/inkling2_attn_nax.metal (metal4.0) | flash attention on the neural accelerators (attn_nax structure, 16x32x16 matmul2d): banded rel bias gathered per score, the online softmax starts at the sink (m = sink, l = 1 on one lane of the row), window / causal mask, gate epilogue. Linear cache: chunk K/V written into the cache first, then read in place. Ring: `ink_kv_stage_bf16` stages [window tail of the ring, chunk] contiguously in the scratch. Registered for T > 16, head_dim 128, G = 3; falls back to the split-KV kernel for a device position, misaligned operands or a window > 4096 (`inkattn::nax_fallbacks()`, 0 in every run here) |

Rows-exact design: the split count and key partition of a row depend only on its position and the window (never on T,
pos0 or the grid), so an R-row call is R one-row calls bit for bit, with host or device positions.

## Admission (`intarsia kernels test --backend metal --op ...`, fleetq jobs #10583, #10645, final run after the last change)

| op | kernel | cases | fail | max_abs (scaled) | rel_l2 | result |
|---|---|---|---|---|---|---|
| ink_kv_prologue | ink_kv_prologue_bf16 | 117 | 0 | 5.46e-3 | 3.99e-4 | ADMITTED |
| rel_sink_attn | rel_sink_attn_bf16 | 63 | 0 | 1.68e-3 | 1.16e-4 | ADMITTED |
| rel_sink_attn | rel_sink_attn_bf16_nax | 7 | 0 | 5.46e-3 | 1.50e-3 | ADMITTED |
| ink_attn_decode | ink_attn_decode_bf16 | 70 | 0 | 1.03e-3 | 7.28e-5 | ADMITTED |

Tolerance: `BF16_TOL`, as the other Metal attention kernels. New sweep cases (crates/intarsia-core/src/inkling2/cases.rs):
real shapes at rows 1 / 4 / 16 and a 2048 chunk, SWA positions 0, 1016-1038 (straddling 1024, wrapping the 1040-row ring),
2100-5000, full positions straddling the 1024 extent and 8180-8192 (tau), a 2048 chunk on each; head_dim-128 mid-size
chunks (6/2 heads, window 64 on a ring of 80, chunks that wrap it, T 33-200) so the NAX kernel gets cases; the fused op's
sweep (small shapes + real shapes, rows 1/3/4/7/16).

Bitwise tests:
- `crates/intarsia-backend-metal/tests/inkling2_attn.rs`: 36 cases (SWA, full, full at 8180 with empty trailing splits;
  T 2/4/7/16): the R-row pair == R one-row pairs (host and device position) == `InkAttnDecode` at R rows and at R x 1 row
  (host and device position), on q / rel / k / v / o, both conv rings and both caches. Passes.
- `crates/intarsia-backend-cpu/tests/ink_attn_decode.rs`: the cpu `InkAttnDecode` == the cpu pair bitwise on all 70
  sweep cases. Passes.

## Decode timing (fleetq #10663, `--quiet gpu,cpu`)

`cargo run --release -p intarsia-backend-metal --example ink_attn_bench`. 16 visits per command buffer, each on its own
operands; 40 passes encoded back to back; GPU time of each command buffer from the trace, divided by its launches (`nd` counts `with_encoder` calls: one per op
call, whatever its number of dispatches). "pair" = prologue
+ attention, interleaved as a decode step runs them. Launch floor in this harness (a 32-element `add_bf16`): 3.5 us.

us per visit:

| layer | ctx | rows | prologue | attn | pair | fused | KV bytes read | KV floor at 614 GB/s |
|---|---|---|---|---|---|---|---|---|
| SWA | 512 | 1 | 12.5 | 16.2 | 30.2 | **22.9** | 2.1 MB | 3.4 |
| SWA | 512 | 4 | 11.5 | 27.6 | 41.0 | **35.8** | | |
| SWA | 512 | 16 | 15.1 | 80.3 | **97.2** | 123.4 | | |
| SWA | 4k / 32k | 1 | 10.5 | 20.3 | 35.2 | **31.4** | 4.2 MB | 6.8 |
| SWA | 4k / 32k | 4 | 10.5 | 43.2 | **56.6** | 60.2 | | |
| SWA | 4k / 32k | 16 | 15.0 | 142.4 | **160.3** | 203.3 | | |
| full | 512 | 1 | 10.8 | 16.4 | 30.8 | **22.8** | 2.1 MB | 3.4 |
| full | 512 | 4 | 10.8 | 27.5 | 41.4 | **35.8** | | |
| full | 512 | 16 | 15.6 | 80.2 | **97.9** | 130.0 | | |
| full | 4k | 1 | 11.0 | 61.0 | 76.0 | **61.8** | 16.8 MB | 27 |
| full | 4k | 4 | 10.9 | 150.3 | 165.6 | **158.1** | | |
| full | 4k | 16 | 15.5 | 496.4 | **515.9** | 553.5 | | |
| full | 32k | 1 | 10.9 | 340.2 | **347.5** | 373.2 | 134 MB | 218 |
| full | 32k | 4 | 11.0 | 952.7 | **970.0** | 1056.9 | | |
| full | 32k | 16 | 15.5 | 3498.6 | **3519.7** | 4507.8 | | |

Launches per visit: the pair is 2 encoder launches (prologue 1 dispatch; attention 1 dispatch, plus the combine dispatch
once a row sees > 64 keys). Fused: 1 launch (1 dispatch, + combine). Per decode token (48 visits): pair 96 launches /
up to 144 dispatches, fused 48 / up to 96.

What limits each:
- Short context (<= 4k SWA, 512 full): latency, not bytes. One visit is 5-9x the 3.5 us launch floor and the K/V is 2-4 MB.
  The prologue (10-12 us) is a single dependent chain: conv, two reductions, clip, the r . relproj tail.
- Full 32k, 1 row: 340 us for 134 MB, 64 % of 614 GB/s.
- More rows: the attention grid has one threadgroup per (kv head, split, row), so each row re-reads the K/V. 16 rows cost
  ~16x one row at long context. That re-read is the price of rows-exact with position-dependent splits.
- Fused vs pair: fused saves a launch and the [T, 24, 1024] rel round trip, but every split threadgroup recomputes q / r
  and the bias of its split, and the threadgroups that hold the call's own keys run the K/V conv. That wins at 1 row up to
  4k, about breaks even at 4 rows, and loses at 16 rows and at full 32k (32 splits each redo the row's q / r).

Split policy sweep (`INTARSIA_INK_SPLIT_KEYS`, same rows-exact rule): 64 keys per split (the default) is best or tied
everywhere at 1 and 4 rows. 128 is +5 us on SWA 1 row; 256 is +10-15 us at short context and -7 us on full 4k 1 row only.
The stage-1 load-batching variants (U = 2, 4, 8 steps of loads issued together) were slower than U = 1 at every shape
(earlier run, harness without pipelining) and were removed.

### Selection rule for the integration lane (InkAttnDecode vs the pair)

From the measured points only. T = 2 and full layers at 8k / 16k were not measured.
- 1 row: fused on every SWA visit (all contexts) and on full visits up to 4k. Pair on full visits at 32k.
- 4 rows: fused only at ctx 512. Pair from 4k.
- 16 rows: pair everywhere.

A 1-row decode step at short context saves ~7 us on each SWA visit and ~8-14 us on each full visit, about 0.35 ms per
token over 48 visits. Both paths give the same bits, so the choice can change from step to step. The model code was not
touched; this rule is for the integration lane.

## Prefill timing (fleetq #10655, `--prefill`): one 2048-token chunk, us

| layer | ctx before the chunk | prologue | attn, split-KV | attn, NAX |
|---|---|---|---|---|
| SWA | 0 | 978 | 12 855 | **871** |
| SWA | 2048 | 979 | 17 043 | **1 200** |
| SWA | 30 720 | 1 013 | 17 065 | **1 149** |
| full | 0 | 1 003 | 17 126 | **1 015** |
| full | 2048 | 997 | 47 269 | **2 513** |
| full | 30 720 | 1 106 | 490 618 | **22 394** |

Per chunk over 48 visits (40 SWA, 8 full) at the start of a prompt: 40 x (0.87 + 0.98) + 8 x (1.02 + 1.00) = ~90 ms, so
attention alone caps prefill near 22 k tok/s. Against the perf-strategy §8 target of 2300-3100 tok/s (660-890 ms per
chunk), that is 10-14 % of the chunk. At 30 k context: 40 x 2.16 + 8 x 23.5 = ~275 ms, 31-42 %, dominated by the 8 full
visits. Full 30 k on NAX: 2048 x 32 768 x 24 heads x 128 x 4 flop = 0.82 Tflop in 22.4 ms, 37 Tflop/s.

Integration requirement: the NAX kernel needs a host `pos0` on prefill chunks. With `OpCall::pos` set it falls back to the
split-KV kernel, which is 15-20x slower per chunk. ink-skel's graph (model/inkling2.rs) sets `pos0` and no `pos` today.

Limits: the prologue at 2048 rows (~1 ms) is half of SWA attention time. It writes the [2048, 24, 1024] bf16 rel tensor
(100 MB), and the NAX kernel then gathers from it per score. Full-layer chunks are compute bound in the NAX loop (scalar
bias gather + mask per score element, no causal tile skipping beyond the key range).

## Next ideas (ranked)

1. Decode, rows > 1 and long full context: one threadgroup per (kv head, absolute key tile) for all rows of the call. Fix the
   key partition to absolute tiles (not per-row splits) so a K/V row is read once for all rows and rows-exact still holds.
   At full 32k x 16 rows this is ~16x less K/V traffic (3.5 ms -> ~0.3-0.4 ms).
2. Prefill: compute the rel bias inside the NAX kernel from r (16 floats per row/head) and relproj (16 x 1024, fits in
   threadgroup memory), and skip the 100 MB rel tensor. This needs a contract change: rel stays an output of InkKvPrologue
   today. It would remove most of the ~1 ms prologue per chunk and the per-score gather.
3. Prologue at decode: split the rel tail over more threads (1024 columns x 3 heads on one 256-thread group) and run the
   conv / reductions with simdgroup-only reductions. 10-12 us -> ~5 us, which also makes the fused path win at 4 rows.
4. Fused kernel: compute q / r and the bias once per (kv head, row) and share them across splits (a single-split fast path
   for vis <= 64 already gives 22.8 us; for more splits a two-phase grid or simdgroup broadcast). That removes the 16-row and
   32k losses.
5. NAX full-layer chunks: skip fully masked tiles and fold the bias into a per-tile band load. At 30 k context that would
   cut the 22 ms full visit at 30 k (not measured against the plain flash NAX kernel at this size).

## Files

- kernels/metal/inkling2_attn.metal, kernels/metal/inkling2_attn_nax.metal (new)
- crates/intarsia-backend-metal/src/inkattn.rs (new); a few lines each in lib.rs (mod + dispatch), kernels.rs (table,
  registration, shader lists, test count), build.rs (watch + METAL4_SOURCES)
- crates/intarsia-backend-metal/tests/inkling2_attn.rs, examples/ink_attn_bench.rs (new)
- Op::InkAttnDecode (75): crates/intarsia-core/src/ops.rs, inkling2/ops.rs (contract, check, shape), inkling2/cases.rs;
  cpu reference crates/intarsia-backend-cpu/src/ink_ref.rs; tests/ink_attn_decode.rs; NOT_ON_CUDA / HIP UNIMPLEMENTED lists
