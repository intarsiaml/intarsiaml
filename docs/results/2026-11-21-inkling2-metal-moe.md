# Inkling-2-Mini MoE kernels on Metal (lane ink-m-moe, endor M5 Max)

Lane `ink-m-moe`, branch `ink/m-moe` from `3feffee6e` (the inkling2 ops commit). Not pushed or merged.
Every GPU run had a `fleetq` slot: correctness runs counted as GPU load, timing runs used `--quiet gpu,cpu`, and builds ran as `--load cpu`.

## Briefs received

- None after the initial brief.

## What is built

| Kernel (registry name) | Op | Rows | Notes |
|---|---|---|---|
| `moe_route_f32` | `MoeRoute` | any | One simdgroup per row (E <= 1024, K <= 32). Supports all three score functions (sigmoid, sqrt-softplus, softmax) and temperature. The bias is used only for selection; ties go to the lower id. Weights are renormalized with `+1e-20` when `norm_topk && K > 1`, then multiplied by `scale`. Uses `precise::` exp/div/sqrt and a cancellation-free log1p, so the ids match the f64 reference exactly. |
| `moe_experts_situ_bf16` | `MoeExperts` with `ink.situ` (NVFP4) | any | `.situ()` and the predicate `s.situ`, so the 35B and Gemma NVFP4 calls keep `moe_experts_bf16` (their pinned selection is unchanged). T <= 16 runs the decode kernels; T > 16 runs the grouped NAX GEMMs (metal4.0), falling back to the decode kernels without metal4.0. |
| `moe_experts_situ_bf16_rows` | same | 1..=16 | `.rows_exact()` and the same launcher. |
| `moe_experts_mxfp8_bf16` | `MoeExpertsMxfp8` | any | silu, gelu (`.ext()`) and situ epilogues. Same structure as the NVFP4 kernel. |
| `moe_experts_mxfp8_bf16_rows` | same | 1..=16 | `.rows_exact()`. |

Files:
- `kernels/metal/inkling2_moe.metal`: router and decode experts.
- `kernels/metal/inkling2_moe_nax.metal`: grouped prefill (added to `METAL4_SOURCES`).
- `crates/intarsia-backend-metal/src/inkmoe.rs`: launchers and registrations (added to `LAUNCHER_SOURCES`).
- `tests/inkmoe.rs` and `examples/inkmoe_bench.rs`.

Edits to shared files are one line or one block each:
- `lib.rs`: the module and one dispatch hook.
- `kernels.rs`: 5 `TABLE` rows, one registration arm, the shader lists, and the pinned kernel list.
- `build.rs`: the two source lists.

The prefill path reuses moe.metal's `moe_group` and the moe_mm.h helpers unchanged.

### Decode experts (T <= 16)

The weights are read in 32-element chunks: NVFP4 chunks are 16 bytes plus 2 e4m3 scales, MXFP8 chunks are 32 bytes plus 1 E8M0 scale.

- **Gate+up:** 8 lanes per weight row; a lane holds the gate row and the up row of one `r`, so x is read once for both. At H = 2304 that is 72 chunks, exactly 9 per lane. Threadgroups have 4 simdgroups covering 16 rows.
- **Down:** 4 lanes per row, again 9 chunks per lane at I = 1152.
- **Reduction:** shuffle-xor in a fixed order (1, 2, 4).
- **Decode:** e2m1 and e4m3 are decoded with the half-bits trick (code / 2^14, byte / 2^8); the power of two is folded back into the block scale exactly.

**Union grouping (Fable's lever 2).** Threadgroup row y is a pair. It works only if it leads its expert, i.e. it is the first pair of each run of 4 pairs of that expert, found by simd ballots over the ids. It then serves those pairs from one read of the expert's weights. With R rows, each expert's weights are read once per call whatever the number of rows that chose it (more than 4 pairs per expert means another leader and another read).

Down has two forms:
- **`dn_f`:** down fused with the k-order token sum, one launch. Used at T = 1, so a one-row visit is 2 launches.
- **`dn_u` + gather:** union-grouped down by pair, then the gather. Used for T >= 2.

**Rows-exact.** A pair's value never depends on T, on the routing of other tokens, on union or no union, or on dn_f versus dn_u + gather. Per (pair, row) the chunk order, the fma chain, the reduction and the roundings are always the same, and the token sum is always in k order. `tests/inkmoe.rs` checks this:
- Every T-row call (T 2, 3, 4, 7, 16), from both the ordinary and the `_rows` kernel, under all 5 decode policies, is bitwise equal to T one-row calls.
- Routings: random, identical, pooled, invalid ids, duplicates.
- bf16 and f32 out; weights at 16-byte and 8-byte alignment; activations aligned and at an odd element.
- Real H and I: 1000 NVFP4 and 3000 MXFP8 multi-row calls.
- Negative control: changing one activation of the last token changes only that token's bytes.

### Prefill (T > 16)

This is moe_nax.metal's grouped GEMM with the format (NVFP4, MXFP8) and the epilogue as template parameters:
- `moe_group` sorts the pairs by expert into tiles of 8, 16, 32 or 64 rows.
- x rows are gathered by pair.
- Each weight block is decoded to bf16 once per tile. This is exact for both formats.
- The accumulation runs on `matmul2d` (the neural accelerators) in f32.
- The situ epilogue multiplies by the routing weight `wts[pair]` in gate+up; then down, then the k-order gather.
- The results are not bitwise equal to the decode kernels (the accelerator has its own K order). This is admitted against the f64 reference as the 35B prefill path is.

## Admission (`intarsia kernels test --backend metal`, full sweep)

| Kernel | Cases | Fail | max_abs | rel_l2 | Result |
|---|---:|---:|---:|---:|---|
| moe_route_f32 | 19 | 0 | 2.0e-7 | 9.8e-8 | ADMITTED |
| moe_experts_situ_bf16 | 9 | 0 | 2.9e-3 | 2.1e-4 | ADMITTED |
| moe_experts_situ_bf16_rows | 7 | 0 | 1.7e-3 | 1.5e-4 | ADMITTED |
| moe_experts_mxfp8_bf16 | 18 | 0 | 3.4e-3 | 2.5e-4 | ADMITTED |
| moe_experts_mxfp8_bf16_rows | 14 | 0 | 2.1e-3 | 1.6e-4 | ADMITTED |

- The tolerance is the Metal MoE budget `Tol(1.5e-2, 1e-3)`; the router uses `Tol(1e-3, 2e-4)`. Ids are compared exactly.
- Cases with f32 activations are skipped (the kernels take bf16 x only).

New cases, all appended on their own streams so the existing cases keep their data:
- **Router**, in `dsv41/cases.rs` (the DSv4.1 cases use sqrt-softplus, E 64, K 6): Inkling's sigmoid, E 128, K 8, scale 8, renorm, at rows 1, 4, 16 and 2048, plus the other score functions and no-renorm / temperature 0.7 at the decode widths.
- **NVFP4 situ**, in `quant/cases.rs`: real H and I at 4 and 16 rows over 32 experts, and a 2048-row chunk over 128 experts (bf16).
- **MXFP8 situ**, in `inkling2/cases.rs`: the same three shapes.

The full-sweep run of the expert kernels takes about 26 s, most of it the f64 reference of the 2048-row chunk.

**Not checked:** the new router cases also run against CUDA's `moe_route` (sigmoid at E = 128), and no CUDA box was used here.

## Timing (examples/inkmoe_bench.rs)

Setup:
- Real shape: H 2304, I 1152, 128 experts, top-8, situ.
- Weight copies rotate so reads miss the SLC: 5 layer copies for NVFP4 (2.9 GB), 3 for MXFP8 (3.2 GB).
- Random routing: distinct top-8 per token, uniform; every call has its own.
- GB/s = the call's actual union x the expert bytes (NVFP4 4.479 MB, MXFP8 8.211 MB), divided by the median wall time of a synced batch of 240 calls (12 for prefill).
- The default policy is used (union gate+up; fused down at T = 1, union down + gather above).

| Rows | Union (mean) | NVFP4 µs / visit | NVFP4 GB/s | MXFP8 µs / visit | MXFP8 GB/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 83 (82-85) | 430 | 128-134 | 490-511 |
| 4 | 29.5 | 268-271 | 488-493 | 454-480 | 505-535 |
| 16 | 83.2 | 750-770 (3 runs); 1030-1135 in 2 noisy runs | 484-497 | 1236-1351 | 506-553 |
| 2048 (prefill chunk) | 128 | 11 900 | 22.5 TFLOP/s bf16-equivalent | 12 700 | 21 TFLOP/s |

Endor's GPU is shared with standing services, and the 16-row numbers moved by up to 45 % between runs on otherwise idle slots; the 1-row and 4-row numbers moved by 1-3 %. The table gives the range over 3-4 runs.

The one-row split (probe runs that launch only one kernel) is:
- NVFP4 gate+up: 50 µs, 480 GB/s.
- NVFP4 down (fused): 33 µs, 355 GB/s.
- MXFP8 gate+up: 95 µs, 455 GB/s.
- MXFP8 down (fused): 50 µs, 435 GB/s.

Policy comparison (all policies give the same bits):
- **One row:** all within 3 %.
- **Four rows:** union and per-pair are within noise.
- **Sixteen rows:** union-grouped beats per-pair by 1.33x (NVFP4) and 1.45x (MXFP8) on a quiet box. Per-pair relies on the cache to share an expert's reads, and at 83 experts it does not.

So at these shapes the union scheme wins, unlike lane mm's 35B result at 8 rows or fewer.

Tried and dropped: loading 3 chunks ahead per lane (`INK_NB = 3`, all loads first) was slower, 83 -> 91 µs at one NVFP4 row. A 256-thread fused down measured the same as 128 threads.

Per token at one row: 44 NVFP4 visits x 83 µs + 4 MXFP8 visits x ~130 µs = 4.17 ms of experts. The perf-strategy estimate was 48 x (58 + 36) = 4.5 ms.

**Decode/NAX crossover.** The threshold is `DEC_MAX_ROWS` = 16; above it, calls take the grouped NAX GEMMs.

| Rows | NVFP4 NAX (µs) | NVFP4 decode, `INTARSIA_INK_MOE_NAX=0` (µs) | MXFP8 NAX (µs) | MXFP8 decode (µs) |
|---:|---:|---:|---:|---:|
| 32 | 1325 | 1202 | 1766 | 1818-2628 |
| 64 | 1574 | 2358 | 2058 | 2836 |
| 128 | 1984 | 5105 | 2845 | 5846 |

NVFP4 at 32 rows is about 10 % better on the decode kernels. A threshold of 32 for NVFP4 is a possible follow-up; the verify and prefill paths rarely land there.

## Limits

- **bf16 activations only.** f32 x is not ported; the model runs bf16.
- **Shape limits.** H and I must be multiples of 32 (16 is the contract minimum for NVFP4), E <= 1024, K <= 32.
- **Alignment.** Weights must be at least 4-byte aligned, and 16-byte alignment takes the faster vector loads. A misaligned x takes a slower bf16-wise load path.
- **E8M0 edge cases.** Scale 255 decodes as NaN. In the decode kernels, an e4m3 NaN code (0x7F) decodes as a finite value; the sweep never generates it.
- **Prefill needs metal4.0.** Without the metal4.0 library, prefill runs on the decode kernels: correct, but about 400 ms per 2048-row chunk.
- **`ink.act_quant` is not implemented.** Only 0 is accepted, which is the contract.

## Next ideas

1. **Router GEMV + MoeRoute fusion.** This needs a new op, i.e. a contract change, so it was not built and is proposed to main instead. The fused op would be bf16 x `[T,2304]` and router w `[128,2304]` in, `MoeRoute` outputs out, one launch. Better still, fold it into the perf-strategy's `ink_mix_route`. It would save one launch (about 5 µs) per visit, 48 per token.
2. **The one-row floor is fixed cost.** gate+up runs at 480 GB/s while the 4- and 16-row calls stream at 530-550 GB/s. Down at one row is 33 µs for 12 MB, about 10 µs over streaming. A persistent gate+up -> down kernel with a device-scope barrier would remove one dependent dispatch per visit (worth trying, but it has no forward-progress guarantee on Metal). So would overlapping the down of visit n with the router of visit n+1.
3. **Activation quant (W4A4 / W8A8) modes.** These are reserved in the contract; on Metal they are not a speed lever for decode.
4. **Fewer decode ALU ops for NVFP4.** At the 16-row union the arithmetic per weight byte is unchanged; the measurements at 4 rows (535 GB/s for gate+up) say memory, not ALU, is the limit there.
