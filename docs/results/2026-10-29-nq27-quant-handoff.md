# Lane NQ27: silu_glu and gated_rmsnorm write the W4A8 GEMM's quantised activations (HIP, 2026-10-29)

Branch `lane-nq27/quant-handoff`, on main `6816f7840` (#248 merged). tatooine, 2x R9700 (gfx1201), `--tp 2 --devices 0,1`, Qwen3.8-27B MXFP4 W4A8, MTP k2, `prefill_chunk 2048`. Radiance stopped by two fleetq jobs that took it down (#7020 gates, #7021 served A/B) plus one short check (#7019); `:8080/v1/models` answered before each release. Data: `docs/results/2026-10-29-nq27-quant-handoff/raw/` (`c1` short check, `w1` gates, `w2` served A/B). Scripts: `scripts/nq27/` (copies of `scripts/fuse27/`; `micro.sh` gained the `act` section).

**Result: exact change ships, `INTARSIA_HIP_ACT_QUANT` (default on, `=0` off).** The prefill `silu_glu` (FFN down projection input) and `gated_rmsnorm` (GDN out projection input) also write the W4A8 direct GEMM's quantised activations from the bf16 values they store, and the GEMM skips its `mx_a8_quant_tiled` pre-pass. 222 of the 258 remaining 4k pre-passes per rank are gone (258 -> 36: 32 of the 36 follow attention, whose `o_proj` inputs come from `flash` / `sigmoid_mul`; the other 4 were not traced). Warm served prefill, 3 interleaved MTP rounds: +2.1 % (1k), +1.6 % (4k), +1.3 % (16k); decode within noise (-0.6 / +0.9 / +0.7 %).

## 1. Change

- `kernels/hip/norm.hip`: `silu_glu_quant_kernel` (one block per row of the 16-padded M, block size `ceil(K / 8 / 4)` rounded to 32 so the 8704-column shard of the 27B fits: 1088 chunks, 288 threads; `ik_silu_glu_quant_bf16`) and `gated_rmsnorm_quant_kernel` (one block per 16-padded TOKEN, one thread per 8-element chunk of the token's `heads * dv` elements, `dv / 8` lanes per head; `ik_gated_rmsnorm_quant_bf16`). Both keep the stored bf16 values in registers, take the order-free row max, and write the e4m3 bytes and f32 row scales in `mx_a8_quant_tiled_kernel`'s layout (pad rows and the K % 64 == 32 pad group zeroed), as `rmsnorm_kernel<.., QUANT>` does. The gated kernel reproduces the plain kernel's per-head sum of squares (its 8-term lane sum, then the xor tree over the head's lanes; the plain 32-lane tree adds exact zeros above them), so every stored element is the plain kernel's bits. Not taken (plain launch): rows that do not take the vector path, `silu_glu` K > 32768, `dv / 8` not a power of two <= 32, `heads * dv / 8 > 1024`.
- `device.rs` `PreQuant`: one slot for three producers (norm, silu_glu, gated_rmsnorm). `last` / `wanted` / `off` are per producer (a model whose silu_glu feeds no W4A8 GEMM never pays for that quant). `gated_rmsnorm` runs on `[tokens * heads, dv]` rows but the GEMM reads `[tokens, heads * dv]`, so its key is the GEMM's: `heads * dv` is learnt from the first direct GEMM that reads the producer's bytes (same buffer, offset, write count, element count), and the "is this a direct-kernel row count" check uses the token count (decode and MTP verify, <= 64 tokens, never quantise). A GEMM reads its activations before the next producer of the stream runs, so one buffer serves all three.
- Knob: `INTARSIA_HIP_ACT_QUANT=0` turns off the two new producers; `INTARSIA_HIP_NORM_QUANT` keeps governing the norms (`docs/tuning.md`).

## 2. Remaining quant passes and kernel time (4k, rank 0, rocprofv3, 3858 tokens; `raw/w1/p-nq27-dev-cli-4k-0.csv.gz` vs FUSE27's `raw/m3` dev trace)

| kernel | main (#248) calls / ms | this change calls / ms |
|---|---|---|
| `mx_a8_quant_tiled_kernel` | 258 / 11.33 | 36 / 0.89 |
| `silu_glu_kernel` | 192 / 18.32 | 65 / 0.26 |
| `silu_glu_quant_kernel` | 0 | 127 / 22.16 |
| `gated_rmsnorm_kernel` | 144 / 6.81 | 49 / 0.19 |
| `gated_rmsnorm_quant_kernel` | 0 | 95 / 4.85 |
| sum of these | 36.46 | 28.35 (-8.1 ms) |
| all kernels, rank 0 | 796.6 | 779.8 |

The plain calls left were not traced to their cause; the handoff starts only after the first direct GEMM read of a producer's output, which accounts for the first layer's calls, and the rest are presumably calls of 64 rows or fewer. Call counts include every pass of the run (warm-up and decode-sized calls), so per-call averages mix sizes; the sum row is the comparison.

## 3. Gates (default on)

| gate | result |
|---|---|
| greedy ids, 18 prompts x (spec off, MTP), 256 tokens (`w1/ids-*`) | 36 / 36 SAME |
| served texts (`w2/texts-*`), 18 prompts, 256 greedy tokens | 18 / 18 identical md5 (main vs dev) |
| GEMM output hashes (`a8_mma_hashes`, base vs dev binary) | 48 / 48 identical |
| `mxfp4_a8` tests under default, `PERGROUP=1`, `DIRECT=0` | 10 passed each (the new `a8_act_quant_is_bitwise_the_pre_pass` is the 10th), the `a8_norm_quant_is_bitwise_the_pre_pass` one stays green |
| `kernels test --backend hip --quick` | exit 0, 75 admitted |
| `plan show` 8B / 35B, base vs dev | byte-identical (1346 / 3184 B) |
| `ar_push` / `comm` / `ar_norm` (default, `AR_DRAIN=1`, `AR_WIDE=0`) | comm 13 passed (4 ignored), ar_norm 4 passed; negative controls caught 1380 wrong results |
| Mac `cargo test --workspace --release` | 1104 passed, 0 failed |

Not run (the next free slot was 65 minutes away behind another job's exclusive window, past the lane's cap): the `a8_` tests under `INTARSIA_HIP_ACT_QUANT=0` / `INTARSIA_HIP_NORM_QUANT=0` (the zero-hits branch of the new test), and a second sample of the spec-off 4k prefill (section 4).

The new test `a8_act_quant_is_bitwise_the_pre_pass` (`tests/mxfp4_a8.rs`) compares against the same producer launched on a side stream (never takes the handoff) and against GEMMs on a copy: silu_glu at M 2048 / 1810 / 300 / 65 (K 8704, K % 64 == 32 tails, a K = 32800 row the kernel does not take); gated_rmsnorm at 2048 / 1810 tokens x 24 heads x 128 and four other head shapes (a 65-head dv 32 K % 64 == 32 tail, dv 256, dv 512 not taken); each with the reads counted, a rewrite in place seen, and the whole chain rmsnorm_add - gate | up pair - silu_glu - down over the one slot, three rounds against a plain-producer chain.

## 4. Served A/B (`w2`, localeval `speed --sizes 1024,4096,16384 --gen 128 --reps 3`, seeded nonce `scripts/nq27/leval.py`)

main = `nq27-base-cli` (6816f7840), dev = `nq27-dev-cli`; MTP: 3 interleaved rounds (main, dev, main, dev, main, dev); spec off one pair. Median of the 3 rounds [min-max] (`scripts/nq27/speedtab2.py`):

| | prefill main tok/s | prefill dev | | decode main | decode dev | |
|---|---|---|---|---|---|---|
| MTP 1k | 4309.6 [4013.1-4334.9] | 4398.2 [4361.2-4402.5] | +2.1 % | 93.5 | 93.0 | -0.6 % |
| MTP 4k | 4832.2 [4823.9-4842.2] | 4909.7 [4907.6-4934.0] | +1.6 % | 87.6 | 88.3 | +0.9 % |
| MTP 16k | 4335.1 [4322.9-4337.9] | 4390.4 [4384.4-4392.6] | +1.3 % | 74.5 | 75.0 | +0.7 % |
| spec off 1k | 4369.6 | 4388.8 | +0.4 % | 55.8 | 55.8 | +0.1 % |
| spec off 4k | 4862.3 | 4678.6 | -3.8 % | 54.9 | 56.0 | +2.0 % |
| spec off 16k | 4356.6 | 4425.6 | +1.6 % | 51.9 | 52.0 | +0.1 % |

Reading: every MTP cell is faster and the ranges do not overlap at 4k and 16k. The spec-off 4k cell is one pair of 3 reps and is the only slower one: dev's reps were 4960 / 4621 / 4679 tok/s against main's 4884 / 4859 / 4862, so one rep is above main's best and two are 5 % below it, a spread the MTP rounds (4907-4934) do not show. It is not explained and was not re-measured (no slot); the same server flags and code run in the MTP 4k rounds. Decode: the change is reached only by rows above 64 per pass (prefill), so decode and MTP verify never run it; the decode deltas are noise.

## 5. Next

The 32 attention `o_proj` pre-passes (`flash` / `sigmoid_mul` output, 0.64 ms at 4k) are the last quant passes with a producer that could write them; they are small. The fused silu kernel was not tuned (one block shape: 288 threads x 4 kept chunks per row); a different block shape could be tried.
