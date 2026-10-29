# Lane FUSE27: all-reduce + residual add + rmsnorm + W4A8 activation quant in the 27B tp=2 prefill (HIP, 2026-10-29)

Branch `lane-fuse27/ar-norm-quant`, on main `781d21833` (merged up to `efb8b4b3b`, which only touches CUDA). tatooine, 2x R9700 (gfx1201),
`--tp 2 --devices 0,1`, Qwen3.8-27B MXFP4 W4A8, MTP k2, `prefill_chunk 2048`. Radiance was stopped for every GPU job (one fleetq job
holding both cards, `--takes-down radiance`) and answered `:8080/v1/models` again before each release.
Raw data: `docs/results/2026-10-29-fuse27-ar-norm-quant/raw/` (m* = short checks, w1 / w2 = the two windows). Scripts: `scripts/fuse27/`
(copies of `scripts/pf27c/`; `micro.sh` gained a `fuse` section and a window-time guard on `ids`).

**Result: one exact change ships, the prefill norm quantises its own output for the W4A8 GEMMs that read it (`INTARSIA_HIP_NORM_QUANT`,
default on, `=0` turns it off). Two attempts to also fold the all-reduce into that norm were bitwise exact but slower, and are not in
the PR (section 3). The quant handoff alone is worth about +1 to +2 % of a warm served
prefill (+1.2 % at 16k, +1.9 % at 4k with spec off, 1k flat with MTP); 4k reads +4.1 % with MTP only because main's three rounds
scattered. Below a clean +2 % at every length, and decode with MTP is within its noise but not shown within 0.5 %: draft PR.**

## 1. What runs at a fusion point on main (4k, rank 0, rocprofv3 `generate` of `4k-0`, 3858 tokens; raw/m2 `p-fuse27-base-cli-4k-0`)

After each pushed row-parallel GEMM (attention o / GDN out, FFN down: 2 points x 64 layers x 2 chunks = 256), main runs:

| step | kernel | calls | time (rank 0 / 1) |
| --- | --- | ---: | ---: |
| reduce of the pushed partial | `ar2p_kernel` (wait for the peer, add its words, store) | 256 | 23.7 / 32.2 ms |
| residual add + rmsnorm | `rmsnorm_kernel<true, true>` (rmsnorm_add) | 256 (+128 decode-sized) | 21.9 / 21.2 ms |
| activation quant of the next GEMM(s) | `mx_a8_quant_tiled_kernel` (the direct pre-pass), one per GEMM call that reads the norm | 384 of 640 | 25.1 / 25.5 ms (all 640) |

The 640 pre-passes: per layer and chunk, the post-attention norm feeds FFN gate | up (one pair call); the input norm feeds GDN qkv | z and
b | a (two pair calls) or attention q and k | v (two calls); the remaining 258 quantise `silu_glu` / `gated_rmsnorm` outputs for FFN down
and the out projections (not norm outputs, out of scope here). There is no wide-AR epilogue left after the push (`ar2p_kernel` is it).

## 2. The change: the norm quantises for the GEMMs (`crates/intarsia-backend-hip/src/device.rs` `PreQuant`, `kernels/hip/norm.hip`)

- `ik_rmsnorm_quant_bf16` (`rmsnorm_kernel<ADD, true, QUANT=true>`): the same norm (the same ss loop, block sum and normed values, at the
  plain launch's block size) that keeps the bf16 chunks it stores in registers (H / 8 <= 4 x block: H <= 8192 at the 256-thread block),
  takes their order-free max, and writes the e4m3 bytes and the row scale exactly as `mx_a8_quant_tiled_kernel` would from the stored
  output: fragment order, 16-padded rows (blocks past M write zero rows), the K % 64 == 32 pad group zeroed. Bitwise the pre-pass.
- `PreQuant` in the backend: a prefill norm launch (eager, stream 0, rows that take the direct kernel, H % 32 == 0) writes into one
  backend buffer and records the key (buffer id, offset, rows, H, the buffer's write count after the norm). A cached-span W4A8 call
  (`matmul_mxfp4_a8_mma` / `_pair_` with `HipBackend::span` Some, the same eager stream-0 direct path) whose `x` has that exact key passes
  the buffer as `xq` to `ik_matmul_mxfp4_a8_mma_cached` / `_pair_`, which then skip `a8_quant`. Every write through the backend bumps the
  write count (`HipBuffer::touch`); the HIP collectives now touch their outputs too (`comm.rs`), so a reduced buffer never matches an old
  key. Producer and readers share stream 0, so the next norm overwrites the buffer only after the readers before it.
- It starts only after a direct GEMM was seen to read the last norm output (`wanted`), so a model whose norms feed no W4A8 GEMM (8B, the
  NVFP4 35B) never quantises: `plan show` 8B / 35B byte-identical, kernels unchanged.

At 4k (raw/m3, memo only): 384 pre-passes are gone (`mx_a8_quant_tiled_kernel` 640 -> 258 calls, 25.1 -> 11.4 ms), the norm costs
+12.6 us per prefill call (83.6 -> 96.2 us, the quant stores), net about 10 ms of kernel time per rank (norm + quant 47.0 -> 36.9 ms);
span 825.2 -> 816.7 ms. Unserved `generate` A/B at 4k, single shots: m3 base 4700 / 4697 vs dev 4780 / 4770 tok/s (+1.6 to +1.8 %),
m4 base 4726 / 4723 vs dev 4758 / 4753 (+0.6 to +0.7 %); base itself moved 0.6 % between the two checks.

## 3. Folding the all-reduce in: two exact attempts, both slower (dead ends; patch in raw/arnorm-prefill-single-pass.patch)

Both deferred only the block-end reductions whose GEMM pushed (a model-side `tp_arnorm_prefill`, independent of `INTARSIA_TP_ARNORM`,
the arm checked from the GEMM's shape so both ranks agree) and passed every gate they were run against (bitwise vs the plain chain on
both ranks, at M 2048 / 1810 / 1144 / 511 / 130 / 1599).

1. Residual add in the reduce (`ar2p_kernel<ADD>`: out and sum stored, then `rmsnorm_bf16` of the sum, which also quantises). raw/m2:
   `ar2p` 23.7 -> 37.0 ms (rank 0) / 32.2 -> 45.7 ms (rank 1), the norm 21.9 -> 14.1 ms: +14.0 / +12.3 ms against -10.4 / -10.0 ms.
   `ar2p` walks the push's fragment order, so the two extra row-major streams (res read, sum write) are touched 16 rows x 16 B per wave
   and cost 25 to 35 % more there than in rmsnorm_add's coalesced row pass. 4k `generate`: 4769 vs 4770 tok/s for memo only (m3), no gain.
2. One pass (`ar2p_wait_kernel` for the handshake, then `ar2p_norm_kernel`: 16 rows per 1024-thread block, the reduce, add, rmsnorm in
   rmsnorm_add's exact reduction order, quant; the reduced rows not stored). raw/m4: 60.5 / 59.9 ms plus a 2.6 / 13.8 ms wait kernel,
   against 23.0 + 24.6 = 47.6 ms (rank 0) for the memo-only pair; 4k `generate` 4721 / 4710 vs 4758 / 4753 tok/s memo only (-0.8 %).
   Sixteen 10 KB rows per block and the per-row block sum in shared memory leave it latency-bound at 2048 rows.

So the exact all-reduce fusion does not pay at prefill sizes on this layout; what the norm can absorb is the quant, and that is shipped.

## 4. Gates (W1, raw/w1, 13 min; Mac)

| gate | result |
| --- | --- |
| greedy ids, 18 prompts x (spec off, MTP), base vs dev, 256 tokens | 36 / 36 SAME, none skipped (`w1/ids-*`) |
| GEMM output hashes (`mxfp4_a8` hash test, base vs dev) | 48 / 48 lines identical |
| W4A8 correctness tests (`mxfp4_a8-dev`, incl. the new `a8_norm_quant_is_bitwise_the_pre_pass`) under default, `PERGROUP=1`, `DIRECT=0`, `NORM_QUANT=0` | 9 passed each |
| `kernels test --backend hip --quick` | exit 0, 75 admitted |
| `plan show` 8B / 35B, base vs dev | byte-identical (1346 / 3184 B) |
| `ar_push` | 4 passed, negative control 102 wrong results |
| `comm`, `ar_norm` under default, `AR_DRAIN=1`, `AR_WIDE=0` | 13 / 4 passed each, negative controls caught |
| Mac `cargo test --workspace --release` | 870 passed, 0 failed |

The ArNorm (decode-sized fused all-reduce + rmsnorm) path is untouched (`ar_norm` tests above); its outputs only gain a write-count touch.

## 5. Served A/B (W2, raw/w2, 17 min; localeval `speed --sizes 1024,4096,16384 --gen 128 --reps 3`, seeded nonce `scripts/fuse27/leval.py`)

Main = `781d21833` (`fuse27-base-cli`), dev = this branch (`fuse27-dev-cli`); MTP 3 interleaved rounds (main, dev, main, dev, ...: each a
fresh server), then one spec-off pair. Medians of the rounds, [min-max] of the rounds (`raw/w2/speedtab.txt`):

| | prefill main | prefill dev | | decode main | decode dev | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MTP 1k | 4292.6 [4283.8-4294.5] | 4291.7 [4254.9-4332.3] | -0.0 % | 92.9 [92.9-94.9] | 93.0 [91.6-93.2] | +0.1 % |
| MTP 4k | 4635.6 [4527.4-4776.4] | 4825.3 [4823.5-4846.5] | +4.1 % | 89.5 [86.2-90.0] | 88.3 [86.9-88.3] | -1.4 % |
| MTP 16k | 4277.8 [4260.4-4282.3] | 4329.7 [4328.7-4333.6] | +1.2 % | 75.0 [74.8-75.0] | 75.0 [73.7-75.0] | +0.0 % |
| spec off 1k | 4288.2 | 4392.4 | +2.4 % | 55.8 | 55.5 | -0.5 % |
| spec off 4k | 4772.0 | 4864.7 | +1.9 % | 54.9 | 54.9 | -0.0 % |
| spec off 16k | 4296.4 | 4349.0 | +1.2 % | 52.0 | 52.0 | +0.0 % |

Served texts (`texts`, 18 gate prompts, 256 greedy forced tokens): 18 / 18 md5 identical, main vs dev.

Reading it: 4k MTP's +4.1 % is mostly main's spread (4527 to 4776); against main's best round it is +1.0 %, against spec off +1.9 %. Decode
cannot take the new path (the quant needs the direct kernel's M > 64 rows; MTP verify and decode steps are 1 to 3 rows), and the spec
off decode is flat to 0.5 %; the MTP 4k -1.4 % sits inside main's own 4 % spread, but three rounds do not show it within 0.5 %. The
lane's window cap (2) was used, so it was not re-measured.

## 6. Next lever (not done here)

The 258 remaining pre-passes quantise `silu_glu` (FFN down) and `gated_rmsnorm` (GDN out) outputs, about 11 ms per rank at 4k. The same
`PreQuant` key would cover them if those two kernels wrote the quant (`silu_glu` H = 8704 per rank is past the 4 x 256 kept chunks: it
needs a different keep scheme).
