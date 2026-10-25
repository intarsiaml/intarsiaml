# Fragment-order repack of the dense fp8 weights: design note (lane wg, 2026-10-25)

Status: note only, nothing built. Written for a user decision. The numbers marked MEASURED are from `raw/`.

## Why it is on the table

The WMMA family needs a lane to hold row `n = lane % 16`. With the checkpoint layout (`[N,K]` row-major) one load instruction of a
wave touches 16 rows x 32 B = 16 cache lines; a coalesced GEMV touches 4. At M = 1 that makes the first WMMA kernel 7-20 % slower
than the current GEMV (MEASURED, `raw/microbench-v1.log`).

MEASURED (`raw/ablation-v1.log`, ablation build, so only ratios count): reading the same weights from a copy in fragment order
`[n/16][k/32][lane = 16 h + col][16 B]` (a wave load is then 512 contiguous bytes) takes the M = 1 kernel from
16.5 -> 11.8 us (2048x2048), 24.4 -> 19.5 (2048x4096), 31.9 -> 23.3 (4096x2048), 57.4 -> 45.4 (12288x2048), 7.1 -> 5.6 (512x2048),
i.e. -20 to -28 %, with bit-identical outputs (0 mismatches). Taking all weight loads out entirely gives 5-13 us, so the
weight stream is the whole story; staging the activations costs about 3 us and the cross-wave sum about 2 us.

Alternative that needs no repack (built in the second kernel version, measured separately): coalesced 4-rows x 128 B loads and a
wave-private LDS transpose into the fragment layout. The result is in the results doc; if it reaches parity with the GEMV at M = 1 this
note is moot for fp8.

## What a repack is

An in-place permutation of each dense fp8 `[N,K]` tensor after upload (and after TP sharding), N % 16 == 0, K % 32 == 0 (the 35B's
shapes: 8192/4096/2048/512 x 2048/4096 all qualify; a TP-halved 256 qualifies; odd shapes keep the checkpoint layout and the old
kernels, flagged per tensor).

* Memory: none net. The permutation needs one scratch buffer of the largest tensor (qkv 8192x2048 = 16.8 MB), allocated once at load
  and freed. A second copy instead would cost about 1.3 GB (the whole dense-fp8 set).
* Load-time cost: one device pass per tensor at memory speed. 1.3 GB read + write at ~500 GB/s is about 5 ms for the model, plus
  launch latency for ~200 tensors. Negligible against a multi-second load.
* Fit: the 35B NVFP4 model is 20.8 GB of weights on a 32 GB R9700 (workspace 0.96 GB, KV 11.9 GB at 16 slots). In place: no change.
  With a second copy: 22.1 GB, KV budget 11.9 -> about 10.6 GB.

## Everything that reads these weights in the original layout

All use `w[n*K + k]` with 16-byte loads along K.

| Reader | Where | What it would need |
|---|---|---|
| fp8 GEMV family (`matmul_fp8_gemv`, `_v2`, `_rows`, pair, rmsnorm_add pair, gated, sigmoid folds) | `kernels/hip/fp8.hip` `fp8_gemv_body`, pair kernels | With the knob off they must run unchanged, so either the repack is knob-gated and the GEMV is never selected for a repacked tensor, or they learn the layout. A warp-per-row GEMV on fragment order reads 16-B pieces 512 B apart: slow, so do not teach it, gate it out. |
| prefill GEMMs `matmul_fp8_mma`, `_v2`, `_f8` | `Fp8Policy::load` (`ik_dq_gemm.h`), `ik_f8_gemm.h:81` | One address formula per policy `load()` (a 16-B piece of (n, k) is at a fixed offset in its 512-B fragment block). Tile loads become contiguous per block, probably faster. Touches the prefill path, which must stay bitwise (the sum order does not change, only addresses). |
| the new WMMA family | `kernels/hip/ik_f8_rows.h` | Weight load becomes one contiguous 16-B load per lane; the LDS transpose goes away. |
| NVFP4 / MXFP4 kernels | `nvfp4.hip`, `moe*.hip` | Not affected: the `IK_DTYPE_FP8_E4M3` there are block scales of NVFP4, not dense weights. The 35B's routed experts and shared experts are NVFP4. |
| GDN / attention kernels | `gdn*.hip`, `attention*.hip` | Not affected (fp8 there is the KV cache). |
| CPU reference, `dequant_ref`, `QTensor::download` | `crates/intarsia-backend-cpu`, `quant/reference.rs` | HIP-only repack leaves them alone, but anything that downloads a HIP QTensor and dequantises would need to un-permute (a layout tag on the tensor). |
| TP sharding | `weights.rs` (`load_tensor_shard`) | Shards slice the host bytes before upload, so repack runs after sharding on the local `[n_local, k_local]`. Safe. |
| tests that build `QTensor.parts` on the host | `rows_exact.rs`, `fusion_p2prime`, `tp_hybrid`, `batch_recurrent`, `mtp` | Need the repack applied (or a layout flag) before launching a repacked-only kernel. |
| the two pair-weight kernels | `fp8_gemv_pair_kernel`, `norm_pair` | Two weight pointers; each tensor is permuted independently, fine. |

There is no existing post-load hook: `quant/mod.rs` says "tensors keep the checkpoint layout (no repacking)", and `wrap_host` is not
overridden on HIP, so HIP weights are always device copies. The work is: a layout tag on `QTensor`, a repack kernel + backend hook
called from `Fp8E4m3::load`, kernel selection by tag in the HIP registry, and the three prefill `load()` changes.

## Size of the work and risk

About 1.5 days: repack kernel + hook (0.5), prefill `load()` changes with their bitwise tests (0.5), registry/tag plumbing and tests (0.5).
Risk: the prefill GEMMs are the default path for every prompt; a bad address formula there would hit all fp8 prefill. Mitigation: the
existing prefill bitwise tests run unchanged on the repacked tensors.

What it buys: the fp8 part of the M = 1 step only. fp8 dense is about 1.3 of 20.8 GB of weights on the 35B; the NVFP4 experts are the
bulk of the bytes (stage 2) and have their own layout, where a repack would be a second, separate decision.

## Recommendation

Wait for the coalesced-load variant's M = 1 number. If it is within 1 % of the GEMV, skip the repack. If it is 1-5 % slower, the repack
would make M = 1 faster than the GEMV by about 15-20 % (extrapolating the ablation), which is worth the 1.5 days only if the NVFP4
stage will also repack; otherwise take the small solo loss against the batched gain and let the user decide.
