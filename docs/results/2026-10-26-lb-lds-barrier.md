# Lane lb: LDS-only block barrier on gfx12, audit and measurement (2026-10-26)

Branch `lane-lb/lds-barrier`. Every base measurement below is `origin/main` 85824cef4; dev is 85824cef4 plus the patch (commit f7f2d1f09 on `lane-lb/lds-barrier-kernels`). The inventory's line numbers are 5300bfde6's, whose kernels are identical to 85824cef4's. Hardware: tatooine, one R9700 (gfx1201), card 0, 35B Qwen3.6-A3B NVFP4 on HIP, ROCm 7.2.4. All fleetq jobs `--takes-down radiance-rank0`, radiance restored after each.

## Result

**Stop rule hit: the served gain is under 2 % on both prefill and decode, so no kernel change ships.** This PR is docs only: the audited inventory, the measurements, and the patch kept as `kernels-lds-barrier.patch` (also on branch `lane-lb/lds-barrier-kernels`). Main's kernels are untouched, `__syncthreads()` stays everywhere.

The patch itself is sound as far as it was tested: 55 literal sites plus the 4 barriers inside `ik_block_sum`/`ik_block_max` (opted in per caller) converted, bitwise identical to main on every gate that ran (table below). The measured gain is just small.

| served localeval, 35B HIP, card 0, `--max-batch 8`, base / dev interleaved x3 (median of 3 runs; the 3 runs) | base | dev | delta |
|---|---|---|---|
| prefill tok/s @ 512 | 4890.0 (4915.0, 4890.0, 4769.0) | 4790.0 (4790.0, 4753.0, 4859.0) | -2.04 % |
| prefill tok/s @ 4096 | 9875.0 (9875.0, 9792.0, 9903.0) | 9968.0 (9968.0, 10034.0, 9925.0) | +0.94 % |
| prefill tok/s @ 16384 | 8804.0 (8829.0, 8804.0, 8780.0) | 8883.0 (8883.0, 8837.0, 8886.0) | +0.90 % |
| decode tok/s @ 512 ctx, 1 stream | 163.6 (163.7, 163.6, 163.2) | 164.0 (164.0, 163.9, 164.0) | +0.24 % |
| decode tok/s @ 4096 ctx, 1 stream | 159.3 (159.7, 159.3, 159.2) | 159.7 (159.7, 159.6, 160.0) | +0.25 % |
| decode tok/s @ 16384 ctx, 1 stream | 144.9 (145.1, 144.9, 144.9) | 145.4 (145.3, 145.4, 145.4) | +0.35 % |
| decode aggregate tok/s, 8 streams (4096-token prompts) | 170.5 (170.5, 169.5, 170.5) | 172.5 (172.5, 172.7, 171.0) | +1.17 % |

The 512-token prefill row is TTFT of 0.10 s per request overhead (localeval marks it "under 1024 uncached prompt tokens"), its -2 % is run noise (3 runs: 4915/4890/4769 vs 4790/4753/4859). Prefill at 4k and 16k: +0.9 %. Decode 1 stream: +0.2..0.35 %. 8 streams aggregate: +1.2 % (170.5/169.5/170.5 vs 172.5/172.7/171.0). Raw outputs: `raw/ab/`. Recipe: `scripts/ab.sh "base dev base dev base dev"`; the dev arm is the same binary plus the patch, same env, same server flags.

The ISA change is real (below) and the kernel-level gain is real but small. A likely reason (not isolated by an experiment): the loads that `s_wait_loadcnt 0` drains are mostly needed right after the barrier anyway, and `global_inv` is cheap when little is cached. l8's +1 to 5 % was a microbenchmark of a loop that is nothing but barrier-separated stages; in the served model the touched kernels are 20 to 70 % of GPU time and the barrier is a small part of each.

## Per-kernel before and after (rocprofv3 kernel trace of the real 35B, not a standalone microbench)

`scripts/prof.sh`: `intarsia generate` of the 35B under `rocprofv3 --kernel-trace`, base, dev, base, dev; per kernel the mean of the 2 runs per side. `pf4k` = 4096-token prompt, 1 new token; `pf16k` = 16384-token prompt; `dec` = 12-token prompt, 192 tokens; `dec4k` = 4k prompt, 128 tokens. Full per-kernel tables (all kernels over 1.5 ms): `raw/prof-summary/kagg-*.md`.

| mode | all kernels base ms | dev ms | delta |
|---|---:|---:|---:|
| pf4k | 388.54 | 384.88 | -0.94 % |
| pf16k | 1708.63 | 1694.31 | -0.84 % |
| dec | 1119.60 | 1118.79 | -0.07 % |
| dec4k | 1202.61 | 1195.38 | -0.60 % |

Touched kernels, summed by family (ms, base -> dev):

| family | pf4k | pf16k | dec | dec4k |
|---|---|---|---|---|
| `moe_wmma_kernel_r7` (routed experts, prefill) | 138.4 -> 136.0 (-1.7 %) | 550.7 -> 541.1 (-1.7 %) | - | 149.4 -> 145.8 (-2.4 %) |
| `f8_gemm_kernel` (dense fp8 prefill GEMM) | 56.7 -> 55.9 (-1.4 %) | 225.4 -> 222.9 (-1.1 %) | - | 64.1 -> 63.5 (-1.0 %) |
| `dq_gemm_kernel` (NVFP4 GEMM) | 21.0 -> 20.9 (-0.4 %) | 84.4 -> 83.8 (-0.8 %) | 7.3 -> 7.2 | 26.9 -> 26.8 |
| `gemm_wmma_kernel` (bf16) | 7.1 -> 6.5 (-8.3 %) | 28.6 -> 26.3 (-8.1 %) | - | - |
| `flash_hpb_kernel` | 17.7 -> 17.7 | 274.4 -> 274.1 | - | - |
| `splitkv_kernel` (decode attention) | - | - | 23.8 -> 23.0 (-3.2 %) | 38.9 -> 38.0 (-2.5 %) |
| `fp8_gemv_*` (decode GEMVs) | 2.5 -> 2.5 | - | 473.7 -> 473.5 (-0.0 %) | 313.5 -> 312.9 (-0.2 %) |
| `nvfp4_gemv_v2_kernel` | - | - | 88.2 -> 88.1 | 58.6 -> 58.6 |
| `rmsnorm_*` | 12.3 -> 12.4 | 49.4 -> 49.1 | 53.6 -> 53.3 | 36.0 -> 35.4 (-1.9 %) |

Why decode barely moves: the hot decode kernels (`moe_block_kernel` 370 ms, 33 % of decode; `fp8_gemv_pair`/`gated`, `gdn_conv_delta`) are either KEEP (ticket protocol) or have one barrier per block against long weight streams (`fp8_gemv_*` 0.0 %). Why prefill gains 1 %: only the k-loop barriers of the WMMA GEMMs gain, 1 to 2.4 % of their time, and the biggest prefill kernels after them (`gdn_delta_rows_v3` ~20 %, `flash_hpb` 16 % of 16k) already use `gdn_lds_sync` (`gdn.hip`, `gdn_fused.hip`) or have 2 barriers per tile.

## ISA check (step 3)

`hipcc -S -O3 --offload-arch=gfx1201` of every touched `.hip`, base vs patched, `scripts/isa.sh`, compared with `scripts/isacmp.py`. Per kernel: number of `s_barrier_signal`, how many have a `s_wait_dscnt` (LDS wait) before them, and how many have `global_inv` in the 5 instructions after them.

Totals over the 10 dumped files (moe, attention_flash, attention_splitkv, fp8, mxfp4, matmul, moe_block_grouped, moe_shared, nvfp4, norm): 2217 barriers in both; `global_inv` **2244 -> 45**. The 45 are the kept sites (41 in `moe.hip`: the `moe_experts_p1_v3_kernel` ticket protocol; 4 in `norm.hip`: `rmsnorm_kernel<ADD=true>`). In the base ISA the converted barriers are preceded by `s_wait_loadcnt_dscnt 0x0` or `s_wait_loadcnt 0x0` and followed by `global_inv`; in the patched ISA by `s_wait_dscnt 0x0` and nothing. Example, `moe_wmma_kernel_r7<true,64,45>`, after `v_cvt_f32_ubyte0`: base `s_wait_loadcnt_dscnt 0x0 ... s_barrier_signal -1; s_barrier_wait -1`, dev `s_wait_dscnt 0x0 ... s_barrier_signal -1; s_barrier_wait -1`.

| file | kernel | barriers | loadcnt 0 before (base / dev) | dscnt wait before (base / dev) | global_inv after (base / dev) |
|---|---|---:|---|---|---|
| moe | `moe_wmma_kernel_r7ILb1ELi64ELi13EEvPK7ik_bf16PKhS4_PKfPKiS8_S8_P` | 21 | 17 / 0 | 17 / 17 | 18 / 0 |
| moe | `moe_wmma_kernel_r7ILb0ELi64ELi13EEvPK7ik_bf16PKhS4_PKfPKiS8_S8_P` | 20 | 16 / 0 | 16 / 16 | 17 / 0 |
| moe | `moe_wmma_kernelILb1ELi32EEvPK7ik_bf16PKhS4_PKfPKiS8_S8_PS0_iiiii` | 7 | 6 / 1 | 6 / 6 | 7 / 0 |
| moe | `moe_wmma_kernel_vILb1ELi64ELi5EEvPK7ik_bf16PKhS4_PKfPKiS8_S8_PS0` | 22 | 17 / 1 | 18 / 18 | 22 / 0 |
| attention_flash | `flash_hpb_kernelILi128ELi64ELi4ELi1EEEvPK7ik_bf16S3_S3_PS1_iiill` | 2 | 2 / 0 | 1 / 1 | 2 / 0 |
| attention_flash | `flash_kernelILi64ELi64EEEvPK7ik_bf16S3_S3_PS1_iiillf` | 2 | 2 / 0 | 1 / 1 | 2 / 0 |
| attention_splitkv | `splitkv_kernelILi8ELi4ELi2ELb1EEEvPK7ik_bf16S3_S3_PfP15HIP_vecto` | 16 | 15 / 0 | 8 / 8 | 16 / 0 |
| fp8 | `f8_gemm_kernelILi4ELi1ELi2ELi1EfEvPKhS1_PKfPT3_iliii` | 2 | 2 / 1 | 2 / 2 | 2 / 0 |
| matmul | `gemm_wmma_kernelILi4ELi2ELi2ELi4EfEEvPK7ik_bf16S3_PT3_iiib` | 2 | 2 / 1 | 2 / 2 | 2 / 0 |
| mxfp4 | `mx_fold_gemm_kernelILi4ELi4ELi4ELi2EfEEvPKhS2_S2_S2_S2_PKfPT3_il` | 2 | 2 / 0 | 1 / 1 | 2 / 0 |
| mxfp4 | `mxfp4_gemv1_lds_kernelIfEEvPK7ik_bf16PKhS5_PT_lS5_S5_S7_lPKfi` | 2 | 1 / 0 | 2 / 2 | 2 / 0 |
| moe_shared | `shared_down_combine_kernelILi1EEEvPK7ik_bf16PKhS5_PKfS3_S3_PS1_l` | 1 | 1 / 0 | 1 / 1 | 1 / 0 |
| moe_block_grouped | `shared_glu_rows_kernelEPK7ik_bf16PKhS4_PKfS4_S4_S6_PS0_ili` | 1 | 1 / 0 | 1 / 1 | 1 / 0 |
| nvfp4 | `nvfp4_gemv_v2_kernelILi1EfLb1EEEvPK7ik_bf16PKhS5_PKfPT0_li` | 1 | 1 / 1 | 1 / 1 | 1 / 0 |
| norm | `rmsnorm_kernelILb0ELb1EEEvPK7ik_bf16S3_S3_PS1_S4_lff` | 2 | 1 / 0 | 1 / 1 | 2 / 0 |
| norm | `rmsnorm_kernelILb1ELb1EEEvPK7ik_bf16S3_S3_PS1_S4_lff` | 2 | 1 / 1 | 1 / 1 | 2 / 2 |
| norm | `gated_rmsnorm_kernelILb1EEEvPK7ik_bf16S3_S3_PS1_lff` | 2 | 1 / 0 | 1 / 1 | 2 / 0 |

In the dev columns a few kernels still show one `loadcnt 0` within 12 instructions of a barrier (f8_gemm, gemm_wmma, nvfp4_gemv, the 6/1 and 17/1 of the `moe_wmma_kernel` rows): those are the compiler's own data-dependence waits on a load the next instruction needs (the first pipeline stage), not a barrier-induced drain. `global_inv` is 0 at every converted barrier and the LDS wait is present at every one.

## Gates (step 4)

All on card 0 with radiance down, dev = main + patch, base = origin/main. Logs: `raw/gates/`.

| gate | result |
|---|---|
| HIP device tests, `cargo test -p intarsia-backend-hip` (all test binaries, 1 thread) | first run: 7 binaries failed. Cause not established: `hipMalloc ... out of memory` at the start of those binaries (the VRAM print was added afterwards; the re-runs saw 57 MiB used at start). `quick_coverage` failed for a different reason, see Open. Re-run of `admission moe_block moe_pairs moe_v3 packed_rows rows_exact` after the card read 57 MiB: all pass (moe_block 11/11, moe_pairs 1/1, moe_v3 7/7, packed_rows 5/5, rows_exact 11 + 2 ignored) except `admission::every_hip_kernel_matches_the_reference` (5 pass, 1 fail). **That one fails identically on origin/main** (`raw/gates/admb.full`): 10 wmma kernels report `NO CASES (untested)`, and `all_pass()` needs every kernel tested. Not caused by the patch. The other binaries of the first run passed (attention, comm, decode_v2, device_pos, flash_v2, fp8_f8, fp8_mma_v2, fp8_pair_shapes, frag_layout, fused_decode, fusion_p1/p2, gdn_v2, gdn_timing, mxfp4_decode/fold/v2, pf_flash_vp, qk_rope_kv, quantmoe, quick_coverage_wmma, round6_selection, row_tables, rows_registry, split_trigger, splitkv_occ/r8, staging, wmma_folds) |
| `intarsia kernels test --backend hip --quick` | exit 0, 66 ADMITTED, 0 FAIL |
| 256 greedy ids, base vs dev, 8B short and 4k | identical (sha 5d882689 / 6203ecdb), plan json and txt byte-identical |
| 256 greedy ids, base vs dev, 35B short and 4k | identical (sha 0d079dea / d626e2f1), plan json and txt byte-identical |
| 256 greedy ids, base vs dev, 27B MXFP4 short and 4k | identical (md5 5ee9c7be / 48e899e4) |
| 35B batched == solo (G-C), `batch_cuda_real` + `batch_recurrent` on HIP, dev | pass on the second run: `batch_cuda_real` 3 passed (`real_35b_batched_decode_is_bitwise_the_solo_step`: 17550 (row, step) pairs bitwise identical, batch 2..16; `real_35b_batched_engine_equals_the_local_engine`; the 8B one SKIPPED, `INTARSIA_QWEN3_DIR` not set), `batch_recurrent` 14 passed. The first attempt (the same GPU window as the OOM failures above) failed after 1.1 s with panics at `batch_cuda_real.rs:107` and `:231`; its log was grep-filtered so the error text was not kept (`raw/gates/g-c-run1-failed.log`); the identical command passed when re-run with 57 MiB VRAM in use at start (`raw/gates/g-c-dev.full`). Not run on base (it is a same-binary equality test, base vs dev ids above cover main) |


Not run: the 8B/27B served A/B (stop rule hit on the 35B first); CUDA/Metal (HIP-only change, `kernels/hip/` only).

## Inventory (step 1)

Every literal `__syncthreads()` under `kernels/hip/` at 5300bfde6 (88 grep hits: 86 code sites, 2 comments), plus the call sites that reach the helper's barriers and the `IK_FP4_LUT` uses. Read site by site, with the surrounding code, not taken from the earlier unreviewed `lane-l8/lds-barrier` branch (its classifications were not reused; every row was classified from the code).

Counts of the 86 literal sites: **70 SAFE** (55 converted outright, 4 helper barriers converted per caller through a template flag, 11 SAFE but not converted because the kernel is cold: `argmax`, `moe_topk*`, `moe_group`, `block_excl_scan`, `gdn_delta_rows_kernel` v1), **16 KEEP** (3 `allreduce.hip`, 1 `gdn_fused.hip`, 7 `moe_block.hip`, 5 `moe.hip` `moe_experts_p1_v3_kernel`). Rows below are one per site or per group of the same site pattern; the helper rows (ik_hip.h) and the `IK_FP4_LUT` use rows are context, the literal barriers they contain are counted once in the 86.

Rule used. SAFE: the data that crosses the barrier moves through LDS (or shuffles) only, and after the barrier no thread reads global memory that another thread of the block wrote before it; the barrier is not part of a cross-block protocol. KEEP: anything else or unsure. Two judgement calls worth a second look: `moe_block.hip:213/288` and `moe.hip:1225` (the table stagings inside the ticket kernels) are LDS-only in data flow, but the kernels are cross-block protocols, so they stay KEEP; `norm.hip` `rmsnorm_kernel<ADD=true>` writes `sum` to global before the barrier and re-reads it after, so it stays KEEP even though the same thread re-reads its own element (the template flag `ik_block_sum<!ADD>` encodes it).

| file:line | kernel / context | class | converted | why |
|---|---|---|---|---|
| ik_hip.h:100 | ik_block_sum (helper, 2 barriers) | SAFE | per caller | red[] in LDS only; converted via `ik_block_sum<true>` by the callers below, default stays __syncthreads |
| ik_hip.h:102 | ik_block_sum (helper, 2 barriers) | SAFE | per caller | red[] in LDS only; converted via `ik_block_sum<true>` by the callers below, default stays __syncthreads |
| ik_hip.h:111 | ik_block_max (helper, 2 barriers) | SAFE | per caller | same, red[] in LDS only |
| ik_hip.h:113 | ik_block_max (helper, 2 barriers) | SAFE | per caller | same, red[] in LDS only |
| attention.hip:56 | attn_kernel: ik_block_max<true> caller | SAFE | yes | sc/red in LDS; K/V/q global read-only inputs; out written once at the end |
| attention.hip:61 | attn_kernel: ik_block_sum<true> caller | SAFE | yes | same; also orders the sc[] write (LDS) before the reads |
| norm.hip:97 | gated_rmsnorm_kernel: ik_block_sum<true> | SAFE | yes | no global write before the barrier; `out` may alias x but each element is read and written by the same thread |
| norm.hip:52 | rmsnorm_kernel<ADD=false>: ik_block_sum<true> | SAFE | yes | no global write before the barrier |
| norm.hip:52 | rmsnorm_kernel<ADD=true>: ik_block_sum<false> | KEEP | no | `sum` is written (global) before the barrier and re-read after it (write-out then re-read): kept even though the same thread re-reads its own chunk |
| fp8.hip:96 | norm_add_stage: ik_block_sum<true> | SAFE | yes | xs/red in LDS; sum/normed are only written (block 0), never re-read |
| matmul.hip:133 | norm_add_stage (gemv_norm_body): ik_block_sum<true> | SAFE | yes | same as fp8.hip |
| argmax.hip:43 | argmax_kernel | SAFE | no | sv/si LDS; not hot (one 1024-thread block per row, nothing in flight) |
| gate_qkr.hip:100 | gate_qk_norm_rope_kv_kernel | SAFE | yes | nrm/fsn/fcs in LDS; gate/cache are written, never re-read; q/k/v inputs read-only |
| elementwise.hip:298 | qk_norm_rope_kv_kernel | SAFE | yes | as gate_qkr |
| allreduce.hip:76 | ar2_kernel (s_seq broadcast) | KEEP | no | cross-device flag protocol kernel (atomics, flags, __threadfence_system) |
| allreduce.hip:99 | ar2_kernel (before the flag store) | KEEP | no | orders the pushed global data before tid 0 stores the flag |
| allreduce.hip:108 | ar2_kernel (after the flag wait) | KEEP | no | other threads then read the peer-written scratch: global read of foreign data |
| attention.hip:33 | attn_kernel (qs staging) | SAFE | yes | qs in LDS, q read-only |
| attention.hip:77 | attn_kernel (sc reuse) | SAFE | yes | WAR on sc[] (LDS) |
| moe_shared.hip:47 | shared_down_combine_kernel (lut) | SAFE | yes | lut in LDS; weight/x prefetch in flight across it is the point |
| attention_flash.hip:84 | flash_kernel (Ks/Vt tile) | SAFE | yes | K/V cache is a read-only input here; tiles go through LDS |
| attention_flash.hip:101 | flash_kernel (Ks/Vt tile) | SAFE | yes | K/V cache is a read-only input here; tiles go through LDS |
| attention_flash.hip:305 | flash_hpb_kernel (register-prefetch ring) | SAFE | yes | same; load_regs(kt+1) stays in flight across 330 |
| attention_flash.hip:330 | flash_hpb_kernel (register-prefetch ring) | SAFE | yes | same; load_regs(kt+1) stays in flight across 330 |
| attention_splitkv.hip:185 | splitkv_body (head merge) | SAFE | yes | s_acc/s_m/s_l in LDS; pacc/pml written, not re-read |
| attention_splitkv.hip:192 | splitkv_body (head merge) | SAFE | yes | s_acc/s_m/s_l in LDS; pacc/pml written, not re-read |
| gdn_fused.hip:173 | gdn_conv_delta_kernel (before the ticket) | KEEP | no | orders the wave's state loads before the global ticket atomic (cross-block protocol) |
| gdn.hip:228 | gdn_delta_rows_kernel (v1) | SAFE | no | sq/sk LDS; fallback kernel, prefetch already consumed before the barrier. v2/v3 already use gdn_lds_sync (prior art, same form) |
| gdn.hip:234 | gdn_delta_rows_kernel (v1) | SAFE | no | sq/sk LDS; fallback kernel, prefetch already consumed before the barrier. v2/v3 already use gdn_lds_sync (prior art, same form) |
| matmul.hip:145 | norm_add_stage (gemv_norm_body) | SAFE | yes | xs LDS; sum/normed written by block 0 only, not re-read |
| matmul.hip:525 | gemm_wmma_kernel (double-buffered tiles) | SAFE | yes | sw/sx in LDS, x/w read-only |
| matmul.hip:545 | gemm_wmma_kernel (double-buffered tiles) | SAFE | yes | sw/sx in LDS, x/w read-only |
| fp8.hip:108 | norm_add_stage (PRO == 3) | SAFE | yes | xs LDS; each thread re-reads its own LDS chunk |
| fp8.hip:229 | fp8_gemv_body (activation staging) | SAFE | yes | xs in LDS; x/pz/pnw read-only inputs |
| nvfp4.hip:106 | nvfp4_gemv_v2_kernel (lut + xs) | SAFE | yes | LDS only |
| nvfp4.hip:198 | nvfp4_glu_kernel (lut) | SAFE | yes | LDS only |
| mxfp4.hip:184 | mxfp4_gemv1_lds_kernel (xs staging) | SAFE | yes | LDS only; first weight batch in flight across it |
| mxfp4.hip:713 | mx_fold_actq_kernel | SAFE | yes | sb_s/red in LDS; xc/aref only written |
| mxfp4.hip:831 | mx_fold_gemm_kernel (double-buffered) | SAFE | yes | sw/sx LDS, inputs read-only |
| mxfp4.hip:854 | mx_fold_gemm_kernel (double-buffered) | SAFE | yes | sw/sx LDS, inputs read-only |
| ik_dq_gemm.h:137 | dq_gemm_kernel (double-buffered) | SAFE | yes | sw/sx LDS, inputs read-only |
| ik_dq_gemm.h:157 | dq_gemm_kernel (double-buffered) | SAFE | yes | sw/sx LDS, inputs read-only |
| ik_f8_gemm.h:116 | f8_gemm_kernel (double-buffered) | SAFE | yes | sw/sx LDS, inputs read-only |
| ik_f8_gemm.h:139 | f8_gemm_kernel (double-buffered) | SAFE | yes | sw/sx LDS, inputs read-only |
| ik_nf4_rows.h:39 | nf4r_body (lut) | SAFE | yes | LDS only |
| ik_nf4_rows.h:97 | nf4r_body (cross-wave sum buffer) | SAFE | yes | sp in LDS; out written after |
| ik_nf4_rows.h:104 | nf4r_body (cross-wave sum buffer) | SAFE | yes | sp in LDS; out written after |
| ik_f8_rows.h:208 | f8r_body (x staging, non-DIRECT) | SAFE | yes | xs LDS; sum/normed (PRO 3) written, not re-read |
| ik_f8_rows.h:278 | f8r_body (restage next window) | SAFE | yes | WAR on xs (LDS) |
| ik_f8_rows.h:293 | f8r_body (cross-wave sum) | SAFE | yes | sp in LDS; out written after |
| ik_f8_rows.h:300 | f8r_body (cross-wave sum) | SAFE | yes | sp in LDS; out written after |
| moe_block_grouped.hip:52 | shared_glu_rows_kernel, shared_down_combine_rows_kernel (lut) | SAFE | yes | LDS only; no ticket, plain grid |
| moe_block_grouped.hip:124 | shared_glu_rows_kernel, shared_down_combine_rows_kernel (lut) | SAFE | yes | LDS only; no ticket, plain grid |
| moe_wmma.h:163 | moe_wmma_kernel, moe_wmma_kernel_v (k loop, sred) | SAFE | yes | sx/sred LDS; src/w/pairs read-only, dst written, not re-read |
| moe_wmma.h:171 | moe_wmma_kernel, moe_wmma_kernel_v (k loop, sred) | SAFE | yes | sx/sred LDS; src/w/pairs read-only, dst written, not re-read |
| moe_wmma.h:183 | moe_wmma_kernel, moe_wmma_kernel_v (k loop, sred) | SAFE | yes | sx/sred LDS; src/w/pairs read-only, dst written, not re-read |
| moe_wmma.h:194 | moe_wmma_kernel, moe_wmma_kernel_v (k loop, sred) | SAFE | yes | sx/sred LDS; src/w/pairs read-only, dst written, not re-read |
| moe_wmma.h:330 | moe_wmma_kernel, moe_wmma_kernel_v (k loop, sred) | SAFE | yes | sx/sred LDS; src/w/pairs read-only, dst written, not re-read |
| moe_wmma.h:338 | moe_wmma_kernel, moe_wmma_kernel_v (k loop, sred) | SAFE | yes | sx/sred LDS; src/w/pairs read-only, dst written, not re-read |
| moe_wmma.h:350 | moe_wmma_kernel, moe_wmma_kernel_v (k loop, sred) | SAFE | yes | sx/sred LDS; src/w/pairs read-only, dst written, not re-read |
| moe_wmma.h:372 | moe_wmma_kernel, moe_wmma_kernel_v (k loop, sred) | SAFE | yes | sx/sred LDS; src/w/pairs read-only, dst written, not re-read |
| moe_wmma_r7.h:99 | moe_wmma_kernel_r7 (lut, k loop, sred) | SAFE | yes | same; the W ring (3-6 loads/lane) is what the old barrier drained |
| moe_wmma_r7.h:197 | moe_wmma_kernel_r7 (lut, k loop, sred) | SAFE | yes | same; the W ring (3-6 loads/lane) is what the old barrier drained |
| moe_wmma_r7.h:207 | moe_wmma_kernel_r7 (lut, k loop, sred) | SAFE | yes | same; the W ring (3-6 loads/lane) is what the old barrier drained |
| moe_wmma_r7.h:221 | moe_wmma_kernel_r7 (lut, k loop, sred) | SAFE | yes | same; the W ring (3-6 loads/lane) is what the old barrier drained |
| moe_wmma_r7.h:243 | moe_wmma_kernel_r7 (lut, k loop, sred) | SAFE | yes | same; the W ring (3-6 loads/lane) is what the old barrier drained |
| moe_block.hip:164 | moe_block_kernel (ticket broadcast) | KEEP | no | tid 0 draws a global ticket (atomic) and the block reads it back from LDS; data is LDS-only but it is the entry of the cross-block ticket protocol. Unsure -> KEEP |
| moe_block.hip:213 | moe_block_kernel (lut + x staging, shared / routed roles) | KEEP | no | LDS-only data flow, but inside the ticket/flag kernel whose producers are ordered by __threadfence + atomics; the per-role weight loads in flight are what the old barrier drains. Not isolated from the protocol -> KEEP (candidate for a follow-up with a role-level proof) |
| moe_block.hip:288 | moe_block_kernel (lut + x staging, shared / routed roles) | KEEP | no | LDS-only data flow, but inside the ticket/flag kernel whose producers are ordered by __threadfence + atomics; the per-role weight loads in flight are what the old barrier drains. Not isolated from the protocol -> KEEP (candidate for a follow-up with a role-level proof) |
| moe_block.hip:248 | moe_block_kernel (producer release half) | KEEP | no | __threadfence() then barrier then tid 0 atomic release: the barrier orders every thread's global h / sa writes before the release |
| moe_block.hip:326 | moe_block_kernel (producer release half) | KEEP | no | __threadfence() then barrier then tid 0 atomic release: the barrier orders every thread's global h / sa writes before the release |
| moe_block.hip:403 | moe_block_kernel (after the acquire spin) | KEEP | no | tid 0 spins on the global counter; the barrier then lets all threads read producer-written global h / sa |
| moe_block.hip:489 | moe_block_kernel (end) | KEEP | no | orders the block's global out writes before tid 0's atomic done-count / counter reset |
| moe.hip:235 | moe_topk_kernel (v1) | SAFE | no | s_val/s_id LDS; not hot, nothing in flight |
| moe.hip:305 | moe_topk_v2_kernel | SAFE | no | same |
| moe.hip:324 | block_excl_scan (moe_group_kernel) | SAFE | no | wsum LDS; one block, not hot |
| moe.hip:326 | block_excl_scan (moe_group_kernel) | SAFE | no | wsum LDS; one block, not hot |
| moe.hip:336 | block_excl_scan (moe_group_kernel) | SAFE | no | wsum LDS; one block, not hot |
| moe.hip:349 | moe_group_kernel | SAFE | no | LDS counters/cursors (LDS atomics); offsets/active/pairs only written; one block, not hot |
| moe.hip:354 | moe_group_kernel | SAFE | no | LDS counters/cursors (LDS atomics); offsets/active/pairs only written; one block, not hot |
| moe.hip:367 | moe_group_kernel | SAFE | no | LDS counters/cursors (LDS atomics); offsets/active/pairs only written; one block, not hot |
| moe.hip:410 | moe_gate_up_kernel, moe_down_kernel (stage_rows) | SAFE | yes | xs/hs LDS staging of read-only rows |
| moe.hip:412 | moe_gate_up_kernel, moe_down_kernel (stage_rows) | SAFE | yes | xs/hs LDS staging of read-only rows |
| moe.hip:477 | moe_gate_up_kernel, moe_down_kernel (stage_rows) | SAFE | yes | xs/hs LDS staging of read-only rows |
| moe.hip:479 | moe_gate_up_kernel, moe_down_kernel (stage_rows) | SAFE | yes | xs/hs LDS staging of read-only rows |
| moe.hip:993 | moe_gate_up_p1_v2_kernel | SAFE | yes | xs + lut LDS; x/ids read-only |
| moe.hip:1088 | moe_down_reduce_p1_v2_kernel | SAFE | yes | hs + lut LDS; h written by the previous launch |
| moe.hip:1179 | moe_experts_p1_v3_kernel (ticket broadcast, lut) | KEEP | no | same structure as moe_block_kernel (global ticket, atomics); kept |
| moe.hip:1225 | moe_experts_p1_v3_kernel (ticket broadcast, lut) | KEEP | no | same structure as moe_block_kernel (global ticket, atomics); kept |
| moe.hip:1263 | moe_experts_p1_v3_kernel (release, acquire, reset) | KEEP | no | cross-block protocol |
| moe.hip:1317 | moe_experts_p1_v3_kernel (release, acquire, reset) | KEEP | no | cross-block protocol |
| moe.hip:1370 | moe_experts_p1_v3_kernel (release, acquire, reset) | KEEP | no | cross-block protocol |
| ik_quant.h:109 | IK_FP4_LUT macro body (the barrier in every use below) | SAFE | yes | fills a 16-entry LDS table then barrier; no global write precedes it; converted once here |
| moe_wmma.h:86 | IK_FP4_LUT at kernel entry: moe_wmma_kernel(_v) | SAFE | yes | macro (ik_quant.h:109) fills a 16-entry LDS table first thing, before any global write |
| moe_wmma.h:250 | IK_FP4_LUT at kernel entry: moe_wmma_kernel(_v) | SAFE | yes | macro (ik_quant.h:109) fills a 16-entry LDS table first thing, before any global write |
| moe.hip:394 | IK_FP4_LUT at kernel entry: moe_gate_up/down(_g2/_p1) kernels | SAFE | yes | macro (ik_quant.h:109) fills a 16-entry LDS table first thing, before any global write |
| moe.hip:463 | IK_FP4_LUT at kernel entry: moe_gate_up/down(_g2/_p1) kernels | SAFE | yes | macro (ik_quant.h:109) fills a 16-entry LDS table first thing, before any global write |
| moe.hip:549 | IK_FP4_LUT at kernel entry: moe_gate_up/down(_g2/_p1) kernels | SAFE | yes | macro (ik_quant.h:109) fills a 16-entry LDS table first thing, before any global write |
| moe.hip:673 | IK_FP4_LUT at kernel entry: moe_gate_up/down(_g2/_p1) kernels | SAFE | yes | macro (ik_quant.h:109) fills a 16-entry LDS table first thing, before any global write |
| moe.hip:842 | IK_FP4_LUT at kernel entry: moe_gate_up/down(_g2/_p1) kernels | SAFE | yes | macro (ik_quant.h:109) fills a 16-entry LDS table first thing, before any global write |
| moe.hip:888 | IK_FP4_LUT at kernel entry: moe_gate_up/down(_g2/_p1) kernels | SAFE | yes | macro (ik_quant.h:109) fills a 16-entry LDS table first thing, before any global write |
| nvfp4.hip:23 | IK_FP4_LUT at kernel entry: nvfp4_gemv_kernel, DqArgs init | SAFE | yes | macro (ik_quant.h:109) fills a 16-entry LDS table first thing, before any global write |
| nvfp4.hip:283 | IK_FP4_LUT at kernel entry: nvfp4_gemv_kernel, DqArgs init | SAFE | yes | macro (ik_quant.h:109) fills a 16-entry LDS table first thing, before any global write |
| mxfp4.hip:86 | IK_FP4_LUT at kernel entry: mxfp4 GEMVs, DqArgs init | SAFE | yes | macro (ik_quant.h:109) fills a 16-entry LDS table first thing, before any global write |
| mxfp4.hip:144 | IK_FP4_LUT at kernel entry: mxfp4 GEMVs, DqArgs init | SAFE | yes | macro (ik_quant.h:109) fills a 16-entry LDS table first thing, before any global write |
| mxfp4.hip:292 | IK_FP4_LUT at kernel entry: mxfp4 GEMVs, DqArgs init | SAFE | yes | macro (ik_quant.h:109) fills a 16-entry LDS table first thing, before any global write |
| mxfp4.hip:402 | IK_FP4_LUT at kernel entry: mxfp4 GEMVs, DqArgs init | SAFE | yes | macro (ik_quant.h:109) fills a 16-entry LDS table first thing, before any global write |

## Patch (not merged)

`kernels-lds-barrier.patch` (21 files, +90 -67, barrier lines only plus the helper). Helper in `kernels/hip/ik_hip.h`:

```
#define IK_LDS_BARRIER() do {{ __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local"); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local"); }} while (0)
```

(same form as `gdn_lds_sync` in `gdn.hip` and radiance's `lds_barrier()`), with a comment stating when it is legal. `ik_block_sum<bool LDS_ONLY = false>` / `ik_block_max<...>` take the flag so the default stays `__syncthreads`.

Restore it with `git apply --3way docs/results/2026-10-26-lb-lds-barrier/kernels-lds-barrier.patch`, or `git merge lane-lb/lds-barrier-kernels`. It was generated against 85824cef4; on the current main (with l5r7's moe-zfast edits) plain `git apply` fails on one hunk of `moe_wmma_r7.h` (the barrier at line 99 sits next to l5r7's `slot` line) and `--3way` applies cleanly (checked with a temporary index, not built). Worth reviving only together with another change that makes the barrier a larger share of a kernel (a deeper register-prefetch ring in the WMMA GEMMs: then the `loadcnt 0` drain is what stops the ring from overlapping, which is l8's case).

## Open

* Not shipped: the converted kernels give -0.9 % kernel time in prefill and -0.1 to -0.6 % in decode; served +0.9 % prefill, +0.3 % decode, +1.2 % at 8 streams.
* `admission::every_hip_kernel_matches_the_reference` fails on main (10 wmma kernels with no admission case, `NO CASES (untested)`); `quick_coverage::every_hip_kernel_has_a_quick_case` fails with `moe_block_bf16_grouped` having no quick sweep case; it needs no GPU and depends only on the Rust kernel registry, which the patch does not touch (re-run on a base build: same failure). Both fail identically on origin/main.
* Scripts used (`scripts/`) are copies of what ran from `~/intarsiaml-lb/dev/scripts/lb/`; to rerun, copy them to `scripts/lb/` of a checkout.
