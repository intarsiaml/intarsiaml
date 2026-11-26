# Inkling-2-Mini on HIP, one R9700 (lane ink-hip, 2026-11-25)

Branch `ink/hip`, base origin/main `072dd0863`. Goal: `intarsia generate --backend hip` on one R9700 (gfx1201, 32 GB), G1 / G2
pass, then first decode / prefill tok/s. W4A16, one rank, plain decode. Not built tonight and refused by guards: batch, TP, MTP /
DSpark, prefix store, W4A4.

## Status

| Step | State |
|---|---|
| Port map + lane skeleton | done (this commit) |
| Checkpoint on tatooine (`/data/models/intarsia/Inkling-2-Mini`, 20 GB; `/data` has 1.5 TB free) | copy running (fleetq, `--load net,disk`) |
| Kernel packages A / B / C / R | open |
| `kernels test`, G1, G2 | open |
| Speed, rocprofv3 profile | open |

## 0. Coordinator brief, 2026-11-25 about 21:50 EST (supersedes section 2's staffing; kept here against context loss)

- Three kernel lanes run in parallel, each from origin/main `d8c3c7148` (includes #496); `ink/hip` is rebased onto it:
  - **ink-hip-attn** (`.worktrees/ink-hip-attn`, `ink/hip-attn`): 70, 71, 75.
  - **ink-hip-mix** (`.worktrees/ink-hip-mix`, `ink/hip-mix`): 67, 68, 69, 74, 73, MoeRoute.
  - **ink-hip-moe** (`.worktrees/ink-hip-moe`, `ink/hip-moe`): NVFP4 situ `MoeExperts`, `MoeExpertsMxfp8` (72).
- Lanes own only new files (`kernels/hip/ink_<area>*.hip`, their ABI headers, `src/ink<area>.rs`, `tests/ink_<area>*.rs`) and add
  one-line `mod` / kernel-list entries to shared files. Each writes its wiring (dispatch arms, ABI symbols, admission cases)
  into `HANDOFF.md` at its worktree root.
- The lead (this lane) owns every shared file (`lib.rs`, `device.rs`, `kernels.rs`, admission / registration, the backend
  checks of `model/inkling2.rs`, loader, recipe), merges the lane branches into `ink/hip`, wires them in, and does not write
  their kernels. The lead owns all tatooine GPU windows for integration (G1, G2, end-to-end tok/s); lanes take their own unit
  windows (20 min or less, at most 4 per lane). No window starts after 06:00 EST; radiance up for good by 07:00.
  Builds: `--load cpu`, no takedown.
- Report when G2 passes, or at 07:00.

Lead's note: `moe_route` was already written here before the brief arrived (`kernels/hip/ink_route.hip`, `src/inkroute.rs`,
`ik_hip_abi_inkroute.h`, lead-owned files); the coordinator was told so the mix lane can skip it or its copy is dropped at merge.

## 1. Port map

### 1.1 What HIP already has (checked against `crates/intarsia-backend-hip/src/kernels.rs` TABLE)

| Model call (`model/inkling2.rs`) | HIP kernel today | Note |
|---|---|---|
| `Embedding`, `RmsNorm` (layer 0, embed norm), `Add` | `embedding_bf16`, `rmsnorm_bf16`, `add_bf16` | as is |
| `wo` (`Matmul` bf16, `[T,3072] -> [T,2304]`) | `matmul_bf16_gemv[_rows]` (T <= 16), `matmul_bf16` / `matmul_bf16_wmma` | as is |
| Router (`Matmul` with f32 out, else `Logits`; 128 x 2304) | `logits_bf16_gemv[_rows]`, `logits_bf16` | the model falls back to `Logits` by itself (`inkling2.rs:1316-1320`) |
| Unembed (`Logits`, 200058 x 2304 bf16) + `Argmax` | `logits_bf16_gemv`, `argmax_f32` | 0.92 GB per token: crosses the R9700's 1 ms dispatch stall (`ik_hip.h`), speed item S5 |
| `InkAttnDecode` (75) | none needed | optional: the model runs the pair 70 + 71 where 75 is not admitted |
| `InkRoute` (76) | none needed | optional (Metal only): the model runs router matmul + `MoeRoute` |
| P12, `HeadScreen`, `LogitsGather`, `ScreenArgmax`, `GumbelArgmax` | none needed | all optional, selected by registry probe or by backend name (cuda / metal only) |

### 1.2 New kernels (none exist on HIP)

"Correctness form" is what G1 / G2 need; "speed form" comes only after G2. All bf16 activations, f32 math, rounding points of
`inkling2/ops.rs`. Every kernel must keep the rows-exact rule (G0): fixed reduction order per row, independent of T.

| Op | Correctness form (template) | Speed form (after G2) | Pkg |
|---|---|---|---|
| `AttnResMix` 67 | port of `ink_row_kernel` (kernels/cuda/ink_mix.cu, SIMT, one block per row, 768 threads) | same | B |
| `SconvRes` 69 | same row kernel + `ink_ring_kernel` | same | B |
| `ConvMix` 74 | same row kernel + ring + the blend kernels | same | B |
| `LoopBlend` 68 | port of `ink_blend_gate_kernel` / `ink_blend_norm_kernel` / `ink_ctl_kernel`; the h GEMV (`[64, 4608]`) as a plain wave GEMV in a fixed order | WMMA h GEMV (`ik_dq_gemm.h` fragment helpers) | B |
| `InkInProj` 73 | port of the SIMT `in_proj_gemv_kernel<R,false>` (kernels/cuda/inkling2_p12.cu:291, bf16 arm only) at every T | T > 16: three `matmul_bf16_wmma`-style GEMMs over the q / k / v segments (kernels/hip/matmul.hip), `Exactness::Fast` like CUDA's cuBLAS kernel | B |
| `InkKvPrologue` 70 | port of `ink_prologue_k` + `ink_ring_write_k` (kernels/cuda/inkling2_attn.cu:244, :305; SIMT) | same | A |
| `RelSinkAttn` 71 | port of the scalar split-KV path `ink_attn_s1_k` + `ink_attn_comb_k` + `ink_kv_write_k` (:496, :526, :552; CUDA's `INTARSIA_INK_DECODE=splitkv` path, no tensor cores) at every T | T > 16: fork of `flash_hpb_kernel` (kernels/hip/attention_flash.hip, 35.8 KB LDS) with the rel bias, the sink and the gate; T <= 16: WMMA tile kernel only if the profile asks for it | A |
| `MoeExperts` situ (NVFP4) | port of the SIMT `ink_moe_group_kernel` / `ink_moe_gu_kernel` / `ink_moe_dn_kernel` / `ink_moe_gather_kernel` (kernels/cuda/ink_moe.cu), which serve any pair count from a pair table; plain layout only (`ink.moe_layout = 0`) | prefill: `moe_wmma_r7.h` with a SiTU epilogue and I = 1152 | C |
| `MoeExpertsMxfp8` 72 | same kernels, FMT = MXFP8 (e4m3 x E8M0 per 32) | prefill: fp8 or bf16 WMMA | C |
| `MoeRoute` | port of `moe_route_kernel` (kernels/cuda/dsv41_misc.cu:32, one wave per row, sigmoid + bias, top-8) | same | R |

### 1.3 CUDA features and their HIP form

| CUDA | Where | HIP / gfx1201 form |
|---|---|---|
| `__shfl_*_sync(0xffffffffu, ..)` | every file | `ik_warp_sum` / `ik_warp_max` or `__shfl_xor(v, o, IK_WARP)` from `ik_hip.h` (ROCm 7.2's `_sync` forms want a 64-bit mask). Wave size 32 on both, so lane mappings carry over |
| `mma.sync m16n8k16`, `ldmatrix`, `movmatrix` (`ik_quant.cuh`) | `ink_ctl_kernel`, `ink_mma_gemv*`, `p12_in_proj`, tile / staged / flash attention, `ink_moe_mma.cu` | gfx12 WMMA 16x16x16 (`moe_wmma.h:43`, `ik_dq_gemm.h`, `ik_f8_gemm.h`). Other tile shape and summation order: bits differ from CUDA, G2 is a tolerance gate. Not used in the correctness forms |
| `mxf4nvf4` FP4 tensor cores (`ink_moe_fp4_sm120a.cu`) | W4A4 prefill | none on RDNA4. W4A4 stays unbuilt and refused (`act_quant` not registered) |
| `cp.async` double buffering | staged tile, flash, `ink_moe_mma` | none; global loads to registers (the design of `moe_wmma_r7.h`) |
| Dynamic shared memory 69.6 KB (attention tile / flash at D = 128), 85-101 KB (`ink_moe_mma`) | speed forms | LDS limit 64 KB per workgroup: those kernels are not ported, the HIP templates above replace them (flash 35.8 KB, r7 18 KB). The correctness forms use < 48 KB (`ink_moe.cu` about 33 KB; `ink_attn_s1_k` dynamic `tb[]` to be sized under 48 KB by the lane) |
| PDL (`griddepcontrol`, `cudaLaunchKernelEx`), `prefetch.global.L2` (`ink_pdl.cuh`) | attention, MoE, P12 | dropped (no-ops); plain launches |
| cuBLAS (`ink_in_proj_bf16_cublas`) | prefill in-projection | `matmul.hip` WMMA GEMM |
| fp8 / fp4 convert intrinsics (`__nv_cvt_*`) | `ink_moe.cu`, `ink_actq` | `ik_quant.h` decoders (e4m3, e2m1); `ink_actq` not ported (W4A4) |
| `atomicAdd` / `atomicMin` (group sort, blend ticket) | `ink_moe.cu`, `ink_mix.cu` | same names in HIP |
| `-ffp-contract`: hipcc contracts `a * b + c` to FMA by default | all | where the CUDA kernel relies on explicit `fmaf` / `__fmul_rn`, keep it explicit; a kernel whose row bits must not depend on the code path goes on `NO_FP_CONTRACT_SOURCES` in `build.rs` (lead edits `build.rs` on request) |
| R9700 dispatch stall (a kernel over about 1 ms is followed by a 256-275 us idle gap, `ik_hip.h`) | prefill MoE / attention, the unembed GEMV | `ik_split_pieces`, as `attention_flash.hip` and `moe.hip` do; speed phase only |

### 1.4 Core and plumbing (lead)

- Done in the skeleton commit: modules `inkattn.rs` / `inkmix.rs` / `inkmoe.rs` (empty tables, `launch` stubs), ABI headers
  `kernels/hip/ik_hip_abi_ink{attn,mix,moe}.h`, the chaining in `kernels::all()` / `full_table()`, the dispatch in
  `device.rs` (`InkArgs`, fall-through arm of `launch`, so trace / status reporting / span touch stay shared), host tests
  (`names_unique_and_ops_covered`, `every_launcher_is_declared_in_the_abi_header`, `intarsia-models/tests/inkling2.rs`).
  `build.rs` globs `kernels/hip/*.hip`: a new file needs no edit.
- No core change is needed for G1 / G2: the backend-name checks in `model/inkling2.rs` (`ink_rows_exact`, the fused attention
  rule, `verify_p12`, `moe_repack_wanted`), `quant/p12.rs`, `screen.rs` all leave an unlisted backend on the plain path
  (pair attention, bf16 weights, plain MoE layout, host argmax). So CUDA and Metal behaviour is untouched by construction.
- Guards (all in core, backend-independent): TP `model.rs:1838`, MTP / non-DSpark draft `:1842`, prefix store
  `prefix_store.rs:321`, packing `model.rs:3593`. Batch: `model/batch.rs:214-219` needs row-table kernels of `ConvMix` and the
  attention ops and errors without them; no lane registers `.row_table()` kernels, so `--max-batch > 1` is refused.
  Graphs: `HipBackend::supports_graphs` is true and graph decode is not refused by core; the first runs use `graphs: false`
  (`--no-graphs`), graph decode is speed item S1 and needs position-independent launchers (below).
- Recipe: `[tuning.hip]` in `recipes/inkling2.toml` only after a measured chunk sweep (builtin 256 until then).

### 1.5 Checked claims of the Neon review (`.worktrees/neon/hip-port/answer.md`)

| Claim | Verdict |
|---|---|
| ABI test reads only two headers | true; the skeleton adds the three lane headers |
| "add every new .hip file to build.rs" | false: `build.rs` globs the directory |
| `supports_row_tables()` true on HIP makes batched serve fail with `NoKernel` | true, and that is the intended refusal for tonight |
| Ring plumbing missing in `state.rs` for HIP | false: rings are core-owned slots, the ops take them as in-place tensors; nothing backend-side |
| `moe_down_g2f` needs I <= 512; Inkling's I = 1152 | not checked; moot for the correctness form (port of `ink_moe.cu`), relevant to the speed form |
| Byte floor 5.49 GB / token | matches port spec section g |
| Rings index with `% cap` (1040 is no power of two) | true (`INK_RING_MARGIN = 16`); lanes must keep `%` |

## 2. Work packages (separate files, no shared edits)

Each package owns one Rust module, one ABI header and its `.hip` files, and edits nothing else. Base: the skeleton commit.
Admission: `intarsia kernels test --backend hip --op <op>` (cases `inkling2/cases.rs`, reference `ink_ref`), tolerances as the
CUDA twin (`BF16_TOL = Tol::new(1.5e-2, 6e-3)`, f32 `1e-3 / 2e-4`, NVFP4 situ `SITU_TOL = Tol::new(1.5e-2, 2e-3)`, MXFP8
`1.5e-2 / 1e-3`). Registration rules: launcher name starts with the op name; `.situ()` on the situ experts and never
`.act_quant()`; no `.row_table()`; `.rows_exact()` twins only where the same launcher is rows-exact; every registered kernel
must be hit by a quick case; `.when(..)` must reject what the kernel cannot take (`AttnResMix` has a Q = 48 case).
Launchers must not size grids or pick variants from the host `pos0` when `pos` is non-null (graph capture later).

| Pkg | Files | Ops | Launchers (names as CUDA) |
|---|---|---|---|
| **A** attention | `kernels/hip/ink_attn.hip`, `ik_hip_abi_inkattn.h`, `src/inkattn.rs` | 70, 71 | `ink_kv_prologue_bf16`, `rel_sink_attn_bf16` (scalar split-KV at every T; head_dim 8..128) |
| **B** mix + proj | `kernels/hip/ink_mix.hip`, `ink_proj.hip`, `ik_hip_abi_inkmix.h`, `src/inkmix.rs` | 67, 68, 69, 74, 73 | `attn_res_mix_bf16`, `sconv_res_bf16`, `conv_mix_bf16`, `loop_blend_bf16`, `ink_in_proj_bf16_gemv` (+ `_rows` twin) |
| **C** experts | `kernels/hip/ink_moe.hip`, `ik_hip_abi_inkmoe.h`, `src/inkmoe.rs` | `MoeExperts` situ, 72 | `moe_experts_situ_bf16`, `moe_experts_mxfp8_bf16` (+ `_rows` twins), W4A16 only |
| **R** router | `kernels/hip/ink_route.hip`, same header and module as C | `MoeRoute` | `moe_route` (lead does this one; it is 60 lines) |

Build and test loop for a lane (tatooine, one directory per lane, never the GPU outside a window):
- compile check, no GPU, radiance stays up (host has 61 GB RAM, about 29 GB available):
  `fleetq run --host tatooine --load cpu --expect 20 --label "<lane>: hip build" -- ssh tatooine 'cd ~/<dir> && INTARSIA_REPO=$PWD deploy/tatooine/dev.sh cargo build --release --features hip -p intarsia-cli < /dev/null'`
- `kernels test` needs the GPU: one `fleetq run --takes-down radiance` window of 45 min or less with a restore trap. To keep the
  number of radiance outages down, lanes hand their branch to the lead, who runs the admission of all packages in one window.

Speed packages, staffed only after G2 (section 4 will carry the numbers that order them): S1 graph decode on HIP;
S2 WMMA prefill experts (`moe_wmma_r7.h` + SiTU, I = 1152, MXFP8); S3 flash prefill attention with rel bias;
S4 in-projection WMMA GEMM for T > 16; S5 unembed split under the 1 ms stall, P12, screened head; S6 fused `InkAttnDecode`.

## 3. Decode byte floor on this card

Port spec section g: 5.49 GB of weights per token (attention projections 2.58, experts 1.84, controllers + router 0.15,
unembed 0.92), plus KV reads (+0.30 GB at 4k). The R9700's measured stream rate is to be re-measured in the first GPU window
(the Neon review quotes 627 GB/s; not verified here). At 627 GB/s: 8.8 ms = 114 tok/s short, 9.2 ms = 108 tok/s at 4k.

## 4. Results

(open)

## 5. Next steps

(open)
