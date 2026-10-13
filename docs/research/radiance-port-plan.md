# Radiance (libr4d) port plan for the R9700 / gfx1201 track

Date: 2026-10-13. Source: `https://codeberg.org/StillDeadcode/radiance`, tag `v1.4.0`, commit
`dc339b4cd239f7124700dd1f52213f73e8939a14` (the image `stilldeadcode/radiance:1.4.0` carries the label
`org.opencontainers.image.revision=dc339b4cd239`, so tag == deployed build). Clone: `tatooine:/data/src/radiance`
(detached at `v1.4.0`; `origin/main` HEAD is the same commit). Kernel library: `libr4d` version `0.5.0`
(`R4D_VERSION` in `libr4d/r4d.h`). Ties to ARCHITECTURE.md section 3.3 (kernel ABI, the `radiance` pack) and
`docs/research/amd-r9700-lessons.md`.

Method and confidence: read from source at the pinned commit (`libr4d/r4d.h`, `libr4d/CMakeLists.txt`, file header
comments of the kernels named below, `docs/OPS.md`, `docs/TOOLS.md`, `NOTICE`) and the `Qwen3.6-35B-A3B` GGUF header on
tatooine. **Nothing was compiled or run from the radiance tree** in this prep; every "serves X" below is a reading of
source comments and entry-point names, to be confirmed by the first port test. Items marked UNVERIFIED are open.

## 1. License and attribution

- `LICENSE`: Apache-2.0, unmodified text (202 lines). `NOTICE`: "Copyright 2026 Deadcode and the radiance contributors",
  followed by a per-contributor credit paragraph and a third-party table.
- Apache-2.0 obligations that apply to a port of libr4d files (section 4 of the license):
  1. ship a copy of the license: `LICENSES/Apache-2.0-radiance.txt` (ARCHITECTURE.md already requires a vendored license
     file per pack in `LICENSES/`);
  2. keep the `NOTICE` attribution text in a `NOTICE` file shipped with any distribution that contains the ported
     files (the "attribution notices" clause): copy the `radiance` NOTICE paragraph verbatim into our
     `kernels/radiance/NOTICE`, keeping the contributor credits that touch what we port (see table below);
  3. every ported source file keeps its existing comments/notices and gets a header: `Derived from radiance v1.4.0
     (dc339b4), libr4d/<file>, Apache-2.0; modified: <what>` (we must state changes);
  4. our own new files in the pack stay under our license; the pack's `LICENSE` note says which files are derived.
- libr4d files carry no per-file SPDX/copyright header (searched `libr4d/` and `abi/`), so the NOTICE file is the only
  attribution carrier: it must be shipped.
- Credits in NOTICE relevant to the kernels we plan to port: tcclaviger (N-rank all-reduce kernels), ggz14 (MXFP4 GEMM,
  A-tiled MXFP4 prefill GEMM, GDN kernels' overflow guards), slydlake (MXFP4 decode and prefill GEMMs, MXFP4 linear),
  keptn42 (routed GEMM decode loads and run search, MoE prefill forms, sorted-operand decode kernel, fused router launch,
  four-rank all-reduce served in place), mechkbfan (token embedding in pinned host pool), Code000 (n-gram gather).
- NOTICE third-party rows (not in libr4d, listed so nobody pulls them in by accident): llama.cpp (MIT; `vendor/llama/`,
  `core/sample/gbnf.*`, `core/sample/host_ref.*`, `core/text/`, `libquant/lq_ggml_*`), nlohmann/json (MIT),
  cpp-httplib (MIT), and `core/server/systemone.cpp` which is under the **LFM Open License v1.0 with a commercial-use
  limit**. Rule: port only from `libr4d/` and `abi/`; never copy from `core/server/`, `core/text/`, `libquant/`.

## 2. Structure of libr4d relevant to porting

- One plugin of 118 "rows" (per `docs/KERNELS.md`: 118 rows, 79 of the 89-op vocabulary, 537 recorded kernel-bench
  cases), built from the units listed in `libr4d/CMakeLists.txt` (`R4D_UNITS`). gfx12 only (wave32, gfx12 WMMA fragment
  layouts, `global_load_tr_b128`); the CMake file skips the whole library for any non-`gfx12*` target. Compile flags that
  matter: every unit `-ffp-contract=off`; `r4d_gdn_chunk_scan_k128_v128_c64_bf16` also `-mcumode`; quantised GEMM units
  need `-DR4D_GEMM_W4_GROUP=128` (and `_W8_`, `_W2A8_`, `_W4A8_`, `_W8A8_`) equal to `R4D_W4_GROUP` in `r4d_args.h`
  (packer and kernel must agree or scales are read wrong).
- **Two entry layers.** (i) `r4d.h`: plain C functions taking raw device pointers (as `long`/pointers), ints, and a stream
  (`hipStream_t` or `long`). No allocation, no sync on launch paths, HIP-graph recordable. This is the layer we port
  against. (ii) `r4d_rows.cpp` (5881 lines): the `(const RadArgs*, RadStream)` shims into radiance's op table. Do **not**
  port layer (ii); it depends on `abi/rad_*.h` and the engine's operand/weight-table model.
- Dependencies: HIP runtime only. Searched the whole of `libr4d/` and `abi/` for `hipblas`, `rccl`, `rocblas`, `torch`:
  the only hits are comments ("torch's grouped norm...", "wheel-ROCm ships no rocThrust, which torch/extension.h
  needs"); **no torch, hipBLASLt, rocBLAS or RCCL link dependency**. Internal headers: `r4d_common.h`, `r4d_args.h`,
  `r4d_gemm_args.h`, `r4d_attn_args.h`, `r4d_dt16.h`, `r4d_gdn_wmma.h`, `r4d_fp8_wmma.h`, `r4d_ar_seq.h`, `r4d_ar_hc.h`,
  `r4d_emit.h` (these travel with whichever kernel includes them).
- **Error convention differs from ours.** r4d entry points either return 0 / negative codes (`-1` shape not served,
  `-100 - hipError` launch error) or **throw `std::runtime_error`** (the GEMM entry points `r4d_gemm_*_m64`, and the
  all-reduce ones, per the `r4d.h` header comment: "the plugin shim wraps in R4D_TRY"). Our `ik_*` launchers must not
  throw: each adapter wraps the call in `try { } catch (...)` and maps to `IK_ERR_SHAPE` / `IK_ERR_LAUNCH`; also call
  `hipGetLastError()` after launch like `CHK4()` does.
- **KV cache layout differs from ours (the main design issue).** All r4d attention kernels read a **paged** cache
  `kv[num_blocks, kv_heads, block_size=16, 2*head_dim]`, K in `[0,hd)` and V in `[hd,2hd)` of each slot, addressed
  through `block_table[num_seqs, max_blocks]` (int32), `seqused_k[num_seqs]`, plus `kv_block_stride`/`kv_head_stride`
  (elements). Our ABI today passes `kc[Nkv,cap,D]`, `vc[Nkv,cap,D]` as separate tensors (ik_abi.h `ik_attn_*`). They are not
  convertible by strides alone (K and V are interleaved inside the slot). Options, in order of preference:
  - A. **Pack-owned paged KV** for the `radiance` pack: add `kv_append`/`attn_*` variants taking `kv[blocks,Nkv,16,2D]` +
    `block_table` (needs an ABI extension, so an ADR; ARCHITECTURE.md phase 1 already lists paged KV + prefix cache, so
    the plan compiler needs it anyway). With a single sequence, a trivial identity block table (`block_table[0][i]=i`,
    built once at plan time into plan-owned device memory) makes the first port work with a contiguous cache.
  - B. Patch the ported kernel to take separate K and V base pointers (changes the slot addressing in the K/V load
    helpers in `r4d_attn_prefill_h256_gqa6.hip` / `r4d_attn_decode_h256_gqa6.hip`). More invasive; loses the shared KV
    path with radiance upstream.
  - Recommendation: A. It also gives fp8 KV (`*_fp8kv` entries take `k_descale`/`v_descale`) for free later.

## 3. Kernel inventory (what we would use)

Legend: "op" = our `Op::as_str()` (`crates/intarsia-core/src/ops.rs`); "ik" = launcher in `kernels/abi/ik_abi.h`. Launch
configs are the entry point's own heuristics unless stated. All activations bf16 unless stated.

### 3.1 Dense linear algebra

| radiance file / entry | purpose | dtypes | constraints | launch | our op / ik | adapter |
|---|---|---|---|---|---|---|
| `r4d_gemm_bf16_nt_m64.hip` / `r4d_gemm_bf16_nt_m64(a,w,c,M,K,N,WV,SK,MB,stream)`; `r4d_gemm_bf16_nt_m64_max_m()` | skinny WMMA GEMM `C[M,N]=A[M,K]@W[N,K]^T`, fp32 accumulate (same layout as `ik_matmul_*`: x[M,K], w[N,K]) | A,W,C bf16 | 1<=M<=`R4D_GEMM64_MAX_M`; `K % (SK*16)==0`; `WV*SK*32<=1024` threads; `MB` in 1..4; throws `std::runtime_error` | grid `(ceil(ceil(N/16)/WV), ceil(ceil(M/16)/MB))`, block `WV*SK*32`, dyn smem `WV*SK*256*4` B | `matmul` -> `ik_matmul_bf16_gemv` (decode, M<=64) and `logits` (needs f32 out, see 3.5) | require contiguous row-major (`IK_ERR_STRIDE` otherwise); pick `(WV,SK,MB)` from a per-(M,N,K) table (radiance tunes these per row; derive ours with the kernel registry autotuner); catch exceptions; N need not be a multiple of 16 (clamped loads). Qwen3-8B K in {4096, 12288}: both fine with SK in {1,2,4}. |
| `r4d_gemm_bf16_nt_tiled.hip` / `r4d_gemm_bf16_nt_tiled(a,w,c,M,K,N,SK,tile,stream)`, `r4d_gemm_bf16_nt_tiled_arm(M,N)` | prefill WMMA GEMM, LDS-staged 8-wave block, 128x128 or 64x64 tile; **byte-identical to the m64 kernel at equal SK** | bf16 | `K % 64 == 0` (`BT_BK`), M,N>=1, `SK` divides `K/16`, `tile` in {1,2}; returns -1 on mismatch; 32-bit index bound checked | grid `(ceil(N/bt), ceil(M/bt))`, block 256 | `matmul` -> `ik_matmul_bf16` (replaces cuBLAS-style "fast" for ROCm; no hipBLASLt needed) | `tile = r4d_gemm_bf16_nt_tiled_arm(M,N)`; SK=1 unless the grid is tiny; negative return -> `IK_ERR_SHAPE`. Header numbers: M 2048,K 2560: N 10240 ~1.6 ms (128x128), ~31 TFLOP/s, "untuned point" per the file comment. |
| `r4d_model_bf16.hip` / `logits_gemm_kernel`, entry `logits_gemm` row (header comment: "the output is f32 and that is the whole reason this kernel exists") | LM-head GEMM with f32 logits | bf16 weights/activations -> f32 | M small (decode/verify) | `__global__ logits_gemm_kernel` in `r4d_model_bf16.hip`, line ~918 | `logits` -> `ik_logits_bf16_gemv` | **only reachable through the row shim** in this file (no `r4d.h` C entry for the bf16 variant); port the kernel body and write our own launcher. For prefill logits use `r4d_gemm_bf16_nt_tiled` + cast, or last-token-only. |

### 3.2 Attention

| radiance file / entry | purpose | dtypes | constraints | launch | our op / ik | adapter |
|---|---|---|---|---|---|---|
| `r4d_attn_paged_h128_gqa4.hip` / `r4d_attn_decode_h128_gqa4_bf16kv(const R4DArgs*, stream)`, `..._fp8kv`, `r4d_attn_decode_h128_gqa4_scratch_bytes(a)`, `r4d_attn_h128_gqa4_have_prefill()` (returns 0) | paged split-KV decode, head_dim 128, **4 q heads per KV head = exactly Qwen3-8B (32 q / 8 kv, hd 128)**; sliding-window bit available (`R4DArgs.window`, 0 = none) | q bf16, kv bf16 or fp8 e4m3 (+`k_descale`/`v_descale`), out bf16 | `q_len*gqa <= 64` (`B_MAX_DECODE_ROWS`); block_size 16; **no prefill instantiation** (file header says decode only on purpose) | grid `(splits, kv_heads, num_seqs)`; `splits=0` lets the split law pick; scratch = f16 partials, size from `*_scratch_bytes` | `attn_decode` -> `ik_attn_decode_bf16_splitkv` (exactly the scratch-taking variant) | fill `R4DArgs` (q,kv,block_table,seqused_k,out,scratch,num_seqs=1,q_len,q_heads=32,kv_heads=8,head_dim=128,block_size=16,max_blocks,strides,scale,splits=0,max_ctx,window=0,causal=1); `KernelImpl::scratch` = `r4d_attn_decode_h128_gqa4_scratch_bytes`; `seqused_k` is a **device** int32 holding `pos0+T`, so the plan needs a tiny per-step device buffer updated by the host (or by a one-thread kernel); needs the paged KV of section 2 option A. |
| `r4d_attn_prefill_h256_gqa6.hip` (+ `r4d_attn_decode_h256_gqa6.hip`) templates; `r4d_attn_paged_gqa8.hip` instantiates **h128 and h64 prefill at gqa 8** (`r4d_rad_attn_prefill_h128_gqa...`) | query-tiled prefill kernel, templated on head_dim and gqa | bf16/fp8 kv | tile = `NWARPS(16)/gqa` query positions per block | per `R4DArgs` | `attn_prefill` -> `ik_attn_prefill_bf16_flash` / `_splitkv` | **Port work, not a drop-in for Qwen3-8B**: add an `h128/gqa4` instantiation of the prefill template (the h128 case already exists for gqa8, so the head_dim path is exercised; only the gqa constant and the tile-per-block rule change). Until then prefill falls back to our own `native` pack attention. |
| `r4d_attn_paged_gqa8.hip` / `r4d_attn_decode_h256_gqa8` family (macro `GQ_SYM`), prefill `h256_gqa8` | the Qwen3.6-35B-A3B full-attention layers: 16 q over 2 kv, head_dim 256 (file header names this model) | as above | gqa exactly 8 | as above | `attn_prefill`, `attn_decode` (35B) | same adapter as above with `head_dim=256, gqa=8`; GGUF header confirms `head_count 16, head_count_kv 2, key_length 256`. |

### 3.3 Small bf16 ops (`r4d_model_bf16.hip`, "the ten small bf16 ops the engine issues between the GEMMs")

These exist only as `__global__` kernels plus row shims in this one 1616-line file; there is no `r4d.h` entry. Port =
copy the kernel + write a 10-line launcher each. File header: arithmetic is deliberately equal to radiance's CPU
reference `libref` (f32 accumulate, bf16 round-to-nearest-even store, **negative index = padding, writes zero /
skips store**).

| kernel | purpose / semantics | constraints | our op | adapter notes |
|---|---|---|---|---|
| `embed_lookup_kernel` | gather rows of the embedding table | negative id -> zero row; optional `mult`, `vocab_offset` (vocab-parallel) | `embedding` / `ik_embedding_bf16` | ours treats ids as valid; keep the padding rule (it is stricter, harmless). Table stride args. |
| `rmsnorm_kernel<WF32,THR>` | `y = x * rsqrt(mean(x^2)+eps) * (w + wadd)`, two passes, f32 accumulate; gain f32 or bf16 (template) | `x_ld`, `y_ld` strides; n = 256 (a QK head) or hidden | `rmsnorm` (`w_offset` == `wadd`), also **`qk_norm`** by launching with rows = tokens*heads, n = head_dim | exact semantic match for `ik_rmsnorm_bf16` (`scale = w + w_offset`). Choose `WF32` from the weight dtype (Qwen3-8B norm weights are bf16). `rmsnorm_add` (ours fused: `sum=x+res`, `out=rmsnorm(sum)*w`) is **not** a radiance kernel; write it on top of this one (radiance composes `add` + `rmsnorm`). |
| `binary_kernel` | elementwise add / mul | | `add` / `ik_add_bf16` | trivial |
| `sigmoid_kernel` | sigmoid | | part of `sigmoid_mul` | radiance has no fused `a*sigmoid(g)` here; the Qwen3.5-style gate is folded into `r4d_qk_norm_rope_gate.hip` / the GDN gated norm. Ours stays a separate small kernel. |
| `silu_mul_kernel` | `y = silu(gu[:, :n]) * gu[:, n:2n]` (kSiluPer=4 elements/thread, split over grid.y) | **takes the fused gate_up row**, gate = first half | `silu_glu` / `ik_silu_glu_bf16` (ours: separate `gate[T,I]`, `up[T,I]`) | change the kernel to take two row pointers (3-line edit), or fuse gate/up weights at load. Fusing at load halves GEMM launches (radiance does this in `rad-convert`) and is the better long-term choice; needs a recipe/loader change, so list as phase 2. |
| `rope_kernel` | in-place rotary over a fused `[n_head + 2*n_kv]` head row (V heads untouched); NeoX (`i`,`i+half`) or GPT-J; partial `rot`; mrope/imrope/axial modes via `plane`-strided positions; YaRN | positions are an **int32 array `pos[M]`**, not a scalar; `rot` may be < head_dim | `rope` / `ik_rope_bf16` (scalar `pos0`, separate `out`, `theta`, `rotary_dim`) | adapter writes `pos0..pos0+T-1` into plan scratch (tiny `iota` kernel or host-built buffer) and aliases out==x (copy first if out != x). Qwen3 uses NeoX with `rotary_dim = head_dim = 128`, theta 1e6 (Qwen3-8B `rope_theta` in `config.json`). |
| `kv_store_kernel` / `kv_store_fp8_kernel` | scatter K,V into the paged cache via `slot_mapping` (negative = padding), bf16 or e4m3 | block 16 | `kv_append` / `ik_kv_append_bf16` | consistent with option A; `slot_mapping[t] = pos0+t` under an identity block table. |
| `fill_kernel` | memset-like fill | | glue | not needed |
| `gather_rows_*`, `scatter_rows_u8_*` | row gather/scatter (MoE/n-gram) | | | later (MoE) |

### 3.4 Sampling

| entry | purpose | our op | note |
|---|---|---|---|
| `r4d_sample_chain_f32(logits, query, u, out, ...)` (`r4d_sample_chain_f32.hip`): one workgroup per position: temperature, softmax, top_k, top_p, min_p, renormalise, draw, plus greedy-draft rejection-sampling outputs | `sample` | vLLM order; temperature 0 keeps lowest-id tie-break. Our op takes explicit `u[R]` f32, which maps onto its `u` argument. |
| `r4d_rad_sample_{temp,mask,penalties,dry,argmax,topk,topp,minp,typical,xtc,...}` (`r4d_sample_stages_f32.hip`, `RadSampleParams` rows) | decomposed 11-op chain | `argmax`, `sample` | `argmax`/`ik_argmax_f32` needs only the argmax stage; tie rule (lowest id) equals ours. Entry layer is the row shim (UNVERIFIED whether a plain-pointer variant exists; port kernel body). |

### 3.5 Gated DeltaNet (Qwen3.6-35B-A3B linear-attention layers)

All GDN kernels are fixed at `K=V=128`, chunk 64 (`k128_v128_c64`); a mismatched geometry returns -1 instead of
running. Qwen3.6-35B-A3B GGUF: `ssm.state_size 128`, `ssm.group_count 16` (k heads), `ssm.time_step_rank 32` (v heads),
`ssm.inner_size 4096`, `conv_kernel 4`: matches.

| file / entry | purpose | dtypes | our op / ik | adapter |
|---|---|---|---|---|
| `r4d_gdn_conv_w4_h128_bf16.hip` / `r4d_gdn_conv_prep_w4_h128_bf16(...)` (prefill) and `r4d_gdn_conv_update_w4_h128_bf16(...)` (decode, speculative window) | depthwise causal conv (width 4, silu) with state cache, **fused** q/k/v split + l2norm of q,k + gate `g = -exp(A_log)*softplus(a+dt_bias)` + cumsum + `beta=sigmoid(b)` | bf16 in, state cache bf16/f32 per call | `gdn_conv` (+ part of `gdn_delta_rule`) | our two ops (`ik_gdn_conv_bf16` writes `y=silu(conv)`, `ik_gdn_delta_rule_bf16` reads qkv/a/b/A_log/dt_bias and does gating + l2norm itself) are **split differently** from radiance's fusion. Two ways: (1) port kernels as-is and expose a **block-scope op** `gdn_layer_decode_fused` (ARCHITECTURE.md explicitly wants block-scope fused ops for the radiance pack); (2) implement our fine-grained ops on top of the pieces. Take (1) for speed and keep our ops for the cpu/native packs. Requires `cache_idx`/`cu_seqlens` style slot indices (can be constant for 1 sequence). |
| `r4d_gdn_recurrent_update_k128_v128_bf16_fp32state(...)` | decode recurrent delta-rule update: gating + qk l2norm + state update + output, against a paged fp32 state cache `[slot, H, V, K]` | bf16 q/k/v, **fp32 state** | `gdn_delta_rule` (decode, T small) | our state is `[Hv,dv,dk]` f32 per sequence: identical to one slot of theirs (`state_slot_stride`, `state_head_stride`). Also fuses the gated RMSNorm (`z_gate`, `norm_weight`) as an epilogue. Header: "state fp16" flag exists (`state_fp16`), keep 0. |
| `r4d_gdn_chunk_scan_k128_v128_c64_bf16` (+`_sidx`), `r4d_gdn_kkt_solve_k128_c64_bf16` | chunked prefill: `kkt_solve` computes `A=(I+tril(beta K K^T e^{g_i-g_j}))^-1`, chunk scan does WY recompute + state scan + output in one kernel; `-mcumode`; `cu_seqlens` ragged batches | bf16 (A bf16, g/beta fp32, h0/ht fp32) | `gdn_delta_rule` (prefill T>1) | cu_seqlens = `[0,T]`; scratch for `A` ([T,H,64] bf16) goes through `ik_scratch`. |
| `r4d_gdn_gated_rmsnorm_h128_bf16(...)` | gated RMSNorm (FLA `RMSNormGated`), `act` 0 silu / 1 swish | bf16 | part of GDN block | folded into the decode kernel epilogue above; standalone for prefill. |

### 3.6 Mixture of experts (Qwen3.6-35B-A3B: 40 layers, 256 experts, top-8, expert ff 512, hidden 2048)

`r4d_moe.hip` (4233 lines; header: "Qwen3.6-35B-A3B: 256 experts, 8 per token, 512 intermediate", grouped GEMM is
bandwidth-bound: 3 MB per expert, 24 MB per layer at decode). It provides router top-k (+scatter), sort/scatter, gather
(`moe_gather`), and a **grouped block-scaled FP8 GEMM** (`moe_gemm_fp8a8`) and a **W4A8 GEMM** (`moe_gemm_w4a8`,
asymmetric int4 group 128/64). There is **no MXFP4 routed-expert GEMM** (0 mentions of mxfp4 in `r4d_moe.hip`).
MXFP4 appears only in the dense GEMMs below and in the packer (`r4d_layout.cpp`, 12 mentions).

| file / entry | purpose | dtypes | our op | note |
|---|---|---|---|---|
| `r4d_moe.hip` router top-k kernels (`router_topk_bf16`, `router_topk_scatter_bf16` rows) | softmax/top-k over 256 logits, renormalised | bf16 logits -> f32 weights, i32 ids | `moe_topk` / `ik_moe_topk_bf16` | semantics need verification against ours (bf16-rounded softmax, tie-break lowest index, `norm`). UNVERIFIED. |
| `r4d_moe.hip` grouped GEMM `moe_gemm_q` rows | experts gate_up/down | fp8 e4m3 block-scaled (128x128) or int4/int8 group 64/128 | `moe_experts` | not our NVFP4/MXFP4 layout; see port order (b). |
| `r4d_gemm_mxfp4a8_nt_m64.hip` (`r4d_gemm_mxfp4a8_nt_m64`, `_max_m`, `_group`), `r4d_gemm_mxfp4a8_decode.hip`, `r4d_gemm_mxfp4a8_tiled.hip` (`r4d_gemm_mxfp4a8_tiled`, `r4d_mxfp4a8_tile_a`) | **dense** OCP-MXFP4 weight x fp8 activation GEMM (e2m1 + E8M0 exponent per 32 K, folded against a per-row reference exponent `wref` so the inner loop has no rescale) | W: fp4 + e8m0; A: fp8 e4m3 + f32 scale per row or per (row,128 K) | `matmul_mxfp4` (new op; `IK_QUANT_MXFP4` is reserved in `ik_abi.h`) | needs `r4d_quant_act_fp8` first (activation quantiser launch) and the `wref` plane built by the packer. W4A8, so numerics differ from our W4A16 oracle: register as `Exactness::Fast`. |
| `r4d_gemm_fp8a16_nt_m1.hip` / `r4d_gemm_fp8a16_nt_m1`, `r4d_gemm_fp8a8.hip` `_nt_m16` / `_tiled`, `r4d_quant_act_fp8.hip` | block-scaled FP8 (128x128 weight blocks) GEMMs for M=1 / M<=16 / any M | fp8 e4m3 + bf16 scale | `matmul_fp8` / `ik_matmul_fp8_*` | our `matmul_fp8` uses 2 scalar scales, not 128x128 blocks; a second variant name `matmul_fp8_block` is needed (`IK_QUANT_FP8_BLOCK` is reserved). |

### 3.7 Collectives (TP=2)

All in `libr4d/r4d_ar_*.hip` (+ `r4d_ar_entry.hip`, 1561 lines, the `r4d.h` shims). Pure HIP, P2P over PCIe, no RCCL.
Design (from `r4d_ar_oneshot_2rank_exact.hip` header): **push model**, one-shot (each rank writes its whole input into the
peer's scratch, then reduces local + received), exactly 2 ranks, multi-block with per-block flags (no grid barrier),
HIP-graph safe via a device-resident per-block sequence counter, scratch double-buffered by launch parity,
**fine-grained (uncached) memory** (`hipDeviceMallocFinegrained`) for scratch+flags so SCOPE_SYS release/acquire skip the
L2 writeback.

| entry | purpose | dtypes | constraints | our op |
|---|---|---|---|---|
| `r4d_ar_oneshot_2rank_exact(peer_scratch, my_scratch, peer_flags, my_flags, seq_ctrs, slot_stride16, inp, out, n_elem, dtype, stream, nblocks, nthreads, drain, acq)` | exact one-shot all-reduce, fp32 accumulate | bf16/fp16/fp32 (dtype 0/1/2) | 2 ranks; scratch sized to full message, 2 parity slots; `r4d_ar_max_blocks()` | new `all_reduce` op (not in `ops.rs` yet) |
| `r4d_ar_oneshot_2rank_wht6(...)` | same, wire is Walsh-Hadamard rotated 6-bit/elem groups of 64, lossy (+ gate/ctr args) | bf16/fp16 | accuracy-gated opt-in only | `all_reduce` variant, `Exactness::Fast` |
| `r4d_ar_ln_had_quant_i8`, `r4d_ar_gather_hc_write`, `ar_rmsnorm_quant_fp8` (from `KERNELS.md` uncovered list) | all-reduce fused with RMSNorm + quantise (the "norm+quant" fusion) | | | block-scope op `all_reduce_rmsnorm_fp8` (ARCHITECTURE.md names it) |
| `r4d_ar_oneshot_Nrank_exact` / `r4d_ar_twoshot_Nrank_exact` / `_ti8` / `r4d_ar_gather_Nrank` / `r4d_ar_gather_2rank` | 3/4/8-rank one-shot, two-shot (reduce-scatter+all-gather), tiered-int8 wire, all-gather | | `R4D_AR_TWOSHOT_WIDE_MAX_ELEMS = 20971520` | out of scope (we have 2 cards) |
| host helpers `r4d_ar_ipc_*`, `r4d_ar_host_alloc`, `r4d_p2p_copy2d*`, `r4d_p2p_fence`, `r4d_ar_ipc_enable_peer` | IPC handle exchange for **one process per rank**, or pinned fine-grained host buffers visible to both cards | | `R4D_AR_HANDLE_BYTES` 64 | See adapter. |

Adapter for the collectives: the buffers are *not* caller-allocated per launch. Our "no allocation inside, scratch passed in"
rule maps to a **comm context** created once at engine start (`ik_comm_init(rank, world, ...)`): allocate fine-grained
scratch + flags + seq counters per rank (`slot_stride16` x 2 slots x message bytes), exchange pointers. radiance uses IPC
handles because it runs one process per rank. IntarsiaML's Rust host can drive both cards from **one process**: after
`hipDeviceEnablePeerAccess` in both directions (measured OK on tatooine, see the prep results) a device pointer from card 1 is
directly usable from card 0's kernels, so no IPC handle exchange is needed. UNVERIFIED that the r4d kernels' fence/flag
discipline and fine-grained alloc behave identically when both pointers are plain peer-mapped pointers instead of
IPC-opened ones; this must be tested early (two-rank bitwise test vs. a host-summed reference over 10^6 iterations
with the seq counters in lockstep, mirroring `r4d_selftest`). `nblocks`/`nthreads`/`drain`/`acq` are tuning knobs with
defaults in the radiance caller; start from the values in `core/` / the plugin row (UNVERIFIED which; read
`r4d_rows.cpp` all-reduce row and `r4d_ar_entry.hip` when porting).

## 4. Formats radiance itself loads

`docs/TOOLS.md`: `rad-convert` takes "a .gguf, a .safetensors, an index.json, or a checkpoint directory" and writes a
`.rad` container, quantised by a recipe; the **engine serves `.rad` files** (production: `/models/qwen3.8-27b-mxfp4.rad`,
from `/v1/models`). `.rad` is an mmap-able container frozen in `abi/rad_format.h` (reader `core/format/radfile.cpp`);
experts in a layer must be same shape/format so addressing is `base + id*stride`. Weight formats imported without
re-quantising ("trivial"): f32/f16/bf16 (GGUF too), block-scaled FP8, Quark's OCP-MXFP4 (`core/format/checkpoint.cpp`
lines ~548-924: two codes/byte, E8M0 exponent plane). GGUF block types are **dequantised** on import (`checkpoint.h`: "a
GGUF block type dequantised") and then quantised to the recipe's encoding. The GGUF type table in `core/text/gguf.cpp`
knows type 39 `mxfp4`. Therefore:

- **Our target MoE GGUF** `tatooine:/data/models/gguf/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-MXFP4_MOE.gguf` (Unsloth;
  `general.file_type 38`, arch `qwen35moe`, 40 layers, hidden 2048, 256 experts / 8 used, expert ff 512, shared-expert
  ff 512, 16 q / 2 kv heads, head_dim 256, rope dim 64, `full_attention_interval 4`). Tensor types counted from the
  header: F32 361, Q8_0 252, MXFP4 78, Q5_K 38, Q6_K 4. So routed experts are MXFP4 (78 tensors = gate/up/down of 26
  layers... UNVERIFIED which layers; the others are Q5_K/Q6_K), attention/dense/shared experts mostly Q8_0.
- radiance can ingest this file only via `rad-convert` (dequantise to bf16, re-quantise to the recipe: its MoE kernels
  serve fp8 block-scaled or int4 `w4`, not MXFP4). That is a **lossy double quantisation** and not what we want for an
  accuracy comparison against the GGUF. UNVERIFIED whether `rad-convert` keeps MXFP4 experts bit-exact (the
  checkpoint code has an MXFP4 passthrough for Quark safetensors, not obviously for GGUF type 39).
- Our own path: read GGUF directly in `intarsia-models` (tensors are already MXFP4 blocks of 32 e2m1 + 1 E8M0 byte, 17
  bytes/block) and write a **new MXFP4 MoE grouped GEMM** for gfx1201 using the dequant logic of
  `r4d_gemm_mxfp4a8_*` as the starting point. Note the dense MXFP4 kernels want a repacked layout
  (`r4d_layout.cpp` packer: WMMA fragment order + `wref`), so the loader needs a "repack at load" step for the
  `radiance` pack (our `Exactness`/layout registry already allows pack-specific weight layouts per ARCHITECTURE.md).

## 5. The `ik_*` adapters, concretely

Rules (ik_abi.h): raw tensors in, `int` status out, explicit stream, no allocation, scratch via `const ik_scratch*`
immediately before `stream`, zero-element tensors are a no-op, row-major contiguous only (else `IK_ERR_STRIDE`).

- Build: one static library `libintarsia_radiance.a` per pack. Sources: the ported `.hip` files (kept byte-for-byte except
  for the named edits), `r4d_*.h` they include, plus our `ik_radiance.cpp` with the `ik_*` launchers. Compile with
  `hipcc --offload-arch=gfx1201 -O3 -ffp-contract=off` (+`-mcumode` for the GDN scan, +`-DR4D_GEMM_W4_GROUP=128` etc.).
  Nothing in `libr4d` needs CMake's `rad_add_plugin` or `r4d_rows.cpp` for the `r4d.h` layer, but several
  units also contain the row-shim half and so `#include` `abi/rad_*.h` (e.g. `r4d_attn_paged_h256_gqa6.hip`,
  `r4d_attn_paged_gqa8.hip`, `r4d_model_bf16.hip`; `r4d_attn_paged_h128_gqa4.hip` includes the h256 ones). In the ported
  copy, remove the shim half (the `r4d_rad_*` entry points) or vendor the small `abi/rad_types.h` subset it needs.
  UNVERIFIED: no unit was compiled outside radiance's CMake in this prep.
- Rust side: `crates/intarsia-backend-hip` (new, mirrors `intarsia-backend-cuda/src/ffi.rs`), `extern "C"` bindings to
  the `ik_*` symbols, `hipStream_t` as `*mut c_void`. No `cudarc` equivalent is needed: use `hipMalloc`, `hipMemcpy`,
  streams, `hipMemGetInfo` through a tiny `hip_rt` FFI (the smoke test in `deploy/tatooine/smoke/smoke.hip` shows the
  calls). `HIP_VISIBLE_DEVICES` and `GPU_MAX_HW_QUEUES=1` are runtime env (lessons doc item 1).
- Scratch declarations (`KernelImpl::scratch`: `ShapeInfo -> bytes`): decode attention = `r4d_attn_decode_*_scratch_bytes`;
  fp8 GEMM split-K = `r4d_gemm_fp8a8_nt_m16_scratch`; GDN `A` matrix = `T*H*64*2`; MoE sort tables = per
  `r4d_moe.hip`; all-reduce is **not** scratch (context-owned, see 3.7).
- Per-step device metadata (`seqused_k`, `slot_mapping`, `pos[]`, `cu_seqlens`, `block_table`): tiny int32 buffers owned
  by the plan, refreshed by the host before the step (a `hipMemcpyAsync` H2D on the same stream) or by a 1-thread
  kernel. This keeps launchers allocation-free and graph-capturable.

## 6. Ranked port order

### (a) Qwen3-8B BF16 decode/prefill (ready now: safetensors in `/data/models/intarsia/Qwen3-8B`)

Model facts (to confirm against `config.json` in the snapshot): hidden 4096, 36 layers, 32 q / 8 kv heads, head_dim 128
(gqa 4), intermediate 12288, vocab 151936, QK-norm, rope theta 1e6.

1. `matmul` via `r4d_gemm_bf16_nt_m64` (decode, M<=64) and `r4d_gemm_bf16_nt_tiled` (prefill). Highest weight: every
   projection, the lm_head, and bandwidth-bound; drop-in layout; no dependency. Gate: matches our cpu-ref within bf16
   tolerance; decode GEMV reaches a large fraction of the **627 GB/s measured read-only bandwidth** per card (see
   `docs/results/2026-10-13-r9700-prep.md`).
2. Small ops from `r4d_model_bf16.hip`: `embedding`, `rmsnorm` (+`qk_norm`), `add`, `rope`, `silu_glu`, `kv_append`
   (with the identity block table), `argmax`/`sample` (3.4). Needed to run a token at all; trivial kernels.
3. `attn_decode` via `r4d_attn_decode_h128_gqa4_bf16kv` (+scratch). Requires the paged KV layout decision (section 2).
   Biggest single speed lever after the GEMMs (split-KV, tuned for RDNA4).
4. `attn_prefill`: instantiate h128/gqa4 prefill (new work from the gqa8 file's h128 pattern); until then use the `native`
   pack's attention.
5. Fusions after the baseline passes golden: gate/up weight fusion + `silu_mul`, `rmsnorm_add`, `qk_norm+rope` (see
   `r4d_qk_norm_rope_gate.hip`, whose "gate" half is Qwen3.5-specific), logits f32 GEMM.
6. fp8 KV (`*_fp8kv`) as an `Exactness::Fast` option.

### (b) Qwen3.6-35B-A3B (MXFP4 GGUF)

1. `attn` h256/gqa8 (decode + prefill) from `r4d_attn_paged_gqa8.hip`: exact geometry match.
2. GDN set (3.5): `conv_prep/update`, `recurrent_update`, `kkt_solve` + `chunk_scan`, gated rmsnorm. 30 of the 40 layers
   are GDN (`full_attention_interval 4`), so this is the dominant non-MoE cost; ship as block-scope `gdn_layer_*` ops.
3. Dense projections (attention q/k/v/o, shared expert, lm_head; GGUF has them Q8_0): needs either dequant-to-bf16 at load
   (fits: dense part of a 35B-A3B is small) with the BF16 GEMMs of (a), or Q8_0 GEMM kernels (not in libr4d: it has w8a8
   `int8 group 128`, a different format; repack Q8_0 -> that is possible but lossy only if scale grouping differs; Q8_0
   group is 32).
4. Routed experts: **write a MXFP4 grouped GEMM** (no radiance equivalent). Use `r4d_moe.hip`'s router/sort/scatter/gather
   unchanged (they are format-independent) and its grouped-GEMM scheduling (run-aligned prefill, (tile,run) decode) with the
   `mxfp4a8` dequant inner loop. Risk: high; budget as the largest task. Interim: fp8 block-scaled experts via
   `rad-convert`-style requant (lossy, but kernels exist) to get end-to-end numbers.
5. `moe_topk`/`moe_combine`/shared-expert gate: port router rows, write `moe_combine` (3-line kernel).
6. Loader: GGUF reader + MXFP4 repack for the radiance pack.

Baselines to beat (lessons doc): MoE on vLLM stack 141 tok/s decode single card; llama.cpp Vulkan Q4_K_XL 112 tok/s on
R9700 (35B-A3B).

### (c) TP=2 one-shot all-reduce

1. Comm context + `r4d_ar_oneshot_2rank_exact` (bf16) behind a new `all_reduce` op and `ik_comm_*` init; single-process
   dual-device with peer access (see 3.7 adapter), correctness harness first.
2. Fused `all_reduce + rmsnorm (+fp8 quant)`: `ar_rmsnorm_quant_fp8` family, as block-scope op; lessons doc estimates
   128 all-reduce calls/step on the 27B, so latency dominates (measured peer `hipMemcpyPeer` of 64 KiB is 11 us, 28 GB/s
   peak; one-shot at decode message sizes is latency-bound, which is why radiance wrote it).
3. `wht6` lossy wire only behind an accuracy gate.
4. Not needed for Qwen3-8B BF16 (16.4 GB fits one 32 GB card); first real use is the 27B-class model.

## 7. Risks and open questions

1. KV layout (section 2): an ABI extension decision is needed before kernel 3 of (a). Owner: architecture ADR.
2. Pointer provenance for collectives in single-process mode (3.7), UNVERIFIED.
3. `rad-convert` MXFP4 passthrough for GGUF, UNVERIFIED (run `rad-convert --help` and a dry run in the 1.4.0 image on a CPU
   fleetq slot before deciding whether the radiance engine itself can serve as a speed/accuracy reference for the 35B).
4. Tuning tables: radiance tunes `(WV,SK,MB)` / tile per shape through its tune cache (`core/format/tunecache.cpp`);
   a ported GEMM without the table can be several times slower. Plan a sweep in `intarsia-bench` on gfx1201.
5. Exceptions escaping `extern "C"` (3. above) - mandatory try/catch in every adapter.
6. License hygiene: only `libr4d/` and `abi/`; never `core/server/systemone.cpp` (LFM license) or llama.cpp-derived
   files.
