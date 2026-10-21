# macOS / Metal backend: scoping

Status: scoping only. No backend code exists or was written. Facts below were read from this repo at `6418ef1`
(branch `arch/plan`; line numbers re-checked at `bf46bad`, which only touched the cpu backend's free-memory code) and from the sources cited in section 3. Estimates are mine and are marked as such.

Host used for local facts: MacBook Pro Mac17,6, Apple M5 Max, 128 GB unified memory (`sysctl hw.memsize` =
137438953472), 40 GPU cores (`system_profiler SPDisplaysDataType`), Metal 4, macOS 27.0, rustc 1.99.

## 0. Summary and recommendation

* Build a native Metal backend crate, `intarsia-backend-metal` (option a), in Rust with `objc2-metal`, with `.metal`
  shaders compiled AOT to a `.metallib` by `build.rs` and embedded with `include_bytes!`. Borrow kernel structure and
  dequant tricks from `ggml-metal` (MIT) where useful (option c as a reference, not as a port of the whole file), and
  offer MPS/`mpp::tensor_ops` GEMM as an optional extra registry entry for prefill (option d as a later add-on).
  Use MLX only as a benchmark oracle, not as a backend (option b).
* Why: the existing seams (`Backend` trait, `KernelImpl` registry with admission, graph capture/replay, `wrap_host`)
  already fit a unified-memory GPU with an explicit launch model. MLX is a lazy array graph with its own allocator and
  does not fit "named kernel launch against `TensorRef` buffers", and it would make the admission and accuracy
  story impossible per kernel.
* Effort (estimates): phase A 50 h, B 60 h, C 70 h, D 40 h; about 220 h total, usable results after A (about 50 h) and B (about 110 h).
* Performance target: at least match llama.cpp Metal on Qwen3-8B BF16 and 35B-A3B NVFP4, and land within 10 percent of
  MLX. On this chip the BF16 8B byte floor is about 37.5 tok/s (section 3.3).
* ADR-0009: write it later, not now. Section 6.

## 1. Approach options

Seams that every option is judged against:

* `Backend` trait, `crates/intarsia-core/src/backend.rs:165`: `alloc`, `upload`, `download`, `copy`, `wrap_host`
  (`:209`), `kernels` (`:215`), `launch(kernel_name, &OpCall, stream)`, `synchronize`, `supports_graphs` (`:226`) /
  `graph_capture` (`:235`) / `graph_replay` (`:240`), `fingerprint`.
* `Buffer` / `TensorRef` carry a byte `offset` (`backend.rs:115`), so a Metal buffer plus offset is natural.
* Kernel registry: `KernelImpl` at `crates/intarsia-core/src/registry.rs:63` (op, name, archs, acts, weights, shape predicate
  `when`, `exact`, `tol`, `priority`, `scratch`); `Registry::select` at `:292` skips impls that admission rejected
  (and untested ones when `require_admitted`), and returns `Error::NoKernel` otherwise. Fused ops fall back to
  the unfused ops this way (`ops.rs` header, the four fused ops are "optional").
* Admission: `crates/intarsia-core/src/admission.rs` (`sweep` at `:78`, `run_admission` at `:594`, `judge` at `:1028`) compares each
  candidate with the `cpu` reference on the model's shapes. The cpu backend already runs on this Mac and is flagged
  `unified: true` (`crates/intarsia-backend-cpu/src/lib.rs:103-120`).
* C ABI `kernels/abi/ik_abi.h`: `ik_tensor`, `ik_scratch`, `ik_<op>_<variant>` launchers taking `void* stream`
  (`:10`, `:96-107`); the device-side position tensor for graph replay is at `:109`. The CUDA and HIP crates link
  static libraries exporting the same `ik_*` symbols, which is why the features are mutually exclusive
  (`crates/intarsia-cli/Cargo.toml:21-25`, `crates/intarsia-cli/src/lib.rs:7` `compile_error!`). Metal shaders are not C-callable, so a Metal crate cannot reuse the `ik_*`
  launcher layer as-is; it should keep the `ik_tensor` struct layout as the shader argument struct and do the launch
  from Rust. That also avoids the exclusive-feature problem because no `ik_*` symbols are exported.

### (a) Native Metal backend crate (`objc2-metal`, AOT metallib). Rank 1, recommended

* Backend trait fit: direct. One `MTLDevice`, one `MTLCommandQueue` per `StreamId`, `launch` encodes a compute
  dispatch for the named pipeline state. `alloc` is `newBufferWithLength` with `MTLResourceStorageModeShared`.
  `kernels()` lists `KernelImpl`s with `backend = "metal"`, `arch` = the GPU family string (for example
  `apple9` / `apple10`, read from `supportsFamily`; M5 is the first with tensor hardware). `fingerprint` should include the
  chip name, macOS version and metallib hash, as the HIP backend does for its toolchain.
* Crate choice: `objc2-metal` (maintained, generated from the SDK headers, tracks Metal 4) over `metal-rs`
  (archived/deprecated upstream; I did not re-verify its current state in this pass, so treat as likely). Either is
  fine behind the backend's own `unsafe` boundary.
* Replay analogue: Metal command buffers are one-shot, so `graph_capture` cannot record a stream the way
  `cudaStreamBeginCapture` does (`crates/intarsia-backend-cuda/src/device.rs:366`). Two routes, both fitting the
  trait: (1) re-encode the whole decode step each token into one command buffer (cost: encoding CPU time; no new
  machinery; `supports_graphs = false`, and the engine falls back to eager, see `crates/intarsia-core/src/model/graph_step.rs:63` `usable`);
  (2) `MTLIndirectCommandBuffer` with compute commands (Metal 3, Apple3+ per Apple's feature-set table; Metal 4 adds
  `executeCommands` on the compute encoder), recorded once inside `graph_capture` by having `launch` write into an ICB
  instead of the encoder. ICB needs all arguments as buffers (no `setBytes`), and buffer addresses fixed, which the
  existing capture contract already demands (`backend.rs:231-236`). The device-side position tensor (`ik_abi.h:109`,
  `graph_step.rs:5`) already removes the host-dependent state. Plan: ship route 1 first, add route 2 in phase D and
  measure the gap before committing.
* Unified memory: `wrap_host` is implementable (details in section 1.5). Weights loaded from mmap'd safetensors
  with no copy (`crates/intarsia-models/src/safetensors.rs:7-14`; consumed at `crates/intarsia-core/src/weights.rs:48` and `:118`).
* Quant: bf16 is native in MSL (`bfloat`, Metal 3.1+). Apple GPUs have no fp8 or fp4 arithmetic that I could verify
  (the researcher found no source for native fp8/fp4 compute on any Apple GPU; a forum post reads Metal 4.1 headers as
  adding FP8/FP4 tensor types but without official docs, so unverified). Plan dequant-in-kernel: e4m3 and e2m1
  decode via bit manipulation or a 16/256-entry threadgroup LUT, scales applied in f32, accumulate in f32. This is
  weight-only (W8A16 / W4A16), same as the ops already specified (`docs/quant-moe.md:13`, `W = e2m1(code) * e4m3(scale) * scale2`).
* Accuracy gate: unchanged tooling. Each Metal kernel gets a `KernelImpl` with a tolerance against the cpu reference
  and is admitted by `intarsia kernels test --backend metal`. Model-level gates run the same goldens. Metal is
  deterministic for a fixed dispatch geometry in practice, so `Exactness::Deterministic` is the likely class (to be
  confirmed per kernel in admission).
* Cons: all kernels are new code; best achievable speed depends on our own GEMV tuning.

### (b) Wrap MLX's C API (`mlx-c`). Rank 4, use as a benchmark oracle only

* Fit: poor. MLX is a lazy array graph with its own allocator and stream/eval model. Our trait wants explicit buffers
  and one named kernel per `launch`. We could wrap each op as an MLX array expression and `eval`, which gives up
  fused-kernel control, per-kernel admission, buffer planning from measured free memory, and graph replay ownership.
* Replay: MLX has `mx.compile` / graph caching but no handle we own; `graph_capture` would map poorly.
* Unified memory: MLX allocates its own Metal buffers; importing our mmap weights goes through `mlx_array_new_data`
  (copies) or custom allocator hooks that I did not verify. Hard to keep the "no copy" property.
* Quant: MLX has its own quantized_matmul layouts (affine group quant, and fp4/mx modes in recent versions: not verified
  here). Our NVFP4 layout (`weight_scale` e4m3 per 16, `weight_scale_2` f32) would need conversion or a custom op.
* Licence: mlx and mlx-c are MIT (researcher confirmed the repo badges, did not read the LICENSE text). Compatible
  with this repo's Apache-2.0 (`Cargo.toml` workspace licence) with attribution.
* Accuracy gate: only end-to-end; the admission tooling cannot compare a kernel it does not own.
* Upside: fastest way to a number. Keep it as a bench reference (`mlx_lm` run next to ours), not a backend.

### (c) Port `ggml-metal` kernels from llama.cpp. Rank 2, as a source of kernel designs

* Licence: llama.cpp is MIT (researcher confirmed). Copying kernels is allowed with the copyright notice kept; the repo
  has a pattern for this: `kernels/hip/NOTICE-radiance`, `kernels/README.md` mentions `packs/` with a LICENSE each.
  Add `kernels/metal/NOTICE-ggml` and cite function names.
* Fit: the shaders are pure MSL, easy to lift, but they are written around ggml's tensor struct (ne/nb strides, quant block
  structs for Q4_K and so on), not our dense row-major `ik_tensor`. Quant formats differ: ggml has no NVFP4 with our
  scale layout (MXFP4 exists), so the quant GEMVs need rewriting anyway. What carries over cleanly: simdgroup
  reductions, RMSNorm, RoPE, flash-attention-style tiling, simdgroup-matrix GEMM for prefill.
* Replay: n/a (shader level). Unified memory: n/a. Accuracy: same as (a) because the kernels end up in our backend.
* Verdict: study and adapt, do not port the dispatcher (ggml graph executor); we keep our own launch model.

### (d) MPSGraph / MPS matmul for GEMM, custom kernels for the rest. Rank 3, optional add-on

* Fit: `MPSMatrixMultiplication` operates on `MTLBuffer`s, so it can be registered as a second `Op::Matmul` impl
  (`priority` above the hand kernel for large M) next to our own GEMV. The registry's `when` predicate handles the
  M threshold (the CUDA/HIP backends already register `matmul_bf16` and `matmul_bf16_gemv` with different predicates,
  `crates/intarsia-backend-hip/src/kernels.rs:48-49`).
* MPSGraph is a graph compiler with its own allocation and cannot be driven one op at a time without per-call compile
  overhead; skip it. Plain MPS matmul is bf16/f16/f32 only, no quantized formats, so it helps BF16 prefill only.
* On M5 the better prefill path is `mpp::tensor_ops` (Metal 4 tensor ops using the GPU neural accelerators; Apple's M5 MLX
  post reports 3.5 to 4x TTFT gains from them), written into our own shader, which keeps the weights in our layout
  and works inside the registry. M4 and earlier run the same API on the normal shader cores (Rigel paper via the researcher, medium confidence).
* Decode is bandwidth bound GEMV where MPS gives nothing.

### 1.5 Cross-cutting: unified memory, weights, memory planner

* `wrap_host(mem, offset, len)` (`backend.rs:209`) is called per tensor with an offset into an mmap. Metal's
  `newBufferWithBytesNoCopy` needs a page-aligned pointer and a length that is a multiple of the page size (16 KB on
  Apple Silicon). Tensor offsets inside a safetensors file are not aligned. Plan: key a cache on the `Arc<dyn HostMemory>`
  pointer, wrap the whole file once (round length up; the mapping is page-granular), and return a `Buffer` whose
  `BufferImpl` stores (shared `MTLBuffer`, base offset). `TensorRef.offset` already stacks on top. The cpu backend
  does the equivalent with `CpuBuffer::host(mem, offset, len)` (`crates/intarsia-backend-cpu/src/lib.rs:150`).
* Open questions to settle by experiment in phase A: `maxBufferLength` for a 22 GB file (3 shards, the checkpoint
  on this Mac is 22 GB: `~/models/Qwen3.6-35B-A3B-NVFP4`); whether no-copy buffers over mmap stay resident or page-fault on
  first touch (a prefault pass or `MTLResidencySet` is likely needed, which shows up as a cold first token otherwise);
  the working-set limit (`recommendedMaxWorkingSetSize`) for the planner.
* Memory planner: `MemoryPlan::compute` takes `DeviceInfo.total_mem` and `free_mem` "measured now"
  (`crates/intarsia-core/src/memplan.rs:1-4`, `:26`). The cpu backend documents that sysinfo's `available_memory()` on macOS reports only the truly-free page list, keeps an
  `INTARSIA_CPU_FREE_MEM_GB` override and now adds reclaimable pages (`crates/intarsia-backend-cpu/src/lib.rs:111-118`, ADR-0008 decision 6, commits `ddac489`/`bf46bad`).
  The Metal backend should report `recommendedMaxWorkingSetSize - currentAllocatedSize` as `free_mem` (that is the
  number the GPU can actually wire), `unified: true`, and `bandwidth_gbs` from a startup copy-kernel measurement
  (`DeviceInfo.bandwidth_gbs`, `backend.rs:47`), used by the bench byte floor.
* `MemKind::Pinned` maps to the same shared storage mode; `upload`/`download` are memcpy against `contents()`.

## 2. Op inventory

The `Op` enum (`crates/intarsia-core/src/ops.rs:51`, `Op::ALL` has 27 entries) is the full list. What the existing
GPU backends register: HIP `crates/intarsia-backend-hip/src/kernels.rs:42-64` (dense) and `quantmoe.rs:13-28` (quantised, fused);
`Op::Sample` is listed `UNIMPLEMENTED` (`kernels.rs:68`) and handled on the host; CUDA mirrors this.

Dense tier (**minimal set to run Qwen3-8B BF16**; 13 ops): `Embedding`, `RmsNorm`, `RmsNormAdd`, `Add`, `QkNorm`,
`Rope`, `Matmul`, `Logits`, `KvAppend`, `AttnPrefill`, `AttnDecode`, `SiluGlu`, `Argmax`. Notes:
* `Matmul` and `Logits` need a GEMV variant for small M (decode) and a tiled GEMM for prefill; HIP has `matmul_bf16`
  and `matmul_bf16_gemv` as separate registry entries (`kernels.rs:48-51`). Start with GEMV plus a simple
  simdgroup-matrix GEMM.
* `AttnDecode` has a split-KV variant on HIP/CUDA (`attention_splitkv`) needing `scratch` (`ik_scratch`, `ik_abi.h:96`); start with the
  plain one. `AttnPrefill` plain first, flash-style later (`attn_prefill_bf16_flash` exists on HIP).
* `Sample` stays host-side; the cpu reference path covers it.
* The minimal set can be confirmed mechanically: run `intarsia plan show` on Qwen3-8B with a Metal backend and read
  the missing-kernel error list (the `kernels` stage fails with a list of missing kernels, `docs/adding-a-model.md` stage table).

Extra set for **Qwen3.6-35B-A3B NVFP4** (checkpoint inspected locally: 40 layers, 3 of 4 linear-attention, 256 experts,
top-8, expert intermediate 512, 16 attention heads, 2 KV heads, head_dim 256; `hf_quant_config.json` is
`MIXED_PRECISION`):
* `GdnConv`, `GdnDeltaRule` (GatedDeltaNet; `docs/hybrid-models.md`, `crates/intarsia-core/src/gdn.rs`), `GateSplit`, `SigmoidMul`
  (gated attention); f32 recurrent state updated in place.
* `MatmulNvfp4` (shared expert and dense parts), `MoeTopk`, `MoeExperts`, `MoeCombine` (routed experts in NVFP4,
  `W4A16_NVFP4`, group 16).
* `MatmulFp8` as well: the checkpoint stores the linear-attention projections (`in_proj_qkv`, `in_proj_z`, `out_proj`)
  as FP8 (read from `hf_quant_config.json`), so fp8 dequant GEMV is not optional for this model.

Optional fused decode ops, each falls back to the unfused ops via `Registry::select` when no kernel is registered:
`MatmulPair`, `MatmulFp8Pair`, `MatmulNvfp4Glu`, `GatedRmsnorm` (`ops.rs` table, HIP registers all four at
`quantmoe.rs:25-28`, `kernels.rs` for `gated_rmsnorm_bf16`). Skip them until phase B/D. They matter more on Metal than on
discrete GPUs because dispatch and barrier overhead per kernel is a larger share of a 7 ms MoE token.

Ops count summary: 13 minimal; +9 for 35B (`GdnConv`, `GdnDeltaRule`, `GateSplit`, `SigmoidMul`, `MatmulFp8`,
`MatmulNvfp4`, `MoeTopk`, `MoeExperts`, `MoeCombine`); 4 optional fused; `Sample` host-side. 13 + 9 + 4 + 1 = 27 = `Op::ALL`.

## 3. Performance targets

### 3.1 What "top 3" means

The comparison set is MLX / mlx-lm, llama.cpp Metal, LM Studio, Ollama. Two of them are wrappers: LM Studio ships a
llama.cpp engine and an MLX engine; Ollama 0.18 runs llama.cpp and 0.19 added an MLX path (Ollama blog, cited below).
So there are really two engines: MLX and llama.cpp. "Top 3" in a field of five (these four plus intarsia) means being at
or above the llama.cpp-based entries (llama.cpp, Ollama-llama.cpp, LM Studio-llama.cpp) and then, to be unambiguously
second, staying within 10 percent of MLX. Defined target, single stream, batch 1, same quant, same prompt length:

* Qwen3-8B BF16 decode: at least llama.cpp Metal, within 10 percent of mlx-lm.
* 35B-A3B class NVFP4/4-bit decode: at least llama.cpp Metal, within 10 percent of MLX (oMLX figure in 3.2).
* Prefill: not a gate until phase D; on M5 the neural accelerators decide it.

### 3.2 Published numbers (batch 1 decode, tok/s)

I found no single source comparing all four on the same chip and model, and no absolute Qwen3-8B BF16 figure for any
Max chip. What exists:

| Model / quant | Chip (bandwidth) | Runtime | Decode tok/s | Source | Confidence |
|---|---|---|---|---|---|
| Qwen3-30B-A3B-Instruct-2507 q4 | M4 Max 64GB (546 GB/s, inferred) | mlx-lm 0.28.2 | 113.3 (2048-token prompt, 128 gen) | https://gist.github.com/awni/c1790e4c3a39be6e8f1c4afd42423d2d | high |
| same, q8 / q6 / q5 | M4 Max | mlx-lm | 83.2 / 94.1 / 101.0 | same gist | high |
| Qwen3-4B bf16 / q8 / q4 (proxy, no 8B in gist) | M4 Max | mlx-lm | 52.5 / 86.9 / 134.5 | same gist | high |
| Qwen3-8B 4-bit | M4 Max 128GB | mlx-lm / llama.cpp / vllm-mlx | 79.9 / 76.9 / 93.3 | https://pub.towardsai.net/i-ran-claude-code-on-my-macbook-with-vllm-mlx-it-embarrassed-llama-cpp-by-87-093e8c777826 | low-medium (paywalled, seen through a summary) |
| Qwen3.5-35B-A3B 4-bit | M5 Max 40c 128GB | oMLX (MLX-based) | about 132 at 1k ctx, about 120 at 4k, about 85 at 32k | https://omlx.ai/benchmarks?page=15462 | medium (community, snippet values, Qwen3.5 not 3.6) |
| Qwen3.6-35B-A3B 4-bit | M4 Max | oMLX | 105.9 at 4k | same | medium |
| Qwen3.5-35B-A3B NVFP4 vs Q4_K_M | Apple silicon, chip not named | Ollama 0.19 (MLX) vs 0.18 (llama.cpp) | 112 vs 58 | https://ollama.com/blog/mlx | high that the page says it; formats differ, chip unstated |
| Llama-2-7B F16 / Q8_0 / Q4_0 (dense 7B proxy) | M5 Max 40c (614 GB/s) | llama.cpp Metal | 37.1 / 72.4 / 119.9 | https://github.com/ggml-org/llama.cpp/discussions/4167 | high |
| same | M4 Max 40c (546 GB/s) | llama.cpp Metal | 31.6 / 54.1 / 83.1 | same | high |
| same | M4 Max 32c (410 GB/s) | llama.cpp Metal | 24.3 / 43.9 / 70.0 | same | high |

Apple's M5 vs M4 MLX post (https://machinelearning.apple.com/research/exploring-llms-mlx-m5; base M5 vs base M4, ratios only): decode 1.24x on
Qwen3-8B bf16 and 4-bit, 1.25x on Qwen3-30B-A3B 4-bit; TTFT 3.5 to 4x from the neural accelerators. Memory: 17.46 GB
Qwen3-8B bf16, 17.31 GB for the 30B-A3B 4-bit. Not found: LM Studio absolute numbers, Ollama GGUF numbers for these
models on a Max chip, and any measured Qwen3-8B BF16 tok/s on a Max chip. Two self-described estimate sites were excluded.
**Phase B must therefore measure MLX and llama.cpp on this Mac itself** with the `local-model-bench` / localeval harness
(see section 4); the table above is context, not a target to copy.

### 3.3 Byte floors for this Mac (M5 Max, 40-core GPU, 128 GB)

Bandwidth: Apple states up to 614 GB/s for the 40-core M5 Max and 460 GB/s for the 32-core (Apple newsroom 2026-03 and
the MacBook Pro specs page; the 32-core number is medium-high confidence). `system_profiler` reports 40 GPU cores here, so 614 GB/s
applies. Practical GPU-visible bandwidth is lower (llama.cpp's Llama-2-7B F16 on this chip reaches 37.1 tok/s for about 13.5 GB,
roughly 500 GB/s or 81 percent).

| Model | Bytes per token | Floor at 614 GB/s | Realistic ceiling (80-90 percent) |
|---|---|---|---|
| Qwen3-8B BF16 | 16.38 GB task size, 15.14 GB actually read (embedding is a one-row gather; figures from `docs/results/2026-10-13-phase2a-hip.md:103,132`) | 37.5 tok/s (40.6 on 15.14 GB) | 30 to 36 tok/s |
| Qwen3-8B 8-bit / 4-bit (reference only) | about 8.7 / 4.9 GB | 70 / 125 tok/s | |
| Qwen3.6-35B-A3B NVFP4 | about 2.3 GB per token (my estimate: 8 of 256 experts per layer at about 0.5625 B/param plus the bf16/fp8 dense parts and a bf16 lm_head; accuracy about 25 percent) | about 270 tok/s | 100 to 160 tok/s |

For the real number, `intarsia bench speed` computes the byte floor from the checkpoint tensor sizes with expert bytes scaled by
`top_k / experts` (`docs/adding-a-model.md`, "Byte floor"); it needs `--bandwidth-gbs`, which the Metal backend's measured
`DeviceInfo.bandwidth_gbs` should feed. The R9700 reference points in this repo: Qwen3-8B BF16 decode 36.9 tok/s at 627 GB/s,
96 percent of a 38.3 floor; 35B-A3B decode 135 tok/s after HIP fusions (git `42d90d5`). A Metal 8B BF16 result of about 33 to 35 tok/s
would be the same quality of result on a 614 GB/s part.

Reading the table: on 8B BF16 every runtime is within a few percent of the floor, so the ranking is decided by GEMV bandwidth efficiency
and per-token overhead; BF16 is the case MLX tunes least (the Apple post only gives ratios). On the MoE the floor is far above what anyone
reaches (MLX about 132 tok/s on M5 Max for a 4-bit 35B-A3B, roughly half the floor), so launch overhead, expert-gather efficiency and
fusion decide it, which is phase D.

## 4. Phased plan with hour estimates

Estimates are for one engineer with Metal experience, include debugging on this Mac, and exclude waiting for runs.

| Phase | Goal | Hours | Exit gate |
|---|---|---|---|
| A | Qwen3-8B BF16 decode and prefill working and accurate | 50 (range 40 to 70) | `qualify --backend metal` passes quick section on 8B; admission passes for the 13 dense ops |
| B | Decode speed within 10 percent of MLX (and not below llama.cpp) on 8B BF16 | 60 (range 40 to 90) | `bench speed` decode at least 90 percent of measured mlx-lm on this Mac; at least 85 percent of byte floor (restated 2026-10-21 by the user: the floor counts KV bytes at the mean context, and only runs at GPU clock ≥ 1300 MHz and CPU load ≤ 0.30 count; see docs/benchmarking.md "Metal speed gate") |
| C | Qwen3.6-35B-A3B NVFP4 | 70 (range 50 to 100) | 35B quick gates pass, `35b-short` is advisory per ADR-0008; decode at least llama.cpp Metal |
| D | Graph replay and fusion | 40 (range 25 to 60) | ICB replay in `graph_capture` (or documented no-go), fused ops registered, 35B within 10 percent of MLX |

Phase A detail (about 50 h):
1. Crate skeleton, `Backend` impl, shared-storage buffers, `wrap_host` over whole-file mmap, `DeviceInfo`; wire `open_backend`
   (`crates/intarsia-cli/src/common.rs:93-118`, add `"metal"` arm and a `metal` feature in `crates/intarsia-cli/Cargo.toml:19-25`) (10 h).
2. `build.rs` metallib compile (`xcrun metal`/`metallib`, env override, stub when absent, mirroring `crates/intarsia-backend-hip/build.rs`) (5 h).
3. 13 dense ops, naive first, each registered with a tolerance and admitted (25 h).
4. Prefill GEMM (simdgroup matrix) and a first plain attention prefill; debug against goldens (10 h).

Phase B: GEMV tuned to bandwidth (wide loads, simdgroup reductions, bf16 dot), split-KV decode attention with `scratch`, fused
`RmsNormAdd`/`SiluGlu` already in the op set, command-buffer encode cost profiled, startup bandwidth measurement; head to head with mlx-lm and
llama.cpp on this Mac via `local-model-bench` (localeval at `/opt/code/deepeval`, per the user's CLAUDE.md rule; this host is not a fleet box, no fleetq slot needed for local-only work, but
confirm before any run against a fleet host).

Phase C: fp8 and nvfp4 dequant GEMV (decode, M small) and mma-style kernels (prefill), GDN conv and delta rule (f32 state in place), MoE topk, grouped
experts, combine; NVFP4 layout per `docs/quant-moe.md:13`.

Phase D: ICB-based `graph_capture`/`graph_replay` (compare with eager re-encode before deciding), fused `MatmulPair`, `MatmulFp8Pair`,
`MatmulNvfp4Glu`, `GatedRmsnorm`, split-KV and flash prefill with `mpp::tensor_ops` on M5.

### Existing tools: unchanged vs needing a Metal arm

Work unchanged once a backend named `metal` is selectable (all take `--backend`):
* `intarsia qualify --backend metal` (`crates/intarsia-cli/src/cmd/qualify.rs`); stages doctor, recipe, plan, kernels, gate, acc-short, bench run as subprocesses of the
  same binary. Mac smoke already exists for cpu (`docs/adding-a-model.md`, "On Mac, `--backend cpu --smoke`").
* `intarsia kernels test --backend metal` and `kernels list` (`crates/intarsia-cli/src/cmd/kernels.rs`): backend is a string, defaulting to `cpu`; the cpu reference runs in-process on the same host.
* `intarsia accuracy` against goldens in `golden/` (`qwen3-8b-fp32.safetensors`, `qwen36-35b-a3b-nvfp4-vllm-top20.safetensors` and the long variants). The Mac cannot
  generate the 35B golden (vLLM) but can consume the committed one.
* `intarsia bench speed`, `serve`, `plan`, `generate`. The graph counters (`intarsia_graph_replays_total`, `crates/intarsia-bench/src/speed.rs:362`) read zero until phase D; `--require-graphs` (`bench.rs:169`) must not be passed.
* `recipe new` is model-side analysis and needs no backend; recipes (`recipes/qwen3.toml`, `qwen3_5.toml`, `qwen3_5_moe.toml`) are shared.

Needs a Metal arm:
* `common::open_backend` and `open_group` (`crates/intarsia-cli/src/common.rs:93-131`): new `"metal"` match arm and feature; update the error string and the `--backend` help text (`common.rs:17`). Exclusive-feature `compile_error!` check in the CLI lib (cuda and hip only today) needs a decision: Metal does not export `ik_*`, so it can coexist.
* `doctor` (`crates/intarsia-cli/src/cmd/doctor.rs`): GPU, driver and bandwidth probes are CUDA/HIP shaped; add a Metal device probe and the `xcrun metal` toolchain check.
* `qualify` byte-floor bandwidth detection (arch table at `crates/intarsia-cli/src/cmd/qualify.rs:1475`): needs the Metal backend's `DeviceInfo.bandwidth_gbs`, else pass `--bandwidth-gbs 614`.
* Per-recipe, per-backend `prefill_chunk` (ADR-0008 decision 2): a Metal entry is needed.
* `--tp` / `Comm`: out of scope (single device); the `Comm` trait stays unimplemented.
* CI: a macOS runner with Xcode (section 5).

## 5. Risks and unknowns

1. **Metal toolchain.** On this Mac `xcrun -f metal` fails with the default developer dir (`xcode-select -p` is `/Library/Developer/CommandLineTools`,
   which has no `metal`). Xcode 27.0 is installed at `/Applications/Xcode.app`, and with `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`
   `xcrun -f metal` resolves (the Metal toolchain is a separate MobileAsset cryptex on recent Xcode). So `build.rs` must honour `DEVELOPER_DIR`, and a
   fresh developer machine or CI runner needs the Metal Toolchain component installed. Fallback to reduce the dependency: compile MSL source at runtime with
   `MTLDevice.newLibraryWithSource` (no Xcode needed at build, costs seconds of startup and slows the first run); keep it as a debug/fallback path behind an env flag, as the HIP build does with a stub when `hipcc` is absent.
2. **Does 35B NVFP4 fit?** The checkpoint is 22 GB on disk (`~/models/Qwen3.6-35B-A3B-NVFP4`), this Mac has 128 GB, and Metal's working-set cap is typically
   about 75 percent of RAM (about 96 GB; to be read from `recommendedMaxWorkingSetSize`). Weights plus KV plus workspace fit with a wide margin. The practical risk is the
   macOS free-memory reading (ADR-0008 decision 6, improved by `ddac489`), not capacity. A 16 GB or 24 GB Mac could not run it; 64 GB could.
3. **Licensing.** mlx, mlx-c and llama.cpp are MIT (researcher confirmed badges; the LICENSE texts were not read). Compatible with the workspace's Apache-2.0 if the notice is retained for any copied source; follow the existing `NOTICE-radiance` pattern. `objc2-metal` is MIT/Apache-2.0 (not re-verified).
4. **fp8/fp4 on Apple GPUs.** No native fp8 or fp4 arithmetic verified on any Apple GPU (a Metal 4.1 header reading suggests FP8/FP4 tensor types, unverified and macOS 27 only). Plan assumes dequant in the kernel; cost is ALU on a bandwidth-bound kernel, probably fine for decode, worse for prefill.
5. **No cross-runtime baseline for our exact cases.** We have no published Qwen3-8B BF16 number for MLX or llama.cpp on a Max chip, so "within 10 percent of MLX" is only checkable by running MLX here. Do that in phase B before promising a target.
6. **Replay on Metal.** `MTLIndirectCommandBuffer` compute support is documented from Metal 3 on Apple3+, but I did not confirm every constraint (no `setBytes`, pipeline-state limits, `maxCommandCount`, performance versus re-encoding). A spike is needed; phase D may conclude "no replay, re-encode is fast enough" (at 300 to 400 dispatches per token the CPU encode cost is tens to a few hundred microseconds per token by my estimate, not measured).
7. **mmap no-copy quirks.** Page alignment, `maxBufferLength`, residency, and cold first token (section 1.5).
8. **Run-to-run determinism.** Reductions through simdgroups are deterministic for a fixed geometry; atomics-based MoE combine would not be. Prefer non-atomic combine so admission can use `Exactness::Deterministic` and the prefix-reuse gate (ADR-0008 decision 4) is not triggered by noise.
9. **Maintenance.** A third kernel pack (cuda, hip, metal) triples the op-level work for each new op. The registry limits the damage (fused ops optional), but every new required op is now three kernels.
10. **Scope of "macOS kernels".** I read the request as a GPU (Metal) backend. CPU-side Accelerate/NEON work on the existing `cpu` backend is a separate, smaller question (the cpu backend is a correctness reference, `crates/intarsia-backend-cpu/src/lib.rs`, with a "fast" path in `fast.rs`) and is not covered.

## 6. ADR-0009: now or later?

Later, but soon, and cheaply gated. Reasons:

* The architecture needs no change: `docs/ARCHITECTURE.md:42` already names "Metal / Vulkan or SYCL backends behind the same trait" as "not built now; the trait is shaped so they slot in", and `:325` lists "Apple/Metal decision" under the phase that chases the leaders. An ADR would only record a decision that the docs already deferred.
* The ADR-worthy open decisions all need data this scoping cannot give: (1) whether ICB replay beats re-encode, (2) the measured MLX and llama.cpp baselines on this Mac, (3) whether `mpp::tensor_ops` pays off on the available hardware. Writing the ADR now would record guesses.
* Suggested trigger: write ADR-0009 at the end of a 10 to 15 hour spike that (a) stands up the backend skeleton with `wrap_host` and one GEMV, (b) measures bandwidth efficiency of that GEMV against the 614 GB/s figure, (c) runs mlx-lm and llama.cpp on Qwen3-8B BF16 here, and (d) checks ICB compute encoding. If GEMV reaches at least 80 percent of bandwidth and MLX BF16 is within reach, accept the plan above; if not, reconsider option b as a pragmatic backend.
* If the operator wants a decision on record now, the minimal ADR is: "Metal backend is a native Rust crate (option a), phased A to D, MLX is a bench oracle only", with the spike as its first consequence.

## Sources

* Apple ML Research, M5 MLX post: https://machinelearning.apple.com/research/exploring-llms-mlx-m5
* llama.cpp Apple Silicon table: https://github.com/ggml-org/llama.cpp/discussions/4167
* awni mlx-lm benchmark gist: https://gist.github.com/awni/c1790e4c3a39be6e8f1c4afd42423d2d
* Ollama MLX blog: https://ollama.com/blog/mlx
* oMLX community benchmarks: https://omlx.ai/benchmarks?page=15462
* Towards AI vllm-mlx article: https://pub.towardsai.net/i-ran-claude-code-on-my-macbook-with-vllm-mlx-it-embarrassed-llama-cpp-by-87-093e8c777826
* Apple M5 Pro/Max newsroom: https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/
* Apple MacBook Pro specs: https://www.apple.com/macbook-pro/specs/
* Apple Metal feature sets: https://developer.apple.com/metal/feature-sets
* Apple Metal Performance Primitives guide: https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf
* Rigel (M4 tensor ops, fp8 emulation): https://arxiv.org/abs/2606.12765
* Cross-framework MLX/llama.cpp/Ollama study (M2 Ultra, Qwen-2.5): https://arxiv.org/pdf/2511.05502v1
* FP8/FP4 in Metal 4.1 (forum, unverified): https://techboards.net/threads/fp8-fp4-tensors-in-metal-4-1.5676/
* Licences: https://github.com/ggml-org/llama.cpp, https://github.com/ml-explore/mlx, https://github.com/ml-explore/mlx-c
