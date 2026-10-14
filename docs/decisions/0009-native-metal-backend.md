# ADR-0009: The macOS backend is a native Metal crate; MLX is a benchmark oracle only

Status: Accepted (operator decision, 2026-10-14: "accept ADR-0009")

Context: `docs/research/macos-metal-backend-scope.md` scoped four options and deferred this ADR until a 10 to 15 hour spike had measured (a) a skeleton with `wrap_host` and one GEMV, (b) that GEMV's bandwidth against 614 GB/s, (c) mlx-lm and llama.cpp on Qwen3-8B BF16, and (d) ICB compute encoding. Section 6 set the gate: "If GEMV reaches at least 80 percent of bandwidth and MLX BF16 is within reach, accept the plan above". Results: `docs/results/2026-10-14-metal-spike.md` (branch `spike/metal`, `fae83b2`). Host: endor, M5 Max 40-core GPU, 128 GB, macOS 27.0.

| # | Decision |
|---|---|
| 1 | Native Rust crate `intarsia-backend-metal` (objc2-metal, AOT `.metallib`), option a |
| 2 | MLX and llama.cpp are benchmark oracles, not backends |
| 3 | Graph replay: re-encode each step; `supports_graphs` stays false; ICB is optional, phase D |
| 4 | Zero-copy weights through `wrap_host`; GEMV keeps an 8-byte-aligned path |
| 5 | Phase plan A to D from the scope doc stands, with the spike's numbers as targets |

## 1. Native Metal crate

**Decision.** The macOS GPU backend is `crates/intarsia-backend-metal`. It implements `Backend` directly on `MTLDevice` / `MTLCommandQueue` / shared-storage `MTLBuffer`s through `objc2-metal`. Its shaders are compiled ahead of time by `build.rs` (`xcrun metal` + `metallib`, honouring `DEVELOPER_DIR`) and embedded with `include_bytes!`. Kernels are registered as `KernelImpl`s and admitted against the `cpu` reference like CUDA and HIP. The crate exports no `ik_*` symbols, so it does not collide with the CUDA/HIP static libraries.

**Evidence.** The default BF16 GEMV (`r1u4s4`: one row per simdgroup, four 16-byte loads per lane) reaches 556.7 / 588.9 / 576.0 / 589.6 GB/s at Qwen3-8B's four GEMV shapes. That is 91 to 96 percent of 614 GB/s, against the 80 percent gate. The full 8B decode-shaped GEMV sequence (253 GEMVs, 15.14 GB) takes 26.07 ms, i.e. 38.4 tok/s GEMV-only, 94.6 percent of peak.

**Consequences.** The backend crate is the third kernel pack after CUDA and HIP. Fused ops stay optional, which limits the per-op cost. The workspace must keep building on Linux, where all Metal code is `cfg(target_os = "macos")`.

## 2. MLX and llama.cpp are oracles

**Decision.** mlx-lm and llama.cpp Metal are measured through localeval as targets. Neither becomes a backend (scope option b), and no MLX code is linked.

**Evidence.** Qwen3-8B BF16, batch 1, decode median tok/s at about 128 / 1k / 4k prompt tokens:

| Engine | ~128 | ~1k | ~4k |
|---|---:|---:|---:|
| mlx-lm 0.31.3 | 34.4 | 33.8 | 31.0 |
| llama.cpp b11146 | 34.8 | 33.4 | 31.7 |

These numbers are 76 to 86 percent of 614 GB/s end to end. Our GEMV alone runs at 94.6 percent, so beating them needs everything outside the matmuls in under 3.0 ms per token at short prompts (6.2 ms at 4k). That is within reach, which meets the gate's second clause. Option b would also have given up per-kernel admission and zero-copy weights (scope doc section 1.2).

## 3. Graph replay: re-encode

**Decision.** The engine encodes each decode step's dispatches into one command buffer per step. `supports_graphs` stays false, so the engine takes the eager path (`graph_step.rs` `usable`). `MTLIndirectCommandBuffer` replay is a phase D option, revisited only if dispatch counts grow past several thousand per step or encode time shows up in profiles.

**Evidence.** ICB replay of the 253-GEMV step is bit-identical to direct encoding. It cuts CPU encode plus commit from 0.107 ms to 0.056 ms per step against 26.6 ms of GPU time. GPU time does not change while barriers are present.

**Consequence.** `launch` must append to an open per-step command buffer rather than commit one per call, as the spike's `launch` still does. This is phase A work.

## 4. Zero-copy weights and alignment

**Decision.** `wrap_host` wraps each mmap'd safetensors file once with `newBufferWithBytesNoCopy`, rounding to the VM page, and tensors are offsets into that buffer. GEMV keeps a 16-byte-load path and an 8-byte-load path, selected by operand alignment. A scalar path remains as a correctness fallback only.

**Evidence.** A read-only `memmap2::Mmap` is accepted with no workaround. GPU reads match host bytes on 150 tensors of the 35B checkpoint, including the last byte of each shard. In Qwen3-8B `model-00001-of-00005.safetensors` the header is 9328 bytes (8 mod 16), so all 53 of its 2-D bf16 weights are only 8-byte aligned. The 8-byte path reaches 558 to 592 GB/s on them; the scalar path reaches 319 to 465 GB/s.

**Open.** Bandwidth from mmap-backed buffers and cold GPU page faults is not measured. Realigning misaligned shards at load time is the fallback if a later kernel cannot take 8-byte operands. The "only `MmapHost` is wrapped" assumption is documented in `host.rs`, not enforced.

## 5. Phase plan

**Decision.** Phases A to D from the scope doc stand:

- A (~50 h): Qwen3-8B BF16 working and accurate, 13 dense ops, `qualify --backend metal`.
- B (~60 h): decode within 10 percent of mlx-lm, i.e. at least about 31 tok/s.
- C (~70 h): Qwen3.6-35B-A3B NVFP4.
- D (~40 h): replay and fusion.

`DeviceInfo.bandwidth_gbs` comes from the copy kernel (546 to 550 GB/s). That under-reports what the GEMV actually reaches by about 7 percent until the copy kernel improves, so byte-floor checks use `--bandwidth-gbs 614` where precision matters.

**Still open.** Whether `mpp::tensor_ops` pays off for prefill on M5. mlx-lm reaches 3.2k tok/s marginal prefill at 4k prompts; llama.cpp reaches 1.6k.
