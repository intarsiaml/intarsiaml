# macOS Metal backend spike (2026-10-14)

The spike gate from `docs/research/macos-metal-backend-scope.md` section 6 asks two things. Does a BF16 GEMV reach at least 80 percent of 614 GB/s, and is MLX BF16 within reach? **Both hold.** The plan in the scope doc stands: build a native `intarsia-backend-metal` crate (option a) and keep MLX as a benchmark oracle only.

Host: endor (MacBook Pro Mac17,6, M5 Max, 40-core GPU, 128 GB, macOS 27.0, Metal toolchain 32023.921). Branch `spike/metal` from `arch/plan` `cf0719c`. Timing runs used exclusive endor slots: fleetq #1763, #1765-1767, #1769, #1771 (GEMV/ICB) and #1772 (baselines). `endor-kev` (18 GB) and `endor-tensorfold` (32 GB) stayed resident and idle. fleetq has no health check for them, so background GPU load from them cannot be ruled out.

Raw logs: `docs/results/raw/2026-10-14-metal-spike/`.

## (a) Skeleton

`crates/intarsia-backend-metal` (objc2-metal; everything is `cfg(target_os = "macos")`):

- `MetalBackend` implements `Backend`.
  - Shared-storage buffers. One queue. `unified: true`.
  - Memory: `total_mem` comes from `recommendedMaxWorkingSetSize`, `free_mem` from the working set minus `currentAllocatedSize`.
  - `bandwidth_gbs` is a copy-kernel measurement, made lazily on the first `info()` call (about 100 ms, 549.7 GB/s here).
  - One `KernelImpl`: a bf16 GEMV for `Op::Matmul` at M=1. `supports_graphs` is false.
  - The crate is not wired into the CLI and exports no `ik_*` symbols.
- `build.rs` compiles `kernels/metal/*.metal` into a metallib and embeds it. It honours `DEVELOPER_DIR` and defaults to `/Applications/Xcode.app/Contents/Developer`.
- `wrap_host` is zero-copy through `newBufferWithBytesNoCopy`.
  - It accepts the loader's read-only `memmap2::Mmap` (`PROT_READ`) with no workaround.
  - The base is rounded down and the end rounded up to the VM page (16 KB). The end cannot leave the mapping, because file mmaps are whole pages.
  - There is one `MTLBuffer` per mapping, and `TensorRef.offset` stacks on top of it.
  - Verified on 150 tensors across all 3 shards of Qwen3.6-35B-A3B-NVFP4, including the last tensor and the last byte of each shard.
  - Caveat: this relies on `MmapHost` being the only `HostMemory`. A heap-backed `HostMemory` must not be wrapped. This is documented in `host.rs`, not enforced.
  - The fallback for a mapping larger than `maxBufferLength` (86.6 GB) is implemented but has never run.
- Tests: `cargo test -p intarsia-backend-metal` gives 9/9 (6 GEMV, 3 `wrap_host`).
  - GEMV is checked against an f64 reference at all Qwen3-8B shapes, odd N/K and unaligned offsets.
  - Worst error is 0.97 of a half bf16 ulp.
  - The Linux build of the workspace has not been checked; this must happen before merge. The Cargo.toml `objc2*` dependencies are under `cfg(target_os = "macos")`, and `build.rs` tests `CARGO_CFG_TARGET_OS`. There is no `rustup` on endor, so cross-checking needs a fleet box.

## (b) GEMV bandwidth

GPU timestamps, 5 warm-up reps, 100 timed reps, rotating over ≥ 4 GiB of distinct weights. GB/s = (W + x + y) / time. The default variant `r1u4s4` puts one row per simdgroup, with four 16-byte loads per lane and 4 simdgroups per threadgroup.

| Shape K→N | median GB/s | p10 | p90 | % of 614 |
|---|---:|---:|---:|---:|
| 4096→4096 | 556.7 | 538.3 | 562.8 | 90.7 |
| 4096→12288 | 588.9 | 582.7 | 591.0 | 95.9 |
| 12288→4096 | 576.0 | 567.2 | 578.4 | 93.8 |
| 4096→151936 | 589.6 | 575.3 | 594.5 | 96.0 |

The copy kernel reaches 546 GB/s (89 percent) and read-only 552-568 GB/s. The GEMV beats both, so these are not tight ceilings. `bandwidth_gbs` will under-report by about 7 percent until the copy kernel improves.

**Qwen3-8B decode-shaped sweep.** 36 layers of q/k/v/o/gate/up/down plus lm_head, in one command buffer: 253 GEMVs, 15.14 GB per step.

- 26.07 ms per step: 38.4 tok/s GEMV-only, 580.8 GB/s, 94.6 percent of 614.
- The byte floor at 614 GB/s is 24.7 ms (40.5 tok/s).

**Alignment finding.** In Qwen3-8B `model-00001-of-00005.safetensors` the header length is 9328, which is 8 mod 16. All 53 of that shard's 2-D bf16 weights therefore start 8 mod 16, and zero-copy weights there cannot use 16-byte loads.

- An 8-byte-load kernel (`gemv_bf16_a8u8s4`, selected automatically) reaches 558-592 GB/s at offset 8.
- The scalar fallback for anything less aligned reaches only 319-465 GB/s.

## (c) Baselines: Qwen3-8B BF16, batch 1, localeval 0.2.1, thinking off

| Engine | ~128 tok | ~1k tok | ~4k tok | prefill @4k (marginal) |
|---|---:|---:|---:|---:|
| llama.cpp 0.5.0 b11146 (Metal, `-fa on`) | 34.8 | 33.4 | 31.7 | 1585 |
| mlx-lm 0.31.3 / mlx 0.32.1 | 34.4 | 33.8 | 31.0 | 3189 |

Decode figures are medians in tok/s. Effective bandwidth is 76-86 percent of 614 GB/s. The llama-bench cross-check gave tg128 = 35.81 and pp512 = 2803 tok/s.

Run IDs:

- `20261014-072555-adhoc-llamacpp-bf16` (unanimis `7b3703a5`)
- `20261014-072755-adhoc-mlx-lm-bf16` (unanimis `f1af95d5`)

The GEMV-only ceiling of 38.4 tok/s (26.07 ms) leaves 3.0 ms per token for norms, RoPE, attention, sampling and launch overhead before we fall below mlx-lm at short prompts (34.4 tok/s, 29.07 ms). At 4k prompts the budget is about 6.2 ms (31.0 tok/s), and that is where the KV reads land. The phase B exit (90 percent of mlx-lm) is about 31 tok/s.

## (d) Indirect command buffers

- `MTLIndirectCommandBuffer` with the 253 recorded GEMVs replays bit-identically to direct encoding (0 of 3,105,536 bytes differ).
- CPU encode plus commit per step is 0.107 ms direct and 0.056 ms ICB. That saves 0.05 ms against about 26.6 ms of GPU time.
- GPU time is unchanged with barriers. Without barriers it is about 3 percent faster, but real decode has data dependencies, so that does not carry over.
- Conclusion: re-encode is enough. ICB replay is low priority for phase D. Recheck only if dispatch counts grow past several thousand per step.

## Not measured

- GEMV speed from mmap-backed buffers, and the cost of cold GPU page faults.
- Thermal behaviour over long runs.
- `launch` commits one command buffer per call. Real decode needs batched encoding in one command buffer, which is phase A work.

## Inputs for ADR-0009

1. Option a is confirmed. The kernel reaches 91-96 percent of peak on its own, and MLX and llama.cpp sit at 76-86 percent end to end.
2. Graph replay: re-encode per step and keep `supports_graphs` false. ICB stays a measured, optional phase D item.
3. Weight alignment: keep the 8-byte-aligned GEMV path, or realign shards whose header is not 16-aligned at load time. Decide in phase A.
4. Still open: whether `mpp::tensor_ops` pays off for prefill on M5. mlx-lm already reaches 3.2k tok/s marginal at 4k.
