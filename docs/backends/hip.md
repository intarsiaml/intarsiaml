# HIP backend (AMD gfx12, R9700)

Status (2026-10-13, Phase 2a): dense BF16 Qwen3 runs end to end on one R9700 (gfx1201) through the torch-free kernel
ABI. 21 kernels, all admitted against the cpu reference. Numbers: [results/2026-10-13-phase2a-hip.md](../results/2026-10-13-phase2a-hip.md).
Qwen3.6-35B-A3B NVFP4 (FP8 / NVFP4 matmuls, sparse MoE, GatedDeltaNet, gated attention) runs on one card, see `docs/results/2026-10-13-phase2a-hip.md` section 9. MXFP4, speculation and TP=2 are later phases; see "Not done".

## Layout

| Path | What |
|---|---|
| `kernels/hip/*.hip` | device code and C launchers (`ik_*`, `kernels/abi/ik_abi.h`), compiled AOT by hipcc for one arch |
| `kernels/hip/ik_hip.h` | wave32 helpers, bf16 on raw 16-bit patterns, launcher validation (`ik_check`, `IK_TRY`) |
| `kernels/hip/ik_runtime.{h,hip}` | thin C wrappers over the HIP runtime (device info, malloc, copies, streams); the Rust side never sees `hipDeviceProp_t` |
| `crates/intarsia-backend-hip` | `build.rs` (hipcc + `ar` into `libintarsia_hip_kernels.a`, links `amdhip64`), `ffi.rs`, `kernels.rs` (the registry table), `device.rs` (`Backend` impl) |
| `crates/intarsia-cli` feature `hip` | `--backend hip` for every command, `intarsia doctor` HIP checks |

The crate compiles to a stub without ROCm (the Mac, CI): `create()` returns `Unsupported`, `cargo check --workspace` passes anywhere.

`build.rs` environment: `ROCM_PATH` / `HIP_PATH` (default `/opt/rocm`, then `hipcc` on `PATH`), `INTARSIA_HIP_ARCH` (default
`gfx1201`; the kernels need gfx12: wave32 and the `__builtin_amdgcn_wmma_*_w32_gfx12` builtins), `INTARSIA_NO_HIP=1` forces the stub.
The `cuda` and `hip` cargo features of `intarsia-cli` are mutually exclusive (compile error): both kernel libraries export the same
`ik_*` and `ik_rt_*` C symbols, and linking both would resolve each launcher to whichever archive the linker meets first.

## Kernels

| Op | Kernel (`ik_` prefix) | Class | Notes |
|---|---|---|---|
| embedding, add, kv_append, argmax | `embedding_bf16`, `add_bf16`, `kv_append_bf16`, `argmax_f32/bf16` | Exact | kv_append also writes an fp8 e4m3 cache (software round-to-nearest-even with saturation, written to match the CUDA conversion; checked only against the cpu reference, which the 4 random fp8 admission cases match exactly, not against CUDA output and not on saturating or subnormal values) |
| rmsnorm, rmsnorm_add, qk_norm, silu_glu, rope, gate_split, sigmoid_mul | `*_bf16` | Deterministic | hipified from the CUDA sources |
| matmul, logits (M <= 4, decode) | `matmul_bf16_gemv`, `logits_bf16_gemv` | Deterministic | native GEMV, `R` rows per warp, `U` 16 B non-temporal loads in flight per lane |
| matmul, logits (any M, prefill) | `matmul_bf16`, `logits_bf16` | Deterministic | native WMMA GEMM (bf16 16x16x16, f32 accumulate, fixed K order), SIMT fallback for K % 32 != 0 or unaligned operands |
| attn_decode (T = 1) and verify (T <= 16) | `attn_decode_bf16_splitkv`, `attn_prefill_bf16_splitkv` | Deterministic | split-KV, one block serves all q heads of a kv group; scratch declared as in the CUDA backend |
| attn_prefill (T > 16) | `attn_prefill_bf16_flash` | Deterministic | WMMA flash attention, head_dim 64 / 128 / 256 |
| attn_prefill / attn_decode fallback | `attn_prefill_bf16`, `attn_decode_bf16` | Deterministic | SIMT, head_dim <= 256 |

Registered since the 35B work: `gdn_conv`, `gdn_delta_rule`, `matmul_fp8_{gemv,mma}`, `matmul_nvfp4_{gemv,mma}`, `moe_topk`, `moe_experts`, `moe_combine` (table in `quantmoe.rs`). Not registered: sampling. `kernels.rs::UNIMPLEMENTED` lists the rest and a unit test keeps the list
honest. No kernel is a radiance port: the Apache-2.0 files under `tatooine:/data/src/radiance` were read for design only, so there is no
`LICENSES/radiance-NOTICE` to add yet (add it with the first ported file).

### Differences from the CUDA sources

* `IK_WARP` is 32. `kernels/common/ik_portable.h` defines 64 for HIP, which is wrong for gfx12 (RDNA4 is wave32 by default, and
  the `_w32` WMMA builtins require it); `ik_hip.h` hard-sets 32, `#error`s on `-mwavefrontsize64`, and the backend refuses a device whose
  `warpSize` is not 32.
* Shuffles are `__shfl_xor(v, o, 32)`: ROCm 7.2 only accepts the `*_sync` forms with a 64-bit mask (`static_assert`).
* bf16 is a plain `struct { uint16_t }` with software round-to-nearest-even conversion, independent of `hip_bf16.h` (its type changed
  between ROCm 6 and 7). `__ldg` is a plain load; `__ldcs` is `__builtin_nontemporal_load`.
* `ik_rt_memcpy_h2d` synchronises the stream after `hipMemcpyAsync`: from pageable memory HIP does not promise the CUDA behaviour that the
  source has been staged when the call returns, and callers drop the source right away.
* Flash prefill is a rewrite: `mma.sync` / `ldmatrix` / `cp.async` are replaced by WMMA. It computes the transposed problem
  (`S^T = K Q^T`, `O^T += V^T P^T`) so the accumulator layout is the B-operand layout of the next product and P never leaves registers.
  V is staged transposed in LDS so every operand is one contiguous 16 B read. Single-buffered (no `cp.async` equivalent used yet).
* GEMM: `D[n, m] = W[n, :] . X[m, :]` (weights are the A operand), so each lane stores 8 contiguous output features. The grid puts token
  blocks on `x`: resident blocks share one weight panel and the weights stream from DRAM once (this alone took M = 1024 gate from 45 to
  73 TFLOPS).
* Device-side position (Phase 1b ABI, `ik_abi.h`): `ik_rope_bf16`, `ik_kv_append_bf16`, `ik_attn_decode_bf16` and `ik_attn_decode_bf16_splitkv` take an optional
  `const ik_tensor* pos` (i32 `[1]`, null = host `pos0`) and the kernels read it when they run, as in the CUDA sources. The backend passes `OpCall::pos`. Split-KV decode with a
  device position launches its grid at the length-independent maximum (the scratch bound) and the blocks past the real split count return, so the partitioning, and every bit of the
  result, equals the host-position launch (`tests/device_pos.rs` checks this bitwise). With a host position the grid is still sized from the key count (unchanged from before the merge).
  HIP does not implement graph capture (`supports_graphs()` is false), so core never passes a device position to it in practice; the path exists to keep the ABI honest.

gfx12 wave32 WMMA 16x16x16 fragment layout (verified on hardware by the admission sweeps): A and B hold 8 bf16 per lane, lane `l` is
row / column `l % 16`, `k = (l / 16) * 8 .. +7`; C / D holds 8 floats, column `l % 16`, rows `(l / 16) * 8 + i`.

## Build and run on tatooine

ROCm lives only in the image `intarsia-hip-dev:rocm7.2.4-rust1.90.0` (`deploy/tatooine/Dockerfile.hip-dev`, base `rocm/dev-ubuntu-24.04:7.2.4` pinned by
digest, Rust 1.90.0; the apt build tools are not pinned and do not touch the kernels), driven by `deploy/tatooine/dev.sh`.
`dev.sh` keeps one cargo target volume **per image build** (`intarsia-cargo-target-<image id>`; `dev.sh --target-vol` prints the name,
`INTARSIA_CLEAN=1` wipes it), so rebuilding the image never reuses a `libintarsia_hip_kernels.a` compiled by the previous hipcc. `build.rs` also
reruns when the hipcc binary or `/opt/rocm/.info/version` changes, and the compiler version is part of the backend fingerprint (below).
The image has no hipBLASLt, rocBLAS or rocWMMA (`hipblaslt-dev` is installable from the ROCm apt repo; see "Not done").

```bash
# on the Mac: sync the worktree, build inside the image (CPU slot, no GPU)
rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv ./ tatooine:~/intarsiaml/
fleetq run --host tatooine --mem 0 --label "build" -- ssh tatooine 'cd ~/intarsiaml && deploy/tatooine/dev.sh cargo build --release -p intarsia-cli --features hip'

# GPU work takes radiance down for a short burst (fleetq lease, restart radiance before releasing)
fleetq acquire --host tatooine --takes-down radiance --mem 30 --label "intarsiaml hip: <what>" --wait
ssh tatooine systemctl --user stop radiance
FLEETQ_JOB=<id> ssh tatooine 'cd ~/intarsiaml && HIP_VISIBLE_DEVICES=0 deploy/tatooine/dev.sh /target/release/intarsia kernels test --backend hip --model /models/Qwen3-8B'
ssh tatooine systemctl --user start radiance   # then wait for curl -s http://tatooine:8080/v1/models
fleetq release <id>
```

`dev.sh` does not publish ports, so a server that other machines reach runs through a plain `docker run --network host` with the same device,
group and mount flags (this is the exact command of the Phase 2a run; stop it with `docker rm -f intarsia-serve` before releasing the lease):

```bash
cd ~/intarsiaml && docker run -d --name intarsia-serve --network host \
  --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video \
  --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=0 \
  -v ~/intarsiaml:/work -v /data/models/intarsia:/models:ro -v "$(deploy/tatooine/dev.sh --target-vol)":/target -w /work \
  intarsia-hip-dev:rocm7.2.4-rust1.90.0 /target/release/intarsia serve --backend hip --model /models/Qwen3-8B \
  --host 0.0.0.0 --port 8090 --served-model-name qwen3-8b-intarsia-hip --max-seq 8192
# bench client, in the same container:
docker exec intarsia-serve /target/release/intarsia bench speed --url http://127.0.0.1:8090 --model qwen3-8b-intarsia-hip \
  --sizes 128,1024,4096 --gen 128 --reps 3 --gb-per-token 16.38 --bandwidth-gbs 627 --label phase2a-hip-qwen3-8b --out-dir /work/bench-out --thinking off
```

Always set `GPU_MAX_HW_QUEUES=1` (radiance measured about 20% on MoE decode
with four hardware queues; `intarsia doctor` warns when it is unset).

Tests (need a GPU, skipped without one; `INTARSIA_HIP_DEVICE` picks the ordinal):

```bash
cargo test --release -p intarsia-backend-hip                       # table tests, admission sweep, attention sweep (+ poisoned-tail and one-hot causal / chunk-edge tests),
                                                                   # f32-output tolerance, MoE batch invariance, thread safety
cargo test --release -p intarsia-backend-hip --test matmul_bench   -- --ignored --nocapture   # GEMV GB/s on the 8B shapes, GEMM TFLOPS
cargo test --release -p intarsia-backend-hip --test attention      -- --ignored --nocapture   # attention throughput
cargo test --release -p intarsia-backend-hip --test quantmoe       -- --ignored --nocapture   # quant / MoE GB/s, hot and DRAM-cold (weights rotate over >= 768 MB)
```

Tuning knobs (read once per process; every one that is set is part of the backend fingerprint, so changing a knob re-runs admission):
`INTARSIA_HIP_GEMV=R,U` (GEMV rows per warp / loads in flight: only the instantiated variants `1,{2,4,8,16}`, `2,{2,4,8}`, `4,{2,4,8}` are accepted, anything
else prints a warning and keeps `1,8`; a launch that matched no instantiation returns `IK_ERR_INVALID_ARG` instead of reporting success),
`INTARSIA_HIP_GEMM_TILE=1..8` (forces a GEMM tile, default auto), `INTARSIA_HIP_FP8_ROWS=1|2|4` (rows per warp of the FP8 GEMV),
`INTARSIA_HIP_MOE_WMMA=0` (batch-invariant MoE, see "MoE and batch size"), `INTARSIA_HIP_DISABLE=name1,name2` (hides kernels from the registry for A/B runs).

## Admission cache

`Backend::fingerprint()` is `hip|<device name>|<gfx arch>|rt<HIP runtime version>|drv<driver version>|pack:<arch the kernels were built for>|cc:<hipcc version>|env:<tuning knobs that are set>`, so a different
card, arch, ROCm runtime, compiler, tuning knob or a rebuild for another arch never reuses cached admission results (`~/.cache/intarsia/admission/`, keyed by this string
plus the executable's size and mtime).

## MoE and batch size

`moe_experts_bf16` has three internal paths, and a token's result depends on which one it ran in:

| tokens T (pairs P = T*K) | path | accumulation |
|---|---|---|
| P <= 16 and K <= 8 (decode, small verify) | `moe_gate_up_p1` + `moe_down_reduce_p1` | per-lane 16-element blocks, warp sum (GEMV order) |
| T < 16 | counting-sort grouping + grouped GEMV-style kernels | the same lane and accumulation order as the p1 kernels |
| T >= 16 (prefill chunks) | grouped bf16 WMMA (`moe_wmma.h`) | tensor-core k order, different rounding |

The first two are bit-identical to each other, so a token decoded alone (T = 1) and in a batch below 16 gives the same bits, as on the CUDA backend
(which has no tensor-core MoE path). The WMMA path is the CUDA-different part: from 16 tokens on, the result differs in the last bits, within the admission
tolerance. Consequences: the same prompt can give slightly different hidden states for a different `--prefill-chunk` or for a remainder chunk of fewer than 16
tokens; prefix-cache reuse versus recompute is not bit-equal; a multi-row verify of 16 or more rows is not bit-equal to serial decode on the MoE layers. The kernel
is `Deterministic` in the registry sense (run-to-run identical for one shape and batch), which is the class definition; it is not batch-invariant.

Gate: `INTARSIA_HIP_MOE_WMMA=0` disables the WMMA path, so every T takes the GEMV-style kernels and a token's result is bit-identical to T = 1 for any batch (MoE at T = 256 takes 3.8 ms
against 1.35 ms with the tensor cores, DRAM-cold; at T = 64 1.49 against 1.23 ms; at T = 16 the GEMV path is even faster, 0.54 against 0.76 ms). The knob is part of the fingerprint. `cargo test -p intarsia-backend-hip --test quantmoe moe_experts_batch_invariance` checks
all three rows of the table (bit-identity below 16 tokens, tolerance above) and re-runs itself with the knob off to check bit-identity at every T. Anything that needs
batch-invariant MoE (lossless speculation claims, `exact` mode) must run with `INTARSIA_HIP_MOE_WMMA=0`; the registry has no per-kernel batch-invariance flag yet (open issue below).
Making the two paths agree is only possible the other way round (the review's longer-term suggestion): GEMV-style kernels that accumulate in the WMMA k-permuted order; a WMMA instruction cannot be made to follow the GEMV order. Not done. `INTARSIA_HIP_MOE_WMMA=0` covers `moe_experts` only: the dense GEMV / GEMM switch (GEMV up to M = 4) and the attention split policy are batch dependent too, so it does not by itself make the whole model batch-invariant.

## Not done (and why)

* **hipBLASLt prefill route.** The dev image has no hipblaslt; adding it means a derived image (`apt-get install hipblaslt-dev`). The WMMA GEMM
  reaches 63-81 TFLOPS on the 8B shapes at M >= 256 (spec-sheet bf16 dense peak is about 190), so a library GEMM is only worth it if it beats that; register it as
  `Exactness::Fast` then.
* **Phase 1b CUDA additions.** CUDA graphs (`graph_capture` / `graph_replay`), the cp.async `*_gemm` launchers and the grouped / two-launch MoE launchers
  (`moe_experts_mma_bf16`, `moe_experts_decode_bf16`) are not ported; the HIP backend keeps its own WMMA GEMM and MoE kernels, whose registered launcher set is unchanged.
* **KV layout.** The contiguous separate K / V cache stays; radiance's paged interleaved layout is a deferred decision.
* **Radiance ports** (MXFP4 x FP8 GEMM, GDN, one-shot all-reduce): Phase 2b onward, see `docs/research/radiance-port-plan.md`.
* **TP=2** over the two cards (P2P 28 GB/s measured).
* **Unifying with CUDA sources.** Candidates for `ik_portable.h`: `norm`, `argmax`, `attention` (SIMT) and `attention_splitkv` differ from the CUDA files by
  2 to 10 lines (stream type, shuffles, `__ldg`); `elementwise` by 34 (fp8 conversion, bf16 type). The blockers are the `IK_WARP` value (32 on
  gfx12 vs the header's 64), the shuffle mask width, the bf16 type, `ik_launch_status` / `IK_STREAM`, and the launch syntax in the runtime wrappers.
  GEMV, the GEMM and the flash kernel stay arch-specific.

## Open issues

* **CUDA backend, same tolerance flaw (finding 8 of `docs/reviews/2026-10-13-opus-phase2a-review.md`).** `crates/intarsia-backend-cuda/src/quantmoe.rs:23` uses one `TOL` (1.5e-2 / 6e-3) for
  bf16 and f32 outputs alike. It needs the same change as the HIP one: an f32-output check at the f32 budget (see the HIP test `quant_f32_outputs_meet_the_f32_tolerance`), and a core
  `judge` / `KernelImpl` that can carry a per-output-dtype tolerance, because `ShapeInfo` has no output dtype and the admission sweep grades every output of a kernel with its single `Tol`.
* **Registry has no batch-invariance flag** (see "MoE and batch size"): `moe_experts` is registered `Deterministic`, which does not say that its result depends on T from 16 tokens up.
* **No per-kernel flag for the dense GEMV vs GEMM paths either**; the same T-dependence exists there and is older than this backend.

## Profiling

`INTARSIA_HIP_TRACE=1` records two stream events per launch (one just before, one just after, per stream) and prints, when the backend is dropped, the GPU time per
kernel (start to stop of that launch, event overhead of about a microsecond included) and, separately, the stream's idle time between launches (batch wall time minus summed
kernel time: host dispatch and bubbles). Earlier traces charged the time between consecutive events to the later kernel, which mixed the two; numbers from those
runs are "inter-launch wall time", not kernel time (see the fix pass in `docs/results/2026-10-13-phase2a-hip.md`). Use with `--profile-every 0`.
