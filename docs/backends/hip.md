# HIP backend (AMD gfx12, R9700)

Status (2026-10-13, Phase 2a): dense BF16 Qwen3 runs end to end on one R9700 (gfx1201) through the torch-free kernel
ABI. 21 kernels, all admitted against the cpu reference. Numbers: [results/2026-10-13-phase2a-hip.md](../results/2026-10-13-phase2a-hip.md).
Qwen3.6-35B-A3B NVFP4 (FP8 / NVFP4 matmuls, sparse MoE, GatedDeltaNet, gated attention) runs on one card, see `docs/results/2026-10-13-phase2a-hip.md` section 9. MXFP4, speculation and TP=2 are later phases; see "Not done".

Decode (2026-10-13, perf/hip-decode): HIP graph capture/replay of the decode step (`supports_graphs`), latency-restructured `*_v2`
decode kernels registered beside their v1 kernels (bitwise equal to them), two launch fusions (conv state, rope tail). 35B decode
97 to 126 tok/s on one R9700; profile, mutation checks, downtime log: [results/2026-10-13-hip-decode.md](../results/2026-10-13-hip-decode.md).

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
honest. Radiance (Apache-2.0) was otherwise read for design only; one kernel is derived: `kernels/hip/allreduce.hip` (Phase 2b), which carries
`kernels/hip/NOTICE-radiance` and `LICENSES/Apache-2.0-radiance.txt` (see "Tensor parallelism" below). Updated 2026-10-21: this paragraph used to say no
kernel was a radiance port and no NOTICE was needed; that was true only before Phase 2b.

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
  Updated 2026-10-21: HIP implements graph capture and replay (`supports_graphs()` is true, merged in `bad0084`, 2026-10-13), and the decode graph is captured once with a fixed grid
  (`docs/results/2026-10-19-hip-round9.md` section 3). This line used to say the opposite (written before the merge).

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
`INTARSIA_HIP_MOE_WMMA=0` (batch-invariant MoE, see "MoE and batch size"), `INTARSIA_HIP_DISABLE=name1,name2` (hides kernels from the registry for A/B runs),
`INTARSIA_HIP_FP8_F8=0` (kill switch for the default W8A8 prefill GEMM `matmul_fp8_mma_f8`: the fp8 projections with M >= 5 go back to
`matmul_fp8_mma_v2` (bf16 tiles on the bf16 WMMA); default on since `docs/results/2026-10-19-hip-round5.md`, not bitwise v2, KL21
numbers in `docs/results/2026-10-19-hip-round4.md` section 14).
`INTARSIA_HIP_FLASH_HPB=1|2|4` (query heads per block of flash v2, capped to divide the GQA group; default 4 since
`docs/results/2026-10-19-hip-round6.md`, every value bitwise), `INTARSIA_HIP_MOE_MW=v|g:d` (variant bits of the grouped MoE
prefill projections, `v` for both or `g` for gate_up and `d` for down; one of 0|1|4|5|13|25|29|41|45. `moe_wmma.h`: 1 skips the
empty 16-pair fragments of a tail tile, 4 aliases the up accumulators onto the activation tiles and compiles for 8 waves per
SIMD, 0 is the round-5 kernel; `moe_wmma_r7.h` (round 7): 8 decodes fp4 with the hardware fp8 conversion instead of the LDS
table, 16 gives down two A fragments per wave, 32 keeps six weight blocks in flight. Defaults since
`docs/results/2026-10-19-hip-round7.md`: gate_up 45 / down 25 at 64 pairs per tile, gate_up 0 / down 41 at 32; `INTARSIA_HIP_MOE_MW=5:5`
runs round 6's 64-pair kernels at both tiles (round 6 itself ran the round-5 kernel at 32). Every value bitwise; read at every
launch).
Round 8 (`docs/results/2026-10-19-hip-round8.md`; every value bitwise, read at every launch): `INTARSIA_HIP_SPLIT_US=n` (default
600, 0 = one launch): the prefill's MoE down projection (over feature tiles) and the GDN delta rule (over token ranges, each piece
continuing the in-place state) run as pieces estimated under n microseconds, because on the R9700 every dispatch longer than
~1 ms is followed by ~275 us in which the GPU starts nothing; gate_up is never split. `INTARSIA_HIP_MOE_REDUCE=0`: the scalar
`moe_reduce_kernel` instead of `moe_reduce_v8_kernel` (8 features per thread, 16-byte loads). `INTARSIA_HIP_SPLITKV_R8=0`: the
split-KV combine one split at a time instead of 16 splits' loads in flight.

Defaults in force on main (checked 2026-10-21 against `crates/intarsia-backend-hip/src/{quantmoe,device,kernels}.rs` and `kernels/hip/attention_splitkv.hip`):
`INTARSIA_HIP_MOE_BLOCK=0` is the kill switch for P2' (`Op::MoeBlock`, top-k + routed experts + shared expert + combine in one launch, bitwise the unfused
chain); it is ON by default since `0cea68fc` (PR #28, `docs/results/2026-10-19-p2prime-soak.md`). `INTARSIA_HIP_FP8_F8=0` is the kill switch for the default
`matmul_fp8_mma_f8` (default since `8624fec1`, 2026-10-18). `INTARSIA_HIP_STAGE=0` (or `=up`) turns off the page-locked staging ring (on by default).
Two opt-ins stay OFF: `INTARSIA_HIP_MOE_V3=1` (one-launch decode experts, +0.8 % decode; the mutant that drops its acquire fence, `v3-nofence`, still
survives every test) and `INTARSIA_HIP_SPLITKV_OCC=1` (split-KV stage 1 at 8 waves per SIMD: +1.5 % 35B decode at about 17.6k tokens, -0.6 % at 128 to
2048; break-even is about 13k tokens of context, so the default stays off; a length-based `auto` choice was rejected because the decode graph is captured once per
process and the position is a device scalar, so the kernel choice would freeze at capture; `docs/results/2026-10-19-hip-round9.md` Decisions 1 and
`docs/results/2026-10-19-hip-round10.md`, split-KV section).

## Admission cache

`Backend::fingerprint()` is `hip|<device name>|<gfx arch>|rt<HIP runtime version>|drv<driver version>|pack:<arch the kernels were built for>|cc:<hipcc version>|env:<tuning knobs that are set>`, so a different
card, arch, ROCm runtime, compiler, tuning knob or a rebuild for another arch never reuses cached admission results (`~/.cache/intarsia/admission/`, keyed by this string
plus the executable's size and mtime).

## Row-exact kernels and batched decode (lane hb)

Batched decode of the 35B (`serve --max-batch N`, N <= 16) runs the `_rows` kernels (`matmul_fp8_gemv_rows`, `matmul_nvfp4_gemv_rows`,
`matmul_bf16_gemv_rows`, `logits_bf16_gemv_rows`, `moe_experts_bf16_rows`, `rows_exact` in the registry): row r of an M-row call is
bitwise row 0 of the 1-row kernel the solo step runs (G-B, `tests/rows_exact.rs`), and none of them takes a WMMA / mma path
(those are M-dependent from 16 tokens). Results: `docs/results/2026-10-23-hb-hip-hybrid-batching.md`.

Lane hr3 adds the row-table launches: one launch per GatedDeltaNet / attention layer for all rows of a step instead of one per row
(`gdn_conv_bf16_rows`, `gdn_delta_rule_bf16_v2_rows`, `kv_append_bf16_rows`, `attn_decode_bf16_splitkv_rows`, `row_table` in the registry, and
`rope_bf16` with one position per token), each row reading its own state or KV cache through a table of device addresses; row r is bitwise the
one-row launch of the solo step (`tests/row_tables.rs`). `moe_block_bf16_rows` serves the fused MoE block at T = 1..16 at top-8 (the ordinary
`moe_block_bf16` stops at T * K <= 16), token t bitwise the T = 1 block (`tests/moe_block.rs`). `INTARSIA_ROW_TABLES=0` keeps one launch per row
per op. Results: `docs/results/2026-10-24-hr3-hip-batching-r3.md`.

Batched decode under tensor parallelism (`serve --max-batch N --tp 2`, lane tp2/batch; HIP and the cpu reference only, CUDA and
Metal keep refusing `--max-batch` with `--tp`): rank 0 runs the batched step and drives the peer through the same step on its own
sub-states (`Model::forward_batch_tp` / `run_peer_batch` in `crates/intarsia-core/src/model/batch.rs`). The peer gets rank 0's
tokens, every row's position and the graph decision, checks its states against them before its first collective, and both ranks
then issue the solo step's collectives once with `b` rows of payload (`[b, hidden]` partials: 128 KiB at b = 16, far under the
4 MiB slot). Each rank has its own row table and `batch_pos`; batched graphs are recorded per rank and row count, rank 0 records
first and the peer follows (a peer that cannot follow fails the step). With the vocab-parallel head each rank projects its
`V / 2` logits of every row, one all-gather fills the tile `[2][b][V / 2]` and rank 0 copies it into `ws.logits` as `[b, V]`.
Every row is bitwise the solo tp=2 step of that sequence (`crates/intarsia-models/tests/tp_batch.rs` on the cpu backend, including
protocol mutants; `INTARSIA_BATCH_TEST_TP=2` runs `batch_cuda_real` on two R9700s; the halved shard shapes are in
`tests/rows_exact.rs`). Window stage: `scripts/tp2-hip/stages-batch.sh`.

## MoE and batch size

`moe_experts_bf16` has three internal paths, and a token's result depends on which one it ran in:

| tokens T (pairs P = T*K) | path | accumulation |
|---|---|---|
| P <= 16 and K <= 8 (decode, small verify) | `moe_gate_up_p1` + `moe_down_reduce_p1` | per-lane 16-element blocks, warp sum (GEMV order) |
| T < 16 | counting-sort grouping + grouped GEMV-style kernels | the same lane and accumulation order as the p1 kernels |
| T >= 16 (prefill chunks) | grouped bf16 WMMA (`moe_wmma.h`) | tensor-core k order, different rounding |

The first two are bit-identical to each other, so a token decoded alone (T = 1) and in a batch below 16 gives the same bits, as on the CUDA backend
below 32 tokens (CUDA has had a tensor-core MoE path, `moe_experts_mma_bf16`, since Phase 1b, so from T >= 32 CUDA is batch-variant too; `docs/quant-moe.md`, ADR-0007 section 5). The WMMA path is the CUDA-different part: from 16 tokens on, the result differs in the last bits, within the admission
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
* **Phase 1b CUDA additions.** The cp.async `*_gemm` launchers and the grouped / two-launch MoE launchers
  (`moe_experts_mma_bf16`, `moe_experts_decode_bf16`) are not ported (CUDA graphs are ported: `graph_capture` / `graph_replay`, `bad0084`); the HIP backend keeps its own WMMA GEMM and MoE kernels, whose registered launcher set is unchanged.
* **KV layout.** The contiguous separate K / V cache stays; radiance's paged interleaved layout is a deferred decision.
* **Radiance ports** (MXFP4 x FP8 GEMM, GDN, one-shot all-reduce): Phase 2b onward, see `docs/research/radiance-port-plan.md`.
* **TP=2** is done for dense attention + dense FFN models (Phase 2b, section below). Not done: GatedDeltaNet and MoE sharding (Qwen3.6-35B-A3B), FP8/NVFP4 weight sharding, an all-gather in the P2P kernel, radiance's fused all-reduce + RMSNorm.
* **Unifying with CUDA sources.** Candidates for `ik_portable.h`: `norm`, `argmax`, `attention` (SIMT) and `attention_splitkv` differ from the CUDA files by
  2 to 10 lines (stream type, shuffles, `__ldg`); `elementwise` by 34 (fp8 conversion, bf16 type). The blockers are the `IK_WARP` value (32 on
  gfx12 vs the header's 64), the shuffle mask width, the bf16 type, `ik_launch_status` / `IK_STREAM`, and the launch syntax in the runtime wrappers.
  GEMV, the GEMM and the flash kernel stay arch-specific.

## Open issues

* **Registry has no batch-invariance flag** (see "MoE and batch size"): `moe_experts` is registered `Deterministic`, which does not say that its result depends on T from 16 tokens up.
* **No per-kernel flag for the dense GEMV vs GEMM paths either**; the same T-dependence exists there and is older than this backend.

## Profiling

`INTARSIA_HIP_TRACE=1` records two stream events per launch (one just before, one just after, per stream) and prints, when the backend is dropped, the GPU time per
kernel (start to stop of that launch, event overhead of about a microsecond included) and, separately, the stream's idle time between launches (batch wall time minus summed
kernel time: host dispatch and bubbles). Earlier traces charged the time between consecutive events to the later kernel, which mixed the two; numbers from those
runs are "inter-launch wall time", not kernel time (see the fix pass in `docs/results/2026-10-13-phase2a-hip.md`). Use with `--profile-every 0`.

## Tensor parallelism: the P2P all-reduce (Phase 2b)

`--tp 2` on `generate`, `serve`, `accuracy`, `plan show` (and `bench speed --tp 2` for the record) opens HIP devices `0,1` (`--devices` overrides) with `intarsia_backend_hip::create_group`; design in [ADR-0006](../decisions/0006-tensor-parallelism.md), numbers in [the Phase 2b results](../results/2026-10-13-phase2b-tp.md). `--comm host` stages through host memory instead (slow, any backend).

`kernels/hip/allreduce.hip` (`ik_ar2_oneshot`, Rust side `crates/intarsia-backend-hip/src/comm.rs`) is a two-rank one-shot push all-reduce after radiance's `r4d_ar_oneshot_2rank_exact` (Apache-2.0, `kernels/hip/NOTICE-radiance`, `LICENSES/Apache-2.0-radiance.txt`). Rewritten for this ABI, not copied; it keeps the design, not the tuning:

* Each rank has a fine-grained (`hipExtMallocWithFlags(hipDeviceMallocFinegrained)`) scratch (2 slots x 4 MiB) and 16 flags, mapped into the peer (`hipDeviceEnablePeerAccess`), plus 16 launch counters in ordinary memory. A collective is one kernel launch per rank on the rank's own stream, from the rank's own thread: push the local partial into the peer's scratch, `s_wait_storecnt` (drain 3, default; `INTARSIA_HIP_AR_DRAIN=1` uses a system fence instead), raise the peer's per-block flag, spin on the own flag, add own partial and the pushed one in f32, round once.
* Scratch is double buffered on the launch counter's parity. 16 blocks, fixed, so both ranks' counters stay equal. Messages over 4 MiB split into several launches.
* The kernel reads a partial in f32 (default: the row-parallel matmuls write f32 through the `logits` op) and writes bf16 into the activation buffer, or reduces bf16 in place (`INTARSIA_TP_PARTIAL=bf16`).
* `ik_ag2_gather` (`ag2_kernel`, same file) is the rank-major all-gather on the same handshake, scratch, flags and counters (so gathers and all-reduces may follow each other in any order; both ranks wait for the peer's flag, which the slot-reuse argument needs): push the local words into the peer's scratch, flag, wait, then write the peer's words from the own scratch next to the local ones (skipped when the send is already this rank's segment of the receive buffer: in place). It carries the vocab-parallel lm_head's logits (`INTARSIA_TP_VOCAB`, ADR-0006); capture-safe like the all-reduce.
* A rank that never arrives makes the other trap after a bounded number of polls (`AR_SPIN_MAX`, about a minute by estimate; the timeout path was never exercised) instead of hanging the card.
* Not ported: the fused RMSNorm epilogue, the N-rank and two-shot variants, the tiered-int8 wire format.

`kernels/hip/ar_norm.hip` (`ik_ar2_add_rmsnorm`, `P2pComm::all_reduce_add_rmsnorm`) fuses that all-reduce with the residual add and the rmsnorm after it (`INTARSIA_TP_ARNORM=1`, off by default until its GPU gates are recorded; `scripts/tp2-hip/stages-arnorm.sh`). Same wire (exact: no lossy transfer), same scratch, flags and counters, so plain and fused collectives interleave and replay in a decode graph. Bit for bit the unfused pair: `y = bf16(a + b)`, `sum = bf16(y + residual)`, the norm reads `sum`, with the norm kernel's block size and reduction order (`tests/ar_norm.rs`). It applies where the norm after a block-end all-reduce is a standalone `RmsNormAdd` (the dense FFN and attention input norms, the final norm, blocks whose first GEMV pair is not folded); a norm already folded into the next block's GEMV stays folded. A message above one scratch slot keeps the unfused pair. The model defers each block-end reduction to the next op (`crates/intarsia-core/src/model/ar_norm.rs`); `plan show` prints how many of a decode step's all-reduces fuse.
