# Kernels

Kernel packs implement ops behind a torch-free C ABI (`kernels/abi/ik_abi.h`). They are compiled by
`nvcc` (or `hipcc`) into a static library and bound from Rust with `extern "C"`. Python is not
involved at run time.

```
kernels/
  abi/ik_abi.h        the C ABI: ik_tensor, ik_quant, status codes, launcher conventions
  cuda/               native CUDA C++ launchers (cuda backend)
  hip/                native HIP C++ launchers for gfx12 / R9700 (hip backend, docs/backends/hip.md)
  common/ik_portable.h  (planned) CUDA/HIP intrinsic shims shared by simple kernels
  packs/              vendored/ported packs (tensorfold, radiance, ...), each with its LICENSE
```

## Adding a kernel

1. **Write the launcher** in C++/CUDA with the ABI signature and the naming convention
   `ik_<op>_<variant>`, for example `ik_matmul_bf16_gemv`. The `<op>` part is the op id from
   `crates/intarsia-core/src/ops.rs` (`Op::as_str`). Rules:
   * raw `ik_tensor` pointers in, an `int` status out (`IK_OK` or an `IK_ERR_*`), an explicit
     `void* stream`, and **no allocation** inside the launcher (the engine owns all memory);
   * asynchronous on `stream`; never synchronise the device;
   * validate dtype, rank and strides and return `IK_ERR_DTYPE` / `IK_ERR_SHAPE` / `IK_ERR_STRIDE`
     rather than faulting; tensors are dense row-major in Phase 0;
   * the per-op tensor layout and attributes are tabulated in `ops.rs`; the cpu-ref implementation in
     `crates/intarsia-backend-cpu/src/reference.rs` is the executable specification.

2. **Register the implementation** from the backend crate's `Backend::kernels()` list, with a
   shape predicate, an exactness class and a tolerance against the reference:

   ```rust
   register_kernel! {
       op: Op::Matmul, name: "matmul_bf16_gemv", backend: "cuda",   // backend == Backend::name()
       arch: ["sm_121"], acts: [DType::Bf16], weights: [DType::Bf16],
       when: |s| s.m <= 8 && s.k % 64 == 0,      // preconditions are code, checked at plan time
       exact: Exactness::Deterministic,           // Fast | Deterministic | Exact
       tol: Tol::new(2e-2, 5e-3),                 // max_abs (relative to output scale), rel_l2
   }
   ```

   `name` is the launcher name without the `ik_` prefix (`ik_matmul_bf16_gemv` -> `matmul_bf16_gemv`).
   `Backend::launch(name, call, stream)` then maps `name` to the launcher and fills the `ik_tensor`
   structs from `call.inputs/outputs/attrs`. Registration is an explicit list, not link-section
   magic, because that breaks with nvcc static libraries.

3. **Run the admission test**:

   ```sh
   cargo run -p intarsia-cli --features cuda -- kernels test --backend cuda --model <hf-dir>
   cargo run -p intarsia-cli -- kernels list --backend cpu
   ```

   `kernels test` exits non-zero when no kernel ran a single case for the requested backend (a filter that
   matched nothing, a stub build, ...) and prints `SKIPPED: kernel <name> ran 0 cases` for any kernel that ran
   none; a run that exercised nothing is not a pass. The GPU `cargo test` files in
   `crates/intarsia-backend-cuda/tests/` print `SKIPPED:` when there is no CUDA device (or when the crate was built
   without CUDA, `tests/gpu_skipped_report.rs`) and fail instead under `CI=1` or `INTARSIA_REQUIRE_GPU=1`; read
   "ok" from a Mac `cargo test` as "no GPU code ran".

   Every registered implementation runs against the cpu-ref reference over a shape sweep that
   includes the model's real shapes and reports `max_abs`, `rel_l2` and pass/fail. A failing
   kernel is marked rejected and is unselectable on the host: this catches "compiles for sm_120,
   runs garbage on sm_121". Shape predicates that exclude every sweep case leave the kernel
   `NO CASES (untested)`, so add the shapes to the sweep (`intarsia_core::admission::sweep`).

Tolerances: `max_abs` is relative to `max(1, max|reference|)` of the output. `Exact` requires
bit-identical results (argmax lowest-index tie-break, embedding, kv_append); `Deterministic` means
run-to-run identical but allowed to differ from the serial reference within tolerance; `Fast` may
vary between runs (atomics, split-K).

## Adding a kernel and its test

A registered kernel must be exercised by `intarsia kernels test --quick` on every backend that registers it. A kernel with
no quick case shows up as `NO CASES (untested)` in the sweep, which is not a failure, so it would otherwise go unnoticed.
Three things make that checkable:

1. **The case comes from the sweep, the tolerance from the registration.** Cases live in `intarsia_core::admission::sweep`
   (`crates/intarsia-core/src/admission.rs`) and, for the quantised and MoE ops, `crates/intarsia-core/src/quant/cases.rs`. The
   admission harness compares the kernel with the cpu reference using the `tol` (and `tol_f32`) declared in `register_kernel!`;
   a case never carries a tolerance of its own. Do not widen a tolerance to make a new case pass.
2. **Make a quick case that your shape predicate accepts.** Read your `when` closure: a threshold such as `s.m >= 256`,
   `s.k % 128 == 0`, `head_dim` in `{64, 128}` or `s.m > 16` needs a quick-sweep shape that satisfies it. Add the smallest such
   shape under the `if quick` branch of the op's sweep (keep the full sweep's cases and random data unchanged: new cases are
   appended). Dtype filters (`acts`, `weights`) count too.
3. **Run the guard.** `cargo test -p intarsia-backend-<cuda|hip|metal> --test quick_coverage` needs no GPU. It calls
   `intarsia_core::admission::kernels_without_quick_case` on the backend's `kernels::all()` and fails, naming the kernels, when
   one has no accepted quick case (the architecture filter is ignored, so a Mac checks CUDA and HIP kernels as well). The
   one exception is a kernel whose predicate depends on a build-time cfg (`moe_experts_nvfp4_mma_bf16` needs `ik_sm120a`);
   the CUDA test skips it when that cfg is off.

Then run the sweep itself on a GPU box (inside a fleetq slot): `intarsia kernels test --backend <cuda|hip|metal> --quick`
must print no `FAIL` row and no `NO CASES`.

## Scratch workspace

"No allocation inside the launcher" also covers kernels that need temporary global memory (split-KV
attention partials, split-K reductions, routing tables). The engine allocates it once and passes it in:

1. **Declare the size on the Rust side**, as a function of the call's shape, next to the registration:

   ```rust
   register_kernel! {
       op: Op::AttnDecode, name: "attn_decode_splitkv_bf16", backend: "cuda",
       arch: ["sm_121"], when: |s| s.kv_len >= 1024,
       // partial (max, sum, acc[D]) per (head, split): f32
       scratch: |s| s.n_heads * 16 * (s.head_dim + 2) * 4,
   }
   ```

   `ShapeInfo` carries `m, n, k, n_heads, n_kv_heads, head_dim, kv_len` and the dtypes.
2. **Take it in the launcher**: `const ik_scratch* scratch` (see `ik_abi.h`) is the parameter right
   before `stream`. Check `scratch && scratch->bytes >= need` and return `IK_ERR_INVALID_ARG`
   otherwise; the contents are uninitialised and do not survive the launch.
3. **Read it in `Backend::launch`**: `call.scratch` is `Some(TensorRef)` (a `u8` view of at least the
   declared bytes, 256-byte aligned) exactly when the selected kernel declared a requirement for this
   shape; map it to `ik_scratch { data, bytes }`.

Plan resolution (below) takes the maximum over every shape the model uses (decode at the first and
last context position, every prefill size probed) and reserves one buffer at load, so there is no
allocation on the serving path and a CUDA graph can capture the launch. All ops share the buffer, which
is safe because ops run in stream order. The admission harness (`run_host`) allocates a private
workspace of the declared size for each case, so `intarsia kernels test` exercises it too.

## Plan resolution and admission at load

When a model loads, `intarsia_core::plan::resolve` runs the real forward path with a recorder in place of
the launcher, for decode (`T = 1`, first and last context position) and prefill chunks of every size
that can occur (all sizes up to 16, then each power of two and its neighbours up to the chunk size,
because predicates switch on `m`). It selects a kernel for every `(op, shape)` and:

* fails the load, listing every `(op, shape)` without a kernel and where it is needed;
* sizes the scratch workspace;
* describes the plan: `intarsia plan show --model DIR --backend cuda [--json]`, `Engine::plan()`,
  `GET /v1/intarsia/plan` (contract on `plan::plan_json`).

`--admit` (always on in `intarsia serve`, together with `require_admitted`) additionally runs every
selected kernel against the cpu reference on the plan's own shapes and marks it admitted or rejected; a
rejected kernel is replaced by the next candidate or the load fails with the reasons. To bound the cost
of the serial f32 reference, rows are clamped to 32, matmul/logits `N` to 16384 and attention positions
to 2000 (`K`, head counts, head dims, dtypes and attributes are exact). Results are cached per host
fingerprint (backend, device, arch, driver/toolkit when the backend reports them, and the executable's
size and mtime, standing in for the kernel-pack hash) in `~/.cache/intarsia/admission/`. Ops without an
exact-shape case builder (`admission::case_from_call`) fall back to the kernel's own sweep, once per
kernel. `--reference` selects only `ref_*` kernels (cpu); backends without reference kernels fail the
load instead of falling back.

## Dtype codes

`IK_DTYPE_*` values in `ik_abi.h` equal `intarsia_core::DType`'s `repr(i32)`; a unit test parses
the header and fails if they drift. Add new dtypes at the end.

## Quantised weights and MoE

HIP ports (gfx1201): `kernels/hip/{fp8,nvfp4,moe,gdn}.hip` with `ik_quant.h`, `ik_dq_gemm.h`, `moe_wmma.h`; see `docs/results/2026-10-13-phase2a-hip.md` section 9. RDNA4 has no fp4/fp8 conversion instructions, so e2m1 is a 16-entry LDS table and e4m3 is decoded with integer arithmetic.

`fp8.cu`, `nvfp4.cu`, `moe.cu`, `moe_grouped.cu`, `moe_decode.cu`, `fp8_gemm.cu`, `nvfp4_gemm.cu`, `wq_gemm.cuh` and `ik_quant.cuh` (fp4/e4m3 decode helpers, ideas from TensorFold's NVFP4 kernels, MIT, see `LICENSES/TensorFold-MIT.txt`). Registered in `crates/intarsia-backend-cuda/src/quantmoe.rs`. Formats, semantics, numerics and measured speeds: `docs/quant-moe.md`.

## Attention (`attention*.cu`)

Causal GQA over the contiguous bf16 KV cache `[Nkv, cap, D]`. Three families, all registered in
`crates/intarsia-backend-cuda/src/kernels.rs`; the plan picks by priority and shape:

| kernel | op | when | notes |
|---|---|---|---|
| `attn_prefill_bf16`, `attn_decode_bf16` | prefill / decode | any `D <= 256`, even | SIMT, one block per (row, head). Priority 0: the fallback and the simplest reference of the cuda pack |
| `attn_decode_bf16_splitkv`, `attn_prefill_bf16_splitkv` | decode / prefill | `T <= 16` rows, `D` in 32/64/128/256, `N/Nkv <= 8` | flash-decoding. Grid (split, kv head, row); one block serves all q heads of a kv head, so K/V are read once per group. Partials (f32 `acc[D]`, `m`, `l`) go to the **scratch workspace**, a second kernel merges them. The number of splits is a function of `kv_len` only (`splitkv_parts`, mirrored in the `.cu`), so results are run-to-run identical. Also serves speculative-decoding verify (`q_len > 1`) |
| `attn_prefill_bf16_flash` | prefill | `T > 16`, `D` in 64/128/256 | FlashAttention-2 style on `mma.sync.m16n8k16` bf16 with f32 accumulate; 64 query rows per block, `cp.async` K/V tiles, Q in registers (`D <= 128`) or smem (`D = 256`). Blocks run longest-first. No scratch |

Measured on GB10 (`cargo test --release -p intarsia-backend-cuda --test attention -- --ignored --nocapture`):
decode reads KV at about the copy bandwidth (250 GB/s at 16k keys, 8B shape; 225 GB/s at 64k keys for the
`D = 256`, 8 q heads per kv head hybrid shape); flash prefill reaches 61-65 TFLOPS (`D = 128`) and 44
TFLOPS (`D = 256`). `INTARSIA_CUDA_DISABLE=name1,name2` hides kernels from the registry for A/B runs.
`tests/attention.rs` (default test) sweeps random GQA group sizes, head dims, row counts and cache offsets
against the cpu reference and checks that repeated launches are bit-identical.
