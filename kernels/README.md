# Kernels

Kernel packs implement ops behind a torch-free C ABI (`kernels/abi/ik_abi.h`). They are compiled by
`nvcc` (or `hipcc`) into a static library and bound from Rust with `extern "C"`. Python is not
involved at run time.

```
kernels/
  abi/ik_abi.h        the C ABI: ik_tensor, ik_quant, status codes, launcher conventions
  cuda/               native CUDA C++ launchers (cuda backend)
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

   Every registered implementation runs against the cpu-ref reference over a shape sweep that
   includes the model's real shapes and reports `max_abs`, `rel_l2` and pass/fail. A failing
   kernel is marked rejected and is unselectable on the host: this catches "compiles for sm_120,
   runs garbage on sm_121". Shape predicates that exclude every sweep case leave the kernel
   `NO CASES (untested)`, so add the shapes to the sweep (`intarsia_core::admission::sweep`).

Tolerances: `max_abs` is relative to `max(1, max|reference|)` of the output. `Exact` requires
bit-identical results (argmax lowest-index tie-break, embedding, kv_append); `Deterministic` means
run-to-run identical but allowed to differ from the serial reference within tolerance; `Fast` may
vary between runs (atomics, split-K).

## Dtype codes

`IK_DTYPE_*` values in `ik_abi.h` equal `intarsia_core::DType`'s `repr(i32)`; a unit test parses
the header and fails if they drift. Add new dtypes at the end.
