# ADR-0002: Torch-free C kernel ABI with TensorFold and radiance as first packs

_Drafted by fleet model `guidance` from the raw note; checked against the note by Claude 2026-10-12_

## Status
Accepted

## Context
IntarsiaML requires a unified interface for GPU kernels to support multiple backends and languages.

## Decision
All GPU kernels, ours and ported, sit behind one torch-free C ABI. The ABI is defined in `kernels/abi/ik_abi.h` with `ik_tensor` containing raw pointer, dtype, shape, stride. Launchers are `extern "C"`. Streams are explicit. No allocation happens inside kernels. Kernels compile with `nvcc`/`hipcc` into a static library per backend. Rust binds via `extern "C"`.

## Alternatives considered
Importing `flashinfer`, `cutlass`, `ggml`, `ck/aiter`, `exl3` as packs. Rejected because all are torch- or ggml-coupled ports. Add only when a measured gap names one. Generic fusion compiler. Rejected because it is a compiler project.

## Consequences
First packs are `tensorfold` (MIT, `/opt/code/TensorFold`: EXL3, NVFP4 qmmf, GDN, DFlash verify, attention; holds the Spark bar 83.4 tok/s Qwen3.8-27B EXL3+DFlash2) and `radiance libr4d` (Apache-2.0 per `codeberg.org/StillDeadcode/radiance LICENSE + NOTICE`, checked 2026-10-12; MXFP4xFP8 GEMM, one-shot all-reduce, GDN, paged attention; holds the R9700 bar 207-226 tok/s Qwen3.8-27B TP=2). `cuBLASLt`/`hipBLASLt` registered as `Exactness::Fast` only. Block-scope fused ops (`gdn_layer_decode_fused`, `all_reduce_rmsnorm_fp8`) are first-class registry entries with the generic op chain as reference and fallback. No generic pattern-rewriting fuser.

## Evidence
Fable found 21 of 28 TensorFold kernel files take `torch::Tensor`. `r9700-engine/DESIGN.md` estimated 7-9 weeks for one model on one platform reusing kernels.

## References
- `docs/reviews/2026-10-12-fable-adversarial-review.md`
- `r9700-engine/DESIGN.md`

<details><summary>Raw note</summary>

Decision (accepted by Claude 2026-10-12 after Fable 5.1 adversarial review, docs/reviews/2026-10-12-fable-adversarial-review.md): all GPU kernels, ours and ported, sit behind one torch-free C ABI (kernels/abi/ik_abi.h: ik_tensor with raw pointer, dtype, shape, stride; extern "C" launchers; explicit stream; no allocation inside). Kernels compile with nvcc/hipcc into a static library per backend, bound from Rust via extern "C".
First packs: tensorfold (MIT, /opt/code/TensorFold: EXL3, NVFP4 qmmf, GDN, DFlash verify, attention; holds the Spark bar 83.4 tok/s Qwen3.8-27B EXL3+DFlash2) and radiance libr4d (Apache-2.0 per codeberg.org/StillDeadcode/radiance LICENSE + NOTICE, checked 2026-10-12; MXFP4xFP8 GEMM, one-shot all-reduce, GDN, paged attention; holds the R9700 bar 207-226 tok/s Qwen3.8-27B TP=2). cuBLASLt/hipBLASLt registered as Exactness::Fast only.
Block-scope fused ops (gdn_layer_decode_fused, all_reduce_rmsnorm_fp8) are first-class registry entries with the generic op chain as reference and fallback; no generic pattern-rewriting fuser.
Alternatives: importing flashinfer, cutlass, ggml, ck/aiter, exl3 as packs (rejected for now: all torch- or ggml-coupled ports; add only when a measured gap names one); generic fusion compiler (rejected: compiler project).
Evidence: Fable found 21 of 28 TensorFold kernel files take torch::Tensor; r9700-engine/DESIGN.md estimated 7-9 weeks for one model on one platform reusing kernels.

</details>
