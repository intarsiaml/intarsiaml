# ADR-0001: Rust host with AOT CUDA and HIP kernels

_Drafted by fleet model `guidance` from the raw note; checked against the note by Claude 2026-10-12_

# Architecture Decision Record: Rust host with AOT CUDA and HIP kernels

## Status
Pending

## Context
Native radiance (C++/HIP host) starts in 11 s vs 160-165 s for vLLM with equal or better step time (20.25 vs 20.47 ms) (docs/research/amd-r9700-lessons.md). Cold Triton compiles take ~2.5 min on GB10, AOT avoids that. Rust has tokenizers, safetensors, minijinja, llguidance, axum, ratatui, prometheus-client, cudarc.

## Decision
Host language is Rust; GPU kernels are AOT-compiled CUDA C++ (sm_121 for GB10) and HIP C++ (gfx1201 for R9700) with a shared ik_portable.h for simple kernels; inline PTX/GCN asm only at profiled hot spots; Python only in offline tools (golden logits, plots), never in the serving path.

## Alternatives considered
C++ host (works for radiance; slower to build server/UI/packaging), Zig host (mlx-serve ~7 MB binary; thin ML library ecosystem), Python host (step overhead, startup, heavy containers), Mojo (TensorFold-mojo worktrees never merged).

## Consequences
One static binary per platform; kernel packs (ggml MIT, flashinfer Apache-2.0, cutlass BSD-3, exl3 MIT, ck/aiter MIT) imported behind op signatures; libr4d not vendored (no license file).

## Evidence
Decision (agent-proposed 2026-10-12, pending Fable adversarial review).

<details><summary>Raw note</summary>

Decision (agent-proposed 2026-10-12, pending Fable adversarial review): host language is Rust; GPU kernels are AOT-compiled CUDA C++ (sm_121 for GB10) and HIP C++ (gfx1201 for R9700) with a shared ik_portable.h for simple kernels; inline PTX/GCN asm only at profiled hot spots; Python only in offline tools (golden logits, plots), never in the serving path.
Why: native radiance (C++/HIP host) starts in 11 s vs 160-165 s for vLLM with equal or better step time (20.25 vs 20.47 ms) (docs/research/amd-r9700-lessons.md). Cold Triton compiles take ~2.5 min on GB10, AOT avoids that. Rust has tokenizers, safetensors, minijinja, llguidance, axum, ratatui, prometheus-client, cudarc.
Alternatives: C++ host (works for radiance; slower to build server/UI/packaging), Zig host (mlx-serve ~7 MB binary; thin ML library ecosystem), Python host (step overhead, startup, heavy containers), Mojo (TensorFold-mojo worktrees never merged).
Consequences: one static binary per platform; kernel packs (ggml MIT, flashinfer Apache-2.0, cutlass BSD-3, exl3 MIT, ck/aiter MIT) imported behind op signatures; libr4d not vendored (no license file).

</details>
