# Adversarial review of ARCHITECTURE.md v0.1 (Fable 5.1, 2026-10-12)

Requested by the user. Reviewer read the plan, both research docs and the cited source trees; read-only. Summary as delivered (lightly condensed). Disposition of each item is at the end.

## 1. Assumption most likely to sink it

"Kernel packs are imports." Every listed pack is a port: 21 of TensorFold's 28 kernel files take `torch::Tensor`; ExLlamaV3 and aiter are torch extensions; flashinfer's usable surface is torch wrappers; ggml-cuda is bound to `ggml_tensor`; CK on gfx1201 is thin. Each needs a torch-free launcher shim, layout adapter and admission tests. The operator's own `r9700-engine/DESIGN.md` estimated 7-9 weeks for one model, one platform, reusing kernels. Realistic failure mode: a correct BF16 engine with a nice registry and no fast path.

## 2. Must-fix (ranked)

1. Packs are ports: define a torch-free C kernel ABI (`extern "C"` launchers: raw pointers, strides, scales, stream). First CUDA pack: TensorFold (MIT; EXL3, NVFP4 qmmf, GDN, DFlash verify, attention). Drop flashinfer/cutlass/ck/aiter/ggml until a measured gap names one.
2. Phase order inverts difficulty: both bars (Qwen3.8-27B: 48 GDN + 16 attention layers, DFlash2, EXL3/MXFP4, TP=2) need GDN state, q_len>1 verify attention, M≤8 quant GEMV. Design the KV manager with recurrent state from day 1; Phase 1 model = Qwen3.8-27B or 35B-A3B on the bar's quant.
3. Check radiance's license today; if permissive it is the HIP pack.
4. Generic pattern-rewrite fusion is a compiler project. Make block-scope fused ops first-class registry entries (e.g. `gdn_layer_decode_fused`) with the generic op chain as fallback.
5. Rust HIP bindings are thin (no RCCL, no hipBLASLt crate): own a hipcc-built C shim bound via `extern "C"`. Verify cudarc on CUDA 13 arm64.
6. Golden-logit gate is on the wrong side: HF BF16 on a 27B/35B on Spark is slow and day-0 models may lack an HF implementation. Quick pass gates on greedy sanity + tiny perplexity; HF golden runs in the deep pass and is skippable with the skip recorded.
7. cuBLASLt picks algorithms by shape and graph capture freezes one: register library GEMMs as `Fast` only. The step watchdog must be a process supervisor (wedged CUDA contexts have sticky errors).
8. Anthropic `/v1/messages` and `/v1/responses` are API surface, not seams: move to Phase 1.

## 3. Keep

- Kernel registry with cpu-ref oracle, admission tests, plan-time shape predicates, no silent reference-on-GPU fallback.
- `QuantFormat` with `dequant_ref`.
- Rust host, AOT kernels, no Python in serving, one static binary.
- Run manifest, per-fingerprint tuning DB, localeval/fleetq contracts.

## 4. Phase 0 mis-scoped (~3× too big)

Cut web UI, Anthropic API, container, paged KV. Keep workspace, cpu-ref, cuda via C-ABI shim, registry, recipe loader, golden compare, localeval speed. Add: same binary runs Qwen3-0.6B on cpu-ref on the Mac and matches golden. "Numbers next to vLLM" is a smoke test only. Measured-memory allocation belongs in Phase 0.

## 5. Verdict

Agree: Rust host + AOT CUDA/HIP C++, inline asm only at profiled hot spots. Disagree with packs as enumerated: TensorFold and radiance are the two that matter; a torch-free C ABI is what makes packs swappable.

## Disposition (Claude, 2026-10-12)

| Item | Decision |
|---|---|
| 1 C ABI + TensorFold first | **Accepted.** `kernels/abi/ik_abi.h`; TensorFold (MIT) is the first CUDA pack. |
| 2 Hard kernels early | **Accepted.** KV/state manager has recurrent-state slots in Phase 0; Phase 1 target is Qwen3.6-35B-A3B NVFP4 (on spark1 disk) then Qwen3.8-27B. |
| 3 Radiance license | **Checked: Apache-2.0**, incl. `libr4d` (codeberg.org/StillDeadcode/radiance, `LICENSE` + `NOTICE`). Radiance becomes the first HIP pack; keep NOTICE. |
| 4 Block-scope fused ops | **Accepted.** |
| 5 HIP C shim | **Accepted**; same shim pattern for CUDA (we link our own `.cu` launchers, cudarc only for driver/NCCL if it builds on CUDA 13 arm64). |
| 6 Golden in deep pass | **Accepted.** |
| 7 cuBLASLt = Fast; supervisor watchdog | **Accepted.** |
| 8 Anthropic to Phase 1 | **Partly.** Responses API → Phase 1. Anthropic `/v1/messages` stays in Phase 0 because it is requirement #6 and costs one parallel agent that blocks nothing. Web UI → Phase 1; Phase 0 ships `intarsia top` (one TUI file) on `/events`. |
| Phase 0 cuts | **Accepted**: no container (plain `cargo build` on spark1), no paged KV, no web UI. Added the Mac cpu-ref golden gate and measured-memory allocation. |
