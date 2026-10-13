# ADR-0008: The 35b-short vLLM-golden gate is advisory; per-backend prefill chunk, HIP decode, prefix-reuse tie rule, test environment

Status: Accepted (operator decision, 2026-10-13: "Accept the `35b-short` gate failure as known noise; record it in an ADR and mark the gate advisory for this model. Kernel-level error checks stay the real guard.")

Scope: the `git log` range ending at `42d90d5` plus the gate change in this commit. Sources: `docs/results/2026-10-14-35b-accuracy.md` (items 1 to 5, "35b-accuracy.md" below), `2026-10-13-prefill-cuda.md`, `2026-10-13-hip-decode.md`, `2026-10-14-hip-decode-2.md`, `2026-10-13-prefix-reuse.md`, `docs/accuracy.md`. All results documents are under `docs/results/`. Accepting these decisions does not make the measurements better than they are; the open items are listed per decision.

| # | Decision |
|---|---|
| 1 | The `35b-short` vLLM-golden gate is advisory on Qwen3.6-35B-A3B-NVFP4 |
| 2 | Per-recipe, per-backend `prefill_chunk`; fast CUDA prefill kernels stay opt-in |
| 3 | HIP decode: graph replay, v2 kernels, fusion ops |
| 4 | Prefix-reuse gate: a rounding tie is judged on the logits |
| 5 | The kernels sweep skips zero-dim cases for MoE-only models |
| 6 | `INTARSIA_CPU_FREE_MEM_GB` for the macOS free-memory reading |

## 1. The `35b-short` vLLM-golden gate is advisory

**Context.** After the grouped MoE mma kernel, `35b-short` against the vLLM golden fails its decode aggregates (mean KL 1.506e-2 > 1.043e-2, tail-bucket KL 1.556e-2 > 1.102e-2, max KL 1.329 > 0.602, `chat_multiturn` 8.66e-2 > 2.92e-2) and has failed the same way since phase1b 10.8. `35b-accuracy.md` asked whether this is a CUDA defect or noise, using the CPU, HIP and CUDA-variant implementations of the same weights as the judge (the vLLM golden is itself a bf16 / NVFP4 / fp8-KV implementation and not a vote).

**Evidence** (`35b-accuracy.md` section 4, items 1 to 5). All four failures are noise, and the fifth item is a comparison error:

* Item 1, decode aggregates: HIP, SIMT attention and (by 1 %) CPU fail or graze the same fixed limits. The old numerics scored on their prefill path at the same 160 positions give mean KL21 1.371e-2 / max 1.139, against 9.58e-3 / 0.241 on the decode path. 8.3e-3 of HEAD's 1.556e-2 mean is the single position of item 2.
* Item 2, `chat_multiturn` decode KL 1.33 at one position: a fragile `assistant` / `<think>` logit. P(`<think>`) ranges 0.008 to 0.132 across equally valid paths of the same weights; vLLM gives 0.388 in this golden and 0.233 in its earlier run.
* Item 3, long-decode KL 5.6e-3 old vs new: HIP is 9.0e-3 from HEAD and 8.8e-3 from the old numerics, further than the two CUDA builds are from each other (5.07e-3).
* Item 4, `chat_reuse_4k`: HIP reproduces the vLLM gap (KL21 0.268, top-1 87.8 %); CUDA vs HIP on this prompt is 0.136, the size of a chunk perturbation (confirms faebb39).
* Item 5, HIP "99.38 % / 9.96e-3": measured against the older golden sha `89d650a8`. Against the golden every CUDA number uses (`b6b51e1b`) HIP is decode top-1 98.75 %, tail-bucket KL 1.242e-2, mean KL 1.132e-2.

**Two measured-noise gates were rejected by their own pre-committed criteria** (no gate was changed):

* Aggregate-envelope rule (pre-registered in 55ea0bc, rejected in 5c5975a): a CPU run with attention output x1.01 (a deliberate bug) is closer to the vLLM golden than the correct CPU run (decode tail-bucket KL 1.037e-2 to 8.39e-3), so the vLLM-relative gate has no resolution at that scale.
* Same-weights CPU-reference gate (pre-registered in 5c5975a): a 5 % attention-output bug passes it (decode mean KL21 1.182e-2 against a budget of 2.06e-2; prefill 6.82e-3 against 9.67e-3), separating from the correct implementations by only about 1.4x. Section 3.6 and 5 of the results document.

So loosening the `35b-short` gate would not detect bugs: a deliberate 5 % attention-output bug passes both gates, and a 1 % MoE-output bug has the same vLLM decode signature as HEAD (tail KL 1.600e-2, max 1.345 against HEAD's 1.556e-2, 1.329). End to end, nothing available resolves a 1 % MoE error.

**Decision.** The `35b-short` baseline gate against the vLLM golden is advisory for this model. `scripts/phase1b-acceptance.sh` reports a FAIL there as `ADVISORY FAIL (ADR-0008)` and does not fail the accuracy stage. Every other gate stays blocking: the `35b-short` floor, prefix reuse, same-weights self-golden and run-to-run determinism, all of `35b-long`, and the 8B fp32 gates. The real guards are the kernel-level accuracy tests (`kernels test`, the CUDA/HIP kernel tests, the grouped MoE sweep) and the `INTARSIA_CUDA_DISABLE` bisect sweep: disabling `moe_experts_mma_bf16` reproduces the old build bit for bit, and the W2 sweep bounds that kernel at rel_l2 1e-4 to 4e-4 against the f64 reference over 192 cases.

**Recorded consequences.**

* The grouped MoE mma kernel is about 2x further from the exact result than the old kernel at 32 to 64 tokens (median off-reference fraction 0.105 % against 0.052 %; rel_l2 median ratio 1.32). 40 of 192 random cases break the bounds of `moe_experts_mma_is_as_accurate_as_old_kernel`, so that test's claim holds for its seed (T = 128, 512), not in general. Not fixed; no new kernel test was added.
* The earlier HIP figure of 99.38 % top-1 was against the old golden sha `89d650a8`; against `b6b51e1b` it is 98.75 %. The old golden file on tatooine (`/models/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors`) should be retired or renamed.
* The HIP decode path is further from the CPU reference than the CUDA variants (KL21 1.03e-2 against 5.2e-3 to 6.7e-3); not investigated.

**Revisit trigger.** A per-layer CPU-reference gate (estimate 3 h), which compares layer outputs rather than end-of-network distributions and so is not limited by the fragile-logit amplification measured above. Other options from the results document: many more positions per mode, a statistic that drops near-tie positions, or k chosen from a calibrated false-pass rate over several known-bad mutants. Remove the advisory handling in `stage_accuracy` when one of these lands.

## 2. Per-recipe, per-backend `prefill_chunk`; fast CUDA prefill stays opt-in

**Decision.** The prefill chunk default comes from the recipe and the backend (63b0f72): the Qwen3.6-35B-A3B uses 2048 on CUDA and 256 elsewhere. The acceptance script keeps `--prefill-chunk 256` on its MoE rows because the baselines were recorded at 256. The GDN-lanes and fp8 tensor-core GEMM prefill kernels are registered, tested and admitted, but opt-in behind `INTARSIA_CUDA_FAST_PREFILL=1` (priority -10 otherwise); they came in with the merge 8626dcd (`2026-10-13-prefill-cuda.md`). Enabling them by default needs an owner decision and a re-calibrated envelope that includes GEMM and recurrence rounding.

## 3. HIP decode

* Graph replay and the v2 decode kernels: merge bad0084 (`2026-10-13-hip-decode.md`).
* Fusion ops `MatmulPair`, `MatmulFp8Pair`, `MatmulNvfp4Glu`, `GatedRmsnorm`: new core ops that only the HIP backend implements; the model asks the registry for the fused kernel and falls back to the unfused ops when there is none. CUDA is untouched. `INTARSIA_NO_FUSION=1` turns fusion off on HIP; it is also off under tensor parallelism. Merge 42d90d5 (`2026-10-14-hip-decode-2.md`): 897 to 677 launches per decode step.
* 35B R9700 decode: 97 tok/s (before) to 126 (bad0084) to 135 (42d90d5; 125.9 to 135.2 tok/s at 128 prompt tokens).
* **Known gap.** HIP failed-capture recovery does not work: after an invalidated graph capture the device keeps returning `hipErrorStreamCaptureInvalidated` (901) for later calls, so a capture that fails would leave the HIP backend unusable although `graph_step.rs` would switch graphs off. Two attempts failed and stopped (`2026-10-13-hip-decode.md`); neither the 35B nor the 8B fails a capture. Graphs are off at TP > 1 (`GraphState::new(opts.graphs && tp.world == 1)`), so the TP path never reaches it; no TP run was made on that branch.

## 4. Prefix-reuse tie rule

**Decision** (1c60449, cd579cb; `2026-10-13-prefix-reuse.md`). A diverging warm/cold pair passes if it is a rounding tie judged on the logits (`judge_streams`, `judge_divergence`), or if it agrees on `--reuse-min-match` leading tokens as before. The logit criterion only rescues a pair, never fails one the leading-token rule passed. A tie is a divergence where every shared row up to and including the divergent one agrees within `--reuse-logit-tol` (default 0.25 nats over the top-5) and `--reuse-kl-tol` (KL(cold || warm) <= 1e-2). The tolerances are calibrated on three ties of one model, shared rows moving 0.09 to 0.14 nats at most, and were checked against a deliberately broken engine; the margin is judged, not measured on a real bug.

## 5. Zero-dim sweep fix for MoE-only models

`kernels test` skips zero-dim sweep cases for MoE-only models, because the CPU reference errors on zero dims (0b0b5ad).

## 6. `INTARSIA_CPU_FREE_MEM_GB`

On this macOS 27 machine `sysinfo::available_memory()` reports 1.8 GB (128 GB machine, about 70 GB unused), which blocks every test that loads a real model ("only 0 MiB free" in `intarsia-models` `engine` and `intarsia-cli` `real_model`). `INTARSIA_CPU_FREE_MEM_GB` (cpu backend only, off by default; f305c79) overrides the reading. **`cargo test --workspace` on the Mac needs `INTARSIA_CPU_FREE_MEM_GB=8`**; with it every test passes, without it the workspace is not green. The failure is pre-existing and environmental (same on a clean worktree of `d256c56`). **Open item:** fix the reading itself so the variable is not needed.
