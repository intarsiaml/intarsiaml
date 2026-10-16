# Spark (GB10), TensorFold, mlx-serve and fleet-tooling lessons

Source: survey of `/opt/code/TensorFold`, `/opt/code/mlx-serve`, `/opt/code/deepeval`, `/opt/code/fleetq`, `/opt/code/missioncntrl_` and the unanimis vault, 2026-10-12.

## The bar to beat on GB10

| Setup | Result | Source |
|---|---|---|
| TensorFold, Qwen3.8-27B EXL3 3.00bpw + DFlash2, 1 Spark | 83.4 tok/s code (sampled), 64.5 greedy | `TensorFold/docs/recipes/qwen3.8-27b.md:114-130` |
| TensorFold, NVFP4 27B | 47 code / 38 chat | same, :147-165 |
| vLLM MTP=3, NVFP4 27B | 23-26 tok/s | same |
| TensorFold, 16 streams | 161.7 tok/s at 25.4 GiB vs vLLM 132.1 at 51.7 GiB | same, :174-180 |
| TensorFold cold prefill (FP8 path) | 1,240-1,870 tok/s, 2k-64k | same |
| 2× Spark TP=2 (TensorFold) | 82.4 code / 71 chat-greedy | same, :189-200 |
| vLLM, Qwen3.6-35B-A3B NVFP4 | 86-90 tok/s solo | unanimis 6e91a8c5 |

Measured GB10 bandwidth ≈ 221 GB/s (unanimis 6a7efe40). Use 221, not the datasheet, for floors.

## Lessons

1. Decode is bandwidth-bound: bytes/token is the lever; drafting multiplies it.
2. Build for `sm_121` explicitly (`sm_120` compiles but does not run; unanimis c1fd18d9).
3. vLLM NVFP4 MoE on sm_121: CUTLASS path emits garbage; marlin costs ~30%; a cuDNN FP8 GEMM via FlashInfer hangs after ~250k steps while `/metrics` still answers. → An engine needs a step-counter watchdog and its own kernel choice per arch, validated against a reference.
4. Unified memory: nvidia-smi reports no GPU memory on GB10; budget from host RAM; measure, do not estimate.
5. Exactness as a contract (TensorFold): multi-row verify must match serial bit-for-bit; fixed reduction trees; rank-ordered gathers; no TF32. Makes drafts, chunking and TP reproducible. Costs some prefill speed → make it a mode, not a global rule.
6. Speed deltas under ~10% are noise unless median-of-N with fresh-nonce prompts and forced lengths.
7. Cold Triton/extension compiles take minutes on GB10 → AOT-compile kernels, persist caches.
8. TP=2 over ConnectX: same config both nodes, `/dev/infiniband`, memlock unlimited, NCCL env; one fleetq job across both hosts.
9. Thinking mode and structured-output defaults change scores; record them in every result.
10. Per-family CUDA engines multiply effort per model (TensorFold) — the main thing IntarsiaML must avoid. Late concurrency support was the other big cost.

## What worked elsewhere

- TensorFold: per-family packages declaring backends; per-family kernel versioning with fingerprints; recipe docs with public prompts; memory-based admission control; exact drafting (DFlash2, MTP, n-gram).
- mlx-serve (Zig, ~7 MB binary, no Python): OpenAI + Anthropic + Responses + Ollama, `/metrics` + `/metrics.json`; fast start. Scope sprawl (TTS, 3D) was a burden.

## Integration contracts

- **localeval**: OpenAI `/v1/chat/completions`, honour `ignore_eos` + `min_tokens`, return `usage` incl. `cached_tokens` on streams, thinking toggle; target entry in `deepeval/targets.toml` with `bandwidth_gbs`; in-flight counter in `/metrics` for the busy check.
- **fleetq**: register service/port in `fleetq/deploy/fleet.toml`; wrap server starts in `fleetq run`; expose a step counter so `WEDGED` detection works.
- **Monitoring**: Prometheus v3 + Grafana 13 in `missioncntrl_/compose.monitoring.yaml`; add a scrape job in `deploy/monitoring/prometheus.yml`.
