# IntarsiaML

A modular inference engine built from interchangeable pieces: easy to plug into, easy to extend.

The engine is under active development. It runs Qwen3 dense models and the hybrid GatedDeltaNet + sparse-MoE family
(Qwen3.6-35B-A3B NVFP4) on a CPU reference backend, on CUDA (DGX Spark, GB10) and on HIP (AMD R9700), behind OpenAI
and Anthropic compatible APIs. Status as measured on 2026-10-13 (`docs/results/`, decisions in `docs/decisions/`):

- spark1, CUDA, 35B NVFP4: decode 80.3 / 79.7 / 77.0 tok/s at 128 / 1024 / 4096 tokens with CUDA graphs and fused decode ops, vs vLLM's 75.9 / 76.7 / 75.1 re-measured 2026-10-13 with the same client (+5.8% / +3.9% / +2.5%; the 4096 margin is within noise)
  (`docs/results/2026-10-14-cuda-decode-fusion.md`). Prefill chunk 2048 is now the
  recipe default on CUDA (63b0f72): 4654 tok/s at 4k tokens against vLLM's 6595, so 1.42x behind (2.75x before); 3891 vs
  5631 at 1k, 4310 vs 6412 at 16k (`docs/results/2026-10-13-prefill-cuda.md`). The Phase 1 speed gate FAILED.
- R9700, HIP, one card: Qwen3-8B BF16 decode 36.9 tok/s, 96% of the byte floor; 35B NVFP4 decode 135.2 tok/s at 128
  prompt tokens, 47.4% of the byte floor (`docs/results/2026-10-14-hip-decode-2.md`).
- R9700, HIP, tp=2: Qwen3-8B BF16 decode 56 tok/s, 1.52x one card. The 35B is not supported at tp=2 yet.
- Accuracy: 8B passes the bf16 floor on CUDA, HIP and tp=2; the CUDA 35B gate is not accepted (ADR-0007), but its `35b-short` vLLM-golden failure was measured as noise and
  that gate is advisory (ADR-0008); kernel-level error checks stay blocking.
- New models: `intarsia recipe new` drafts a recipe and `intarsia qualify` runs the staged checks (8B example run: quick
  section 3 min 8 s, 6 min 34 s with the in-depth section, per `docs/adding-a-model.md`; the 35B on HIP passes the quick
  section in 1 min 34 s, `docs/results/2026-10-13-qualify/35b-hip-quick/quick-report.md`).

See `docs/ARCHITECTURE.md` for the plan and `docs/results/` for what was measured and what is still missing.

Built-in web UI (chat, live dashboard, live architecture map) at `/` when serving; `--no-ui` turns it off. See `docs/architecture-diagram.md`.

![Chat](docs/assets/ui-chat-dark.png)
![Dashboard](docs/assets/ui-dashboard-dark.png)
![Architecture map](docs/assets/ui-arch-dark.png)

## Quickstart

Needs Rust 1.90 or newer. The CUDA backend also needs `nvcc` (CUDA 13); without it the crate builds as a stub.

```sh
cargo build --release                      # cpu backend only
cargo build --release --features cuda      # with the CUDA backend (run on the GPU machine)
cargo test                                 # whole workspace
```

The binary is `target/release/intarsia`. `MODEL` below is a Hugging Face model directory (config.json,
tokenizer.json, safetensors) of a supported model (Qwen3, Qwen3.5/3.6 dense and MoE; the recipe is matched by the
`architectures` entry of config.json, see `recipes/`).

```sh
intarsia doctor                                            # host fingerprint and known-pitfall checks
intarsia generate --model $MODEL --backend cpu -p "The capital of France is" --max-tokens 32   # or --backend cuda
intarsia kernels test --backend cuda                       # check every kernel against the cpu-ref oracle
intarsia plan show --model $MODEL --backend cuda [--admit] # kernel chosen for every op and shape (JSON with --json)
intarsia accuracy --model $MODEL --backend cpu --quick     # day-0 gate: perplexity and sanity prompts
intarsia accuracy --model $MODEL --backend cpu --golden golden/<name>.safetensors   # golden-logit gate (prefill + decode)
intarsia generate --model $MODEL --backend cuda --chat --no-think -p "What is the capital of France?"
intarsia serve --model $MODEL --backend cuda --mem-budget 38 --fp8-act static   # lease-sized KV; FP8 W8A8 (static) or W8A16 (none)
intarsia serve --model $MODEL --backend cuda --port 8090 --served-model-name my-model --max-seq 8192
intarsia top --url http://127.0.0.1:8090                   # live terminal dashboard (q to quit)
intarsia bench speed --url http://127.0.0.1:8090 --model my-model --sizes 128,1024,4096 --gen 128
```

`serve` speaks `POST /v1/chat/completions`, `/v1/completions`, `GET /v1/models`, `POST /v1/messages` and
`/v1/messages/count_tokens` (streaming and not), plus `/health`, `/metrics` (Prometheus), `/events` (SSE) and
`/v1/intarsia/plan` (the resolved kernel plan).
The default context is 4096 tokens; raise it with `--max-seq` for long-prompt benchmarks.

Top-k goldens from a running vLLM server (for quantised checkpoints that HF cannot run): `tools/golden_vllm.py`
(`docs/accuracy.md`).

Golden logits come from `cd tools && uv run golden.py --model $MODEL --out ../golden/<name>.safetensors` (add
`--dtype float32` for an fp32 reference, which is the fairer gate for small models).

More: `docs/accuracy.md`, `docs/benchmarking.md`, `docs/observability.md`, `docs/telemetry.md`,
`docs/adding-a-model.md`.
