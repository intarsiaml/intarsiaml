# IntarsiaML

A modular inference engine built from interchangeable pieces: easy to plug into, easy to extend.

The engine is under active development. Phase 0 (the seams) runs Qwen3 dense models on a CPU reference backend and on
CUDA (tested on a DGX Spark, GB10), behind OpenAI and Anthropic compatible APIs. Phase 1a adds the hybrid
GatedDeltaNet + sparse-MoE family with FP8 and NVFP4 weights: Qwen3.6-35B-A3B NVFP4 runs end to end on the GB10 at
63-65 tok/s decode (vLLM: 77-78) and tracks vLLM's logits (top-1 98.4%); prefill is the open gap. See
`docs/ARCHITECTURE.md` for the plan, `docs/results/2026-10-12-phase0.md` and `docs/results/2026-10-12-phase1a.md` for
what was measured and what is still missing.

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
