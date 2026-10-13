# IntarsiaML

A modular inference engine built from interchangeable pieces: easy to plug into, easy to extend.

The engine is under active development. Phase 0 (the seams) runs Qwen3 dense models on a CPU reference backend and on
CUDA (tested on a DGX Spark, GB10), behind OpenAI and Anthropic compatible APIs. See `docs/ARCHITECTURE.md` for the plan
and `docs/results/2026-10-12-phase0.md` for what was measured and what is still missing.

## Quickstart

Needs Rust 1.90 or newer. The CUDA backend also needs `nvcc` (CUDA 13); without it the crate builds as a stub.

```sh
cargo build --release                      # cpu backend only
cargo build --release --features cuda      # with the CUDA backend (run on the GPU machine)
cargo test                                 # whole workspace
```

The binary is `target/release/intarsia`. `MODEL` below is a Hugging Face model directory (config.json,
tokenizer.json, safetensors) of a Qwen3ForCausalLM model.

```sh
intarsia doctor                                            # host fingerprint and known-pitfall checks
intarsia generate --model $MODEL --backend cpu -p "The capital of France is" --max-tokens 32   # or --backend cuda
intarsia kernels test --backend cuda                       # check every kernel against the cpu-ref oracle
intarsia accuracy --model $MODEL --backend cpu --quick     # day-0 gate: perplexity and sanity prompts
intarsia accuracy --model $MODEL --backend cpu --golden golden/<name>.safetensors   # golden-logit gate
intarsia serve --model $MODEL --backend cuda --port 8090 --served-model-name my-model --max-seq 8192
intarsia top --url http://127.0.0.1:8090                   # live terminal dashboard (q to quit)
intarsia bench speed --url http://127.0.0.1:8090 --model my-model --sizes 128,1024,4096 --gen 128
```

`serve` speaks `POST /v1/chat/completions`, `/v1/completions`, `GET /v1/models`, `POST /v1/messages` and
`/v1/messages/count_tokens` (streaming and not), plus `/health`, `/metrics` (Prometheus) and `/events` (SSE).
The default context is 4096 tokens; raise it with `--max-seq` for long-prompt benchmarks.

Golden logits come from `cd tools && uv run golden.py --model $MODEL --out ../golden/<name>.safetensors` (add
`--dtype float32` for an fp32 reference, which is the fairer gate for small models).

More: `docs/accuracy.md`, `docs/benchmarking.md`, `docs/observability.md`, `docs/telemetry.md`,
`docs/adding-a-model.md`.
