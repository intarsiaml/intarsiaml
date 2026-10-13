# Benchmarking

Two tools, same methodology, so numbers are comparable:

* `intarsia bench speed` (built in, no Python, ships with the binary). Use it for quick checks, CI
  smoke runs and comparing against vLLM from any machine that can reach the server.
* `localeval` (`/opt/code/deepeval`). Use it for the logged, fleet-aware runs: it takes its own
  fleetq slot, refuses busy servers, and records every run in unanimis. Quality evals live there too.

Both measure end to end over HTTP against an OpenAI-compatible `/v1/chat/completions` stream, so
the same command works for IntarsiaML, vLLM, llama.cpp and oMLX.

## Methodology

* Prompt sizes 128 / 1K / 4K / 16K tokens. The tool calibrates tokens per filler sentence from the
  server's own `usage.prompt_tokens`, so sizes are right for any tokenizer. The row reports the
  measured `prompt` token count.
* Every prompt starts with a fresh random nonce: no prefix-cache or KV-reuse hit can inflate prefill.
* Output length is forced with `ignore_eos`, `min_tokens` and `max_tokens` (a row whose `gen` column
  is below `--gen` means the server ignored them; a warning is printed). Servers that reject these
  parameters with HTTP 400/422 are retried without them and a warning says so.
* One discarded warmup call per size, then the median of `--reps` measured calls.
* TTFT = request sent to first content delta. Prefill tok/s = prompt tokens / TTFT (includes queueing
  and HTTP, so small prompts look slow). Decode tok/s = (completion tokens - 1) / (end - first token).
* Concurrency rows (`--concurrency 1,4`): N simultaneous streams at `--conc-size` tokens, a warmup
  round, then per-stream and aggregate decode tok/s (aggregate = total decoded tokens over the
  window from the first token to the last stream finishing).
* Bandwidth floor: decode is memory bound, so `bandwidth_gbs / gb_per_token` is a ceiling on single
  stream tok/s. Pass both and the table adds `% of floor`. Above 100 means a measurement bug or a
  wrong `gb_per_token` (MoE and quantised models read far fewer bytes than the file size; use the
  bytes actually read per token). For Qwen3-8B bf16 on the GB10: `--gb-per-token 16.4
  --bandwidth-gbs 221` gives a floor of 13.5 tok/s.

## Built-in: `intarsia bench speed`

```sh
# Our engine on spark1 (start it first, inside a fleetq slot; see below)
intarsia bench speed --url http://spark-4e77:8090 --model Qwen3-8B \
    --sizes 128,1024,4096,16384 --gen 128 --reps 3 --concurrency 1,4 \
    --gb-per-token 16.4 --bandwidth-gbs 221 --label qwen3-8b-cuda-bf16 \
    --fingerprint-file doctor.json

# The vLLM on spark1, for comparison. Needs the guidance slot: it is a production role.
fleetq run --uses guidance --label "intarsiaml bench smoke" -- \
    intarsia bench speed --url http://spark-4e77:8000/v1 --model guidance \
    --sizes 128,1024 --reps 2 --gen 64 --thinking off
```

Output is a table on stdout and `results/<label>.json` (`--out-dir`, `--no-save`, `--json`). The JSON
holds the config, the calibration, one row per size, one row per concurrency level, warnings and a
`fingerprint` object. Fill the fingerprint with `intarsia doctor --json > doctor.json` then
`--fingerprint-file doctor.json` (stored under `fingerprint.host`), and/or `--fingerprint key=value`
for things the tool cannot know (`engine=vllm-0.12`, `quant=nvfp4`, `git=<sha>`). Always do so for a
number you will quote: GPU clocks, driver, THP and free memory change results.

Other flags: `--thinking off` sends `chat_template_kwargs.enable_thinking=false`;
`--key-env VAR` reads a bearer token from an environment variable; `--timeout-s`; `--warmup N`.
Plain `http://` only (the client is built without TLS); tunnel for https.

### Fleet rules

Anything that loads or runs GPU work on spark1 takes a fleetq slot first (`fleetq status`).
The benchmark client itself only issues HTTP, but it loads the server, so for a shared server use
`--uses <role>` for vLLM or hold the slot that owns our engine:

```sh
fleetq acquire --host spark1 --mem 24 --label "intarsiaml serve 8b" --wait   # note the job id; JOB=<id>
FLEETQ_JOB=$JOB ssh spark1 '~/intarsiaml/target/release/intarsia serve --backend cuda \
    --model /home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
    --port 8090 --host 0.0.0.0'      # in another shell, or nohup
intarsia bench speed --url http://spark-4e77:8090 --model Qwen3-8B ...
fleetq release $JOB
```

Never use port 8000 for our server (production vLLM).

## localeval

`localeval` reads targets from `/opt/code/deepeval/targets.toml`. Add the engine once by pasting
`deploy/localeval/targets-snippet.toml` (target `spark-intarsia`, `bandwidth_gbs = 221`), or use it
ad hoc without editing anything:

```sh
cd /opt/code/deepeval

# by registry name (after pasting the snippet)
uv run localeval speed spark-intarsia --sizes 128,1024,4096,16384 --gen 128 --reps 3 --concurrency 1,4

# ad hoc, any server
uv run localeval speed --url http://spark-4e77:8090/v1 --model Qwen3-8B \
    --label intarsia-qwen3-8b-bf16 --thinking off --sizes 128,1024,4096,16384 --gen 128 --reps 3
```

Flags (from `uv run localeval speed --help`): `--url`, `--model`, `--key-env`, `--thinking
{on,off,default}`, `--structured`, `--label`, `--wait-idle SECONDS`, `--ignore-busy`, `--no-queue`,
`--no-unim`, `--sizes`, `--gen`, `--reps`, `--concurrency`. localeval takes its own fleetq slot
(`--no-queue` is for non-fleet servers only; do not use it on spark1 without sign-off), waits for the
server to be idle, and logs the run to unanimis. If it complains the box is busy, the engine or
another chat holds a slot: `fleetq status`.

Quality (golden logits and benchmarks) is a separate gate: `docs/accuracy.md`.

## Comparing runs

* Same prompt sizes, `--gen`, and thinking mode on both sides, same day, same GPU state.
* Compare decode tok/s at 128 and 1K prompts (weight streaming) and prefill tok/s at 4K and 16K
  (compute bound). TTFT at 128 tokens mostly shows scheduling and launch overhead.
* Check `% of floor`: a healthy bf16 dense decode on a GB10 should land at 70-90%. Far below means
  launch overhead or an unfused path (see `intarsia top` kernel shares); above 100 means a bug.
* `intarsia top` (or the Grafana dashboard) alongside a run shows which kernels the time goes to.
