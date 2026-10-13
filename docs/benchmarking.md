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
  parameters with HTTP 400/422 *naming them* (or saying "unknown/unsupported parameter") are retried without
  them, and the "rejected optional parameters" warning appears only when that retry worked. A length error
  ("maximum context length", `max_model_len`) is never treated as a parameter problem.
* The server's context window is read from `GET /v1/models` (`max_model_len`). A size whose estimated prompt
  plus `--gen` (+2% and 16 tokens of slack) exceeds it is **skipped before anything is sent**, with a message
  and a `skipped` row in the JSON; the calibration probe also shrinks to fit a small window.
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

Anything that loads or runs GPU work on spark1 takes a fleetq slot first (`fleetq status`). The slot
must cover the **whole life of our server**, and the benchmark must run inside that same slot:
`localeval` takes its own exclusive slot on the box behind its `--url`, so it must be started with
`FLEETQ_JOB` set (it then sees it is already inside a covering slot and carries on) and pointed at
`http://spark-4e77:8090`, the server's real host (not a loopback tunnel such as `127.0.0.1:18090`,
which fleetq reads as "this machine" and locks the wrong box). Never start the server outside the
lease and leave it running after `fleetq release`.

`--mem-budget GIB` ties the KV cache to the lease: weights + workspace + KV stay within it, whatever
the box has free (the default sizes KV from measured free memory, so a large `--max-seq` could
otherwise take memory another lease owns). Use the same number as the lease's `--mem`.

One command: `deploy/localeval/serve-and-bench.sh` starts the server on spark1, waits for `/health`,
runs the benchmark, and stops the server on exit, all inside one `fleetq run` lease:

```sh
fleetq run --host spark1 --exclusive --mem 24 --label "intarsiaml: qwen3-8b speed" -- \
    deploy/localeval/serve-and-bench.sh --mem-budget 24 -- \
    uv run --project /opt/code/deepeval localeval speed spark-intarsia --sizes 128,1024,4096 --gen 128 --reps 3
```

By hand with a lease (every command carries `FLEETQ_JOB`, the server is stopped before the release):

```sh
fleetq acquire --host spark1 --exclusive --mem 24 --label "intarsiaml: qwen3-8b speed" --wait   # prints the job id
export JOB=<id>
FLEETQ_JOB=$JOB ssh spark1 'cd ~/intarsiaml && nohup target/release/intarsia serve --backend cuda \
    --model /home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
    --port 8090 --host 0.0.0.0 --mem-budget 24 > /tmp/intarsia-serve.log 2>&1 & echo $! > /tmp/intarsia-serve.pid'
until curl -sf http://spark-4e77:8090/health >/dev/null; do sleep 2; done
FLEETQ_JOB=$JOB intarsia bench speed --url http://spark-4e77:8090 --model Qwen3-8B ...
FLEETQ_JOB=$JOB sh -c 'cd /opt/code/deepeval && uv run localeval speed spark-intarsia ...'   # same slot, no second queue
FLEETQ_JOB=$JOB ssh spark1 'kill $(cat /tmp/intarsia-serve.pid)'   # SIGTERM reaches the supervisor and its child
fleetq release $JOB
```

`localeval` refuses a busy server by reading `vllm:num_requests_running` and
`vllm:num_requests_waiting` from `/metrics`; `intarsia serve` exports both. If a run dies and leaves
the server busy, `curl spark-4e77:8090/metrics | grep vllm:` shows what is in flight (a disconnected
client cancels its request, so this clears within a prefill chunk).

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
another chat holds a slot: `fleetq status`. Inside a lease you hold (see Fleet rules) run it with
`FLEETQ_JOB` set so it does not queue behind your own server.

Quality (golden logits and benchmarks) is a separate gate: `docs/accuracy.md`.

## Comparing runs

* Same prompt sizes, `--gen`, and thinking mode on both sides, same day, same GPU state.
* Compare decode tok/s at 128 and 1K prompts (weight streaming) and prefill tok/s at 4K and 16K
  (compute bound). TTFT at 128 tokens mostly shows scheduling and launch overhead.
* Check `% of floor`: a healthy bf16 dense decode on a GB10 should land at 70-90%. Far below means
  launch overhead or an unfused path (see `intarsia top` kernel shares); above 100 means a bug.
* `intarsia top` (or the Grafana dashboard) alongside a run shows which kernels the time goes to.
