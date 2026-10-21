# Benchmarking

Two tools, same methodology, so numbers are comparable:

* `intarsia bench speed` (built in, no Python, ships with the binary). Use it for quick checks, CI
  smoke runs and comparing against vLLM from any machine that can reach the server.
* `localeval` (`/opt/code/deepeval`). Use it for the logged, fleet-aware runs: it takes its own
  fleetq slot, refuses busy servers, and records every run in unanimis. Quality evals live there too.

To deploy a pinned commit to a box, stamp the toolchain and pack a run so another host can replay it, see
[Reproducing results](reproducing-results.md).

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

Prefill tok/s depends on the server's prefill chunk, which a recipe may set per backend (the 35B recipe uses 2048 on CUDA,
256 elsewhere; `serve` prints `prefill chunk N (source)` at load). The client cannot see it, so pass
`--prefill-chunk N` (the value `serve` printed) to store it as `fingerprint.prefill_chunk`, and compare only runs
whose chunk matches. `accuracy` baselines refuse a chunk mismatch and name the recipe default; pin
`--prefill-chunk 256` to compare against baselines recorded before the recipe default existed.

Other flags: `--thinking off` sends `chat_template_kwargs.enable_thinking=false`;
`--key-env VAR` reads a bearer token from an environment variable; `--timeout-s`; `--warmup N`.
Plain `http://` only (the client is built without TLS); tunnel for https.

### Graph replay, variance and profiling in the result (Phase 1b review findings 4, 5, 6)

`intarsia serve` exports `intarsia_graph_replays_total`, `intarsia_graph_captures_total`,
`intarsia_decode_eager_steps_total` and `intarsia_decode_profiled_steps_total` in `/metrics` (docs/telemetry.md). `bench speed`
reads them before and after the measured reps of each size (warmup excluded) and writes, per row and summed at the top level:

* `graph`: `{available, replays, captures, eager_steps, profiled_steps}` as deltas over the measured reps. `available: false`
  (all null) when the server does not export the counters (vLLM, llama.cpp).
* `decode_tok_s_reps`, `decode_tok_s_min`, `decode_tok_s_max`, `decode_tok_s_stdev` next to the median `decode_tok_s`.
* warnings: `0 graph replays during the measured reps (decode ran eagerly)`, and `timing includes N profiled eager steps`.
* `--require-graphs`: exit non-zero (after saving) if the server does not export the counters or any measured size replayed
  zero graphs. The acceptance procedure passes it for our server. 0 replays means the run measured the Phase 1a eager path.

**Profiled steps bias the number.** With `--profile-every N` (default 16 on GPU, 1 on cpu, 0 for `serve` on metal) every Nth decode step runs eagerly
with per-op synchronisation: about 6% of the steps at N = 16, each far slower than a replay. A throughput measurement runs the
server with `--profile-every 0` so that no step is profiled, on every arm. vLLM has no such steps, so profiling only ever
penalises our side. The as-shipped default can be reported as an extra, separately labelled arm, never as the A/B number.

**One run per arm is not a result.** Run-to-run and server-restart variation on the GB10 is of the order of the margin the
Phase 1 gate is decided on (76 against 78 tok/s). The A/B uses interleaved rounds (below), a discarded warmup call per size,
five measured calls per size per round, the per-round difference against vLLM, and a decision rule written down before the run.

### A/B against vLLM on spark1: `scripts/phase1b-acceptance.sh`

The whole procedure is one script (stages: preflight, build, lease, kernels, accuracy, graphs, ab, verdict; `STAGES=ab,verdict`
runs a subset). What it enforces, and why:

1. **vLLM is stopped while ours runs.** The `guidance` vLLM reserves 64 GB (`gpu_memory_utilization 0.5`) on the 128 GB unified
   memory; a 38 GB server or an accuracy job beside it is what hung spark1 at 01:50 on 2026-10-13 (an 8B HF golden job next to
   it). The script takes one exclusive lease with `fleetq acquire --host spark1 --exclusive --takes-down guidance`, stops the
   container (`VLLM_CONTAINER`, the name from `docker ps` on the box), and waits for the memory to come back. Nothing else runs
   in the lease: never a golden or HF job, never a build. Builds happen in an earlier small shared slot.
2. **Restore before release.** An EXIT/INT/TERM trap stops our server (SIGTERM, wait up to 60 s for the pid, then `kill -9`;
   `deploy/localeval/serve-and-bench.sh` `stop()` now waits too), then restarts the container and waits for `/health` and
   `/v1/models`, then releases the slots. fleetq counts the role back only when it sees the expected model again.
   Dry-run check: `DRY_RUN=1 FAIL_AT=accuracy scripts/phase1b-acceptance.sh` shows the stop, start and release commands being
   issued after a forced failure (no box is touched in a dry run).
3. **Memory pre-check.** Before every model load (kernels, each accuracy case, graphs, each A/B round) the script reads
   `MemAvailable` on the box and aborts if it is below model size + 20 GB (35B: 38 GB, 8B: 17 GB; override with
   `MODEL_GB_35B` / `MODEL_GB_8B`).
4. **Interleaved rounds.** `ROUNDS` (default 3) rounds, each: ours-new and ours-old (order alternates per round), then vLLM up,
   vLLM, vLLM down. vLLM and ours never run at the same time, and drift in the box (thermal state, clocks) hits all arms alike.
5. **Valid round.** The 1 Hz poller must never see `running > 1` or `waiting > 0` on the server under test (the bench is its only
   client); ours-new must show replays > 0 and 0 eager or profiled steps for every size. Invalid rounds do not count; up to two
   extra rounds replace them.
6. **Decision rule** (header of the script; written before any number existed). Per size S and round r:
   d(S,r) = ours-new median decode tok/s minus vLLM median in the same round. PASS(S): d >= 0 in every valid round.
   MARGINAL(S): median d >= 0 but some round below 0. FAIL(S): median d < 0. Fewer than 3 valid rounds: INCONCLUSIVE.
   The Phase 1 speed gate passes only if 128, 1024 and 4096 all PASS; MARGINAL or INCONCLUSIVE sizes give "NOT DECIDED" and the
   numbers (min / median / max of d) are reported as they are. If no ours-new round is valid because of 0 replays or eager
   decode, the A/B fails outright: those numbers would be Phase 1a numbers.
7. The verdict stage (`verdict.py`, written into the work directory) is the only thing that says PASS. The script exits 0 only
   if it and the accuracy stages pass.

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

### Metal speed gate (restated 2026-10-21 12:25 EDT)

Decided by the user ("1 yes", `docs/decisions/LOG.md`). The evidence is `docs/results/2026-10-21-g3-metal.md` section 2: on endor, decode follows the GPU clock (r = +0.94), and the clock falls when other processes load the CPU (r = −0.93). A flat floor therefore failed builds that had not changed.

The gate uses three parts:

1. **Floor:** decode must reach at least 85 % of the **context-aware** byte floor. The floor is bandwidth / (weight bytes + KV bytes at the mean context). The mean context is the prompt plus half the generated tokens. For Qwen3-8B BF16 that is 15.1368 GB plus 147,456 B per position at 614 GB/s, for example 39.0 tok/s at a 4k prompt. The old flat 40.6 tok/s ignored the KV bytes.
2. **State admission:** a run counts for the floor only when its macmon samples show a median GPU clock of at least 1300 MHz (busy samples) and a median CPU load of at most 0.30.
   - A run outside that window is STATE-INVALID: re-run it; it is not a failure.
   - On a normally used endor, many runs fall outside the window.
3. **Same-slot ratios:** these apply to every intarsia arm and do not depend on the clock. Decode must be at least 0.90× mlx-lm and at least 1.00× llama.cpp, measured in the same exclusive slot.

To run it, use `scripts/metal-speed-gate/sameslot.sh <binary> <out-dir>` inside one exclusive endor slot. It runs an ABBA of intarsia, mlx-lm, llama.cpp and intarsia with localeval and macmon, then `gate.py`. `gate.py` exits with:
- 0 for PASS;
- 1 for FAIL;
- 3 for STATE-INVALID (the ratios passed but no arm was admitted).

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

* Same prompt sizes, `--gen`, thinking mode and prefill chunk (`fingerprint.prefill_chunk`) on both sides, same day, same GPU state.
* Compare decode tok/s at 128 and 1K prompts (weight streaming) and prefill tok/s at 4K and 16K
  (compute bound). TTFT at 128 tokens mostly shows scheduling and launch overhead.
* Check `% of floor`: a healthy bf16 dense decode on a GB10 should land at 70-90%. Far below means
  launch overhead or an unfused path (see `intarsia top` kernel shares); above 100 means a bug.
* `intarsia top` (or the Grafana dashboard) alongside a run shows which kernels the time goes to.
