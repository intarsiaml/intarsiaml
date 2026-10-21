# Terminal UI (`intarsia tui`)

A terminal twin of the built-in web UI: the live architecture map, a streaming chat and a metrics pane, in one screen
over SSH. Dark by default (`--light` for a light theme).

```sh
intarsia serve --model $MODEL --backend metal --profile-every 16 --max-batch 4    # the server
intarsia tui --url http://127.0.0.1:8090                                          # the UI (Esc or Ctrl-C quits)
```

Keys: type and `Enter` to chat, `/clear` or `Ctrl-L` clears the chat, `PgUp`/`PgDn` scroll it, `Esc`/`Ctrl-C` quit.
Flags: `--url`, `--light`, `--thinking` (let a thinking model think in the chat; default is `enable_thinking: false`),
`--max-tokens N` (chat turn, default 256), `--exit-after SECS`.

## What is on screen

```
# frame "reply finished": Qwen3-8B on Metal, --max-batch 4, 150x52, empty rows cropped
 intarsia tui  http://127.0.0.1:59736  Qwen3-8B  metal Apple M5 Max  live
╭ architecture · 36 layers · exactness deterministic ──────────────────────────────╮╭ metrics ───────────────────────────────────────────────────────╮
│╭ request queue / scheduler ─────────────────────────────────────────────────────╮││decode    21.1 tok/s      ▂▂▇██▆▆▆                              │
││running 0 · waiting 0                                                           │││prefill   0.0 tok/s       ██                                    │
││batch queue depth 0 · requests 1                                                │││waiting   0 (run 0)                                             │
│╰────────────────────────────────────────────────────────────────────────────────╯││KV used   0.8%                   █                              │
│                                         ▼                                        ││TTFT      p50 200ms  p95 245ms  p99 249ms  (30s, n=1)           │
│╭ prefill ──────────────────────────────╮╭ decode ───────────────────────────────╮││ITL       p50 43ms  p95 82ms  p99 152ms  (30s, n=98)            │
││exactness deterministic                ││now 21.1 tok/s · last 23.1             │││last req  TTFT 156ms · decode 23.1 tok/s · prefill 173 tok/s    │
││now 0.0 tok/s                          ││batch size 1.0 · steps 15.2/s          │││batch     avg 1.0 rows · step 46.0 ms                           │
││last 173 tok/s · cached 0              ││                                       │││queue     running 0 · waiting 0 · depth max 1                   │
│╰───────────────────────────────────────╯╰───────────────────────────────────────╯││steps     15.2/s · total 107                                    │
│                                         ▼                                        ││tokens    prompt 27 · cached 0 · generated 107                  │
│╭ [A] x36 ───────────────────────────────────────────────────────────────────────╮││memory    weights 16.4GB · free 96.4GB of 115.4GB               │
││attention > dense_ffn                                                           │││uptime    57s · 1 requests                                      │
││time share n/a (per-op below)                                                   ││╰────────────────────────────────────────────────────────────────╯
│╰────────────────────────────────────────────────────────────────────────────────╯│╭ chat · Qwen3-8B · last 107 tok in 4.8s (22.5 tok/s) ───────────╮
│ AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA                                             ││you Explain in a few sentences how a KV cache speeds up         │
│                                         ▼                                        ││    transformer decoding.                                       │
│ kernels (9) · share of sampled op time                                           ││ai  A KV cache speeds up transformer decoding by storing the    │
│ embedding         1.3% █░░░░░░░░░        rmsnorm           0.3% █░░░░░░░░░       ││    previous keys (K) and values (V) from each decoding step.   │
│ rmsnorm_add      14.0% ███░░░░░░░        matmul           31.6% ███████░░░       ││    During the generation of a sequence, the model needs to     │
│ attn_prefill      0.1% █░░░░░░░░░        attn_decode       6.5% ██░░░░░░░░       ││    attend to all previous tokens, which can be computationally │
│ silu_glu          6.9% ██░░░░░░░░        matmul_pair      28.9% ██████░░░░       ││    expensive. Instead of recomputing the keys and values for   │
│ qk_norm_rope_kv   8.1% ██░░░░░░░░                                                ││    all previous tokens at each step, the KV cache reuses the   │
│                                         ▼                                        ││    already computed K and V from earlier steps. This           │
│╭ KV cache ───────────────╮╭ sampler ─────────────────╮╭ output ─────────────────╮││    significantly reduces redundant computation and allows the  │
││133 / 16384 tok (0.8%)   ││logits 2.3% · argmax 0.0% ││generated 107 tok        │││    model to generate the next token efficiently, improving both│
││19.6MB of 2.4GB          ││                          ││finished stop 1          │││    speed and efficiency during sequence generation.            │
│╰─────────────────────────╯╰──────────────────────────╯╰─────────────────────────╯││                                                                │
│                                                                                  ││> message, Enter to send                                        │
╰──────────────────────────────────────────────────────────────────────────────────╯╰────────────────────────────────────────────────────────────────╯
 Enter send  /clear or Ctrl-L clear chat  PgUp/PgDn scroll chat  Esc / Ctrl-C quit
```

- **Architecture map (left).** Request queue / scheduler, then prefill and decode, then the layer stack, the kernels, and
  KV cache, sampler and output. Each node carries live values from the server; a node whose box is highlighted is active
  now. The row of letters under the layer boxes has one letter per layer (same letter = same block signature).
- **Metrics (top right).** Sparklines (two minutes) for decode tok/s, prefill tok/s, waiting requests and KV fill, then
  TTFT and ITL percentiles, the last request, batch size, queue, steps, token totals, memory.
- **Chat (bottom right).** Streams from `POST /v1/chat/completions`; the title shows tokens so far while streaming and the
  reply's token count and rate afterwards.

## Where every number comes from

No new server endpoints; the TUI reads the ones the web UI's map and Prometheus already use.

| Overlay | Source | Note |
| --- | --- | --- |
| Map structure (layer groups, kernels) | `GET /v1/intarsia/plan` | once; with a `null` plan the map is built from the telemetry op names |
| running / waiting, requests, steps, KV tokens and bytes, last prefill/decode tok/s, finished reasons, memory | `GET /events` (the telemetry snapshot, 5 Hz) | same stream as `intarsia top` and the web UI |
| prefill / decode tok/s "now" | with `--max-batch N`: deltas of the scheduler counters in `/metrics` (`intarsia_batch_decode_rows_total`, `intarsia_batch_prefill_tokens_total`) between two scrapes, so the value is an average over the last second; without it: deltas of the snapshot token counters while a request runs | the engine books (seen on Metal, --max-batch 4; the web UI notes the same) the telemetry token counters, KV tokens and `last_*` values when a request finishes, so on the single-sequence engine `~` marks the step rate standing in for decode tok/s (as the web UI does) and KV/token totals move at the end of a reply, not during it |
| steps/s | deltas of `steps_total` over ~1.2 s | |
| Kernel time share | snapshot `ops[*].time_ms`, scaled by calls/timed_calls | only when the server times ops; `n/a` otherwise (see below) |
| TTFT and ITL p50/p95/p99 | `GET /metrics` histograms `intarsia_ttft_seconds`, `intarsia_itl_seconds` (batch variants as fallback) | scraped once a second; percentiles of the last 30 s when there were new observations, else since server start (the label says which) |
| Batch size, queue depth | `/metrics` `intarsia_batch_*` | exist only with `--max-batch N`; `n/a` on the single-sequence engine |
| Chat model name | `GET /v1/models` | first id |

`n/a` is shown instead of a guess when the server does not export a value: per-block (attention/GDN/MoE/FFN) time share
(the plan has no op-to-block attribution, so layer nodes show `n/a`; the kernel row shows per-op share), batch size
without `--max-batch`, and every kernel share when op timing is off. Metal serving defaults to `--profile-every 0`
(no op timing, it costs ~12% decode); pass `--profile-every 16` to get kernel shares in the map.

## Modularity

The map is built by `intarsia_tui::layout::build(plan, telemetry_ops)`, a pure function. Layers are grouped by their
`blocks` signature (so `[linear_attention, moe]` and `[attention, moe]` become two nodes with counts), kernels come from
`ops[]`. A block kind or an op the TUI has never seen gets its own node without a code change. The only built-in knowledge
is the fixed spine (queue, prefill/decode, layers, kernels, KV/sampler/output) and a short prefix table
(`layout::OWNERS`) saying that `kv*` ops belong to the KV node and `argmax`/`sample*`/`logits`/`softmax`/`top*` to the
sampler; any other op is a plain kernel.

## Headless snapshots (CI, docs, logs)

`--snapshot` renders the same UI off-screen and prints text frames; no terminal is needed and raw mode is never entered.

```sh
intarsia tui --url http://127.0.0.1:8090 --snapshot --size 150x48                     # one frame after --hold seconds
intarsia tui --url http://127.0.0.1:8090 --snapshot --chat "Explain KV caches" --size 150x48
```

With `--chat` it waits for telemetry and the plan, sends the message, and prints a frame mid-stream (after `--mid-tokens`
chunks), one right after the reply, and one a few seconds later when the latency percentiles have been scraped.

## Build

The TUI is the `intarsia-tui` crate behind the `tui` feature of `intarsia-cli` (on by default).
`cargo build --release -p intarsia-cli --no-default-features` leaves the `tui` subcommand and crate out. `intarsia top`
(the plain telemetry dashboard) is separate and always built, so `ratatui` stays linked either way; the feature drops
the TUI, not the dependency.
