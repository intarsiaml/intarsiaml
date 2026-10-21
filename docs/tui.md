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
@@SNAPSHOT@@
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
| prefill / decode tok/s "now", steps/s | deltas of the snapshot counters over ~1.2 s | token counters are booked when a request ends, so while one runs `~` marks the step rate standing in for decode tok/s (as the web UI does) |
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
