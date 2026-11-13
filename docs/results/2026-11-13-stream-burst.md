# stream-burst: where streamed tokens are held back (2026-11-13, spark1)

## Brief (as given)
Lane `stream-burst`, branch `stream-burst` from origin/main `10b235785`. fp8-decode (`2026-11-13-fp8-decode.md`, PR #355) found a
single stream decodes at a steady ~15.6 ms/step on a GB10 while the client sees bursts (ITL p99 ~80 ms, p50 ~15-18 ms). Find
where tokens are held back between the engine and the HTTP client (engine -> server channel, detokenizer holdback, parsers,
SSE writer / flushing, tokio/hyper, Nagle, the localeval client) and fix it if it is ours. Gates: `cargo test -p intarsia-server`,
`cargo test -p intarsia-core --lib`; before/after chunk-arrival histogram; streamed text byte-identical for the same seed.
No fix of ours -> report the evidence, open no PR.

## Verdict: the transport is not the cause; no PR
Three things make the client see bursts, none of them a delivery bug:
1. **The reasoning parser holds trailing whitespace (thinking on), by design.** `ReasoningParser::push` (`structured.rs`) keeps
   trailing whitespace and a partial `</think>` until the next token, because the reasoning is trimmed at `</think>`; releasing
   it early would change the bytes. 400 generated tokens arrive as 328 SSE content chunks; the 72 whitespace tokens (newlines,
   indents) ride with the next token, so those chunks are 24 / 36 ms apart instead of 12.
2. **localeval's ITL math inflates it.** `itl_stats` / `_chunk_weights` (`/opt/code/deepeval/src/localeval/client.py`) divide each
   chunk delta by tokens-in-chunk, estimated from character counts whenever chunks != tokens (case 1, spec decode). A short chunk
   gets a weight well under one token, so its "ITL" is multiplied. Same marks: raw chunk gap p99 35.7 ms, localeval p99 50.9 ms,
   p50 11.9 vs 13.5 ms.
3. **Intended buffering elsewhere.** MTP (the default, `--spec mtp --spec-k 2`) emits up to 3 tokens per verify pass back to
   back: 400 tokens in 166 socket reads, ~24 ms apart, ~0 ms within a pass. A tool call is emitted whole at `</tool_call>`
   (53 tokens, first event at ~310 ms, the second call right after), as the qwen3_xml/hermes parsers need a complete call.

Thinking off, plain decode (`--max-batch 2`, no spec): perfectly even. 400 events, p50 11.8 ms, p99 12.4-12.9 ms, max 13.5 ms,
one event per socket read, on loopback and over the LAN.

## Path read
engine `Seq::on_token` -> `tx.send(GenEvent::Token)` (std mpsc, one per token, no batching; `engine.rs` ~1894) -> `intarsia-bridge`
thread (`api.rs`, forwards each event at once) -> tokio unbounded channel -> `sse_from` (`unfold`, one `Event` per event, axum
flushes each) -> hyper. No buffering in any of these. TCP_NODELAY was not set on accepted sockets (`axum::serve` default).

## Measurements
Qwen3.6-35B-A3B-NVFP4 (1355db6a), spark1, `--max-seq 32768 --mem-budget 40`, one fleetq slot per run, vLLM untouched, port 8193.
Client `2026-11-13-stream-burst/sse_probe.py`: raw socket, 400 tokens forced (`ignore_eos`), T 0.7, top_p 0.95, seed 7; stamps
each SSE event when read. Diagnostic only, no speed claims. Raw files: `2026-11-13-stream-burst/raw/`.
Histogram buckets of inter-event gap (ms): `<1,<5,<10,<14,<17,<20,<30,<50,<80,>=80`.

| setup | events / 400 tok | gap p50 / p99 / max (ms) | histogram |
|---|---|---|---|
| batch 2, think off (loopback) | 400 | 11.8 / 12.4 / 12.5 | [0,0,0,399,0,0,0,0,0,0] |
| batch 2, think on (loopback) | 328 | 11.9 / 36.0 / 36.4 | [0,0,0,277,0,0,28,22,0,0] |
| batch 1 + MTP k=2, think off (loopback) | 400 (166 reads) | 0.0 / 29.4 / 41.9 | [235,2,1,1,0,1,155,4,0,0] |
| batch 2, tool request (2 calls) | 3 | one 310 ms gap (whole call buffered) | |

**Before / after TCP_NODELAY**, same setup (batch 2, client on spark2 over the LAN, 3 runs each, `lan.sh`): think off before
p99 12.9 / 13.1 / 12.9, after 12.7 / 12.7 / 12.7 ms; think on before p99 35.7 / 36.1 / 36.0, after 35.8 / 35.4 / 35.5 ms; the
histograms are identical ([0,0,0,399,...] and [0,0,0,277,0,0,28,22,0,0]). Mac loopback and Linux loopback mock
(`stream_probe`, 15 ms cadence): p99 15.8 vs 15.7 ms. Setting NODELAY changes nothing here, so it is not shipped as a fix.
Streamed text is byte-identical before / after (`cmp` on the text of think-off, think-on, seed 7), and between repeated runs.

## Recommendations (not done: outside what is ours or by design)
- localeval: when chunks != tokens, report ITL per chunk (or per verify step), not char-weighted; or take token counts from
  `return_token_ids`. This alone moves the thinking-on p99 from ~50 to ~36 ms.
- To smooth thinking-on delivery the parser would have to emit trailing whitespace early and retract it at `</think>`, which
  SSE cannot do; the alternative is to stop trimming the reasoning's end (changes bytes, differs from vLLM).

## Checks
- `cargo test -p intarsia-server`: pass (39 + 51 + 3). `cargo test -p intarsia-core --lib`: 201 pass, 2 ignored.
- Branch commits (not pushed): the ignored `stream_probe` test (`tests/api.rs`, `Mock.paced_ms`), `intarsia_server::nodelay`
  and its use in `serve` (no measured effect; drop it if unwanted).
- spark1 build dir `~/intarsia-stream-burst` (bin `intarsia-before` = 7b347ec59, `intarsia-after` = d997ed855) left in place;
  server stopped, spark2 probe dir removed.
