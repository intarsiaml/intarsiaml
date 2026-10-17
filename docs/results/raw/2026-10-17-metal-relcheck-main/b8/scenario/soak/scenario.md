# cbsoak scenario

http://127.0.0.1:8143 model `Qwen3-8B` started 2026-10-17T16:13:13-04:00: 7 streams x ~1000-token prompts decoding (max_tokens 800), one ~4000-token prompt (max_tokens 400) sent at t = 5.0 s, 5 repetitions (5 valid), fresh nonce per prompt, warm-up request right before each repetition. after window = [long first token + 1 s, + 8.0 s] with the long request still decoding (rows = streams + 1).

| window | n gaps | p50 ms | p95 ms | p99 ms | max ms |
|---|---|---|---|---|---|
| steady before (streams only) | 1893 | 36.2 | 37.8 | 38.1 | 73.7 |
| during the long prefill | 693 | 126.4 | 139.7 | 147.7 | 148.0 |
| steady after | 6283 | 43.5 | 50.0 | 51.3 | 85.8 |

long request TTFT under load: p50 2097.0, max 2601.4 ms (n 5); solo (warm, alone) TTFT: p50 970.7 ms (n 3); ratio of medians 2.16x (design bound 2.5x).
per-repetition p95 during the prefill (ms): 118.5, 144.3, 144.6, 137.8, 145.8.
server decode passes by rows over the whole scenario run (delta of /metrics): {'1': 69.0, '2': 43.0, '3': 42.0, '4': 42.0, '5': 43.0, '6': 48.0, '7': 1867.0, '8': 1995.0}.

| rep | valid | long TTFT ms | gaps b/d/a | during p50 | during p95 | during max | before p50 | after p50 | notes |
|---|---|---|---|---|---|---|---|---|---|
| 0 | yes | 2097.0 | 400/147/1330 | 106.6 | 118.5 | 119.3 | 34.6 | 41.5 |  |
| 1 | yes | 1958.1 | 300/119/1292 | 128.2 | 144.3 | 144.6 | 36.5 | 43.0 |  |
| 2 | yes | 2485.6 | 316/147/1225 | 129.8 | 144.6 | 147.7 | 37.3 | 44.9 |  |
| 3 | yes | 2601.4 | 449/154/1253 | 129.6 | 137.8 | 140.0 | 35.4 | 43.7 |  |
| 4 | yes | 1951.5 | 428/126/1183 | 126.5 | 145.8 | 148.0 | 35.3 | 47.0 |  |
