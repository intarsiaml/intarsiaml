# cbsoak scenario

http://127.0.0.1:8107 model `Qwen3-8B` started 2026-10-16T10:03:44-04:00: 7 streams x ~1000-token prompts decoding (max_tokens 800), one ~4000-token prompt (max_tokens 400) sent at t = 5.0 s, 5 repetitions (5 valid), fresh nonce per prompt, warm-up request right before each repetition. after window = [long first token + 1 s, + 8.0 s] with the long request still decoding (rows = streams + 1).

| window | n gaps | p50 ms | p95 ms | p99 ms | max ms |
|---|---|---|---|---|---|
| steady before (streams only) | 1893 | 35.7 | 37.4 | 37.6 | 74.0 |
| during the long prefill | 700 | 125.4 | 140.8 | 145.6 | 148.5 |
| steady after | 6353 | 43.9 | 45.4 | 47.2 | 87.7 |

long request TTFT under load: p50 2109.6, max 2603.6 ms (n 5); solo (warm, alone) TTFT: p50 969.9 ms (n 3); ratio of medians 2.17x (design bound 2.5x).
per-repetition p95 during the prefill (ms): 122.3, 142.5, 139.9, 140.8, 145.4.
server decode passes by rows over the whole scenario run (delta of /metrics): {'1': 68.0, '2': 45.0, '3': 44.0, '4': 44.0, '5': 45.0, '6': 47.0, '7': 1864.0, '8': 1995.0}.

| rep | valid | long TTFT ms | gaps b/d/a | during p50 | during p95 | during max | before p50 | after p50 | notes |
|---|---|---|---|---|---|---|---|---|---|
| 0 | yes | 2109.6 | 372/147/1323 | 107.3 | 122.3 | 129.8 | 34.9 | 41.7 |  |
| 1 | yes | 1997.4 | 286/126/1278 | 129.4 | 142.5 | 144.8 | 36.8 | 43.6 |  |
| 2 | yes | 2535.8 | 358/154/1232 | 133.8 | 139.9 | 148.5 | 37.2 | 45.1 |  |
| 3 | yes | 2603.6 | 442/154/1246 | 130.6 | 140.8 | 141.4 | 35.8 | 44.6 |  |
| 4 | yes | 1904.0 | 435/119/1274 | 125.2 | 145.4 | 145.6 | 34.7 | 43.3 |  |
