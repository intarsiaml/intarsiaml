# cbsoak scenario

http://127.0.0.1:8143 model `Qwen3-8B` started 2026-10-16T15:18:16-04:00: 7 streams x ~1000-token prompts decoding (max_tokens 800), one ~4000-token prompt (max_tokens 400) sent at t = 5.0 s, 5 repetitions (5 valid), fresh nonce per prompt, warm-up request right before each repetition. after window = [long first token + 1 s, + 8.0 s] with the long request still decoding (rows = streams + 1).

| window | n gaps | p50 ms | p95 ms | p99 ms | max ms |
|---|---|---|---|---|---|
| steady before (streams only) | 2068 | 34.6 | 37.5 | 39.5 | 69.1 |
| during the long prefill | 686 | 111.5 | 133.2 | 142.0 | 143.5 |
| steady after | 6564 | 42.6 | 45.6 | 47.3 | 80.7 |

long request TTFT under load: p50 2039.6, max 2480.4 ms (n 5); solo (warm, alone) TTFT: p50 970.5 ms (n 3); ratio of medians 2.10x (design bound 2.5x).
per-repetition p95 during the prefill (ms): 113.4, 124.8, 122.6, 141.7, 122.4.
server decode passes by rows over the whole scenario run (delta of /metrics): {'1': 68.0, '2': 44.0, '3': 44.0, '4': 44.0, '5': 44.0, '6': 47.0, '7': 1865.0, '8': 1995.0}.

| rep | valid | long TTFT ms | gaps b/d/a | during p50 | during p95 | during max | before p50 | after p50 | notes |
|---|---|---|---|---|---|---|---|---|---|
| 0 | yes | 2039.6 | 379/147/1344 | 106.5 | 113.4 | 115.2 | 34.3 | 41.0 |  |
| 1 | yes | 1709.8 | 405/119/1363 | 110.4 | 124.8 | 124.9 | 34.5 | 40.1 |  |
| 2 | yes | 2201.2 | 414/147/1288 | 115.1 | 122.6 | 131.7 | 35.6 | 43.1 |  |
| 3 | yes | 2480.4 | 421/154/1288 | 121.5 | 141.7 | 143.5 | 36.7 | 42.8 |  |
| 4 | yes | 1684.4 | 449/119/1281 | 108.9 | 122.4 | 122.4 | 34.3 | 42.8 |  |
