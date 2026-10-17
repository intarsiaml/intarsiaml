# cbsoak scenario

http://127.0.0.1:8107 model `Qwen3-8B` started 2026-10-17T15:58:21-04:00: 3 streams x ~1000-token prompts decoding (max_tokens 800), one ~4000-token prompt (max_tokens 400) sent at t = 5.0 s, 5 repetitions (5 valid), fresh nonce per prompt, warm-up request right before each repetition. after window = [long first token + 1 s, + 8.0 s] with the long request still decoding (rows = streams + 1).

| window | n gaps | p50 ms | p95 ms | p99 ms | max ms |
|---|---|---|---|---|---|
| steady before (streams only) | 1917 | 29.3 | 29.8 | 30.8 | 59.0 |
| during the long prefill | 295 | 109.6 | 127.2 | 130.2 | 132.5 |
| steady after | 3758 | 31.7 | 32.7 | 33.7 | 63.7 |

long request TTFT under load: p50 1969.7, max 2290.1 ms (n 5); solo (warm, alone) TTFT: p50 972.7 ms (n 3); ratio of medians 2.02x (design bound 2.5x).
per-repetition p95 during the prefill (ms): 109.6, 128.9, 127.2, 119.2, 116.9.
server decode passes by rows over the whole scenario run (delta of /metrics): {'1': 69.0, '2': 48.0, '3': 1952.0, '4': 1995.0, '5': 0.0, '6': 0.0, '7': 0.0, '8': 0.0}.

| rep | valid | long TTFT ms | gaps b/d/a | during p50 | during p95 | during max | before p50 | after p50 | notes |
|---|---|---|---|---|---|---|---|---|---|
| 0 | yes | 1969.7 | 378/64/765 | 100.5 | 109.6 | 110.1 | 29.1 | 31.1 |  |
| 1 | yes | 2290.1 | 388/63/753 | 118.1 | 128.9 | 130.0 | 29.3 | 31.6 |  |
| 2 | yes | 2153.6 | 372/60/743 | 118.7 | 127.2 | 132.5 | 29.5 | 31.8 |  |
| 3 | yes | 1633.4 | 391/51/744 | 105.6 | 119.2 | 121.6 | 29.3 | 32.1 |  |
| 4 | yes | 1839.6 | 388/57/753 | 108.0 | 116.9 | 117.7 | 29.1 | 31.7 |  |
