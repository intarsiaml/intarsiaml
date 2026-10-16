# cbsoak scenario

http://127.0.0.1:8107 model `Qwen3-8B` started 2026-10-16T10:54:16-04:00: 3 streams x ~1000-token prompts decoding (max_tokens 800), one ~4000-token prompt (max_tokens 400) sent at t = 5.0 s, 5 repetitions (5 valid), fresh nonce per prompt, warm-up request right before each repetition. after window = [long first token + 1 s, + 8.0 s] with the long request still decoding (rows = streams + 1).

| window | n gaps | p50 ms | p95 ms | p99 ms | max ms |
|---|---|---|---|---|---|
| steady before (streams only) | 1868 | 29.7 | 30.6 | 31.3 | 61.7 |
| during the long prefill | 294 | 111.8 | 130.8 | 137.1 | 145.1 |
| steady after | 3689 | 32.3 | 33.5 | 34.2 | 68.0 |

long request TTFT under load: p50 1977.5, max 2323.1 ms (n 5); solo (warm, alone) TTFT: p50 969.2 ms (n 3); ratio of medians 2.04x (design bound 2.5x).
per-repetition p95 during the prefill (ms): 110.2, 130.9, 131.7, 136.2, 123.1.
server decode passes by rows over the whole scenario run (delta of /metrics): {'1': 71.0, '2': 50.0, '3': 1950.0, '4': 1995.0, '5': 0.0, '6': 0.0, '7': 0.0, '8': 0.0}.

| rep | valid | long TTFT ms | gaps b/d/a | during p50 | during p95 | during max | before p50 | after p50 | notes |
|---|---|---|---|---|---|---|---|---|---|
| 0 | yes | 1977.5 | 370/63/753 | 101.7 | 110.2 | 111.6 | 29.6 | 31.7 |  |
| 1 | yes | 2323.1 | 376/63/738 | 120.5 | 130.9 | 133.6 | 29.9 | 32.2 |  |
| 2 | yes | 2281.7 | 358/60/719 | 127.8 | 131.7 | 136.5 | 30.4 | 33.1 |  |
| 3 | yes | 1665.6 | 385/51/744 | 106.2 | 136.2 | 145.1 | 29.4 | 32.0 |  |
| 4 | yes | 1865.6 | 379/57/735 | 107.5 | 123.1 | 124.3 | 29.6 | 32.4 |  |
