# cbsoak scenario

http://127.0.0.1:8107 model `Qwen3-8B` started 2026-10-16T12:56:24-04:00: 3 streams x ~1000-token prompts decoding (max_tokens 800), one ~4000-token prompt (max_tokens 400) sent at t = 5.0 s, 5 repetitions (5 valid), fresh nonce per prompt, warm-up request right before each repetition. after window = [long first token + 1 s, + 8.0 s] with the long request still decoding (rows = streams + 1).

| window | n gaps | p50 ms | p95 ms | p99 ms | max ms |
|---|---|---|---|---|---|
| steady before (streams only) | 1868 | 30.1 | 31.6 | 32.3 | 63.0 |
| during the long prefill | 294 | 108.5 | 128.2 | 131.2 | 133.5 |
| steady after | 3722 | 32.0 | 34.1 | 34.8 | 64.4 |

long request TTFT under load: p50 2002.3, max 2166.1 ms (n 5); solo (warm, alone) TTFT: p50 972.8 ms (n 3); ratio of medians 2.06x (design bound 2.5x).
per-repetition p95 during the prefill (ms): 111.3, 120.2, 128.6, 110.5, 129.3.
server decode passes by rows over the whole scenario run (delta of /metrics): {'1': 70.0, '2': 49.0, '3': 1951.0, '4': 1995.0, '5': 0.0, '6': 0.0, '7': 0.0, '8': 0.0}.

| rep | valid | long TTFT ms | gaps b/d/a | during p50 | during p95 | during max | before p50 | after p50 | notes |
|---|---|---|---|---|---|---|---|---|---|
| 0 | yes | 1978.8 | 355/63/744 | 103.0 | 111.3 | 111.7 | 30.6 | 31.8 |  |
| 1 | yes | 2127.0 | 382/63/720 | 111.0 | 120.2 | 120.9 | 30.2 | 33.0 |  |
| 2 | yes | 2166.1 | 352/60/731 | 119.7 | 128.6 | 133.5 | 31.1 | 32.6 |  |
| 3 | yes | 1558.3 | 388/51/768 | 101.7 | 110.5 | 112.5 | 29.1 | 30.9 |  |
| 4 | yes | 2002.3 | 391/57/759 | 114.5 | 129.3 | 131.0 | 29.1 | 31.4 |  |
