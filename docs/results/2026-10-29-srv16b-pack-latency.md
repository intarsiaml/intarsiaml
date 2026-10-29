# srv16b: bounded packed prefill on the Spark 35B (row cap + half-queue rule)

Lane srv16b, 2026-10-29, branch `lane-srv16b/pack-latency` (on #235 = `main` ebf8192a0). spark1 (spark-4e77, GB10 sm_121),
Qwen3.6-35B-A3B NVFP4, tp1, `serve --backend cuda --max-batch 16 --max-seq 8192 --mem-budget 40`, no MTP.
**Status: draft PR.** The new default bounds the decode stall to 0.47 s (#235: 1.1 s) and keeps TTFT at 4 / 8 streams at or below
pre-#235 main, but 16 streams land at 289.4 tok/s: 0.2 % under the 290 gate, and 2.6 % below #235's 297.1.

## 1. Problem

#235 packs up to 8192 rows (seven ~1k-token prompts) into one prefill pass. Every decoding stream waits for the whole pass:
about 1.1 s once, where pre-#235 main stalls about 0.2 s per prompt. Every prompt in a pass also gets its first token only when the
pass ends. At 4 streams, the three prompts that queue behind the first one pack together, so median TTFT goes from 0.48 to 0.65 s.

Cost model from SRV16's TTFTs and this lane's W1:
- A solo ~1k prefill plus the interleaved decode step costs about 0.196 s.
- A pass of k prompts costs about 0.04 + 0.137k s.
- So most of the gain is already there at k = 3 (0.150 s per prompt) compared with k = 7 (0.143 s per prompt).

## 2. Change

Two scheduler knobs (`SchedulerConfig::pack_rows`, `pack_half`; read in `BatchedOptions::new`), both default-on. They only change
which prompts share a pass; the packed pass itself is #235's.

- `INTARSIA_PACK_ROWS`: the most body rows a packed pass may hold. The default is 3072 (`PACK_ROWS_DEFAULT`), which is three
  ~1k prompts. `0` means only the workspace bound applies (8192, #235).
- `INTARSIA_PACK_HALF`: a pass takes at most half (rounded up) of the prompts waiting to prefill. Default on; `0` turns it off.
  With it on, 3 waiting prompts run as 2 + 1 instead of 3, and 15 waiting prompts run as 3 + 3 + 3 + 3 + 2 + 1.
- `/metrics` gains `intarsia_batch_pack_passes_total` and `intarsia_batch_packed_chunks_total`.

## 3. Sweep (W1, one round, `raw/w1/`)

Throughput comes from localeval (`--sizes 1024 --gen 256 --concurrency 1,4,8,16`, prompts of about 994 tokens). Latency comes from
`scripts/srv16b/lat.py`: one burst per stream count, with distinct prompts of 870-1030 tokens and 256 forced tokens. Its gaps are
client-side inter-token gaps pooled over all streams. The pass counts come from the server's counters.

| arm | tok/s 1 / 4 / 8 / 16 | localeval TTFT median 4 / 8 / 16 | gap p99 / max (16 streams) | gaps > 0.5 s (16) | passes / chunks (16) |
|---|---|---|---|---|---|
| base (pre-#235 main, 200be3f05) | 80.0 / 168.9 / 223.2 / 278.3 | 0.47 / 0.88 / 1.67 | 217 / 228 ms | 0 | 0 / 0 |
| p8k (#235: 8192 rows) | 79.9 / 173.1 / 233.8 / 297.7 | 0.65 / 1.23 / 1.36 | 52 / 1084 ms | 10 | 2 / 15 |
| r2k (2048 rows: pairs) | 79.7 / 171.9 / 227.4 / 289.6 | 0.51 / 0.82 / 1.48 | 325 / 355 ms | 0 | 7 / 14 |
| r3k (3072: triples) | 79.5 / 173.0 / 230.6 / 290.2 | **0.65** / 0.91 / 1.56 | 52 / 488 ms | 0 | 5 / 15 |
| r4k (4096: quads) | 79.9 / 173.0 / 239.7 / 290.4 | **0.65** / 0.78 / 1.38 | 52 / 600 ms | 15 | 4 / 15 |
| h (half rule, 8192) | 80.4 / 181.0 / 229.6 / 297.9 | 0.51 / 0.85 / 1.35 | 52 / **1094** ms | 10 | 3 / 14 |
| **h3k (half rule + 3072)** | 79.9 / 171.4 / 226.7 / 291.2 | 0.51 / 0.82 / 1.57 | 230 / 477 ms | 0 | 5 / 14 |
| h4k (half rule + 4096) | 79.6 / 170.0 / 227.6 / 288.1 | 0.51 / 0.79 / 1.38 | 224 / 609 ms | 15 | 4 / 14 |

Every arm meets the p99 aim (at most ~0.5 s). Pooled over 4080 gaps, p99 hides a stall that only a few streams take. The arms
differ on the worst single stall:
- **8192-row passes** (p8k, h) leave a ~1.1 s stall.
- **4096-row passes** (r4k, h4k) leave ~0.6 s.
- **3072 rows** keep it under 0.5 s.

A cap without the half rule fails TTFT at 4 streams:
- r3k and r4k pack the three waiting prompts together, so median TTFT is 0.65 s against main's 0.47 s.
- r2k passes TTFT but runs pairs only and has the highest gap p99 (325 ms).

**h3k** is the only arm that meets every latency target. On throughput:
- At 16 streams it is level with r3k and r2k.
- At 1, 4 and 8 streams it stays above pre-#235 main.

## 4. A/B (W2, 3 interleaved rounds, round 2 reversed, `raw/w2/`)

`dev` is the branch default (h3k). `p8k` is the same binary with `INTARSIA_PACK_ROWS=0 INTARSIA_PACK_HALF=0`, i.e. #235.

| streams | base r1, r2, r3 (mean) | p8k (mean) | dev (mean) | dev vs base | dev vs p8k (#235) |
|---|---|---|---|---|---|
| 1 | 80.1, 80.0, 79.3 (79.80) | 78.7, 79.8, 79.5 (79.33) | 80.0, 80.1, 80.0 (80.03) | +0.3 % | +0.9 % |
| 4 | 169.6, 168.9, 169.4 (169.30) | 176.7, 172.5, 170.8 (173.33) | 173.1, 172.1, 174.3 (173.17) | +2.3 % | -0.1 % |
| 8 | 221.8, 225.9, 217.1 (221.60) | 233.6, 234.9, 232.2 (233.57) | 225.4, 228.7, 227.5 (227.20) | +2.5 % | -2.7 % |
| 16 | 277.9, 278.2, 277.6 (277.90) | 294.6, 298.9, 297.8 (297.10) | 289.0, 291.8, 287.5 (289.43) | +4.1 % | -2.6 % |

Latency, per arm. The localeval medians are taken over r1-r3. The lat.py values are the same in all three rounds to within 0.03 s
and 25 ms.

| | base | p8k (#235) | dev |
|---|---|---|---|
| localeval TTFT median 4 / 8 / 16 | 0.48 / 0.88 / 1.67 | 0.48-0.65 / 1.19 / 1.32-1.56 | 0.51 / 0.81-0.83 / 1.55-1.56 |
| lat.py TTFT median / p90, 4 streams | 0.57 / 0.77 | 0.63 / 0.63 | 0.51 / 0.70 |
| lat.py TTFT median / p90, 8 streams | 0.95-0.99 / 1.55-1.58 | 1.13 / 1.13 | 0.93-0.94 / 1.34 |
| lat.py TTFT median / p90, 16 streams | 1.76-1.78 / 3.04-3.05 | 1.27-1.41 / 2.26-2.27 | 1.53-1.54 / 2.34 |
| gap p99 / max, 8 streams | 193-198 / 207-209 ms | 33 / 946-961 ms | 34 / 439-460 ms |
| gap p99 / max, 16 streams | 216-220 / 225-228 ms | 51 / 1088-1106 ms | 223-227 / 468-475 ms |
| gaps > 0.5 s, 16 streams (of 4080) | 0 | 10-12 | 0 |

## 5. Gates

| gate | result |
|---|---|
| 16 streams >= 290 tok/s | **miss**: 289.43 (rounds 289.0 / 291.8 / 287.5; W1 291.2). It keeps 59 % of #235's gain over pre-#235 main |
| no stream count below main by more than 1 % at 1 / 4 / 8 | **pass** against pre-#235 main (+0.3 / +2.3 / +2.5 %). **Miss** against main = #235 at 8 streams (-2.7 %), the price of the bound |
| p99 inter-token gap <= ~0.5 s | **pass**: 0.23 s at 16 streams. Max 0.47 s (#235: 1.1 s) |
| TTFT at 4 / 8 no worse than main by more than ~0.05 s | **pass**: localeval median 0.51 vs 0.48 (+0.03) and 0.82 vs 0.88 |
| served greedy tokens (equal.py, 21 cells) base vs dev | **pass**: 21/21, 9408 ids (`raw/ident-w2/compare.log`) |
| long prompts (300..1300 tokens, 4 and 16 at once) | **pass**: 4/4, 2560 ids (`raw/ident-w2/compare-long.log`) |
| multi-chunk mix (3k..7.6k beside 1k, 4 and 8 at once) | **pass**: 6/6, 2304 ids (`raw/ident-w2-mix/compare-long.log`) |
| `real_35b_packed_prefill_equals_the_local_engine` (default knobs) | **pass**: 8 slots, 7 passes / 16 chunks; 16 slots, 8 passes / 21 chunks (`raw/w2-batch-real-packed.txt`) |
| `real_35b_batched*` | **pass**: 2 passed, 2 ignored as on main (`raw/w2-batch-real.txt`) |
| `rows_exact` | **pass**: 9 passed (`raw/w2-rows-exact.txt`) |
| `plan show` 8B / 35B / Gemma-4, json + txt | **pass**: byte-identical, 6/6 (`raw/build2.log`) |
| Mac `cargo test --workspace --release` | **pass**: exit 0 (`raw/mac-cargo-test.log`) |

## 6. Notes

- **What the 16-stream loss comes from.** Of 15 queued prompts, the first pass under #235 packs 7 while only one stream decodes.
  That pass is the whole stall (10 gaps over 0.5 s). Any rule that bounds the worst stall of each stream gives this pass up.
  Within that bound, pairs, triples and the half rule all land at 288-291 tok/s.
- **Tests.**
  - The mock scheduler (`crates/intarsia-core/tests/scheduler.rs`) can now pack. `packed_passes_follow_the_row_cap_and_the_half_rule`
    checks the pass sizes, and `batched_equals_solo_randomized_with_packing` checks identity and trace invariants under every knob
    combination.
  - The trace's prefill-budget check now allows one chunk per packed prompt.
  - With the half rule, 2 slots never pack. `batch_recurrent`'s packed test therefore sets `pack_half = false`, so that it still
    tests packing.
- **Item 2 was not attempted.** Item 2 is running a prompt's last token inside the next batched decode step, estimated at about
  +1.3 % at 16 streams. Three reasons:
  1. The first token would wait for a whole decode step (~44 ms) instead of 13 ms per packed prompt. That would push the 4-stream
     TTFT (+0.03 s now) past the 0.05 s budget.
  2. It changes `Seq`'s first-token path (the prepared token, `accept_first`, MTP / spec planning).
  3. It needs a new exactness proof: a batched decode row against the eager 1-row last-token forward.

## 7. Process

GPU windows: **2 of 3**.
- #6769 (W1, `--quiet gpu,cpu --mem 50`, 01:29-01:46, 17 min): 8-arm sweep.
- #6788 (W2, 01:57-02:29, 32 min): batch_cuda_real packed and batched, rows_exact, served identity (plain and mix), then the 3-round A/B.

There were no extra short GPU checks. CPU jobs (`--load cpu`):
- #6765: build base + dev.
- #6780: rebuild dev, build the test binaries, `plan show`.

spark1-vllm stayed up.
