# Radiance on tatooine: what context lengths does it actually serve?

Date: 2026-10-19. Read-only investigation. Nothing restarted or reconfigured, no prompts sent, no GPU load, no fleetq slot needed.

## Bottom line

There is no organic traffic history to size a decode-kernel default against. Of the 438 requests logged since 2026-10-08 that are not restart smoke tests, 328 are benchmark sweeps from endor (this Mac) at hand-picked fixed lengths (about 0.5k, 4k, 16.5k, 33k, 63k and 170k prompt tokens), and 78 are tatooine-local probes. The only plausibly real client is Mission Control (`missioncntrl`, 192.168.1.212): 32 requests on 2026-10-08, -11 and -12, which carry about 3.6 % of the non-smoke decode tokens. All 32 are long-context agent turns (see below). That is too small a sample to decide on, but it points the same way as the "agentic means long" assumption. The last non-smoke request was 2026-10-17 15:11; every request on 10-18 and 10-19 is a restart smoke test.

Mix of the logged decode tokens, in order of what they say about demand:

| Source | Requests | Gen tokens | What it measures |
|---|---|---|---|
| Restart smoke tests (193 starts x 19) | 3667 | 501 606 | nothing: 17-300 token prompts, health check |
| endor benchmark sweeps | 328 | 162 351 | what was benchmarked, not what is demanded |
| tatooine-local probes | 78 | 14 838 | partial smoke/probe runs, all <= 1066 tokens |
| missioncntrl (agent) | 32 | 6 593 | the only organic-looking client |

## Caveats on the question itself

- The service does not serve Qwen3.6-35B-A3B. `docker inspect radiance` shows `--model /models/qwen3.8-27b-mxfp4.rad --tp 2 --max-num-seqs 16 --max-model-len 200000 --kv-cache-dtype fp8 --num-speculative-tokens 7 --served-model-name Qwen3.8`, i.e. Qwen3.8-27B MXFP4, TP=2 over both R9700s (fleetq: `radiance-rank0`, `radiance-rank1`). Anything measured here is a 27B dense-model workload.
- The container image is `sha256:51b3fdfd...` run via `radiance-native.sh` (the unit description says "stilldeadcode/radiance"). I did not verify that this is the intarsiaml HIP engine. Check this before reading the numbers as intarsiaml's.

## Sources used

1. Live `/metrics` (`http://tatooine:8080/metrics`, one GET). Prometheus-style, vLLM-compatible names. It has `vllm:request_prompt_tokens` and `vllm:request_generation_tokens` histograms. **Not useful for history**: the engine restarted at 2026-10-19 21:40:19 EDT, three minutes before the query. The counters hold only that boot's smoke test: 19 requests, 3031 prompt tokens, 2605 generated, all 19 prompts <= 500 tokens.
2. `docker logs radiance`: the container is three minutes old, so no history.
3. `journalctl --user -u radiance` on tatooine, copied read-only over ssh: 257 272 lines, 2026-09-25 20:25 to 2026-10-19 21:44 EDT. This is the useful source.
   - Native engine era (2026-10-08 09:35 onward): one line per finished request with the client address, `prompt=N (cache M) gen=K` and finish reason. 4105 requests. `prompt` includes the cached prefix, so prompt + gen is the final KV length. This is the dataset below.
   - vLLM era (2026-09-25 to 2026-10-11): only HTTP access lines (no token counts) and 10-second throughput lines. POST /v1/chat/completions by client: endor 10 787, tatooine-local 4660, docker bridge 1166, missioncntrl 376, iphone 11. No per-request lengths exist for this era.
4. Fleet Prometheus on `obs:9090` (v3.15, 180 d retention): scrape jobs are grafana, llamacpp (tatooine:8010-8012, down), loki, mc-exporter (down), prometheus and `vllm` on spark-4e77:8000. **tatooine:8080 is not scraped**, so there is no Prometheus history for radiance. Loki on `obs:3100` returns no labels, so it holds no logs.
5. Experiment that failed: I tried to infer decode-weighted context for the vLLM era from "GPU KV cache usage x KV capacity / running requests" in the 10 s throughput lines. It reports a mean context of at least 12k in every window, including days with only 300-token smoke traffic, so the KV percentage includes retained prefix-cache blocks. I discarded it.

## Results (native engine era, 2026-10-08 to 2026-10-19)

Smoke tests are the fixed 19-request sequence run after every start (prompts 17, 42 x3, 53 x2, 300 x9, 18, 21 x3). Radiance was started 193 times in these 11.5 days, from 203 engine PIDs.

Decode share means the share of generated tokens produced while the KV length was at least the threshold. It is token-weighted, not step-weighted: with 7 speculative tokens one step emits several tokens, so treat it as a close proxy for the decode-step share.

| Slice | Requests | Prompt p50 / p90 / max | Gen p50 / p90 / max | Total ctx p50 / p90 / max | Gen tokens |
|---|---|---|---|---|---|
| All incl. smoke | 4105 | 300 / 301 / 169 846 | 27 / 1024 / 1748 | 327 / 1066 / 170 358 | 685 388 |
| Excluding smoke | 438 | 991 / 21 982 / 169 846 | 512 / 1024 / 1748 | 1503 / 22 102 / 170 358 | 183 782 |
| endor (benchmarks) | 328 | 992 / 16 655 / 169 846 | 512 / 1024 / 1024 | 1504 / 16 911 / 170 358 | 162 351 |
| tatooine-local (127.0.0.1) | 78 | 42 / 300 / 301 | 27 / 1024 / 1024 | 326 / 1066 / 1066 | 14 838 |
| missioncntrl | 32 | 21 982 / 52 451 / 54 817 | 50 / 298 / 1748 | 22 102 / 52 481 / 54 848 | 6 593 |

Share of decode tokens at KV length at least T, and share of requests whose final length is at least T:

| Slice | Decode >= 4k | >= 8k | >= 12k | >= 16k | Requests final ctx >= 4k / 8k / 12k / 16k |
|---|---|---|---|---|---|
| All incl. smoke | 10.6 % | 2.8 % | 2.8 % | 2.8 % | 3.4 / 1.5 / 1.5 / 1.5 % |
| Excluding smoke | 39.5 % | 10.5 % | 10.5 % | 10.5 % | 32.2 / 14.2 / 14.2 / 14.2 % |
| endor (benchmarks) | 40.8 % | 7.9 % | 7.9 % | 7.9 % | 34.8 / 10.7 / 10.7 / 10.7 % |
| tatooine-local | 0 % | 0 % | 0 % | 0 % | 0 % |
| missioncntrl | 97.3 % | 97.3 % | 97.3 % | 97.3 % | 84.4 % |

The 8k, 12k and 16k columns are identical because benchmark prompts jump from about 4k straight to 16.5k, 33k, 63k and 170k; nothing sits between 8k and 16k. Below 2k the non-smoke decode share is only 60 % (the 2k column is 39.8 %), and the smoke-inclusive stream is 89 % below 4k.

Final-context histogram, non-smoke only:

| Final ctx | Requests | Gen tokens |
|---|---|---|
| < 1k | 96 | 9 913 |
| 1k - 2k | 201 | 100 781 |
| 2k - 4k | 0 | 0 |
| 4k - 8k | 79 | 53 858 |
| 8k - 16k | 0 | 0 |
| 16k - 32k | 21 | 3 680 |
| 32k - 64k | 25 | 7 358 |
| 64k - 200k | 16 | 8 192 |

What the long requests are:

- endor: 78 requests near 4k (benchmark), 6 at 16.5k, 6 at 33k, 6 at 62.9k (2026-10-12), and 16 at 169.7-169.8k prompt with 512 generated (2026-10-16 14:10-14:27, a max-context sweep). The p90 prompt of 16.6k comes from this sweep design.
- missioncntrl: 2026-10-08 21:57 is a tool-call loop with growing context from 19k to 55k, 15-250 tokens generated per turn, prefix cache hits of 16k-52k per turn (20 of the 32 requests). The other turns are one-offs at 19-22k. Per-turn generation is short, so even in that agent loop decode is a minor share of time; prefill and cache hits dominate.

## Per-day activity (non-smoke requests)

endor: 10-08 1, 10-09 12, 10-12 139, 10-13 116, 10-16 60. tatooine-local: 10-11 27, 10-13 16, 10-17 35. missioncntrl: 10-08 23, 10-11 1, 10-12 8. Nothing after 10-17 15:11.

## What this means for the decode-kernel default

- No evidence that most traffic is long context. The recorded long-context decode tokens come almost entirely from a benchmark sweep chosen by the person asking the question.
- The only organic-looking client is entirely at 19k+ context, but it produced 6.6k decode tokens in total (1 % of all logged decode tokens, 3.6 % of non-smoke).
- If you must pick today: either behaviour is within about 1.5 % at the lengths measured, and the log does not show demand at lengths where the difference matters. Mission Control's agent turns (about 22k mean total ctx, short generations) argue for the long-context-favourable default; the 4k/0.5k benchmarking and smoke test argue for neutral. Collect real data before treating this as a decision.

## Data missing, and the smallest change that would record it

Not recorded anywhere that persists across restarts:

1. Radiance restarts about 17 times a day (193 starts in 11.5 days, benchmark rounds), which resets the in-process histograms every time. Nothing scrapes `tatooine:8080/metrics`.
2. The existing histogram buckets are coarse above 2k (`le` 2000, 5000, 10000, 20000, 50000...), so the 12k and 16k thresholds can only be bracketed, never read.
3. Nothing separates organic from benchmark traffic except the source IP, which is only in the journal.
4. Decode-step context is not recorded; only per-request prompt and generation counts are.

Smallest changes (none made):

- Add a Prometheus scrape job for `tatooine:8080` on obs (`/etc/prometheus` on `obs`; same shape as the `vllm` job for spark-4e77). Prometheus handles counter resets, so `increase(vllm:request_prompt_tokens_bucket[7d])` then works across restarts. This alone gives the per-request prompt and completion distributions, bucketed at 5k/10k/20k.
- Optional: add `le` buckets at 4096, 8192, 12288 and 16384 to the engine's `request_prompt_tokens` and `request_generation_tokens` histograms, plus a "request client" label or a header-derived `x-client` tag, to read the thresholds directly and filter benchmark traffic.
- Without any change, rerun `tools/radiance_ctx_stats.py` on a fresh journal export; the journal (`journalctl --user -u radiance`) already keeps the per-request line and has not been rotated since 2026-09-25. Journald rotation could still drop it eventually.

## Reproduce

```
ssh tatooine 'journalctl --user -u radiance --no-pager -o short-iso' > journal.txt
python3 -I tools/radiance_ctx_stats.py journal.txt --markdown \
  --clients "100.92.253.46=endor,192.168.1.212=missioncntrl,127.0.0.1=tatooine-local"
```

The script is `tools/radiance_ctx_stats.py`. The tables above come from it; the final-context histogram and the vLLM-era access-log client counts came from throwaway scripts in the session scratchpad.

## Side effects of the investigation

A handful of read-only HTTP GETs: tatooine:8080/metrics, obs:9090 and obs:3100 (Prometheus and Loki APIs), and obs:3000 (Grafana login redirect). While looking for a Prometheus I also sent unanswered connection attempts (GET /-/healthy on ports 9090 and 3000) to localhost, spark1 and endor; all failed with no connection, so nothing was read or changed. Anything on spark1 or endor was outside the scope you gave me, so I note it here.
