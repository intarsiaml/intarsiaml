# sk3: Spark decode, intarsia against vLLM, and why the lane stops here (2026-10-26)

Lane `sk`, branch `lane-sk/spark-decode`, origin/main `85824cef` (#165). spark2 (GB10, sm_121), `Qwen3.6-35B-A3B-NVFP4`, default recipe, thinking off.
**Result: intarsia is ahead of vLLM at every measured point, so by the stop rule no profile and no code were produced.** One finding came out of the
measurement: `serve --spec mtp` failed to start on main (a regression of #162); fix is PR #166.

## Method

`localeval speed --sizes 1024 --gen 256 --reps 3` (+ `--concurrency 4,8`) for decode tok/s and per-stream rates, `localeval suite --tier adhoc --cells speed-plain`
for ITL p50 / p99 of one stream (the 1024-token row; localeval reports p99, not p95, and does not report ITL at concurrency, so concurrent ITL is given as
the mean 1000 / per-stream tok/s). Fresh `intarsia serve` per round, vLLM = spark2's own `retro`-role server (`--max-num-seqs 2 --max-num-batched-tokens 8192
--kv-cache-dtype fp8 --gpu-memory-utilization 0.32 --enable-prefix-caching`, no speculative decoding), three interleaved rounds (round 2 reversed), one
`--quiet gpu,cpu` slot each (`scripts/sk/dec-window.sh`, `dec-agg.py`; raw in `2026-10-26-sk3-spark-decode/raw/`). The `base1` window lost its MTP arm
(server failed to start, below) and its vLLM concurrency (arm spec missed `--concurrency`), so `base2` repeats b8 + vLLM with concurrency and `mtp1`
repeats plain / MTP / vLLM at one stream with the fixed build.

## Numbers (medians of 3 rounds, min-max in brackets)

| arm | 1 stream decode tok/s | ITL p50 ms | ITL p99 ms | 4 streams: aggregate tok/s (per stream) | 8 streams: aggregate tok/s (per stream) | TTFT 4 / 8 streams s |
|---|---|---|---|---|---|---|
| intarsia `serve --max-batch 8` | 84.0 (83.4-84.5) | 11.9 | 14.8 | **161.9** (43.9) | **208.9** (28.8) | 0.48 / 0.88 |
| intarsia single-sequence, plain (`mtp1`) | 83.6 (83.5-84.6) | 12.0 | 15.0 | - | - | - |
| intarsia `serve --spec mtp` (k default, `mtp1`) | **89.9** (88.2-93.3) | 0.0 (bursts) | 26.7 | - | - | - |
| vLLM `guidance` (spark2, 2 seqs) | 80.1 (80.1-80.2) | 12.5 | 15.9 | 116.1 (61.6) | 116.0 (61.4) | 2.52 / 6.92 |
| intarsia b8 vs vLLM | +4.9 % | -0.6 ms | -1.1 ms | +39 % | +80 % | |
| intarsia MTP vs vLLM | +12.4 % | | | | | |

Sources: rows 1, 4 `base2` (1-stream ITL from its suite runs); row 2, 3 and MTP-vs-vLLM from `mtp1` (vLLM there 80.0 tok/s, ITL 12.5 / 15.3); `base1` agrees (b8 83.7,
vLLM 79.9, 12.0 vs 12.5 ms). MTP delivers tokens in bursts, so its ITL p50 is 0 (several tokens per chunk) and p99 is the time between bursts; the mean
ITL is 1000 / 89.9 = 11.1 ms. That tail is inherent to speculative decoding and is not a throughput gap.

Caveat for the 4 / 8 stream rows: the fleet's `guidance` vLLM runs `--max-num-seqs 2`, so streams 3 to 8 queue (TTFT 2.5 / 6.9 s) and its aggregate is flat at
116 tok/s. That is a comparison with the deployed configuration, not with vLLM's best at 8 sequences; a vLLM with `--max-num-seqs 8` would need a second server
beside `retro` (container image not reachable from the `nvidia` user; not attempted). Proposed as an option if the 4 / 8 stream ranking matters.

## Stop rule and what is not done

* Rule 5.1 (ahead of vLLM at every point): met at 1, 4 and 8 streams, ITL p50 and p99, and with MTP. No gap to profile, so no per-kernel profile or lever plan
  was written; the earlier decode work is in `2026-10-14-cuda-decode-fusion.md`, `2026-10-22-mtp-spec-decode.md` and the 10-22 standings.
* No code in the decode path changed. Decode itself is unchanged by anything in this lane.

## Finding: `serve --spec mtp` did not start on main (fixed by #166)

Since #162 (macro prefill default 8192 rows on the CUDA 35B) `intarsia serve --spec mtp` exits at start: `invalid op call: MTP pass of 8192 rows at 0 (cap 20480, chunk
2048)` (`raw/base1/`, server log). The macro plan probe is sized above the MTP drafter head's chunk-sized workspace. Workaround on main: `INTARSIA_PREFILL_MACRO=0`.
PR #166 sets the macro rows to 0 when the MTP head is loaded (a speculative sequence never took macro passes anyway); the CPU test
`mtp_model_ignores_the_macro_prefill_knob` fails before and passes after, and the `mtp1` window ran `serve --spec mtp` on the fixed build (89.9 tok/s above). No test
had covered `serve --spec mtp` plus the recipe default on GPU; the 35B MTP check is `mtp1`.

## Fleet

#5558 build main; #5560 `base1`; #5575 `base2`; #5580 build fix; mtp1 window; probe suite #5555. Build dirs `~/intarsiaml-sk-main` and `-fix` deleted (none left on spark2). `retro` untouched.
