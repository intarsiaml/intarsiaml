# ab35b: cutover follow-up measurements, Qwen3.6-35B-A3B NVFP4, intarsiaml vs production vLLM (spark2)

## User go-ahead

2026-11-12 ~19:50 EST, verbatim: "use fable and go." This approves taking spark2's vLLM (role `retro`) down for up to 90 minutes in total for the three runs below (at most two fleetq jobs with `--takes-down retro --quiet gpu,cpu`; restore on any exit and wait for `/v1/models` 200 before releasing). Lane `ab35b`, branch `ab35b/cutover-followup` from origin/main `2939d30ac`, fleetq owner `claude-ab35b`. Build under test: `ab35b-3b2f22fef119` (origin/main at `2939d30ac` plus this lane's docs and scripts; no engine code changed). That base **predates PR #349** (merge `30bd4b05c`, 20:25:58 EST, after this lane's build and after window 1 had started), so the build lacks the chat-template fix for string tool-call `arguments` (`82d0090f6`); see the tool-call section. The build stamp is recorded here and set by `scripts/ab35b/build.sh` from the worktree HEAD; the raw files themselves do not carry it.

## Question

`docs/results/2026-11-12-ab35-intarsia-vs-vllm.md` (PR #348) measured greedy, uncached, short-output speed and left the cutover open. This lane closes its biggest caveats on the same box (spark2, GB10) with the same model, one engine at a time: multi-turn prefix-cached traffic, exact-reply equality and fixed-prompt accuracy, sampled decode, and a tool-call smoke test. A measurement; no default is changed and no cutover is recommended beyond what the numbers show.

## Setup

- **A**: production vLLM (`missioncntrl-guidance`, role `retro`, arguments as in the ab35 doc: fp8 KV, `--max-num-seqs 2`, `--enable-prefix-caching`, `qwen3_xml` tool parser, no speculation), measured live under a `--quiet gpu,cpu` slot (fleetq #9168, #9171) while it was up. `cached_tokens` comes from the usage field (`--enable-prompt-tokens-details`).
- **B**: intarsiaml `serve --backend cuda`, built on spark2 in `~/intarsia-ab35b` (fleetq #9166, `--load cpu`, `MAX_JOBS=4`, 5 min 57 s, vLLM up), `--max-seq 131072 --mem-budget 40 --profile-every 0`, shipped CUDA defaults otherwise (MTP k2, bf16 KV, prefill chunk 2048, macro 8192, tool parser `hermes`). **131072 fits**: every arm logged `context 131072 tokens`, lowest MemAvailable during any arm 78.3 GB (B-store, with its 8 GiB host store; the other arms 82.3 to 85.7 GB), 0 watchdog kills, earlyoom active. Four B arms, each a fresh server on :8191:
  - **B-batch2**: `--max-batch 2` (the arm the task asked for; it is also what ab35 measured). MTP on while one sequence decodes. Per `docs/results/2026-10-23-rb2-cuda-batching.md` batched slots keep no recurrent snapshots, so the scheduler never reuses a prefix: `cached_tokens` is 0.
  - **B-specoff**: B-batch2 with `--spec off` (GSM8K only).
  - **B-single**: no `--max-batch` (default 1 = the single-sequence engine), no store. MTP on; reuse is only the one held `SequenceState` plus device snapshots.
  - **B-store**: `--prefix-store on` (defaults: 8 GiB host). `serve --prefix-store` is **single-sequence only** (refused with `--max-batch` above 1), so this arm cannot be run with `--max-batch 2`. A configured store **disables the MTP default** (recipe, `docs/tuning.md` `INTARSIA_SPEC_DEFAULT`): plain decode. Knob: `--prefix-store on` or `INTARSIA_PREFIX_STORE`.
- Tool: localeval 0.4.0 (`/opt/code/deepeval`, git `d96f899`) for GSM8K (`quality --bench gsm8k --pin-gsm8k`, **no `--nonce`**). localeval has no multi-turn mode, no sampled-temperature speed mode and no tool-call check (`agent` drives coding harnesses, not these), so the other three measurements are small scripts in `scripts/ab35b/` that import localeval's own streaming `Client` (same TTFT, decode tok/s and `cached_tokens` definitions as `localeval speed`) and call the same OpenAI endpoint on both engines: `replay.py`, `sampled.py`, `toolsmoke.py` (stdlib only), with `compare_replies.py` for reply equality and `tables.py` for the run 1 tables. localeval's `--recipe` literal fails as a path when long ("File name too long"), so every run passes a short recipe file (`scripts/ab35b/lib.sh`).
- Windows: **vLLM was down 1281 s (20:19:05 to 20:40:26, fleetq #9169) and 1226 s (20:40:58 to 21:01:24, #9170) = 2507 s = 41.8 min of the 90**; both restored by the script's trap and waited for `/v1/models` 200 (about 3 min each, most of the figure). The scripts' per-window caps were 50 and 55 min; the actual total stayed at 41.8 min. vLLM answered 200 after both, and after the last localeval run (#9171).

## Run 1: multi-turn prefix-cached replay

Workload (`scripts/ab35b/replay.py`): 8 conversations x 6 turns, no nonce, temperature 0, thinking off, `max_tokens` 160, one request at a time. Turn 1 is a system prompt plus a notes section of about 8k tokens; every later turn appends the previous assistant answer and a new section of about 4.8k tokens plus a question, so a conversation's prompt grows from about 8k to about 33k tokens; 48 requests and about 989k prompt tokens per run. Two orders with different seeds (so no run hits the cache of another): **interleaved** (turn 1 of every conversation, then turn 2, ...; 7 other conversations come between a conversation's turns, like several users) and **sequential** (all turns of one conversation, then the next). The assistant answers in the history are replayed from vLLM's recorded replies, so every arm saw byte-identical requests (`raw/replay-trace-seed*.json`); the model still generates and is timed on every turn.

Each cell: median over the 8 conversations (min-max) for TTFT; `cached tok` is the median `cached_tokens` the server reported (fraction of the prompt).


#### interleaved (seed 0)

| turn | prompt tok | A vLLM: cached tok, TTFT s, decode tok/s | B batch2 (--max-batch 2, MTP, no reuse): cached tok, TTFT s, decode tok/s | B single (default engine, MTP, held-state reuse): cached tok, TTFT s, decode tok/s | B store (--prefix-store on, plain decode): cached tok, TTFT s, decode tok/s |
|---|---|---|---|---|---|
| 1 | 8045 | 0 (0.00), 1.177 (1.172-1.238), 75.9 | 0 (0.00), 1.131 (1.128-1.2), 106.8 | 0 (0.00), 1.137 (1.132-1.202), 108.2 | 44 (0.01), 1.258 (1.249-1.459), 42.9 |
| 2 | 13065 | 6288 (0.48), 1.141 (1.135-1.191), 74.3 | 0 (0.00), 1.926 (1.915-1.971), 102.5 | 0 (0.00), 1.94 (1.929-2.021), 105.7 | 8041 (0.61), 1.019 (1.013-1.071), 51.2 |
| 3 | 18090 | 12576 (0.69), 1.062 (1.053-1.136), 73.3 | 0 (0.00), 2.753 (2.736-2.804), 103.3 | 0 (0.00), 2.758 (2.75-2.822), 102.7 | 13061 (0.72), 1.106 (1.089-1.173), 43.9 |
| 4 | 23122 | 16768 (0.72), 1.291 (1.276-1.349), 71.1 | 0 (0.00), 3.632 (3.622-3.644), 105.5 | 0 (0.00), 3.64 (3.629-3.693), 105.5 | 18086 (0.78), 1.181 (1.17-1.245), 41.5 |
| 5 | 28146 | 23056 (0.82), 1.176 (1.164-1.224), 71.4 | 0 (0.00), 4.582 (4.571-4.651), 100.5 | 0 (0.00), 4.583 (4.575-4.651), 100.7 | 23118 (0.82), 1.263 (1.247-1.303), 46.8 |
| 6 | 33160 | 27248 (0.82), 1.427 (1.409-1.463), 70.3 | 0 (0.00), 5.638 (5.599-5.719), 96.0 | 0 (0.00), 5.644 (5.599-5.741), 96.0 | 28142 (0.85), 1.345 (1.319-1.399), 47.2 |

| arm | requests | uncached prompt tok | sum TTFT s | median decode tok/s | wall s |
|---|---|---|---|---|---|
| A vLLM | 48 | 301501 of 988989 | 58.58 | 73.0 | 165.1 |
| B batch2 (--max-batch 2, MTP, no reuse) | 48 | 988989 of 988989 | 157.73 | 101.3 | 232.9 |
| B single (default engine, MTP, held-state reuse) | 48 | 988989 of 988989 | 158.08 | 102.2 | 233.3 |
| B store (--prefix-store on, plain decode) | 48 | 265143 of 988989 | 58.06 | 44.9 | 225.7 |

#### sequential (seed 1)

| turn | prompt tok | A vLLM: cached tok, TTFT s, decode tok/s | B batch2 (--max-batch 2, MTP, no reuse): cached tok, TTFT s, decode tok/s | B single (default engine, MTP, held-state reuse): cached tok, TTFT s, decode tok/s | B store (--prefix-store on, plain decode): cached tok, TTFT s, decode tok/s |
|---|---|---|---|---|---|
| 1 | 8042 | 0 (0.00), 1.176 (1.165-1.214), 75.6 | 0 (0.00), 1.134 (1.128-1.289), 105.2 | 0 (0.00), 1.139 (1.134-1.211), 107.1 | 44 (0.01), 1.258 (1.246-1.339), 79.1 |
| 2 | 13078 | 6288 (0.48), 1.145 (1.134-1.155), 74.9 | 0 (0.00), 1.93 (1.928-2.009), 102.7 | 0 (0.00), 1.957 (1.918-2.014), 104.4 | 8038 (0.61), 1.008 (1.005-1.163), 77.3 |
| 3 | 18084 | 12576 (0.69), 1.061 (1.058-1.065), 73.7 | 0 (0.00), 2.751 (2.737-2.802), 100.6 | 8192 (0.45), 1.649 (1.646-1.657), 100.2 | 13074 (0.72), 1.079 (1.054-1.169), 75.2 |
| 4 | 23103 | 16768 (0.73), 1.301 (1.264-1.393), 71.6 | 0 (0.00), 3.632 (3.618-3.732), 105.6 | 14336 (0.62), 1.607 (1.592-1.654), 106.2 | 18080 (0.78), 1.156 (1.148-1.303), 72.9 |
| 5 | 28118 | 23056 (0.82), 1.168 (1.157-1.183), 70.9 | 0 (0.00), 4.564 (4.558-4.597), 102.1 | 20480 (0.73), 1.499 (1.46-1.539), 102.7 | 23099 (0.82), 1.23 (1.212-1.39), 70.7 |
| 6 | 33136 | 27248 (0.82), 1.409 (1.401-1.505), 70.8 | 0 (0.00), 5.589 (5.559-5.652), 95.7 | 20480 (0.62), 2.537 (2.483-2.625), 96.2 | 28114 (0.85), 1.299 (1.274-1.403), 68.0 |

| arm | requests | uncached prompt tok | sum TTFT s | median decode tok/s | wall s |
|---|---|---|---|---|---|
| A vLLM | 48 | 301134 of 988622 | 58.35 | 73.3 | 161.5 |
| B batch2 (--max-batch 2, MTP, no reuse) | 48 | 988622 of 988622 | 157.34 | 102.1 | 231.2 |
| B single (default engine, MTP, held-state reuse) | 48 | 480718 of 988622 | 83.37 | 102.7 | 157.2 |
| B store (--prefix-store on, plain decode) | 48 | 264989 of 988622 | 57.19 | 72.9 | 163.9 |


Run ids and raw: `raw/<arm>-<order>-seed<seed>.{jsonl,summary.json}` (per request rows with replies), `raw/<arm>-replay-*.out`, plan `raw/replay-plan.json`, `raw/serve-B-*.{log,metrics}`.

What it shows (all measured, 48 requests per cell, no repeats of a whole run):

- **Shipped batched path, no reuse.** B-batch2 recomputes every prompt: TTFT grows 1.13 s to 5.64 s with the prompt, against vLLM's 1.18 s to 1.43 s (vLLM caches 48 to 82 % of the prompt). Summed TTFT over the run 157.7 s vs 58.6 s (2.7x). Its decode is 96 to 107 tok/s (MTP) against vLLM's 70 to 76.
- **B-single** (the default engine, no store) reuses only the single held sequence: interleaved it reuses nothing (identical to B-batch2); sequential it reuses from turn 3 (8k to 20k tokens cached at 45 to 73 %) with an MTP decode of 96 to 107 tok/s; sum TTFT 83.4 s vs vLLM's 58.4 s. Turns 1 and 2 of a conversation get no reuse and turn 6 only 62 % (the held sequence's snapshot cut), so the TTFT at turn 6 is 2.54 s vs vLLM's 1.41 s.
- **B-store** (prefix store on) gets cache fractions above vLLM's (61 to 85 % of the prompt from turn 2) and TTFT within vLLM's range in both orders (turn 6: 1.35 s and 1.30 s vs 1.43 s and 1.41 s; sum 58.1 s and 57.2 s vs 58.6 s and 58.4 s). Its **decode is not**, and the drop has two separate parts:
  - **Losing MTP** (a configured store skips the MTP default): 96 to 107 tok/s falls to plain decode. The sequential order shows that plain rate with nothing else in the way: 79.1 / 77.3 / 75.2 / 72.9 / 70.7 / 68.0 tok/s at 8k / 13k / 18k / 23k / 28k / 33k. That is the engine's plain-decode curve measured elsewhere (ab35 `--spec off`: 83.3 at 4k, 75.4 at 16k; this doc's sampled B-batch2, which never drafts: 80.6 at 4k, 65.9 at 34k), and it is vLLM's range (70 to 76). So without MTP intarsiaml decodes at parity with vLLM, not ahead.
  - **A further drop to 41 to 51 tok/s in the interleaved order, which is not MTP.** Same server, same plain decode, same prompt sizes; even turn 1 (nothing restored) is at 42.9 against 79.1. The peak 32-token rate inside those requests is still 80 to 125 tok/s (`peak_decode_tok_s` in `raw/B-store-interleaved-seed0.jsonl`), so decode stalls rather than runs slowly: each request's decode window is about 3.5 s against about 2.2 s sequential (165.8 s against 105.1 s summed over the 48 requests, 61 s extra). The server's own counters point at the store's write side (`raw/serve-B-store.metrics`, both orders together): 394 inserts, 24.9 GB written, `insert_seconds_total` 103.4, `settle_seconds_total` 73.8; restores are negligible (54 host hits, 1.16 s in total). In the sequential order the held sequence serves every later turn (40 `local_wins`), in the interleaved order every turn changes conversation (40 host restores). The slowdown also carried into the start of the sequential run on the same server (conversation 0: 43.8 to 64.8 tok/s) before it settled. That the settle work lands inside the decode window is the likely cause, **not isolated**: `INTARSIA_PREFIX_STORE_DEBUG` was not run.
  - The 8 GiB host tier **was full and evicting** (8.58 of 8.59 GiB, 128 entries, 266 host evictions), without costing hits (4 misses of 98 lookups). A larger store was not tried.
  - **The store changes greedy replies.** B-batch2 and B-single gave identical text on 48 of 48 requests in both orders (so held-state reuse is exact here), but B-store matches them on only 20 and 21 of 48, with 0 of 8 identical at turn 1. This is the documented effect of the store's boundary-cut prefill schedule on a chunk-sensitive backend (`docs/tuning.md`, `INTARSIA_PREFIX_STORE`), not a restore error; warm-against-cold identity with the store on was not measured in this lane (the `prefix_store_cuda` gate covers it).
- **Store versus batching**: the store works only on the single-sequence engine, so the one arm that matches vLLM's multi-turn TTFT cannot serve 2 streams, and it gives up MTP by default (an explicit `--spec mtp` with the store is accepted, but a drafting request bypasses the store, so a request gets one or the other; not measured).

Verdict run 1: **vLLM is ahead for multi-turn traffic.** The shipped batched default is 2.7x worse on TTFT (3.9x at 33k tokens); the prefix store closes the TTFT gap but not as a 2-stream engine, and without MTP: plain decode is 68 to 79 tok/s (vLLM 70 to 76), and it falls to 41 to 51 when conversations interleave, a store-side stall that is separate from the MTP loss.

## Run 2: fixed-prompt GSM8K (no nonce, n=200, thinking off, temperature 0)

`localeval quality --bench gsm8k --limit 200 --max-tokens 1024 --pin-gsm8k` (pinned, hashed prompt template, identical questions on every arm; no `--nonce`). Accuracy is localeval's `gsm8k_flexible`; the replies are compared as exact text per question id (`scripts/ab35b/compare_replies.py`, `raw/gsm8k-compare.json`). The style columns classify each reply text directly: a reply that is only a number, or a worked solution.

| arm | run id | accuracy (n=200) | number-only replies: right of count | worked replies: right of count | mean output tokens |
|---|---|---|---|---|---|
| A vLLM, first | `20261112-201151-adhoc-ab35b-A1-gsm8k` | 0.610 | 75 of 150 | 47 of 50 | 43.0 |
| A vLLM, second (same request set again) | `20261112-201412-adhoc-ab35b-A2-gsm8k` | 0.585 | 66 of 147 | 51 of 53 | 44.6 |
| B-batch2, MTP (default) | `20261112-202746-adhoc-ab35b-B-default-gsm8k` | 0.710 | 58 of 114 | 84 of 86 | 72.5 |
| B-batch2 `--spec off` | `20261112-203318-adhoc-ab35b-B-specoff-gsm8k` | 0.710 | 58 of 114 | 84 of 86 | 72.5 |

| pair | identical reply text | same correctness |
|---|---|---|
| **B-default vs B-specoff** | **200 of 200** | 200 of 200 |
| **A1 vs A2 (vLLM, same request twice)** | **128 of 200** (72 differ) | 177 of 200 |
| A1 vs B-default / B-specoff | 113 of 200 | 162 of 200 |
| A2 vs B-default / B-specoff | 112 of 200 | 161 of 200 |

- **MTP is exact under greedy decoding in this test**: all 200 replies identical between the shipped default and `--spec off`, including the 1 truncated reply. This is the batched engine, which has no cache state to vary; it says nothing about the prefix store, which does change greedy replies (run 1). MTP did draft in the default arm: `raw/serve-B-default.metrics` after the whole window (replay, GSM8K, sampled, tools) shows 10851 verify steps, 21679 draft tokens proposed, 18590 accepted (all from greedy requests; the sampled run added none), and `raw/serve-B-specoff.metrics` shows 0 steps, 0 proposed, 0 accepted.
- **vLLM did not reproduce its own replies at temperature 0**: the same 200 requests sent again gave 72 different replies, 38 of them from the first character and 27 with a change of style (number-only in one run, worked in the other; 15 one way, 12 the other). **The cause is not established.** localeval sends these requests one at a time, so the lane's own traffic was not batched with itself. What differed between the runs and was not controlled: the prefix cache (the second run finds every full prompt cached, so prefill resumes from other points), any missioncntrl request batched alongside (the fleetq slot does not stop them), and the fp8 KV cache makes near-ties more likely to flip. None of these was tested.
- **Accuracy, paired on identical prompts** (`raw/compare-A1-vs-Bdefault.txt`, `compare-A2-vs-Bdefault.txt`, `compare-A1-vs-A2.txt`, localeval `compare`):

| pair | accuracy | localeval verdict | questions only the first is right / only the second is right |
|---|---|---|---|
| A1 vs A2 (vLLM twice) | 0.610 vs 0.585 | noise (2 SE 0.098) | 14 / 9 |
| A1 vs B-default | 0.610 vs 0.710 | **B better, +0.100 (2 SE 0.094)** | 9 / 29 |
| A2 vs B-default | 0.585 vs 0.710 | **B better, +0.125 (2 SE 0.095)** | 7 / 32 |

  vLLM against itself is small and balanced; vLLM against intarsiaml is larger and one-sided (also by a paired test: 9 against 29 and 7 against 32 discordant questions), so it is not the same kind of difference as A against itself.

  **What the gap is.** It is entirely the mix of reply styles. In this pinned-prompt, thinking-off regime a number-only reply is right about half the time on both engines (75 of 150 and 66 of 147 on A, 58 of 114 on B) and a worked reply nearly always (47 of 50 and 51 of 53 on A, 84 of 86 on B). B works the problem out on 86 questions, A on 50 and 53. Where both engines answer in the same style they score the same (both worked: 40 of 42 each against A1; both number-only: 58 and 57 of 106). Against A1, B switches 44 questions from number-only to worked and 8 the other way.

  **What can be claimed.** No sign that intarsiaml is less accurate than vLLM on this benchmark. Not that it is more capable: the same weights choose differently at the first token, the absolute level (0.59 to 0.71) is low because half the replies skip the working, and a regime where the model always works the problem out (thinking on, or a prompt that asks for steps) would probably erase the gap; that was not run. Why the two engines lean differently is **not explained** (bf16 against fp8 KV cache is one candidate, `--kv-dtype fp8` not tried). A systematic difference in output between engines on identical prompts is an open item, favourable here, of unknown sign on other prompts. An earlier version of this section labelled the replies that are not a correct bare number as "worked replies" (142 on B, 125 and 134 on A); the table above replaces it, and `compare_replies.py` now classifies by reply text.

Verdict run 2: MTP-vs-off exactness holds (200/200, batched engine) and MTP demonstrably ran; on identical prompts intarsiaml scores higher than vLLM by more than 2 SE (0.71 vs 0.585 to 0.61), all of it from answering with a worked solution more often (86 vs 50 to 53 of 200), which is not a capability claim; vLLM does not reproduce its own replies, cause not established.

## Run 3: sampled decode and tool calls

### Sampled decode (temperature 0.7, top-p 0.95, single stream, 3 reps, 256 forced tokens, thinking off)

`scripts/ab35b/sampled.py`; a fresh nonce-prefixed prompt per rep (no cache hit: `cached_tokens_max` 0 on every cell). localeval's size calibration lands on 4.07k, 33.9k and 68.9k real prompt tokens for the 4k, 32k and 64k targets. Decode tok/s median (the 3 reps), TTFT s median:

| size (real prompt tokens) | A vLLM decode (reps) | B-batch2 decode (reps) | B-single decode (reps) | B-single MTP acceptance (accepted / proposed, tokens per verify) | TTFT A / B-batch2 / B-single |
|---|---|---|---|---|---|
| 4k (4070) | 75.1 (74.3-75.6) | 80.6 (80.5-80.8) | 76.8 (74.5-80.9) | 57.2 %, 2.14 | 0.63 / 0.60 / 0.59 |
| 32k (33.9k) | 69.2 (69.2-69.2) | 65.9 (65.6-67.0) | 68.5 (66.1-71.6) | 62.9 %, 2.25 | 6.08 / 5.72 / 5.71 |
| 64k (68.9k) | 62.7 (62.4-63.3) | 54.8 (54.2-54.8) | 52.3 (49.3-56.6) | 63.9 %, 2.28 | 15.84 / 14.48 / 14.43 |

- **B-batch2 never speculates on sampled requests**: the spec counters did not move over the whole sampled run (0 verify steps; `raw/B-default-sampled.json`). Its sampled decode is plain batched decode. Sampled speculation (`Match`, the server log line) is a single-sequence-engine feature, so MTP acceptance at each size could only be measured on **B-single** (57 %, 63 %, 64 %), and there the sampled decode is no faster than B-batch2's plain batched decode (76.8 / 68.5 / 52.3 vs 80.6 / 65.9 / 54.8; the reps overlap at 4k and 32k). That compares two engines; a single-engine `--spec off` sampled control was not run, so this does not show what MTP alone does for sampled requests.
- Against vLLM: B-batch2 +7 % at 4k (under the 10 % noise line), -5 % at 32k (same), **-12.6 % at 64k** (54.8 vs 62.7); prefill TTFT is 5 to 9 % lower on B at every size. This is the opposite of the greedy picture (ab35: B-default +36 to +41 % over vLLM at 1k to 4k) and is the same regime a request without a temperature gets (checkpoint defaults temperature 1, top_p 0.95).

### Tool-call smoke (each engine with its own parser: vLLM `qwen3_xml`, intarsiaml `hermes`)

`scripts/ab35b/toolsmoke.py`: three tools (nested object and array arguments in `create_event`), `tool_choice` auto, temperature 0, thinking off. Per tool: 5 non-stream and 2 stream calls; each call must return exactly one `tool_calls` entry with the right name, `finish_reason` `tool_calls`, `arguments` that parse as strict JSON and hold the right values, and no raw tool markup in `content`; then the tool result goes back as a `role: tool` message (the assistant turn carrying the OpenAI-standard string `arguments`) and the final reply must contain the result value ("receipt"). Also `response_format` json_schema 5 times.

| check | A vLLM (rerun #9171) | B (B-batch2 and B-single, identical) |
|---|---|---|
| `add_numbers`: call (non-stream / stream) | 5/5, 2/2 | 5/5, 2/2 |
| `create_event`: call (non-stream / stream) | 5/5, 2/2 | 5/5, 2/2 |
| `get_weather`: call (non-stream / stream) | 1/5, 0/2 (replies in plain text "The current weather in Paris is 15°C.", a made-up answer) | 0/5, 0/2 (the same plain-text answer) |
| tool receipt, `add_numbers` / `create_event` | 5/5 and 5/5 (stream 2/2, 2/2) | **0/5 and 0/5 (stream 0/2, 0/2): HTTP 400** |
| `response_format` json_schema, strict JSON with the required keys | 5/5 | 5/5 |

- Where a tool call came back, both engines gave strict JSON with correct values (for the two working tools 14 of 14 on A's run and 14 of 14 on each of B's two runs). `get_weather` is a model prompt effect: both engines mostly answer without a call (vLLM's first run of this script gave 0/5, the rerun 1/5, so it is not deterministic either). It is not a parser difference.
- **B fails the receipt turn on every request**: `HTTP 400 tokenizer/template error: chat template: invalid operation: cannot convert value into pairs (in chat:120)` (`raw/B-single-tools.out`). The model's chat template (line 120) iterates `tool_call.arguments|items`, which needs an object, and the standard OpenAI client sends `arguments` as a JSON string; vLLM converts it before templating, and this build did not.
- **This is a known bug of the build under test, already fixed on main, not yet re-tested.** The build (`ab35b-3b2f22fef119`, base `2939d30ac`) predates PR #349 (merge `30bd4b05c`, 2026-11-12 20:25:58 EST), whose commit `82d0090f6` makes `ChatTemplate::render` parse string `arguments` that hold a JSON object when the template rejects the messages as sent; `git merge-base --is-ancestor 82d0090f6` is false for this branch. The fix has a unit test against the same template construct and the same error text. It has **not** been run against this model on a Spark: **a re-test of the receipt turns on a build that contains `30bd4b05c` is pending** (`toolsmoke.py`, one short window). Until then the measured state is: any agent loop that sends tool results back breaks on the measured build.

## Verdict per run

| run | verdict |
|---|---|
| 1 multi-turn prefix-cached replay | vLLM ahead. Shipped batched default: TTFT 2.7x worse over the run (5.6 s vs 1.4 s at 33k), no reuse. Prefix store matches vLLM's TTFT but only single-sequence and without MTP: plain decode 68 to 79 tok/s (vLLM 70 to 76), and 41 to 51 when conversations interleave (a store-side stall, not the MTP loss; cause not isolated) |
| 2 GSM8K n=200 fixed prompt | MTP and `--spec off` identical (200/200), MTP verified running. vLLM differs from itself on 72 of 200 repeated requests (cause not established). B scores higher on identical prompts (0.71 vs 0.585 to 0.61, outside 2 SE), all from more worked replies (86 vs 50 to 53); same accuracy within each style, not a capability claim |
| 3 sampled decode + tools | Batched B never drafts on sampled requests; single-engine MTP acceptance 57 to 64 %, no speed advantage over batched plain decode (no same-engine off control). B vs vLLM: +7 % at 4k, -5 % at 32k (both under the 10 % line), 13 % behind at 64k. Tool calls parse as strict JSON on both, but the measured intarsiaml build fails every tool-result turn (HTTP 400); it predates the fix in PR #349 (`30bd4b05c`), re-test pending |

## Cutover readiness (updated from the ab35 list)

Settled by ab35 and ab35b (measured, same box, same harness):

- Greedy, uncached, short-output decode and prefill: intarsiaml as shipped is ahead or at parity up to 16k (ab35).
- 131072 context loads and serves with `--mem-budget 40` (MemAvailable fell from about 110 GB with vLLM down to 82 GB at load; lowest during any arm 78.3 to 85.7 GB); 64k prompts prefill at the same speed as vLLM (14.4 s vs 15.8 s).
- MTP is exact against `--spec off` under greedy decoding (200 of 200 replies identical, batched engine), which ab35 had left unchecked.
- vLLM did not repeat itself at temperature 0 here (128 of 200 repeats identical, cause not established), so exact-reply parity with vLLM is not a testable bar. intarsiaml's replies are not invariant across its own configurations either: the prefix store changes 27 and 28 of 48 greedy replies against the store-off arms (documented chunk-schedule effect).
- The shipped multi-turn path (batched, no reuse) is far behind vLLM on TTFT; the prefix store fixes TTFT only in the single-sequence engine.
- Sampled decode at temperature 0.7: B has no decode lead over vLLM (batched B +7 % at 4k, -5 % at 32k, -13 % at 64k), and the batched engine never speculates on sampled requests.
- Tool call generation parses as strict JSON on both engines.

## Cutover blockers, ranked

What intarsiaml must gain before it can replace vLLM for this model on the Sparks, most blocking first. Sizes are rough guesses of the work, not estimates anyone committed to.

| # | blocker | evidence | size |
|---|---|---|---|
| 1 | **Tool-result turns must work on the box.** | Every receipt turn returned HTTP 400 on the measured build (28 of 28 over the two B arms). The build predates the fix (PR #349, `82d0090f6`, merge `30bd4b05c`); the fix is unit-tested only. | Small: rebuild from current main and re-run `toolsmoke.py` in one short window. A blocker until that passes, because agent traffic fails outright. |
| 2 | **Prefix reuse with 2 streams.** Production runs `--max-num-seqs 2` with prefix caching; intarsiaml has reuse only on the single-sequence engine. | Batched path: 0 cached tokens, TTFT 5.64 s vs 1.43 s at 33k, 157.7 s vs 58.6 s summed (2.7x). `--prefix-store` is refused with `--max-batch` above 1; batched slots keep no recurrent snapshots. | Large: engine work in the batched scheduler (per-slot snapshots, store lookup and capture), then this replay again with 2 concurrent conversations. |
| 3 | **Store decode stall when conversations interleave.** | Plain decode 41 to 51 tok/s interleaved against 68 to 79 sequential on the same server; peak rate unchanged (80 to 125), `settle_seconds_total` 73.8 and `insert_seconds_total` 103.4 for 61 s of extra decode time. Interleaving is the normal case with several users. | Small to find (one run with `INTARSIA_PREFIX_STORE_DEBUG`), medium to fix if the capture work has to leave the decode path. |
| 4 | **MTP together with prefix reuse.** | With the store a request either drafts or uses the store. Without MTP decode is 68 to 79 tok/s, vLLM's range (70 to 76): parity, so this blocks the speed argument for the cutover (96 to 107 with MTP), not parity. | Medium. |
| 5 | **Sampled decode at long context, and drafting on sampled requests in the batched engine.** Requests without a temperature are sampled (checkpoint default 1.0), so this is the common production case. | Temperature 0.7: +7 % at 4k, -5 % at 32k (both inside the 10 % noise line), **-12.6 % at 64k** (54.8 vs 62.7). The batched engine drafted 0 tokens on sampled requests; the single engine drafts (57 to 64 % accepted) with no measured gain, and no same-engine `--spec off` control exists. | Small first steps: `--kv-dtype fp8` at 64k, and the single-engine `--spec off` sampled control. Medium if batched sampled drafting has to be built. |
| 6 | **Unmeasured load and soak.** | Never run on intarsiaml for this model: more than 2 streams, concurrent multi-turn load, a soak, thinking-on quality and reasoning-parser equivalence, real missioncntrl traffic. | Medium, measurement only; mostly waits on 2. |
| 7 | **Unexplained output differences against vLLM.** Low: nothing measured points to worse quality. | GSM8K: B writes a worked answer on 86 of 200 against 50 to 53, same accuracy within each style. The prefix store changes greedy replies against store-off (documented). | Small: GSM8K with `--kv-dtype fp8`, and one thinking-on quality run. |

Already fine (measured, no work needed for a cutover):

- Greedy, uncached decode and prefill up to 16k: ahead or at parity (ab35).
- 131072 context with `--mem-budget 40`: loads and serves, lowest MemAvailable 78.3 GB, 0 watchdog kills; 64k prefill 14.4 s vs 15.8 s.
- MTP exactness under greedy decoding: 200 of 200 replies identical to `--spec off`.
- Single-stream multi-turn TTFT with the prefix store: at vLLM's level (1.35 s vs 1.43 s at 33k; summed 58.1 s vs 58.6 s), with higher cache fractions.
- Plain (no-MTP) decode: at vLLM's level up to 33k (68 to 79 vs 70 to 76 tok/s).
- Tool-call generation (strict JSON, right values, stream and non-stream) and `response_format` json_schema: same as vLLM.
- GSM8K accuracy on identical prompts: not below vLLM.

## Fairness and limits

- Every B cell is one run per order (48 requests); the spread shown is across the 8 conversations, not repeated runs. A was run once per order and, for GSM8K, twice. A ran before the windows and was not re-measured after the restart (ab35 showed at most 2 % drift).
- The replay history uses vLLM's recorded answers for all arms; B's own answers differ in wording and are not fed back, so B's cache behaviour is for those prompts, not for B's own continuations (with a store, a continuation sharing B's own tokens could hit differently on the token before the generation prompt).
- Prompts are synthetic filler notes, not production traffic; decode tok/s uses (n-1) over the decode window with natural stops (max 160 tokens), so it is not the forced-256 figure of ab35.
- Sampled runs use a nonce-prefixed prompt per rep (no cache) and force exactly 256 tokens (`ignore_eos`), a different regime from natural-length chat; each cell is 3 reps.
- `get_weather` and the receipt turn's wording depend on the prompt text I wrote; counts are small (5 and 2 per tool).
- Review corrections (2026-11-12, after the runs): the GSM8K style table and the reading of the accuracy gap, the split of the prefix-store decode drop into MTP loss and a store-side stall (with the store counters, which had been reported as not checked), the store's effect on greedy replies, the cause of vLLM's self-mismatch stated as open, and the tool-result failure tied to the build predating PR #349. `raw/gsm8k-compare.json` was missing and is regenerated from the committed `result.json` files; two stray `compare-A* 2026...` files holding only a traceback were removed. No new measurement was run.
- The `--quiet gpu,cpu` slot stops other fleetq jobs only, not missioncntrl requests; every localeval activity check on A read idle (`activity` in `raw/A-*summary.json`).

## Raw results

- `docs/results/2026-11-12-ab35b-cutover-followup/raw/`: replay (`*-{interleaved,sequential}-seed*.{jsonl,summary.json}`, `*-replay-*.out`, `replay-plan.json`, `replay-trace-seed*.json`), GSM8K (`<run id>/{summary.md,result.json}`, `*-gsm8k.out`, `gsm8k-compare.json`, `index.txt`), sampled (`*-sampled.{json,out}`), tools (`*-tools.{json,out}`; `A-tools-run1.*` is the first A run before the diagnostics were added), windows (`window-w1.log`, `window-w2.log`, `serve-B-*.{log,metrics}`, `arm_a.log`).
- localeval originals: `/opt/code/deepeval/results/<run id>/`; unanimis ids: A1 7f2f96ed, A2 0e9e7886, B-default c6f7dd25, B-specoff 5c3f6997.
- fleetq jobs (owner `claude-ab35b`): #9166 build, #9167 replay smoke, #9168 arm A, #9169 window 1, #9170 window 2, #9171 A tool rerun.
- Scripts: `scripts/ab35b/` (`build.sh`, `memwatch.sh`, `lib.sh`, `arm_a.sh`, `window.sh`, `replay.py`, `sampled.py`, `toolsmoke.py`, `compare_replies.py`, `tables.py`).

## Decisions

None to flip. The user's go-ahead (top of this file) is the only decision; one row in `docs/decisions/LOG.md` records it.
