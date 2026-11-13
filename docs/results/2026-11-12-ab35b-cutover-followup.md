# ab35b: cutover follow-up measurements, Qwen3.6-35B-A3B NVFP4, intarsiaml vs production vLLM (spark2)

## User go-ahead

2026-11-12 ~19:50 EST, verbatim: "use fable and go." This approves taking spark2's vLLM (role `retro`) down for up to 90 minutes in total for the three runs below (at most two fleetq jobs with `--takes-down retro --quiet gpu,cpu`; restore on any exit and wait for `/v1/models` 200 before releasing). Lane `ab35b`, branch `ab35b/cutover-followup` from origin/main `2939d30ac`, fleetq owner `claude-ab35b`. Build under test: `ab35b-3b2f22fef119` (origin/main plus this lane's docs and scripts; no engine code changed).

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
- **B-store** (prefix store on) gets cache fractions above vLLM's (61 to 85 % of the prompt from turn 2) and TTFT within vLLM's range in both orders (turn 6: 1.35 s and 1.30 s vs 1.43 s and 1.41 s; sum 58.1 s and 57.2 s vs 58.6 s and 58.4 s). Its **decode is not**: plain decode (no MTP) measured 41 to 51 tok/s median in the interleaved order and 68 to 79 sequential, against vLLM's 70 to 76. The interleaved figure is far below `--spec off` decode in ab35 (about 86 at short prompts), and the cause was not isolated (a hypothesis, untested: the store moves deferred captures off the device after the first token, inside the decode window; `INTARSIA_PREFIX_STORE_DEBUG` was not run). Defaults only (8 GiB host); whether a larger store changes anything was not tried, and no eviction counters were checked.
- **Store versus batching**: the store works only on the single-sequence engine, so the one arm that matches vLLM's multi-turn TTFT cannot serve 2 streams, and it gives up MTP.

Verdict run 1: **vLLM is ahead for multi-turn traffic.** The shipped batched default is 2.7x worse on TTFT (3.9x at 33k tokens); the prefix store closes the TTFT gap but not as a 2-stream engine, and with plain decode (41 to 51 tok/s interleaved, 68 to 79 sequential, vLLM 70 to 76).

## Run 2: fixed-prompt GSM8K (no nonce, n=200, thinking off, temperature 0)

`localeval quality --bench gsm8k --limit 200 --max-tokens 1024 --pin-gsm8k` (pinned, hashed prompt template, identical questions on every arm; no `--nonce`). Accuracy is localeval's `gsm8k_flexible`; the replies are compared as exact text per question id (`scripts/ab35b/compare_replies.py`, `raw/gsm8k-compare.json`).

| arm | run id | accuracy (n=200) | bare-number replies | worked replies right | mean output tokens |
|---|---|---|---|---|---|
| A vLLM, first | `20261112-201151-adhoc-ab35b-A1-gsm8k` | 0.610 | 75 | 47 of 125 | 43.0 |
| A vLLM, second (same request set again) | `20261112-201412-adhoc-ab35b-A2-gsm8k` | 0.585 | 66 | 51 of 134 | 44.6 |
| B-batch2, MTP (default) | `20261112-202746-adhoc-ab35b-B-default-gsm8k` | 0.710 | 58 | 84 of 142 | 72.5 |
| B-batch2 `--spec off` | `20261112-203318-adhoc-ab35b-B-specoff-gsm8k` | 0.710 | 58 | 84 of 142 | 72.5 |

| pair | identical reply text | same correctness |
|---|---|---|
| **B-default vs B-specoff** | **200 of 200** | 200 of 200 |
| **A1 vs A2 (vLLM, same request twice)** | **128 of 200** (72 differ) | 177 of 200 |
| A1 vs B-default / B-specoff | 113 of 200 | 162 of 200 |
| A2 vs B-default / B-specoff | 112 of 200 | 161 of 200 |

- **MTP is exact under greedy decoding in this test**: all 200 replies identical between the shipped default and `--spec off`, including the 1 truncated reply. MTP did draft in the default arm: `raw/serve-B-default.metrics` after the whole window (replay, GSM8K, sampled, tools) shows 10851 verify steps, 21679 draft tokens proposed, 18590 accepted (all from greedy requests; the sampled run added none), and `raw/serve-B-specoff.metrics` shows 0 steps, 0 proposed, 0 accepted.
- **vLLM is not reproducible even at temperature 0 here**: the same 200 requests sent again gave 72 different replies; most differ from the first character (a bare number in one run, a worked solution in the other), i.e. near-ties at the first token flip with the batch and cache state. Cache and batch state differ between the two runs (vLLM's prefix cache warms on the shared few-shot prefix), one possible cause, untested.
- **Accuracy, paired on identical prompts** (`raw/compare-A1-vs-Bdefault.txt`, `compare-A2-vs-Bdefault.txt`, `compare-A1-vs-A2.txt`, localeval `compare`):

| pair | accuracy | localeval verdict | questions only the first is right / only the second is right |
|---|---|---|---|
| A1 vs A2 (vLLM twice) | 0.610 vs 0.585 | noise (2 SE 0.098) | 14 / 9 |
| A1 vs B-default | 0.610 vs 0.710 | **B better, +0.100 (2 SE 0.094)** | 9 / 29 |
| A2 vs B-default | 0.585 vs 0.710 | **B better, +0.125 (2 SE 0.095)** | 7 / 32 |

  vLLM against itself is small and balanced; vLLM against intarsiaml is larger and one-sided, so this is not the same kind of difference as A against itself. B scores higher on identical prompts, by more worked-out answers and more of them correct (84 of 142 worked answers right on B vs 47 to 51 of 125 to 134 on A; bare-number replies 58 vs 66 to 75, right in every case). That is a reply-style difference in this pinned-prompt, thinking-off regime, not shown to be a capability difference; it favours intarsiaml, it is not a risk. The A-vs-B reply-text differences (87 and 88 of 200) are larger than A against itself (72 of 200).

Verdict run 2: MTP-vs-off exactness holds (200/200) and MTP demonstrably ran; on identical prompts intarsiaml's GSM8K accuracy is higher than vLLM's by more than 2 SE (0.71 vs 0.585 to 0.61), through the reply style described above; vLLM does not reproduce its own replies.

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
- **B fails the receipt turn on every request**: `HTTP 400 tokenizer/template error: chat template: invalid operation: cannot convert value into pairs (in chat:120)` (`raw/B-single-tools.out`). The model's chat template (line 120) iterates `tool_call.arguments|items`, which needs an object, and the standard OpenAI client sends `arguments` as a JSON string; vLLM converts it before templating, intarsiaml's server evidently does not. That the fix is only a server-side conversion is a reading of the error and the template, not tested. **Any agent loop that sends tool results back breaks on intarsiaml as it stands**; this is the main cutover finding of the tool test.

## Verdict per run

| run | verdict |
|---|---|
| 1 multi-turn prefix-cached replay | vLLM ahead. Shipped batched default: TTFT 2.7x worse over the run (5.6 s vs 1.4 s at 33k), no reuse. Prefix store matches vLLM's TTFT but only single-sequence, plain decode (45 to 79 tok/s vs 70 to 76), cause of the interleaved slowdown unknown |
| 2 GSM8K n=200 fixed prompt | MTP and `--spec off` identical (200/200), MTP verified running. vLLM differs from itself on 72 of 200 repeated requests. B scores higher on identical prompts (0.71 vs 0.585 to 0.61, +0.10 and +0.125, outside 2 SE; reply-style driven, not shown to be capability) |
| 3 sampled decode + tools | Batched B never drafts on sampled requests; single-engine MTP acceptance 57 to 64 %, no speed advantage over batched plain decode (no same-engine off control). B vs vLLM: +7 % at 4k, -5 % at 32k (both under the 10 % line), 13 % behind at 64k. Tool calls parse as strict JSON on both, but intarsiaml fails every tool-result turn (HTTP 400) |

## Cutover readiness (updated from the ab35 list)

Settled by ab35 and ab35b (measured, same box, same harness):

- Greedy, uncached, short-output decode and prefill: intarsiaml as shipped is ahead or at parity up to 16k (ab35).
- 131072 context loads and serves with `--mem-budget 40` (MemAvailable fell from about 110 GB with vLLM down to 82 GB at load; lowest during any arm 78.3 to 85.7 GB); 64k prompts prefill at the same speed as vLLM (14.4 s vs 15.8 s).
- MTP is exact against `--spec off` under greedy decoding (200 of 200 replies identical), which ab35 had left unchecked.
- vLLM is not deterministic at temperature 0 here (128 of 200 repeats identical), so exact-reply parity with vLLM is not a testable bar.
- The shipped multi-turn path (batched, no reuse) is far behind vLLM on TTFT; the prefix store fixes TTFT only in the single-sequence engine.
- Sampled decode at temperature 0.7: B has no decode lead over vLLM (batched B +7 % at 4k, -5 % at 32k, -13 % at 64k), and the batched engine never speculates on sampled requests.
- Tool call generation parses as strict JSON on both engines.

Still open:

1. **Tool-result turns fail (HTTP 400) on intarsiaml**: the chat template needs an object for `tool_call.arguments`; needs a server fix and a re-run of the receipt test. Blocker for agent traffic.
2. Prefix reuse with more than one stream: no combination of `--max-batch 2` and a prefix store exists; batched slots never reuse. Blocker for multi-turn traffic at the vLLM setting (`--max-num-seqs 2`).
3. Prefix-store decode: plain decode, 41 to 51 tok/s interleaved (far below vLLM's 70 to 76) and 68 to 79 sequential (vLLM's range), interleaved slowdown unexplained; store sizing (8 GiB default) and eviction not examined.
4. Sampled MTP: the single-sequence engine drafts (57 to 64 % acceptance) but showed no gain against the batched plain path; a same-engine `--spec off` sampled control was **not measured**, so MTP's value on sampled requests is open. The 64k sampled decode is 13 % slower than vLLM (bf16 vs fp8 KV is a hypothesis, `--kv-dtype fp8` not tried).
5. More than 2 streams, concurrent multi-turn load, soak and stability, thinking-on quality and reasoning parser equivalence, `--kv-dtype fp8`, and production client traffic (missioncntrl) are still unmeasured.
6. Accuracy: on identical prompts B scores higher than vLLM on GSM8K n=200 (+0.10 to +0.125, outside 2 SE), driven by reply style (more worked answers, more of them right); no equivalence and no capability difference is shown.

## Fairness and limits

- Every B cell is one run per order (48 requests); the spread shown is across the 8 conversations, not repeated runs. A was run once per order and, for GSM8K, twice. A ran before the windows and was not re-measured after the restart (ab35 showed at most 2 % drift).
- The replay history uses vLLM's recorded answers for all arms; B's own answers differ in wording and are not fed back, so B's cache behaviour is for those prompts, not for B's own continuations (with a store, a continuation sharing B's own tokens could hit differently on the token before the generation prompt).
- Prompts are synthetic filler notes, not production traffic; decode tok/s uses (n-1) over the decode window with natural stops (max 160 tokens), so it is not the forced-256 figure of ab35.
- Sampled runs use a nonce-prefixed prompt per rep (no cache) and force exactly 256 tokens (`ignore_eos`), a different regime from natural-length chat; each cell is 3 reps.
- `get_weather` and the receipt turn's wording depend on the prompt text I wrote; counts are small (5 and 2 per tool).
- The `--quiet gpu,cpu` slot stops other fleetq jobs only, not missioncntrl requests; every localeval activity check on A read idle (`activity` in `raw/A-*summary.json`).

## Raw results

- `docs/results/2026-11-12-ab35b-cutover-followup/raw/`: replay (`*-{interleaved,sequential}-seed*.{jsonl,summary.json}`, `*-replay-*.out`, `replay-plan.json`, `replay-trace-seed*.json`), GSM8K (`<run id>/{summary.md,result.json}`, `*-gsm8k.out`, `gsm8k-compare.json`, `index.txt`), sampled (`*-sampled.{json,out}`), tools (`*-tools.{json,out}`; `A-tools-run1.*` is the first A run before the diagnostics were added), windows (`window-w1.log`, `window-w2.log`, `serve-B-*.{log,metrics}`, `arm_a.log`).
- localeval originals: `/opt/code/deepeval/results/<run id>/`; unanimis ids: A1 7f2f96ed, A2 0e9e7886, B-default c6f7dd25, B-specoff 5c3f6997.
- fleetq jobs (owner `claude-ab35b`): #9166 build, #9167 replay smoke, #9168 arm A, #9169 window 1, #9170 window 2, #9171 A tool rerun.
- Scripts: `scripts/ab35b/` (`build.sh`, `memwatch.sh`, `lib.sh`, `arm_a.sh`, `window.sh`, `replay.py`, `sampled.py`, `toolsmoke.py`, `compare_replies.py`, `tables.py`).

## Decisions

None to flip. The user's go-ahead (top of this file) is the only decision; one row in `docs/decisions/LOG.md` records it.
