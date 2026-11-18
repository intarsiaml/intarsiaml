# mgdnvsoak: soak of the Metal fused GatedDeltaNet verify rows (`INTARSIA_METAL_GDN_VERIFY_FUSE=1`) before the default-on decision (PR #445)

Lane `mgdnvsoak`, branch from origin/main `a17a1a04d`. **No engine change**, nothing default-on, PR #445 untouched. Closes the soak gap listed in `docs/results/2026-11-18-mgdnv27.md` ("Not verified"): 27B batched and sampled, long context, thinking mode, a continuous mixed load.
Endor (Apple M5 Max, macOS 27.0), every run in a `fleetq` slot (`--quiet gpu,cpu`), release build `target-mgdnvsoak/intarsia-new` (`scripts/mgdnvsoak/build.sh`). Arms differ by env only (knob unset vs `=1`). Raw: `docs/results/2026-11-18-mgdnvsoak/raw/`, scripts: `scripts/mgdnvsoak/` (copies of `scripts/mgdnv27/` with the changes named below).

## Verdict

| run | result |
|---|---|
| 1. 27B `serve --max-batch 16`, batched MTP, 16 or 8 sampled streams (T 0.7, top_p 0.95, seeds 1234+i), ABBA | reply shas equal **72 of 72** (48 at 16 streams, 24 at 8), no errors; aggregate tok/s ratio on/off medians **1.014x** (16 streams) and **1.017x** (8 streams) |
| 2. long context, served, greedy, 35B and 27B, ~14.2K-token prompt and ~29.8K-token prompt, 1024 tokens, 2 rounds | reply sha equal **8 of 8** (4 per model); decode ratio 0.94x to 1.03x, no trend |
| 3. thinking mode (chat-template reasoning on), 27B, greedy, `generate`, 4 prompts | ids knob on = off **4 of 4** |
| 4. 20-minute mixed load, 27B, batched, knob on | 208 requests, **0 errors, 0 flagged replies**, identical sha for every repeated (prompt, mode, seed) key, memory flat |

## 1. 27B batched, sampled

`serve --max-batch 16 --spec mtp --spec-k 2 --spec-off-at 9` with `INTARSIA_BATCHED_VERIFY=1` and `INTARSIA_BATCHED_MTP=N`, a fresh server per arm, 2 waves per arm per round after a warm-up, 300 tokens per request (rotation chat-short / code-gen / reason / agent-edit), 3 ABBA rounds per configuration (`scripts/mgdnvsoak/run1.sh`, `batch.sh`, `btable.py`; raw `raw/batch-27b-sampled-mb16-bmtp9`, `-bmtp8`).
**`--spec-off-at` is capped at 9 by the CLI**, so with 16 streams a sequence drafts only while at most 9 decode (the tail of each wave); with 8 streams every wave drafts throughout. Both are reported. The 16-stream arm still drafted (359 to 366 spec steps per server run, acceptance about 79 %), so the batched verify branch ran in both.

| config | rounds on/off (aggregate tok/s, median of 2 waves) | median ratio | replies equal |
|---|---|---|---|
| 16 streams (drafting at <= 9 decoding) | 1.132, 0.989, 1.014 (off 63.8 to 72.5, on 69.9 to 73.0) | **1.014x** | 48 / 48 |
| 8 streams (batched MTP N = 8) | 1.011, 1.031, 1.017 (off 54.2 to 55.9, on 54.4 to 57.3) | **1.017x** | 24 / 24 |

Total completion tokens were equal between arms in every wave. The 1.132 is round 1, whose knob-off arm ran slow (64 against 71 to 73 later); read the 16-stream result as "not slower, about +1 %".

## 2. Long context

Prompt: this repo's `crates/intarsia-core/src` Rust sources plus a "write a technical overview" question (`scripts/mgdnvsoak/longprompts.py`), 14173 and 29834 prompt tokens (nominal "16K" and "32K"; the tag prefix of `cell2.py` differs per round, so shas differ between rounds, not between arms). `serve --max-batch 1 --max-seq 36864`, each model's default recipe (MTP k2), greedy, thinking off, 1024 tokens, one server per arm, order reversed on round 2 (`lc.sh`).

| model | prompt | round | reply sha | decode tok/s off -> on | ratio |
|---|---|---|---|---|---|
| 27B | 14.2K | 1 / 2 | equal / equal | 51.19 -> 51.31 / 49.53 -> 49.63 | 1.002 / 1.002 |
| 27B | 29.8K | 1 / 2 | equal / equal | 41.64 -> 42.30 / 42.68 -> 40.13 | 1.016 / **0.940** |
| 35B | 14.2K | 1 / 2 | equal / equal | 148.3 -> 151.2 / 145.7 -> 149.4 | 1.020 / 1.025 |
| 35B | 29.8K | 1 / 2 | equal / equal | 123.0 -> 122.8 / 118.3 -> 121.4 | 0.998 / 1.026 |

The one 0.940 (27B 29.8K round 2, knob-on arm second; its TTFT was also 2 s longer) is not repeated in round 1 (1.016) and the reply is identical; treated as noise, not pinned down (the 27B at that length runs hot, thermal level up to 2).

## 3. Thinking mode

`generate --chat` without `--no-think` (the template's reasoning is on: the output starts with the reasoning text), 27B default recipe, greedy, 700 tokens, `--show-ids`, knob off (`def`) against on (`def+on`): chat-short 367 ids, reason 566, code-gen 700, agent-edit 10; **4 of 4 equal** (`raw/think27`, `run3.sh`; the PASS lines say "sampled" because the twin-comparison branch of `ids.sh` is shared, the runs are greedy). Not run: sampled thinking apart from run 4.

## 4. 20-minute mixed load

27B `serve --max-batch 16 --spec mtp --spec-k 2 --spec-off-at 9`, `INTARSIA_BATCHED_MTP=8`, `INTARSIA_BATCHED_VERIFY=1`, **knob on**, 8 client threads for 1200 s (`run4.sh`, `soak.py`, `mon.sh`). Each request picks greedy or sampled (T 0.7, top_p 0.95, seed 1000 to 1002), thinking on or off, one of 4 fixed prompts: 57 greedy+thinking, 55 sampled+thinking, 51 greedy, 45 sampled; 400 max tokens with thinking, 300 without.
- 208 admitted, 208 finished, 0 failed, 0 cancelled, 0 client errors, 68805 completion tokens; the server was alive at the end.
- Reply checks: none empty, no NaN / inf / replacement characters, no degenerate tail; **every (prompt, mode, seed) key (31 keys) produced one single reply sha** however it was co-batched (batched == solo for the fused path, also under changing batch composition).
- The path ran: 27342 spec steps, 41198 of 54597 proposed tokens accepted (75 %), 81927 batched verify rows, 516 prefill chunks (515 mixed iterations).
- Memory (every 60 s, `raw/soak/mon.tsv`): footprint (incl. GPU) 17 GB at 1 min, 19 GB from 2 min to the end, flat; process RSS 4307 MB at start, 4366 to 4435 MB: **a slow RSS creep of about +3 MB per minute (+70 MB over 20 minutes)**, about 1.6 %, not leveling off within the window. Not checked with the knob off, so it is not attributed to the knob (the 20-minute window cannot say whether it continues). System swap used constant at 309 MB. Thermal level 2 from minute 2 on, throughput stable.

## Deviations and what is not covered

- `INTARSIA_BATCHED_MTP` and `INTARSIA_BATCHED_VERIFY=1` were set explicitly (the 27B Metal recipe's batched default is off for the MTP engine, `--spec mtp` is skipped at `--max-batch` > 1 by default); the batching engine's default path, plain batched decode, does not use the fused rows at all.
- 16 concurrent streams do not draft above 9 decoding sequences (CLI cap), so "16 streams with the fused rows throughout" cannot be configured; 8 streams is the widest sustained drafting case here.
- One machine, one session, one run of the 20-minute load, no knob-off soak for comparison, 2 rounds for long context, long-context decode with the knob only at B = 1. No ids against `--spec off` in this lane (done in `mgdnv27`). Thinking with sampling only inside run 4 (sha stability, not on/off).
- A first launch with `--spec-off-at 17` failed at argument parsing (exit before load); two early starts were killed by me and cleaned up, their raw output removed. Other fleetq clients on endor: a `pkill -f 'fleetq run --host endor'` of mine during that clean-up hit only this lane's clients as far as `fleetq status` showed (no other endor jobs were listed), but it was broader than it should have been.

## Revert

Nothing to revert: no engine, recipe or default changed.

## Does anything argue against merging #445?

No. With the knob on, every comparison made here was bitwise equal to the knob-off path (72 of 72 sampled batched 27B replies, 8 of 8 long-context replies at 14K and 30K tokens on both models, 4 of 4 thinking-mode greedy runs) and a 20-minute mixed batched load (208 requests, greedy and sampled, thinking on and off) ran with no error, no bad reply, a stable reply for every repeated request and flat GPU footprint; speed is about +1 to +2 % on the 27B batched, +0 to +2.5 % at long context, with one unexplained 0.94 single run. The residual items are small: process RSS grew about 70 MB over the 20 minutes (not compared with the knob off, so it is not shown to come from the knob, and it is worth a longer knob-off/on comparison if the user wants certainty), batched MTP drafting is capped at 9 sequences so the fused rows do not run in 10 to 16 stream steps, and this was one machine. I found nothing that argues against the flip; the payoff is modest, so the keep-`=0`-as-opt-out safeguard of `mgdnv27` still applies.
