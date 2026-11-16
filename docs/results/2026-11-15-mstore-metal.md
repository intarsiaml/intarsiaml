# mstore: the prefix store with MTP on Metal (roadmap idea #8)

Lane `mstore`, branch `mstore` from origin/main `5c9cb111e`, worktree `.worktrees/mstore`, box endor (M5 Max, 128 GB,
macOS 27.0). Raw files: [`2026-11-15-mstore-metal/raw/`](2026-11-15-mstore-metal/raw/). Scripts: [`scripts/mstore/`](../../scripts/mstore/).

## Brief + progress (kept at the top so it survives compaction)

Brief (coordinator, 2026-11-15 evening, condensed):

- Goal: make the prefix store work with MTP on Metal, then build the case for default-on for `serve` on Mac. Idea #8 of
  `docs/design/2026-11-15-mac-roadmap.md` (section 2.1). CUDA side: lane mtp-store, PR #371 (`MtpState::base`: the head's
  context restarts at the restore point; no drafting inside a parked prefill chunk). Metal guard: `load.rs` near line 710
  (a configured store turns drafting off; `INTARSIA_MTP_STORE=1` forces it on).
- Step 1, gates on Metal with `INTARSIA_MTP_STORE=1`, 35B and 27B: ids warm == cold with MTP (greedy and seeded sampled,
  single and `--max-batch`); the store's reuse gates; a restart test (the store survives a server restart and restores);
  acceptance after a restore. Fix failures bitwise-neutrally.
- Step 2, measure: TTFT for a 4k / 32k shared prefix cold vs restored (MTP on); decode tok/s after a restore vs cold;
  the cost of the head catch-up at restore; memory and disk footprint at a sensible default budget.
- Step 3, only if measurement says so: persist the MTP head's KV in the entry; an ahead-of-time head catch-up.
- Step 4: PR 1 = fixes + this doc, default unchanged (coordinator merges on green gates). PR 2, separate, unmerged:
  Metal `serve` defaults to MTP-with-store (`INTARSIA_MTP_STORE` unset = on for Metal) and, if the data supports it, the
  store on by default for `serve` on Metal with a disk budget; title "... (USER DECISION, do not merge)", numbers in the body.
- **Stop rule:** if the head catch-up at restore costs more than 20 % of the saved prefill, write it up and stop.
- Rules: a restore never changes tokens; knobs unset leave `plan show --json` byte-identical; worktree and store dirs only
  under `.worktrees/mstore`; fleetq for every GPU run (`--quiet gpu`, jobs <= 32 GB, never `--force` / `--no-queue` /
  `--exclusive`); ABBA with fresh servers; plain per-chunk gap p99, not localeval's weighted ITL p99; lanes mtt / mnr / mka
  edit `model/mtp.rs`, `model/spec_pipe.rs`, `engine.rs`, `engine/batch.rs` and Metal kernels: keep edits there small.


## Summary

- **Correctness: MTP drafting on prefix-store requests works on Metal.** New gate `tests/mtp_store_metal.rs`, 35B and
  27B, `INTARSIA_MTP_STORE=1`: every restored reply equals the cold one (greedy and seeded sampled, single-sequence and
  `--max-batch 2`, and after a restart from disk); greedy ids with MTP on equal MTP off. No engine fix was needed for that.
- **Stop rule fired on the 35B at 4k.** The restart design gives the MTP head only the tokens prefilled after the
  restore point (133 in the measured turn), so acceptance after a restore drops from 60 % to 42 % on identical tokens and
  decode from 148 to 128 tok/s. That costs **23 to 32 %** of the prefill time the restore saves (break-even reply
  800 to 1100 tokens). Everywhere else it is small: 27B at 4k 3 to 4 %, both models at 32k <= 1 %. Per the brief: written
  up, neither step 3 opt-in (head KV in the entry, ahead-of-time catch-up) was built.
- **Restore speed (Apple silicon):** a 32k prefix comes back in 0.69 s (35B, 36.5k tokens from disk after a restart,
  TTFT) against 13.4 s cold with the store, 11.5 s cold without; the 27B in 1.7 s against 53 s. The disk read is from the
  page cache (no `purge` without sudo), not a cold SSD read.
- **The store costs every miss:** cold TTFT with the store on against off, mean (range over rep pairs): +30 % (25 to
  34, 35B 4k), +16 % (4 to 32, 35B 32k), +35 % (20 to 51, 27B 4k), 0 % (-7 to +6, 27B 32k); and on Metal the store-on chunk schedule changes the text against store off (3 of 4 windows, 5 of
  6 gate replies). **So no store-on default is proposed** (PR 2 only flips `INTARSIA_MTP_STORE` for Metal).
- **Fix (PR 1):** queued disk writes were lost when a store was dropped or `serve` stopped; now flushed. Bitwise neutral,
  `plan show --json` byte-identical (8B, 27B, 35B).

## Gates (step 1)

`crates/intarsia-models/tests/mtp_store_metal.rs` (new): two chats, three turns each through the chat template, thinking
on, the chats alternating so every later turn restores. Arms: reference (store on, spec off); single-sequence engine
store on + MTP default; per later prompt a fresh single and a fresh `--max-batch 2` engine (cold); store off + MTP
(reported); `--max-batch 2` store on + MTP (both turns together, then each alone); restart (disk tier, new engine).
`INTARSIA_MTP_STORE=1`; the 27B with `INTARSIA_BATCHED_MTP=1` (its recipe has no `batched_mtp` on Metal). Script
`scripts/mstore/gate.sh`. First run on `7f806cc51` (#9498, #9499; raw `raw/gate-*-premerge.log`); rerun on the branch after
the merge of main `0f1fa5ccf` (lanes mtt and mtp-snap changed MTP) with the fix (#9642, #9643; raw `raw/gate-*.log`): the same
verdicts, every check true, the same acceptance numbers.

| check | 35B (#9498, #9642; 160 tokens/reply) | 27B (#9499, #9643; 128 tokens/reply) |
|---|---|---|
| single engine: greedy ids and cached tokens, MTP on == MTP off (6 replies, 4 restored) | pass | pass |
| greedy restored == cold with MTP on (fresh engine) | pass | pass |
| seeded sampled restored == cold, single engine (4) | pass (also == the spec-off sampler) | pass |
| control: two cold seeded sampled runs agree | pass | pass |
| `--max-batch 2`: greedy ids == reference (6 together, 6 alone) | pass | pass |
| `--max-batch 2`: seeded sampled restored == cold (4) | pass (also == the spec-off sampler) | pass |
| restart: new engine restores from the disk tier, ids == reference | pass (1871 cached, 1 disk hit) | pass (1909 cached) |
| store off vs store on, same prompts (reported) | 1 of 6 equal (chunk schedule) | 3 of 6 equal |

Acceptance (the gate's prompts):

| | 35B | 27B |
|---|---|---|
| cold turns | 77.8 % | 77.3 % |
| restored, a new user turn after the restore point | 77.5 % | 61.0 % |
| store off, the head's whole context (later turns, other tokens) | 81.0 % | 66.3 % |
| all but the last 2 tokens restored from the store | 58.7 to 68.7 % | 45.2 % sampled / 54.4 % greedy |

Store reuse gate (`accuracy --mode reuse`, `scripts/mstore/reuse.sh`, raw `raw/reuse/`): see the progress notes below.
Store on, every judged row agrees 32 of 32 with the cold run, but the gate reports FAIL on its own bookkeeping (its cold
run restores from the store too; reuse lands on the store's cut points). The harness cannot judge a configured store as
written; the fresh-engine comparisons above are the store's warm == cold gate. Store off on the 35B fails `long_code_4k`
as on main (lane hrr).

## Speed (step 2)

`scripts/mstore/window.sh` (#9567 to #9570), pinned binary `345aec626` (`target/mstore-run/intarsia-measure`), endor,
`serve` single sequence, MTP default, `--max-seq 40000`. Prompts (`scripts/mstore/prompts.py`): P = system + a
document (corpus docs + code) of about 4.5k / 36.5k tokens + a question; Q = P + a fixed assistant reply + a new question
(133 tokens after P). Greedy, thinking off, `ignore_eos`, 256 tokens. A fresh server per run, order A B0 B B B0 A:
A = store off (today's default); B0 = store on, empty store; B = store on with a copy of a store holding P, after a
restart: Q (restores P from disk), an unrelated request U, Q again (restores all but 4 tokens from the host tier).
Means of 2 reps. Rep-to-rep spread: cold TTFT up to 24 % (35B 32k store off: 10.3 / 12.8 s; 27B 4k store on: 7.5 /
6.2 s), restored TTFT up to 16 %, decode up to 13 %; the catch-up costs compare B0 with B, whose decode reps agree within 8 %. Client `scripts/mstore/req.py` (per-SSE-chunk gaps, not a
weighted ITL). Table: `scripts/mstore/table.py raw`.

| model, prefix | arm | TTFT | decode tok/s | acceptance | chunk-gap p99 ms | cached / prompt |
|---|---|---|---|---|---|---|
| 35B 4k | A cold, store off | 915 ms | 146.2 | 61.1 % | 17.5 | 0 / 4619 |
| 35B 4k | B0 cold, store on | 1186 ms | 148.2 | 59.9 % | 15.4 | 0 / 4619 |
| 35B 4k | B restored from disk, new turn | **323 ms** | 128.0 | 42.0 % | 14.8 | 4486 / 4619 |
| 35B 4k | B repeat, from host | **41 ms** | 128.4 | 42.0 % | 14.8 | 4615 / 4619 |
| 35B 32k | A cold, store off | 11530 ms | 86.8 | 57.8 % | 28.2 | 0 / 36646 |
| 35B 32k | B0 cold, store on | 13400 ms | 87.8 | 64.7 % | 29.1 | 0 / 36646 |
| 35B 32k | B restored from disk, new turn | **688 ms** | 83.1 | 47.8 % | 27.1 | 36513 / 36646 |
| 35B 32k | B repeat, from host | **108 ms** | 85.9 | 48.6 % | 26.5 | 36642 / 36646 |
| 27B 4k | A cold, store off | 5063 ms | 52.5 | 64.4 % | 45.0 | 0 / 4623 |
| 27B 4k | B0 cold, store on | 6855 ms | 47.9 | 60.4 % | 48.4 | 0 / 4623 |
| 27B 4k | B restored from disk, new turn | **847 ms** | 46.3 | 50.0 % | 44.2 | 4486 / 4623 |
| 27B 4k | B repeat, from host | **67 ms** | 45.7 | 48.4 % | 44.3 | 4619 / 4623 |
| 27B 32k | A cold, store off | 53283 ms | 36.4 | 75.0 % | 74.8 | 0 / 36650 |
| 27B 32k | B0 cold, store on | 53047 ms | 36.6 | 75.0 % | 72.4 | 0 / 36650 |
| 27B 32k | B restored from disk, new turn | **1714 ms** | 34.1 | 53.8 % | 61.8 | 36513 / 36650 |
| 27B 32k | B repeat, from host | **167 ms** | 33.8 | 52.9 % | 61.2 | 36646 / 36650 |

B0 and B generate the same text (hash equal) in every window: a restore changes no token. A differs from B0 in 3 of 4
windows (the store's boundary-cut chunk schedule changes prefill bits on Metal), so A-vs-B comparisons are on other tokens.

### Head catch-up cost (the stop rule)

The restart design (lane mtp-store) has no catch-up compute: after a restore the MTP head's context starts at the
restore point. A catch-up over the restored positions is not possible from the entry (head row p needs the main model's
final hidden state of p, which the store does not keep). So the cost is the lower acceptance. Measured as (decode time
of the 256-token reply restored - cold) / (TTFT cold - TTFT restored), against B0 (identical tokens):

| model, prefix | case | extra decode (256 tokens) | TTFT saved | cost | break-even reply |
|---|---|---|---|---|---|
| **35B 4k** | new turn, from disk | 0.27 s | 0.86 s | **32 %** | 808 tokens |
| **35B 4k** | repeat, from host | 0.27 s | 1.15 s | **23 %** | 1100 tokens |
| 35B 32k | new turn, from disk | 0.16 s | 12.71 s | 1 % | 19980 tokens |
| 35B 32k | repeat, from host | 0.06 s | 13.29 s | 0 % | 53131 tokens |
| 27B 4k | new turn, from disk | 0.19 s | 6.01 s | 3 % | 8193 tokens |
| 27B 4k | repeat, from host | 0.26 s | 6.79 s | 4 % | 6721 tokens |
| 27B 32k | new turn, from disk | 0.52 s | 51.33 s | 1 % | 25282 tokens |
| 27B 32k | repeat, from host | 0.58 s | 52.88 s | 1 % | 23216 tokens |

Against today's default A (store off; other tokens, so approximate) the 35B 4k cell is worse: TTFT saved 0.59 s (disk)
/ 0.87 s (host) for about 0.25 s more decode, 28 to 42 %. The acceptance drop is deterministic (139 / 232 cold against
116 / 276 restored, the same in both reps), not window drift. The measured answer quotes the document; with only the 133
new tokens in its context the head cannot propose those quotes. **The 35B at a few thousand tokens is the main Mac chat
shape, so the stop rule's verdict stands for it; the passing cells do not offset it.** The follow-up that removes the
cost is persisting the head's KV rows in the entry (35B: about 2 KB per token, 9 MB at 4.5k; it needs a store format bump
and a restore path for `MtpState`). Not built (stop rule).

### What the `INTARSIA_MTP_STORE` flip changes for a store user (PR 2)

Today on Metal a configured store skips the MTP default, so a store user gets restore + plain decode. `scripts/mstore/flip.sh`
(#9639, #9640): store on, a fresh server per run on a copy of the same store, order M P P M; M = `INTARSIA_MTP_STORE=1`, P = `=0`
(today). All runs generate the same text (hash equal, M and P). Means of 2 reps:

| model 4k | case | TTFT M / P | decode tok/s M / P | acceptance M | request total M / P |
|---|---|---|---|---|---|
| 35B | same prompt again, restored from the host tier | 40 / 43 ms | **126.8 / 112.4 (+13 %)** | 42.0 % | 2051 / 2320 ms |
| 27B | same prompt again, restored from the host tier | 68 / 82 ms | **45.7 / 31.5 (+45 %)** | 48.4 % | 5642 / 8192 ms |
| 35B | new turn after a restart | 343 ms / 1164 ms (P: no restore, see below) | 124.0 / 120.8 | 42.0 % | 2399 / 3275 ms |
| 27B | new turn after a restart | 848 ms / 6077 ms (P: no restore) | 45.6 / 31.7 | 50.0 % | 6443 / 14126 ms |

So drafting after a restore beats today's plain decode after a restore even in the cell where the stop rule fired (35B 4k,
+13 %), and by +45 % on the 27B. The flip helps store users; the stop rule is about restored-with-MTP against cold-with-MTP.

The store's build key includes the resolved plan, which lists the MTP head's ops, so a server with MTP and one without use
different store directories (`build 84df166f...` with MTP, `673989410...` without; P found 0 entries in the store M wrote,
hence its cold "new turn" row). The text is identical, so the key is conservative; the consequence for PR 2 is that a Metal
store written before the flip is not reused after it (one cold miss per prefix).

### Footprint (default budget)

Disk tier after storing P (`raw/win-*/setup.store.json`; the store keeps an entry at each message boundary and every
2048-token grid point inside a long message; KV pages are shared between entries, the recurrent blob is per entry):

| model, prefix | entries | disk | per 32k-class conversation at the 64 GiB `disk_mb` default |
|---|---|---|---|
| 35B 4.5k | 2 | 213 MiB | |
| 35B 36.5k | 9 | 1282 MiB | about 49 fit |
| 27B 4.5k | 3 | 732 MiB | |
| 27B 36.5k | 18 | 5000 MiB | about 12 fit |

Host tier the same size (write-through); `host_mb=8192` held every run. Insert time 0.03 to 0.59 s and settle 0.02 to
0.12 s per stored prompt, so the cold-TTFT cost of the store is not the captures: most likely the extra, shorter prefill
passes of the boundary-cut schedule (on the 27B each MXFP4 pass decodes its weights once, roadmap section 1.4). Restore
time per run (two hits, disk + host): 0.06 s (35B 4k, 303 MiB) to 0.90 s (27B 32k, 4.9 GiB). Memory pressure stayed at or
above 46 % free in every run; swap used stayed at 88 to 96 MB (`raw/*/mem.log`).

Shutdown: a server stopped right after its reply (no wait for the writer) left every entry on disk at these sizes, on
the binary without the fix below (the `F` runs: `F2` restores 4486 of 4490 / 36513 of 36517 tokens). So the window cannot demonstrate the
fix; it is covered by the unit test and by reading the shutdown path.

## Fix (PR 1)

Writes still queued for the disk tier were lost when a `PrefixStore` was dropped, and at `serve` shutdown the store was
never dropped at all (the watchdog thread keeps an engine handle). `PrefixStore` now flushes in `Drop` (bounded like
`flush`, 60 s), and `serve` calls the new `Engine::flush_prefix_store` (default no-op; single-sequence and batching engines
flush their store) after the HTTP server stops. Test `store_drop_waits_for_the_disk_writer` (`tests/prefix_store.rs`):
fails without the fix (entries missing after the drop), passes with it. No computed bit changes; `plan show --json`
byte-identical to the pinned main-equivalent binary on the 8B, 27B and 35B (#9594, `raw/plan-show/`).

## Deviations and limits

- Two reps per arm, one window per cell; endor drifts 10 to 20 % window to window (lane m27). Every comparison above is
  inside one window; rep-to-rep spread is up to 24 % on cold TTFT (so the store's cold overhead is a range, not a point).
- The disk restore reads the page cache; a cold SSD read was not measured.
- `--max-batch` was gated (ids) but not timed.
- The 27B batched arms need `INTARSIA_BATCHED_MTP=1` (opt-in on the 27B).
- The reuse gate cannot judge a store (above); a store-aware reuse mode is a leftover.
- Gate thinking on, window thinking off (the window's question asks for quotes, the case where the head's lost context
  costs most).

## Progress notes

- 21:04 build #9478 (CLI), #9494 (gate test). New gate `crates/intarsia-models/tests/mtp_store_metal.rs`; scripts
  `scripts/mstore/{gate.sh,window.sh,req.py,prompts.py}`.
- Queued: gates #9498 (35B), #9499 (27B); windows #9502 (35B 4k), #9503 (35B 32k), #9504 (27B 4k), #9505 (27B 32k).
- Definition of the head catch-up cost (the restart design has no catch-up compute; the head starts empty at the restore
  point, so the cost is lower acceptance): decode time over the same tokens restored minus cold, divided by TTFT cold minus
  TTFT restored, per cell (4k / 32k x new turn after the restore point / repeated prompt).
- 21:50 gate 35B #9498: **PASS** (1 passed, 199 s; `raw/gate-35b.log`). Single greedy, cold-engine greedy, single
  sampled restored == cold, batched greedy, batched sampled restored == cold, restart from disk: all true. Store off gives
  other ids than store on (5 of 6), as documented for Metal (the boundary-cut schedule changes prefill bits).
  Acceptance: cold turns 77.8 %, restored turns with a new turn 77.5 %, store-off whole head 81.0 % (later turns);
  all but 2 tokens restored from the store 58.7 to 68.7 % (63.0 % greedy batched, 63.3 % sampled).
- 22:0x gate 27B #9499 (`INTARSIA_BATCHED_MTP=1` for the batched arms, opt-in on the 27B): **PASS** (1 passed, 385 s;
  `raw/gate-27b.log`), all six checks true. Acceptance: cold turns 77.3 %, restored turns with a new turn 61.0 %
  against 66.3 % with the whole head context (store off; not the same tokens), all but 2 restored 45.2 % (sampled) /
  54.4 % (greedy, batched).
- 22:45 reuse gate 35B #9514 (`raw/reuse/35b-*.out`, chunk 1024, 4k golden): store off (main's default, MTP on, the
  engine's own hybrid reuse) FAILS `long_code_4k continue` (3 of 32 agree, KL 1.6e-1), the known failure on main (lane hrr,
  `2026-10-29-hrr-hybrid-reuse.md`). Store on: all 9 rows agree **32 of 32** with the "cold" run, but the gate reports FAIL
  on its own bookkeeping, not on tokens: "the cold run was not cold" (its cache flush replaces the held sequence, not the
  store, so the cold run restores too), "reused N > the shared prefix" (an earlier case of the same prompt left a longer
  entry; the store compares every token id, so it is the same prefix), "reuse covered 1024 < 2000" (the store resumes only
  at its own cut points). So `accuracy --mode reuse` cannot judge a configured store as written; the store's warm == cold
  gate is the fresh-engine comparison of `tests/mtp_store_metal.rs`. Leftover: a store-aware reuse mode.
- 22:40 pinned measurement binary `target/mstore-run/intarsia-measure` from `345aec626` (no local crate changes).
- PR 2 draft on branch `mstore-default` (worktree `target/pr2wt`): `mtp_store_default_for` = cuda | metal, tests and docs.
- 23:0x reuse gate 27B #9515 (`raw/reuse/27b-*.out`, chunk 512; the golden's longest prompt is 700 tokens, so only
  `long_prose_700` is judged): store off PASS; store on: restore ok, aligned ok (logits bit-identical to cold), continue
  32 of 32 agree but FAIL "reuse covered 512 < the 698 tokens of turn 1" (the store resumes at its cut point).
- PR 2 cpu tests (`target/pr2wt`, #9586 and rerun): `tp_mtp` 13 passed, `mtp` 51 passed (1 ignored), `docs_env_vars` 2
  passed after adding rows for this lane's two test knobs.
- 23:2x **fix** `ad5870fb0`: writes still queued when a store is dropped (or a server stops: its watchdog keeps an engine
  handle, so the store was never dropped) were lost. `PrefixStore` now flushes on drop, `serve` calls the new
  `Engine::flush_prefix_store` after the HTTP server stops. New test `store_drop_waits_for_the_disk_writer` fails without
  it, passes with it. CPU suites on the branch (#9592): core lib 204, `prefix_store` 34, `mtp` 51, `batch_engine` 9,
  server 53 / 58 / 3, `docs_env_vars` 2, all passed. `plan show --json` check queued (#9594).
- 01:0x PR 1 = #398 (this branch: gate, fix, results; default unchanged). PR 2 = #399 (`mstore-default`, based on
  `mstore`: `INTARSIA_MTP_STORE` unset = on for Metal; USER DECISION, do not merge). No store-on default proposed.
