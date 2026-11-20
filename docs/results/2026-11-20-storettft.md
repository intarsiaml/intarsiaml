# storettft: why uncached prompt TTFT is worse with `--prefix-store on`, and an opt-in partial fix

Lane `storettft`, branch `storettft` from origin/main `6d2eda2e4`, 2026-11-19 23:30 to 2026-11-20 03:30 EST, fleetq owner `claude-overnight-storettft`, box spark2 (GB10, driver as in ab35d, vLLM `retro` resident and untouched, no vLLM traffic during any cell: the clock guard of every cell logs the vLLM request counter delta, all 0). Raw: `docs/results/2026-11-20-storettft/raw/`. Scripts: `scripts/storettft/`.

## Verdict

1. **The gap is real and has three separate causes, none of them dominant.** Shipping config `serve --max-batch 2 --prefix-store on`, one uncached request at a time, TTFT over the store-off control (same binary), fresh server: **+34 / +31 / +27 %** at 4k / 16k / 32k (w9; w2: +31 / +30 / +27 %; ab35d's cell, reproduced: ab35d B was +22 / +27 / +12 % against vLLM, and the store-off control was 4 to 8 % better than vLLM).
   - **A. The recipe's 8192-row macro prefill is switched off with the store** (the store's own chunk schedule runs chunk by chunk): 0.03 / 0.24 / 0.46 s, **and with it the store captures after every 2048-token chunk** instead of every 8192 tokens, which is the other half of cause C. Fix below.
   - **B. The store's schedule itself** (cuts at message boundaries: 2 to 3 extra tiny passes per request, each a full weight pass): 0.05 / 0.06 / 0.04 s. This is what the store is for (restore points); not changed.
   - **C. The settle of the previous request lands in the next request's TTFT, and in a fresh server it is slow because the host tier's pages are new.** The settle moves a request's captures from the device to host buffers; a never-used host buffer costs a page-populate (about 3.6 GB/s) plus a first-use device-to-host copy at about 7 GB/s (a reused buffer copies at 43 to 55 GB/s), so the settle after a 16k request moves 641 MiB in 600 ms (1 GB/s). One request's settle runs on the scheduler thread, so the next request waits for it: 0.10 / 0.47 / 0.44 s of TTFT. Once the host tier has been through its first eviction (buffers recycled) the same server is at **+13 / +12 / +11 %**. Captures past the 1 GiB device deferral cap download to host inside the prefill (at 32k the 14th to 33rd capture of the request, `insert ... host` in the debug log), which lands in new host pages too (about 0.6 s at 32k with an idle gap). Both are the same fresh-host-memory cost, in the prefill and after it; moving it from one to the other does not help (a 4 GiB cap alone changed nothing, below).
2. **Opt-in fix: `INTARSIA_PREFIX_STORE_MACRO=1`** (default off, docs/tuning.md row). It keeps the recipe's macro prefill on with the store, for a request that prefills alone, on the request's own store schedule (bitwise the same chunks, run as one pass; the store captures at the pass end and at message cuts). Result, fresh server, back-to-back cold requests, final binary: **+31 / +19 / +19 % (was +34 / +31 / +27 %)**; in steady state (host tier warm) **+13 / +4 / +2 %**; with an idle gap between requests **+14 / +4 / +2 %**.
3. **Gate (a), cold TTFT within 3 % of store-off at 4k / 16k / 32k, is NOT met** in the regime ab35d measured (fresh server, back-to-back). It is met at 32k only with a warm host tier or an idle gap, and never at 4k (B: the tiny schedule passes, unchanged). Gates (b), (c), (d) pass. Not claimed as a fix for the ab35d regression; it removes about half of it.
4. **What would remove the rest (not done, no safe small change found):** make the settle's host landing not pay fresh-page costs on the scheduler thread (pre-populate and first-use-copy the host tier's buffers on a helper thread, or pinned host buffers); the measured upper bound is the warm-tier column below. Cause B is the store's design.

## Method

- Cell: ab35d's. `localeval speed --structured prompt --thinking off --sizes 4096,16384,32768 --gen 256 --reps 3` (localeval 0.4.0, `/opt/code/deepeval`), nonce first in every prompt (no prefix reuse, `cached max 0` in every cell), one request at a time, back-to-back (the next request is sent the moment the previous stream ends). TTFT is localeval's (first content chunk). Each visit is a fresh `intarsia serve` (the binary built from this worktree, CUDA, `--max-seq 131072 --mem-budget 40 --profile-every 0 --max-batch 2`, plus the arm's store flag or env), 3 reps per size, the median of the 3 is the visit's value, the arm value is the mean of its visits, visits in ABBA order (A B C C B A for three arms). Run order, recipe line and env of every visit: `raw/recipes/*.txt`, `raw/window-*.log`.
- Clock guard (lane rules addendum) on every cell: clk.sh samples SM clock, utilisation and power every 5 s; cells abort on SM < 2000 MHz at > 90 % util (3 samples) or SW power cap growth > 1 s. No cell tripped: min SM 2346 MHz over all cells, mean about 2385 MHz at > 90 % util, SW power cap delta 0 us in every cell, max power 78 to 91 W (`raw/window-*.log`, `raw/clk-*.tsv`, `raw/marks-*.tsv`).
- Memory: lowest MemAvailable during any visit 35.6 GB (replay visits, store on), 40 to 47 GB in the speed cells; KILLED lines 0 everywhere; builds ran under a Spark-side watchdog (kill at < 15 GB available; lowest 71 GB). vLLM was never touched.
- Two binaries: the first build had a temporary deferral-cap knob that turned out to do nothing (arms `defer` / `smacrodefer`, windows w4, w6, w7, w8-discarded, w5); it was removed, the final build (this PR's code) ran w9 to w12. `smacro` in w7 (first build) and w9 (final) are the same arm, and agree.
- Discarded: an ids window (w8) whose prompts differed between arms (localeval's calibration request carries a random nonce, which moved the sentence count by one); replaced by w11 with a fixed sentence count. Raw of w8 removed.

## Arms

| arm | `serve` flags / env | what it isolates |
|---|---|---|
| off | `--max-batch 2` | the control: store off, recipe macro prefill on |
| offm0 | off + `INTARSIA_PREFILL_MACRO=0` | cost of losing macro prefill, store off |
| nocap | `--prefix-store min=100000000` | store lookup / schedule / settle machinery, no capture (prompts shorter than min) |
| gridoff | `--prefix-store grid=100000000` | captures at message cuts only, not every 2048 tokens |
| on | `--max-batch 2 --prefix-store on` | the shipping config |
| smacro | on + `INTARSIA_PREFIX_STORE_MACRO=1` | the fix |
| dbg / dbgsync | on + `INTARSIA_PREFIX_STORE_DEBUG=1` / `=sync` | per-insert / per-settle timing lines (not for ratios) |
| gap | any arm, `scripts/storettft/ttft_gap.py --gap 5` | same prompts and TTFT definition, 5 s idle after every request, so the previous settle finishes in idle time |
| FILL | 8 x 32k requests before the measured cell, `on,host_mb=4096` | warm host tier: buffers recycled, as in a long-running server |

Startup log with the store on and the recipe: `prefix store: the recipe's macro prefill (8192 rows) is off with --max-batch > 1` and `the backend's packed prefill is off with --max-batch > 1` (the two suspects of the brief). Packed prefill needs two prompts at once; a single request at a time never uses it, so it plays no part in this cell (it is not tested here, `INTARSIA_PACK_PREFILL=1` with the store is still refused).

## Attribution (fresh server, back-to-back, ABBA; windows w2 and w3)

TTFT s, arm mean of 2 visits; ratio to off in brackets:

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 0.590 | 2.535 | 5.700 |
| offm0 | 0.620 (1.05) | 2.750 (1.08) | 6.155 (1.08) |
| nocap | 0.670 (1.14) | 2.805 (1.11) | 6.195 (1.09) |
| gridoff | 0.755 (1.28) | 3.220 (1.27) | 7.160 (1.26) |
| on | 0.775 (1.31) | 3.300 (1.30) | 7.245 (1.27) |
| on, 5 s idle between requests (w3, 2 visits) | 0.671 (1.14) | 2.827 (1.13) | 6.810 (1.19) |
| off, 5 s idle (control) | 0.587 | 2.510 | 5.720 |

Where the seconds go (differences of the rows above, s per request):

| component | 4k | 16k | 32k | isolated by |
|---|---|---|---|---|
| A. macro prefill off with the store | 0.03 | 0.22 | 0.46 | offm0 - off |
| B. the store's chunk schedule (extra tiny passes at message cuts) | 0.05 | 0.06 | 0.04 | nocap - offm0 |
| C1. captures inside the prefill (past the 1 GiB deferral cap: synchronous download into new host pages) | 0.00 | 0.02 | 0.62 | on with gap - nocap |
| C2. the previous request's settle, still running when the next one arrives | 0.10 | 0.47 | 0.44 | on back-to-back - on with gap |
| total | 0.19 | 0.78 | 1.55 | on - off |

`nocap` has no captures, hence no settle: its extra 0.05 s over `offm0` is the schedule alone (the chunk counter confirms it: 121 prefill chunks for the 11 requests of a cell against 88 with the store off and macro off, 29 with macro on: `intarsia_batch_prefill_chunks_total`, `raw/snap-*`). `gridoff` (captures at message cuts only) saves almost nothing, so it is not the number of capture points per se but the bytes and the macro schedule.

Store counters per cell of 11 requests (`snap-*.after/before.gz`): on 88 inserts, 6.7 GB captured, insert time 20.5 s (the time the scheduler thread spends in capture, a part of it waiting on the GPU queue, per insert up to 130 ms on 2048-token chunks), settle 4.2 s; smacro 33 inserts, 4.9 GB, insert 12.7 s, settle 5.4 s (more bytes still on the device at the settle, fewer downloaded inside the prefill). TTFT sum over the 11 requests: off 26.8 s, on 32.0 s, smacro 28.4 s (w9).

### The settle in a fresh server (mechanism of C)

- `INTARSIA_PREFIX_STORE_DEBUG=1` (w1, w3 dbgsync): `settle (after done) moved 641 MiB off the device in 618 ms` after a 16k request (600 ms again for the next ones), 150 MiB in 118 to 130 ms after 4k. Per object: `demote 32160 KiB: alloc 9.1 ms, download 4.9 ms` (alloc is `host_buf`: calloc plus one write per 4 KiB page).
- Microbenchmark `scripts/storettft/d2h.cu` / `d2h2.cu` on spark2 (`raw/d2h*.txt`), per 32 MiB buffer: page-populate plus copy into a NEW pageable buffer 13.8 ms; the same copy into a buffer that was copied into before 0.75 ms (43 GB/s); pinned reused 0.56 ms (57 GB/s); `cudaHostAlloc` fresh 17 ms; `MAP_POPULATE` fresh 8.8 ms. So the cost is the first touch of a buffer (kernel page-populate about 3.6 GB/s), and it vanishes when buffers are recycled: the host tier recycles on eviction (`take_host_buf`), which a fresh server has not reached.
- Same cell, same binary, warm host tier (w4: 8 requests of 32k first, `host_mb=4096`): on is **0.675 / 2.830 / 6.295** (1.13 / 1.12 / 1.10 of off) against 0.775 / 3.300 / 7.245 fresh; settle seconds per cell 0.33 s against 4.2 s.
- ab35d's B (0.775 / 3.310 / 6.854) is the fresh regime. Its 32k cell varied 7.29 s (B1) to 6.42 s (B2) between visits; I do not know why (here the 32k cell of `on` is 7.2 to 7.3 s in all 9 fresh visits).

## The fix and its numbers

`INTARSIA_PREFIX_STORE_MACRO=1` (code: `SeqStore::macro_len` in `engine.rs`, the scheduler's `BatchSeq::macro_len` and `BatchModel::store_macro_rows`, `store_combinations` in `engine/batch.rs`, the recipe switch in `intarsia-models/src/load.rs`, the opt-in in `prefix_store.rs`):

- With `--prefix-store` and `--max-batch` > 1, the recipe's macro prefill stays on (the workspace is planned with it, about 1000 MiB more on the 35B) and an explicit `INTARSIA_PREFILL_MACRO` > 0 is allowed.
- A request that prefills alone (no other request decoding, none other prefilling) takes up to `macro_rows / chunk` (4) consecutive WHOLE chunks of its own store schedule as one layer-by-layer pass; never the prompt's last chunk (it runs whole with the head, as before), never across a message cut (the step ends at the cut, where the store captures). The pass is bitwise those chunks one by one (the existing macro-prefill property, `recipe_macro` tests), so the chunk boundaries inside it are the schedule's and every state byte is the one the store-on schedule produces.
- The store captures at the pass end and at message cuts instead of after every chunk: a cold prompt leaves entries every 8192 tokens instead of every 2048 (coarser restore points when a later prompt shares only a prefix with it: the shared prefix is restored to the last entry at or below it). Beside another request (decoding or prefilling) nothing changes: own chunks one by one as today.
- Nothing is on by default; with the variable unset the code paths are the old ones (`macro_len` is never called, `store_macro_rows` returns 0, the recipe switch is untouched).

### TTFT, final binary (w9, fresh server, back-to-back, ABBA, 3 reps)

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| off | 0.590 | 2.555 | 5.715 |
| on | 0.790 (1.34) | 3.355 (1.31) | 7.270 (1.27) |
| smacro | 0.770 (1.31) | 3.045 (1.19) | 6.780 (1.19) |

(w7, first build, same arm: smacro 0.775 / 3.035 / 6.735 = 1.31 / 1.21 / 1.18 of off; w6: 1.24 / 1.20 / 1.18.) Against ab35d's vLLM A (0.635 / 2.610 / 6.102): smacro is +21 / +17 / +11 % (on: +24 / +29 / +19 % in this run's numbers).

Other regimes (first build, with a deferral-cap knob that had no effect, so these are smacro):

| regime | arm | 4096 | 16384 | 32768 |
|---|---|---|---|---|
| warm host tier (w4, FILL 8 x 32k, `host_mb=4096`) | off | 0.595 | 2.520 | 5.695 |
| | on | 0.675 (1.13) | 2.830 (1.12) | 6.295 (1.10) |
| | smacro | 0.675 (1.13) | 2.625 (1.04) | 5.820 (1.02) |
| 5 s idle between requests, fresh server (w3 / w12, final binary) | off | 0.587 / 0.589 | 2.510 / 2.511 | 5.720 / 5.720 |
| | on (w3) | 0.671 (1.14) | 2.827 (1.13) | 6.810 (1.19) |
| | smacro (w12) | 0.673 (1.14) | 2.618 (1.04) | 5.815 (1.02) |

The deferral-cap knob (`INTARSIA_PREFIX_STORE_DEFER_MB`, 4096) alone: no change (w4: 1.13 / 1.14 / 1.10; w6: 1.35 / 1.33 / 1.28); on top of smacro: no change (w7: 1.30 / 1.21 / 1.17 against 1.31 / 1.21 / 1.18). It was removed from the code. A bigger cap alone does not help because the captures it keeps on the device just move the cost to the settle.

## Gates

(a) Cold TTFT within 3 % of store-off at 4k / 16k / 32k: **fail** in ab35d's regime (fresh server, back-to-back: 1.31 / 1.19 / 1.19); 32k passes (1.02) with a warm tier or an idle gap, 16k is 1.04 there (just outside), 4k is 1.13 / 1.14 in every regime (cause B and A at 4k, not touched).

(b) Reply ids / text:
- 2-stream replay (ab35d's plan and traces, seeds 0 and 1, 48 greedy requests per run, vLLM's recorded answers as history, `scripts/interleave/replay2.py`): smacro vs on **48 of 48 identical per run, both seeds, all 8 comparisons** (`compare_replay.py`; w5 first build and w10 final binary), and identical to ab35d's own B replies in both seeds. Cached-token fraction 0.732 in every run (the same restores).
- Cold prompts, fixed greedy prompts of 1.5k to 33k tokens (12 sizes), each sent cold and again (second one restored from the store), 64 tokens (`ids_check.py`, w11): smacro vs on **24 of 24 identical** (cold and restored replies), and the cached-token counts of the restored sends are identical for all 12 sizes (1441, 2920, ..., 34064). store off vs store on: 0 of 24 identical, off vs off (same arm twice) 24 of 24: that is the documented store-vs-off chunk schedule (LOG 2026-11-17 18:30, lane storeids), unchanged by this knob; smacro's reference is store on, which it reproduces.
- Why smacro can be bitwise store-on: its macro steps contain only whole chunks of the store schedule and never cross a cut, and a macro pass is bitwise the chunks one by one (the recipe macro property, tests `recipe_macro`, `batch_recurrent`).

(c) 2-stream replay TTFT sum and decode vs ab35d's B (seeds 0 / 1): ab35d B 79.4 s / 75.1 s TTFT sum, decode median 46.0 / 47.8 tok/s, combined 62.8 / 65.1 tok/s. This lane, final binary (w10, ABBA on smacro smacro on, 2 visits per arm): on 80.2 and 80.8 s / 74.7 and 73.7 s; **smacro 75.3 and 80.0 s / 76.4 and 73.1 s**, i.e. both seeds summed 152.4 s against on 154.7 s (-1.5 %, inside the visit-to-visit spread of 4.7 % on one cell); decode median 44.5 / 46.2 and 48.5 / 47.4 against 45.7 / 45.8 and 47.9 / 47.0; combined 62.2 / 62.7 and 65.2 / 65.3 against 62.6 / 62.3 and 65.2 / 65.0; no errors. Not worse. (Two streams rarely give a request that prefills alone, so the replay mostly exercises the "nothing changes beside another request" path, which is the point of the gate; w5 with the first build agrees.) Single-stream decode in the speed cells: smacro 118 / 100 / 84 tok/s against off 120 / 100 / 82 (4k / 16k / 32k, medians over all reps of all visits).

(d) Tests: `cargo test -p intarsia-core -p intarsia-models` plus `cargo test -p intarsia-cli --lib --tests` (includes `docs_env_vars`): **1039 passed, 0 failed, 27 ignored** on the Mac in the worktree (CPU / mock tests; the CUDA-real ones are ignored without a model). New: scheduler test `store_macro_takes_whole_own_chunks_only_for_a_request_that_prefills_alone` (alone, four-chunk prompt keeps its last chunk, model switch off, decoder beside), and `engine::store_macro_tests` over `SeqStore::macro_len` (the steps reproduce the schedule's chunk ends, never the last chunk, stop at message cuts and at cuts that fall on a chunk end). The CUDA behaviour was checked on spark2 by the cells above.

## Not verified / limits

- One model (35B NVFP4), one box, CUDA only. `INTARSIA_PREFIX_STORE_MACRO=1` on HIP (also has a macro pass) and with MTP (`--spec mtp` with the store, `mtp-store`) was not run; the knob prints a note when the model has no macro pass.
- Entries every 8192 tokens instead of every 2048 for a cold prompt: coarser restore granularity for a later prompt sharing only part of it. Not measured on the multi-turn replay beyond the cached-token fraction (0.732, identical; its prompts are conversations whose reuse points are message cuts, which are kept). A workload that shares long prefixes not ending at a message cut would restore less.
- The macro workspace (about 1000 MiB) is allocated with the knob on; the ab35d memory margins are not re-measured with it (lowest MemAvailable 35.6 GB in w5, the same as ab35d's 35.7 GB).
- Packed prefill (two prompts at once) was not tested with the store: not applicable to the single-request cell, still refused with the store.
- Idle-gap and warm-tier cells used the first build (a no-op deferral-cap knob that was removed) except the w12 gap cell. I did not rerun the warm tier with the final binary.
- vLLM comparison numbers are ab35d's, not re-measured here.

## Revert

Unset `INTARSIA_PREFIX_STORE_MACRO` (default). The change is additive: `git revert` of the commit removes the knob, the two trait hooks and the tests; nothing else depends on it.

## Raw

`raw/window-w*.log` (run logs with clock guard lines), `raw/recipes/*.txt` (serve line and env per visit), `raw/w*-speed.out` and `raw/2026*/summary.md` (localeval cells), `raw/w3-*-gap5.out` / `raw/w12-*-gap5.out` (idle-gap cells), `raw/w5-*`, `raw/w10-*` (replay), `raw/w11-*-ids.json` (reply ids), `raw/snap-*` (store and scheduler counters before / after each cell), `raw/serve-*.log`, `raw/d2h*.txt`, `raw/tables-w*.md` (generated by `scripts/storettft/tables.py`).
