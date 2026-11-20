# storeflip: `INTARSIA_PREFIX_STORE_PREFAULT` and `INTARSIA_PREFIX_STORE_MACRO` default-on on CUDA

Lane `storeflip`, branch `storeflip` from origin/main `a2c83ca47`, 2026-11-20 06:55 to 10:30 EST, fleetq owner `claude-storeflip`, box spark2 (GB10, vLLM `retro` resident and untouched, 0 vLLM requests in every cell), Qwen3.6-35B-A3B NVFP4, `serve --max-batch 2 --prefix-store on` (+ `--max-seq 131072 --mem-budget 40 --profile-every 0`). Raw: `docs/results/2026-11-20-storeflip/raw/`. Scripts: `scripts/storeflip/` (copies of `scripts/storeprefault/` with arms `dflt` / `zero` / `ref` and a `restore` mode).

The user approved the flip (2026-11-20 ~06:55 EST, "yes flip prefault and macro"). This lane made it and ran the gates. **Gate 3's restore-point check shows a cost of the MACRO half the user may not have in mind, so the PR is NOT merged; see Verdict 2.**

## Verdict

1. **The code does what was asked.** Unset = on on CUDA for both knobs (`PREFAULT` unset = `1`, huge-page mode), `=0` = off (the old behaviour), `PREFAULT=4k` kept. On Metal, HIP and cpu the unset default stays OFF: both knobs were measured on CUDA / GB10 only, and the loader records the backend (`prefix_store::set_backend`, same pattern as `spec::set_copy_backend`) before any store or macro plan is made. An explicit `1` still turns either on on any backend.
2. **Gate 3 (MACRO's restore-point check) is the problem.** With MACRO on, a cold single long message leaves entries only every 8192 tokens (the macro pass ends), not every 2048. A later prompt that shares only the first part of that message restores to the last entry at or below the shared length:

   | shared prefix (tokens) | cached, `=0` (entries every 2048) | cached, default (MACRO) | TTFT `=0` | TTFT default |
   |---|---|---|---|---|
   | 2930 | 2051 | **0** | 0.22 s | 0.53 s |
   | 4992 | 4099 | **0** | 0.23 to 0.30 s | 0.80 s |
   | 9078 | 8195 | 8195 | 0.26 s | 0.26 s |
   | 13160 | 12291 | **8195** | 0.29 s | 0.90 s |
   | 17276 | 16387 | 16387 | 0.32 to 0.34 s | 0.31 to 0.32 s |

   (2 visits per arm, ABBA, one long 20k-token message per shared size, 1 reuse request each; the two visits of an arm agree to 0.01 s and to the token.) Reuse falls to **0 where it was 2051 and 4099 tokens** (shares under 8192 tokens of a long cold single message), and loses 4096 tokens (0.29 s to 0.90 s TTFT) at 13k. The brief names "reuse drops to 0 where it was > 0" as a failure of this gate. Multi-turn chats are not affected (their reuse points are message cuts, which MACRO keeps: replay cached fraction 0.732 in every arm), nor is a share that lands on or just above an 8192 multiple. Pasted-document workloads where later prompts share a prefix of a long first message do lose reuse. So the default flip is a trade: cold-prompt TTFT -13 / -21 / -20 % (Verdict 4) against coarser restore points for long single messages. **The user should decide this with the table in view; `INTARSIA_PREFIX_STORE_MACRO=0` keeps old reuse and still gets the PREFAULT part of the gain** (below: `m0`).
3. **Gate 1 met** (log lines, both default and `=0`; below).
4. **Gate 4 met** (cold TTFT, fresh server, back-to-back, ABBA): default against `=0` **-13.2 / -21.4 / -20.2 %** at 4k / 16k / 32k (ratio 0.868 / 0.786 / 0.798; 4 visits per arm over windows w3 and w5); #467's direction reproduced and larger, because `=0` here is the full old default (store, no macro, no prefault) where #467 compared against macro-only.
5. **Gate 2 met** (replies identical, below).
6. **Gate 5 met** (tests, below).

## Gate 1: startup log, env unset and `=0`

Serve logs `raw/serve-w1-v1-dflt.log` (nothing set) and `raw/serve-w1-v2-zero.log` (`INTARSIA_PREFIX_STORE_MACRO=0 INTARSIA_PREFIX_STORE_PREFAULT=0`):

| | nothing set | both `=0` |
|---|---|---|
| macro | `prefix store: macro prefill (8192 rows) stays on for a request that prefills alone (INTARSIA_PREFIX_STORE_MACRO, on by default on CUDA; =0 turns it off) ...` | `prefix store: recipe's macro prefill (8192 rows) off with --max-batch > 1 (the workspace is planned without it)` |
| workspace | `workspace 1727 MiB` | `workspace 678 MiB` (the default costs +1049 MiB of workspace) |
| host buffers (new startup line, `StorePolicy::announce`) | `prefix store: new host buffers: page-aligned mappings, 2 MiB-aligned with MADV_HUGEPAGE (1) (INTARSIA_PREFIX_STORE_PREFAULT)` | `prefix store: new host buffers: off (vec, 16 bytes into a page) (INTARSIA_PREFIX_STORE_PREFAULT)` |

The build fingerprint of the store differs between the two (`f166bc11...` against `85e2d8d2...`): a disk tier written by one is not read by the other.

## Gate 2: replies identical to #467's explicit `MACRO=1 PREFAULT=1` arm

- Window w2 (`MODE=ids`, storettft's `ids_check.py`: 12 fixed greedy prompts of 1.5k to 33k tokens, each sent cold then again restored from the store, 64 tokens): default vs `ref` (explicit `MACRO=1 PREFAULT=1`, same binary) **24 of 24 identical**; default vs `zero` (both `=0`) 24 of 24 identical; default vs #467's own recorded `w2-v2-pf-ids.json` (storeprefault lane's binary) 24 of 24 identical; cached-token counts of the restored sends identical at all 12 sizes (1441, 2920, ..., 34064). `raw/compare-ids-w2.txt`.
- Window w4 (`MODE=replay`, ab35d's plan and traces, 2-stream interleaved, seeds 0 and 1, 48 greedy requests each): default vs `ref` **48 of 48 identical in both seeds**, and identical to #467's recorded `w3-v2-pf` replay in both seeds. Cached fraction 0.732 in every run. `raw/compare-replay-w4.txt`.

Replay TTFT sum and decode (not a gate here, same cell as #467's gate 4):

| visit | seed | errors | TTFT sum s | TTFT p50 / p90 s | decode tok/s per stream (median) | combined out tok/s |
|---|---|---|---|---|---|---|
| default | 0 | 0 | 69.5 | 1.39 / 1.57 | 47.9 | 66.5 |
| default | 1 | 0 | 73.4 | 1.46 / 2.12 | 47.3 | 65.5 |
| ref | 0 | 0 | 73.5 | 1.48 / 2.08 | 47.6 | 66.8 |
| ref | 1 | 0 | 74.6 | 1.45 / 2.44 | 48.7 | 65.6 |

## Gate 3: restore points

`scripts/storeflip/restore_point.py`, window w1 (`dflt zero zero dflt`, fresh server each): per shared size K a separate 20k-token single user message A_K (own filler seed, nonce-free), sent cold, then a prompt that is the first K tokens of A_K followed by a different tail; 6 s pause after every request (the settle is over). Table in Verdict 2. Also: A_K again (full repeat) cached 20262 of 20266 in both arms; the cold 20k request's TTFT (3.23 to 3.25 s default against 3.55 to 3.58 s `=0`, the first request of each server 3.74 / 4.13 s).

Why: with MACRO a request that prefills alone runs up to 4 consecutive 2048-token chunks as one layer-by-layer pass (`SeqStore::macro_len`) and the store captures at the pass end, at message cuts and at nothing in between (the intermediate recurrent states of a macro pass are not materialised per chunk). The single message of a pasted document has no cut inside it, so its entries are at 8192, 16384, ...; with `=0` the store captures after every 2048-token chunk. The first header cut is at about 3 tokens, below `min_tokens` 32: no entry.

### Options measured for the user's decision (not the default; windows w6, w7)

`m0` = `INTARSIA_PREFIX_STORE_MACRO=0` with PREFAULT left at its new default; `mac4k` = default plus `INTARSIA_PREFILL_MACRO=4096` (macro pass of 2 chunks, so entries every 4096 tokens; the workspace is planned for 4096 rows). Restore points (cached tokens, TTFT s of the share request; w6, one visit per arm; `m0` equals the `=0` rows above):

| shared prefix | `=0` / `m0` | default (macro 8192) | `mac4k` |
|---|---|---|---|
| 2930 | 2051 (0.22) | 0 (0.53) | 0 (0.52) |
| 4992 | 4099 (0.23 to 0.30) | 0 (0.80) | 4099 (0.23) |
| 9078 | 8195 | 8195 | 8195 (0.35) |
| 13160 | 12291 (0.28) | 8195 (0.90) | 12291 (0.28) |
| 17276 | 16387 | 16387 | 16387 |

Cold TTFT (w7, ABBA `m0 mac4k mac4k m0`, 2 visits per arm; s, mean; the same cell as gate 4; the `=0` and default rows are the 4-visit values above):

| arm | 4096 | 16384 | 32768 |
|---|---|---|---|
| `=0` (old) | 0.775 | 3.322 | 7.290 |
| `m0` (PREFAULT only) | 0.670 (0.865) | 2.835 (0.853) | 6.365 (0.873) |
| `mac4k` | 0.670 (0.865) | 2.695 (0.811) | 6.010 (0.824) |
| default (macro 8192) | 0.673 (0.868) | 2.612 (0.786) | 5.817 (0.798) |

(ratios to `=0`.) So PREFAULT alone gives about -13 to -15 % on cold TTFT at every size and keeps every restore point; MACRO adds a further -8 % at 16k and 32k (nothing at 4k, where there is no macro step) and is what costs the restore granularity; a 4096-row macro pass gets about half of that extra gain and loses only the shares under 4096 tokens. Not tested with `mac4k`: the effect on the replay or on other prompt shapes. Switching to one of these defaults is a one-line change (`INTARSIA_PREFIX_STORE_MACRO=0` in the server environment for `m0`).

## Gate 4: cold TTFT, default vs `=0`

Windows w3 (ABBA `dflt zero zero dflt`) and w5 (`zero dflt dflt zero`), 2 + 2 visits per arm, each visit a fresh server; `localeval speed --structured prompt --thinking off --sizes 4096,16384,32768 --gen 256 --reps 3` (nonce first, one request at a time, back-to-back); TTFT s, visit medians, arm mean (ratio to `=0`):

| arm | visits | 4096 | 16384 | 32768 |
|---|---|---|---|---|
| `=0` (old default) | 4 | 0.775 | 3.322 | 7.290 |
| default | 4 | 0.673 (0.868) | 2.612 (0.786) | 5.817 (0.798) |

Per visit: default 0.67 / 0.68 / 0.67 / 0.67, 2.61 / 2.62 / 2.61 / 2.61, 5.81 / 5.82 / 5.81 / 5.83; `=0` 0.74 / 0.80 / 0.78 / 0.78, 3.29 / 3.33 / 3.32 / 3.35, 7.24 / 7.30 / 7.32 / 7.30 (w3 alone: 0.877 / 0.790 / 0.800). Decode tok/s (w3): default 121.0 / 104.2 / 82.2, `=0` 118.8 / 102.3 / 81.6 (no change). Clock guard every cell of both windows: min SM 2372 MHz, mean 2384 to 2389 MHz at > 90 % util, 0 samples under 2000 MHz, SW power cap delta 0 us, vLLM request delta 0 (`raw/window-w3.log`, `raw/window-w5.log`, `raw/clk-*.tsv`, `raw/tables-w3w5.md`).

## Gate 5: tests

- Mac, worktree: `cargo test -p intarsia-core -p intarsia-models -p intarsia-cli --lib --tests` **1043 passed, 0 failed, 27 ignored** (`raw/mac-tests-1.log`); after the docs edits `cargo test -p intarsia-cli --lib --tests` again incl. `docs_env_vars`: all ok (`raw/mac-tests-cli.log`). New test `prefix_store::host_buf_tests::prefix_store_knob_defaults_are_cuda_only_and_zero_is_off` (unset: on only on CUDA, `0` off, `1` on everywhere, `4k` kept).
- spark2 (Linux, the `MADV_HUGEPAGE` branch compiles), `rtest.sh` in the build's `--load cpu` slot, the same source as the served binary: `cargo test --release -p intarsia-core --lib prefix_store` **9 passed, 0 failed** (`raw/build.out`). The CUDA-real prefix-store tests (`crates/intarsia-models/tests/prefix_store_cuda.rs`) need a checkpoint and were not run; the served cells above exercise the CUDA store end to end.

## Memory

The default adds the macro workspace (+1049 MiB: 1727 against 678 MiB). Lowest MemAvailable during any visit beside vLLM: 37.2 GB (replay, both arms), 43.0 GB (restore visits, both arms), kill line 15 GB, no KILLED line. Host buffers: the same bytes as before (mappings trimmed to the page; #467).

## Not verified / limits

- One model (35B NVFP4), one box, CUDA only. Metal, HIP and cpu stay off by default and are untested with the knobs on beyond #466 / #467's own notes; the dense models (no macro pass, no recurrent state) run as before with MACRO "on" (`store_macro_rows` is 0 for them).
- `--spec mtp` with the store at `--max-batch` > 1 was the replay's config (recipe MTP N = 2): covered. Single-sequence engine (`--max-batch 1`) with the store gets PREFAULT by default too (host buffers only); not separately timed.
- vm.max_map_count: one VMA per host object (about 4500 for the default 8 GiB tier of the 35B against 65530), see #467.
- No soak with the default on.

## Revert

`INTARSIA_PREFIX_STORE_MACRO=0` and / or `INTARSIA_PREFIX_STORE_PREFAULT=0` in the server environment, or `git revert` of the PR (the code change is the two resolvers and the backend hook; the knobs themselves stay).

## Raw

`raw/window-w*.log` (run logs with clock guard lines), `raw/recipes/*.txt`, `raw/serve-*.log`, `raw/w1-*-restore.{json,out}` (restore points), `raw/w2-*-ids.json`, `raw/compare-ids-w2.txt`, `raw/w4-*` (replay), `raw/compare-replay-w4.txt`, `raw/replay-table-w4.md`, `raw/w3-*-speed.out`, `raw/tables-w3.md`, `raw/build.out`, `raw/mac-tests-*.log`, `raw/job-w*.out`. fleetq jobs: build #10382, w1 #10388, w2 #10389, w3 #10390, w4 #10391 (exit 2: I edited `window.sh` while bash was reading it, after both visits had finished and their data were copied; not a measurement failure), w5 #10402, w6 #10403, w7 #10404.
