# Inkling-2-Mini on HIP (R9700): speed table, main7 vs main after flip B and #552

Lane ink-hip-table, 2026-11-27. A measurement only: no code change, no default change.

## Why

The last HIP table is from main7 (`4b629184c`, 11-27 02:50, `docs/results/2026-11-26-inkling2-hip-specflip/raw/m7-analysis.txt`). Since then:

- **Flip B** (#550, `b2ce42d23`): greedy requests draft with DFlash2 k4 by default instead of DSpark k5.
- **#552** (`351657cc1`): a drafting request no longer replays a decode graph recorded without draft taps.

So the table's default-spec rows are stale.

## Method (fixed before the window)

- **Binaries:** `main7` = `4b629184c` and `now` = `351657cc1`, each from `git archive`, built on tatooine in this lane's volume
  (`raw/prep.sh`, `raw/build.sh`).
- **Stage:** `raw/stage_m7b.sh`, lane ink-hip-specflip's main7 protocol unchanged. Both cards at once, each its own ABBA, fresh processes,
  greedy, 256 tokens, EOS honoured, `generate --max-seq 20480`.
  - **Default-spec decode** (no `--spec`): `speed-m7d.json` (warm, 3 short, 1k, 4k, 16k).
  - **Plain decode** (`--spec off`): `speed-m7p.json` (warm, 3 short, 4k, 16k).
  - Card 0 reps r1 r2, card 1 reps r3 r4.
- **Window:** one radiance window (`raw/window.sh`, CAP 285 s; 11-27's M2 took 252 s). The user approved it at 11-27 ~20:27 EST
  ("Yes, when ready").
- **Analysis:** `raw/speed.py` (decode ratio `main7 / now` with CI, tok/s per class, verify counters, ids) and `raw/prefill_m7.py`.

What to expect:
- `--spec off` rows: `main7` = `now` (no change touches plain decode).
- Default rows: `now` faster (flip B's ABBA).
- Greedy ids: identical in both arms (drafts only propose).

## Results

Window: fleetq #12808. Radiance stopped 20:29:18 and answered again 20:33:42: **4 min 24 s down**. Stage 249 s, all 16 processes exit 0.
Binaries: `main7` md5 `d644af53…` (the same bytes as 11-27's main7 build, so the build reproduces), `now` md5 `eba34e36…`.
Full output: `raw/analysis.txt`; logs: `raw/logs/`.

**HIP speed, Inkling-2-Mini on one R9700 (tok/s, median of 4 processes):**

| Cell | main7 `4b629184c` | now `351657cc1` | now / main7 [95 % CI] |
|---|---:|---:|---:|
| Default decode, short (3 prompts) | 98.4 | **112.1** | **1.140** [1.138-1.143] |
| Default decode, 1k | 102.8 | **116.9** | **1.138** [1.134-1.142] |
| Default decode, 4k | 117.7 | 116.4 | **0.991** [0.987-0.995] |
| Default decode, 16k | 91.2 | **100.3** | **1.101** [1.098-1.105] |
| Plain decode (`--spec off`), short / 4k / 16k | 83.6 / 80.3 / 75.3 | 83.6 / 80.3 / 75.3 | 1.000 / 1.000 / 1.000 |
| Prefill 1k / 4k / 16k | 2820 / 3481 / 3282 | 2805 / 3464 / 3267 | 0.999 / 0.999 / 0.997 (CIs include 1) |

- **Greedy ids:** identical across both arms and all reps, default and `--spec off`.
- **Draft work:** tokens per verify pass rose from 2.31 / 2.68 / 3.22 / 2.47 to 2.81 / 3.19 / 3.23 / 3.00 (short / 1k / 4k / 16k).
- **Default decode** is 10-14 % faster everywhere except 4k. This matches flip B's own gate run (#541: short 1.189, 1k 1.142,
  4k 0.991 [0.987-0.995], 16k 1.101), including its one cell under 1. At 4k, DFlash2 k4 accepts no more tokens per pass than DSpark
  k5 (3.23 vs 3.22), so the cell is a wash.
- **Plain decode and prefill** did not move, as expected: neither change touches them.
- **#552's own effect is not in this table.** `generate` runs only greedy prompts, so no non-drafting request ever records a graph
  for a drafting one to replay. #552 matters on a server that mixes request types. Here it only shows that the fix costs nothing on
  the greedy path.
