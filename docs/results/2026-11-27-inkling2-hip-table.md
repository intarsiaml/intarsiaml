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

(pending)
