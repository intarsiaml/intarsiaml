# FX5c: is the warm (resumed) `continue` path less accurate than cold? Against a CPU oracle, no. (2026-10-22)

Lane `fx5c`, stacked on PR #87 (`lane-fx5/hybrid-reuse-ci`, head `3f30059c`, which carries fx5b). **Measurement only**: no gate, tolerance
or engine change. The one code change is two opt-in diagnostic flags on `accuracy` (`--reuse-dump`, `--reuse-seq`, section 6). Until #87
merges the diff of this PR includes #87's changes.

Question (from fx5b): on Qwen3.6-35B-A3B NVFP4 the `long_docs_4k` `continue` case shows warm (resumed, decode-written KV) against cold
(full prefill) KL up to 8.5e-2. Is warm LESS ACCURATE than cold, or are both equally close to the truth and just different?

**Answer: noise.** Scored against an independent CPU oracle on the same 32 teacher-forced rows, warm is no further from the oracle than cold
on CUDA (mean KL 0.83x) and within the 1.5x rule on Metal (1.09x, 95 % CI 0.70 to 1.68). Warm and cold are about as far from each other as
each is from the oracle: two samples of one noise cloud, not one of them displaced from the truth. Caveats in section 5.

## 1. Method

* **Sequence** (the exact one fx5b teacher-forced; CUDA spark1, 35B NVFP4, `long_docs_4k`, chunk 1024, split 2000, 16-token reply, resume at 2015):
  `turn2` (4080 tokens = turn-1 prompt 2000 + reply 16 + remaining prompt 2064) followed by the 32 tokens of the CUDA cold engine's greedy
  stream, 4112 tokens in all (`raw/cuda/dump/seq.json`). Row `j` (0..31) is the distribution the cold run sampled generated token `j` from,
  position `4079 + j`; row 0 is the last prefill row, rows 1 to 31 are decode steps. These are fx5b's 32 rows.
* **Engine rows**, full vocabulary, teacher-forced on that stream through `Engine::forward_script` (fx5b's `script_rows`):
  `cold` (full chunked prefill), `warm` (turn 1 prefilled, the 15 reply tokens decoded one at a time, resume at 2015: decode-written KV),
  `warm_pf` (the reply prefilled instead of decoded, then the same resume: isolates the resume-alignment part), and on CUDA `cold_engine`
  (the engine's own `generate` rows, a second route to cold). Dumped by `accuracy --mode reuse --reuse-diag --reuse-dump`.
  Sanity gate: the dump reproduces fx5b's warm-vs-cold line exactly on CUDA (KL mean 1.37e-2, max 8.54e-2, 3 argmax flips; raw/cuda/run.log).
* **Metal** (endor, M5 Max) ran the same `forward_script` rows with the same tokens forced (`--reuse-seq`; Metal's own cold stream differs from CUDA's,
  `cold_engine` is therefore not comparable and not scored).
* **Oracle**: the f64 CPU oracle of FX1 (`accuracy --backend cpu --mode prefill`, chunk 256, top-64 self-golden), one prefill over the 4112-token
  sequence (`scripts/fx5c/make_oracle_input.py` wraps it as a one-prompt golden with filler reference arrays; the tool's verdict on the filler is
  meaningless). **Run on spark1's CPU, not endor**: endor's queue was strictly gated all morning (a stale cpu job and the #4051 reservation);
  spark1 took it at once, 15 min 36 s, 37.8 GB RSS (raw/oracle/oracle.log). Alignment check: the oracle's top-1 equals the forced stream token at 28 of 32
  rows (not off by one).
* **Score** (`scripts/fx5c/score.py`): KL(oracle || engine) over the oracle's top-64 ids plus a tail bucket (the quantity `accuracy` gates, FX1's
  scale; KL21 is also in the raw score files), mean and max over the 32 rows; top-1 against the oracle's argmax; KL(cold || warm) over the full
  vocabulary. Slots: all numerics only; none is a speed number.

## 2. Result (long_docs_4k, 32 rows)

KL in nats. `o` = oracle.

| backend | KL(o \|\| cold) mean / max | KL(o \|\| warm) mean / max | warm / cold (mean; max) | top-1 vs o cold / warm | KL(cold \|\| warm) mean / max | KL(warm \|\| cold) mean / max |
|---|---|---|---|---|---|---|
| CUDA GB10 (spark1) | 1.83e-2 / 1.16e-1 | 1.52e-2 / 1.25e-1 | 0.83; 1.08 | 30/32 / 29/32 | 1.47e-2 / 1.18e-1 | 1.37e-2 / 8.54e-2 |
| Metal M5 Max (endor) | 1.46e-2 / 5.50e-2 | 1.59e-2 / 1.18e-1 | 1.09; 2.16 | 30/32 / 28/32 | 1.93e-2 / 9.96e-2 | 1.89e-2 / 1.12e-1 |

Supporting rows:

| backend | path | KL(o \|\| path) mean / max | top-1 vs o |
|---|---|---|---|
| CUDA | `warm_pf` (reply prefilled, resumed at 2015) | 1.94e-2 / 1.22e-1 | 29/32 |
| CUDA | `cold_engine` (engine `generate` rows) | 1.62e-2 / 8.52e-2 | 28/32 |
| Metal | `warm_pf` | 1.46e-2 / 5.50e-2, bit-identical to `cold` (32/32 rows) | 30/32 |

Paired over rows (raw/score-*.txt): CUDA warm is the worse of the pair on 14 of 32 rows, paired mean difference (warm - cold) -3.0e-3 +- 3.5e-3 (s.e.);
Metal 17 of 32, +1.4e-3 +- 3.6e-3. Bootstrap (20000 row resamples) of mean(warm)/mean(cold): CUDA median 0.83, 95 % CI [0.55, 1.21]; Metal median 1.07, 95 % CI [0.70, 1.68].
Medians: CUDA cold 1.12e-2, warm 7.8e-3; Metal cold 8.7e-3, warm 9.7e-3. Mean without the two largest rows: CUDA cold 1.29e-2, warm 1.08e-2; Metal cold 1.20e-2, warm 1.17e-2.

## 3. Reading

* **Verdict by the rule** (warm within about 1.5x of cold's oracle KL means noise): CUDA 0.83x, Metal 1.09x on the mean. Noise.
* **Warm and cold differ from each other by as much as each differs from the oracle** (1.4e-2 to 1.9e-2 against 1.5e-2 to 1.8e-2). If warm were displaced from
  the truth by a bug, KL(o || warm) would exceed KL(o || cold) by about KL(cold || warm); it does not (CUDA -3.0e-3, Metal +1.4e-3, against 1.5e-2 to 1.9e-2).
* **The heavy rows are shared noise, not a warm-only effect.** KL per row ranges from 3e-6 to 1.2e-1; the three largest rows carry 31 to 39 % of each mean.
  Row 3 is bad for CUDA cold (1.16e-1 against warm 3.7e-2); row 8 is bad for warm on both backends (1.25e-1 CUDA, 1.18e-1 Metal; cold 8.1e-2 and 3.0e-2), and there
  warm's argmax differs from the oracle's on both backends while cold's agrees. The oracle's own gap between its top two tokens at row 8 is 0.40 nats, a near-tie, as at every
  row where any path leaves the oracle's argmax (gaps 0.03 to 0.40 nats; 6 of the 32 rows have a gap below 0.3, the median gap is 1.6). Argmax disagreements with the oracle: warm 3 (CUDA) and 4 (Metal),
  cold 2 and 2; too few, and all on near-ties, to call a difference. The max over 32 rows is one or two rows, so
  "max warm / max cold" is not a usable statistic (Metal 2.16, CUDA 1.08, from the same two kinds of rows). The mean is.
* **Components** (fx5b's decomposition, now with the truth): resume at the unaligned offset 2015 is not a bug on either backend. On CUDA `warm_pf` (no decode-written
  KV) reads 1.94e-2, no better than warm (1.52e-2) and no worse than cold (1.83e-2). On Metal the resume alone is bit-identical to cold, so the whole warm-minus-cold
  difference (KL 1.9e-2 between them) is the 15 decode-written KV rows plus the recurrent state advanced by decode steps; against the oracle that costs
  +9 % on the mean (CI 0.70 to 1.68), i.e. not detectable at 32 rows. There is no component to point at; none of the paths is closer to the oracle than another by more than the row-to-row noise.
* **Absolute level**: all paths sit at 1.5e-2 to 1.9e-2 here, above FX1's prefill floor (9.3e-3 CUDA, 9.9e-3 Metal, gate 2e-2). 31 of the 32 rows are decode steps, which FX1's
  oracle never covered ("decode: no oracle"). The decode path is therefore about 1.5 to 2x the prefill distance from the oracle on this model, and the 8.5e-2 of fx5b is
  within the spread of one engine path against the truth (max 5.5e-2 to 1.25e-1 per row for every path, cold included).
* **The oracle is not exact across machines, and its row-level swings are as large as the engines'.** Rows 0..1999 of the sequence are the committed endor oracle's `long_docs_4k`
  rows 0..1999 (same tokens, same chunk 256). The spark1 oracle differs from them by KL mean 3.1e-3 but **max 2.4e-1**, up to 0.72 nats on the top-1 log-probability, top-1 agreement 98.5 %
  (raw/oracle-xcheck.txt): bf16 activations make the oracle deterministic per machine, not portable, and near-tie rows (MoE routing, close top-2) move it by as much as they move an engine.
  Consequences: (a) the mean is common-mode (warm and cold are scored against the same oracle rows), so the mean ratio stands, with 3e-3 of oracle noise against 1.5e-2 to 1.9e-2 of distance;
  (b) per-row readings (row 8 worse for warm on both backends, the argmax disagreements) are inside the oracle's own row-level noise and are not findings; (c) the "decode rows are 1.5 to 2x the prefill distance"
  comparison with FX1 (9.3e-3 / 9.9e-3, scored against the endor oracle on different rows) is confounded by the oracle machine as well as the row set, and is indicative only.

## 4. Proposed gate (not implemented; a decision for the user)

"warm-vs-oracle <= cold-vs-oracle x margin", per backend, on the mean over the teacher-forced rows:

* mean over rows of KL(oracle || warm) <= 1.5 x mean KL(oracle || cold) (+ a small absolute term, say 1e-3, so that two near-exact paths do not fail on ratios of tiny numbers).
  Observed ratios 0.83 and 1.09; a bit-identical resume (Metal `warm_pf`) gives 1.00. Do not gate the max over rows (one or two rows decide it, section 3).
* Add an absolute backstop on the warm side (a wrongly restored state moves KL by orders of magnitude, `restore`/`aligned` already catch the exact cases): e.g. warm mean KL(o || warm) <= the FX1 floor
  class bound x the decode factor measured here (about 2e-2 x 2). To be fixed with the user's floor policy, not here.
* **It is only practical with a committed oracle on a fixed token list.** The `continue` case today free-runs the engine, so the reply and the stream are the engine's own and no
  oracle can exist ahead of time; a 16-min CPU run per qualify is out. What works: freeze `turn2` and the 32 stream tokens (this run's `seq.json`, 4112 ids) and commit only
  the oracle's 32 rows (32 x top-64: about 16 KB), then the gate is the `--reuse-dump` path (teacher-forced `forward_script` rows, which is what the engine measures anyway) scored
  against those rows, with no CPU oracle at qualify time. That replaces, for models whose KL floor is wide (MoE, hybrid, NVFP4), the current stream-tolerance check that fx5b showed cannot pass (0.25 nats / 1e-2 KL calibrated on dense Qwen3-8B / HIP).
* **What this gate tests, and what it does not.** `--reuse-dump` scores the `forward_script` replay of the continue sequence of operations, not `generate` reading the real prefix cache: fx5b measured the engine's
  own warm rows against their scripted replay at KL 3.25e-2, so a cache-lookup or snapshot-selection bug inside `generate` is not caught by it. The exact `restore` / `aligned` checks (bit-identical logits, `cached == split`) stay as the guard for the cache logic;
  the oracle gate only bounds the numerical side of a resumed state. A forced-token `generate` would close the gap and does not exist.
* Sample size: 32 rows from one text is thin (row-to-row heavy tails; rows share state, so the effective n is smaller than 32 and the s.e. above is optimistic). A gate wants 64 or more rows and two
  prompts; the CI of the ratio at n = 32 (Metal up to 1.68) would false-fail a 1.5 margin by chance.

## 5. Caveats, not done

* One prompt on Metal and two on CUDA (section 5b), one sequence and 32 rows each. The sequence was chosen by the CUDA cold engine's greedy decoding, so it is the cold path's own context; both warm and cold are scored on it and the oracle row
  is conditioned on it, but a context picked by one path is a mild favour to that path. Metal teacher-forces CUDA's tokens, which differ from its own stream.
* Oracle on spark1 CPU, once; it differs from the committed endor oracle at 3e-3 (above). Not repeated on endor (blocked by the queue).
* No prefill-only comparator on the same positions, so "decode rows are 1.5 to 2x the prefill distance" compares different rows and is indicative.
* No per-layer attribution (the repo has no activation dump; fx5b said the same).

## 5b. Second prompt, `long_code_4k`, CUDA only

Same recipe on a code prompt (fleetq #4122: CUDA dump and CPU oracle in one spark1 slot; the sequence is this CUDA engine's own, split 2000, turn2 4080 + 32 stream tokens;
oracle 15 min 16 s, 37.8 GB; the oracle's top-1 equals the stream token at 32/32 rows). Metal could not run a second prompt: endor's queue was gated by a stale cpu job and the #4051 reservation.

| path (CUDA, long_code_4k) | KL(o \|\| path) mean / max | top-1 vs o |
|---|---|---|
| cold | 2.80e-3 / 3.36e-2 | 32/32 |
| warm | 3.09e-3 / 4.97e-2 | 32/32 |
| `warm_pf` | 3.35e-3 / 5.90e-2 | 32/32 |
| `cold_engine` | 3.29e-3 / 4.34e-2 | 32/32 |

warm / cold: 1.10 on the mean (bootstrap 95 % CI [0.65, 1.46]), 1.48 on the max; warm is the worse of the pair on 14 of 32 rows; KL(cold || warm) 3.2e-3 mean / 2.6e-2 max, KL(warm || cold) 3.2e-3 / 2.8e-2
(raw/score-cuda-code.txt). Same verdict, and the same shape as `long_docs_4k`: warm and cold are as far from each other as each is from the oracle (3.2e-3 against 2.8e-3 to 3.1e-3), warm is within the 1.5x rule on the mean.
Code is low entropy, so every distance is about 5x smaller than on the prose prompt; the gate margin is a ratio, which is why it transfers where the absolute 1e-2 does not.

## 6. Code and files

* `crates/intarsia-cli/src/cmd/accuracy.rs`: `--reuse-dump DIR` (with `--reuse-diag`: write `cold`, `warm`, `warm_pf`, `cold_engine` rows as raw little-endian f32 and `seq.json`, then exit)
  and `--reuse-seq FILE` (teacher-force another run's `seq.json`). Opt-in diagnostics; no default, gate, tolerance, or engine path changes. `cargo check -p intarsia-cli` clean; the flags were exercised on CUDA and Metal as above.
* `scripts/fx5c/`: `cuda-dump.sh`, `metal-build.sh`, `metal-dump.sh`, `oracle.sh`, `make_oracle_input.py`, `score.py`, `oracle_xcheck.py`, `code-chain.sh`.
* `docs/results/2026-10-22-fx5c-reuse-vs-oracle/raw/`: run logs, `seq.json`, the oracle dump (2 MB, top-64 for all 4112 rows), score files. The 32 x 248320 f32 row dumps (127 MB per backend) are not in the repo.
* Fleet jobs (owner claude-f31ccf91-fx5c): #4084 (spark1, build + CUDA dump, non-exclusive), #4089 (spark1, CPU oracle, non-exclusive, 40 GB), #4083 (endor, Metal build),
  #4108 (endor, Metal dump). Cancelled before running: #4109 and #4115 (a Metal-generated `long_code_4k` sequence, unreachable). #4122 (spark1, `long_code_4k` CUDA dump + oracle): done, exit 0 (oracle exit 1 is the tool's verdict on the filler reference, expected).
  One `cargo check` on endor ran before I took a slot (no fleet role affected). Callers of the changed `reuse_diag` signature: the two known sites in `accuracy.rs`, no others in `crates/`; `AccuracyArgs` has no struct-literal constructors (tests parse it with clap), so `--tests` was not separately built.
