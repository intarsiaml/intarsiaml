# Inkling-2-Mini on Metal: DFlash2 k7 + confidence cut 0.3 and the prefill prologue split by default (lane ink-metal-flip2, 2026-11-27)

Status: **all seven gates PASS (slot F, endor exclusive #12925). Metal decode 1.171x the old default (DFlash2 k5) over the six cells,
prefill 1.02-1.07x, greedy and sampled ids identical, serve one-sequence and batch-4 replies unchanged.** PR `ink/metal-flip2`.

User decision 2026-11-27 ~23:05 EST, "Flip both", on the two opt-in levers measured today:

| Lever | Opt-in result | PR |
|---|---|---|
| DFlash2 `--spec-k 7` + `INTARSIA_DFLASH_CUT=0.3` (greedy decode) | 1.126x DFlash2 k5 over the six cells, ids identical, slot R ready gates pass (`2026-11-27-inkling2-dflash-cut.md`) | #556 |
| Prefill KV prologue split `INTARSIA_INK_PROLOGUE_SPLIT=1` | bitwise, prefill +3.1-3.3 % at 1k / 4k / 16k (`2026-11-27-inkling2-metal-prologue-split.md`) | #555 |

## The change (branch `ink/metal-flip2`, on main `e76cbd89b`)

Two commits, so either can be reverted alone:

1. `4f306174c`: the prologue split is on unless `INTARSIA_INK_PROLOGUE_SPLIT=0` (`crates/intarsia-backend-metal/src/inkprefill.rs`). Metal only
   (the kernel exists only there), prefill chunks of more than 16 rows.
2. `30420f61c`: the DFlash2 cut defaults to 0.3 on Metal when `INTARSIA_DFLASH_CUT` is unset or blank (`dflash2.rs`
   `CUT_DEFAULT_METAL`, `cut_resolve`; `=0` turns it off; HIP, CUDA and cpu stay off unless set), and `recipes/inkling2.toml`
   `[tuning.metal] dflash2_k = 7` (was 5). The tiny-fixture DFlash2 tests set `INTARSIA_DFLASH_CUT=0`: the random model's 16
   near-uniform candidates (~1/16 each) would cut every draft and make the drafting checks vacuous on `INTARSIA_INKLING2_BACKEND=metal`.

Revert: `INTARSIA_INK_PROLOGUE_SPLIT=0`; `--spec-k 5` (or `dflash2_k = 5`) with `INTARSIA_DFLASH_CUT=0`.

## Gates (pre-registered before slot F)

One binary (the branch), one endor `--exclusive` fleetq slot, script `raw/slotF.sh`. Arms: `flip` = no flags, no env (the new
default); `main` = today's main default through the revert switches (`--spec-k 5`, `INTARSIA_DFLASH_CUT=0`,
`INTARSIA_INK_PROLOGUE_SPLIT=0`); `off` = `--spec off`.

| # | Check | Rule |
|---|---|---|
| F0 | Prologue split bitwise on the GPU | `ink_prologue_split` test exe passes (built from the branch) |
| F1 | Greedy ids + what loaded | g3.json (code, len4k, len16k), 256 tokens, `--ignore-eos`: md5 `off` = `main` = `flip` = the goldens (`c69d1438...`, `9e85ba11...`, `89994b87...`). `flip` loads DFlash2 at k 7 with "dflash: cut 0.3 (the Metal default ...)" and traces `prologue split`; `main` shows k 5, no cut line, no split trace |
| F2 | Sampled ids | seed 7, 32 tokens, T 1.0 (100 prompts) and T 0.6 (60): `flip` = `off` on every prompt |
| F3a | Decode | ABBA x 2 over speed.json (`raw/rule.py flip main`): mean of the six cells >= 1.03, ids identical in every process |
| F3b | Prefill | ABBA x 2 over pf3.json, 1 token (`raw/pfabba.py`, off = main, on = flip): on / off of the medians >= 0.98 at each prompt |
| F4a | Serve, one sequence | `serve --max-batch 1`, flip (no `--spec`) vs `--spec off`: 12 / 12 turns' ids and `cached_tokens` equal |
| F4b | Serve, batching | `serve --max-batch 4` (flip): solo = concurrent x 2, and solo = #551's committed batch-4 replies (`2026-11-27-inkling2-metal-dflash2-flip/raw/G/b4-solo.json`, before the split existed) |

F3 is a confirmation with two reps per arm, not a new A/B: both levers were measured on their own with more reps (#555: 6 runs per
arm; #556: 4 reps). Its bars only catch a regression from combining them. 45 s GPU idle before every timed run, macmon logged.

**Decision rule:** all pass, so a PR and merge on green CI (the user decided the flip). Any F1, F2 or F4 ids mismatch: no merge, report.
F3 below its bar: no merge, report the numbers and ask.

## Results (slot F, endor exclusive fleetq #12925, 23:16:44-23:36:41)

Binary `intarsia-f2` md5 `b3fe395f5f39e90fb417c98f837e8203` (`30420f61c`; `780fc4fc4` after it changes a test only), test exe `t-ps`
md5 `19afa5113e4fd8e801ee5f6534d3b784`. Log `raw/slotF-12925.log`, per-run files `raw/F/`, build `raw/build-12923.log` (core dflash2
3/3, models `inkling2_dflash2` 13/14 at first: the recipe test still expected k5 at line 780, fixed in `780fc4fc4`, then 14/14 and
`recipe_ink_sampled_dflash` 3/3).

| # | Result | Evidence |
|---|---|---|
| F0 | **PASS** | `ink_prologue_split` 1/1 on the M5 Max GPU, 1.98 s |
| F1 | **PASS** | code `c69d1438...`, len4k `9e85ba11...`, len16k `89994b87...` in `off`, `main` and `flip` (the goldens). `flip`: "--spec-k 7", "dflash: cut 0.3 (the Metal default; INTARSIA_DFLASH_CUT=0 turns it off)", `prologue split` traced, 193 passes / 3.93 tokens a pass; `main`: "--spec-k 5 (k from --spec-k)", no cut line, no split trace, 203 / 3.76. Peak 20.29 / 20.31 / 20.32 GB (off / main / flip) |
| F2 | **PASS** | 160 / 160 prompts' ids equal (T 1.0: 100, T 0.6: 60); 0 verify passes in both arms (sampled requests do not draft on Metal) |
| F3a | **PASS** | mean of the six cells **1.171** (bar 1.03), ids identical in all four processes |
| F3b | **PASS** | on / off of the medians p1k **1.022**, len4k **1.065**, len16k **1.072** (bar 0.98) |
| F4a | **PASS** | 12 / 12 turns' ids and `cached_tokens` equal to `--spec off`; the server loaded DFlash2 at k7 with the cut |
| F4b | **PASS** | batch 4: solo = concurrent x 2 (4 / 4 each), solo = #551's committed replies (4 / 4) |

**Decode tok/s (F3a, geometric mean over each cell's prompts and the two reps; speed.json, 256 tokens, EOS honoured):**

| Cell | main (k5, no cut, no split) | flip (k7 + cut 0.3 + split) | flip / main | per rep |
|---|---:|---:|---:|---|
| short | 110.5 | **128.2** | **1.161** | 1.192 / 1.130 |
| 1k | 84.4 | **99.2** | **1.175** | 1.210 / 1.142 |
| prose | 106.7 | **133.6** | **1.252** | 1.271 / 1.234 |
| chat | 107.5 | **122.6** | **1.141** | 1.169 / 1.113 |
| 4k | 106.4 | **122.4** | **1.150** | 1.150 / 1.150 |
| 16k | 79.4 | **91.2** | **1.149** | 1.146 / 1.152 |

Overall decode: main 103.43 / 107.29 tok/s, flip 123.22 / 122.79 (1215 vs 1195 verify passes, 6042 vs 4244 drafts proposed, 3.50 vs
3.54 tokens a pass).

**Prefill tok/s (F3b, median of 2 runs, 1 token):**

| Prompt | main (split off) | flip (split on) | flip / main |
|---|---:|---:|---:|
| p1k (1000) | 1571 | **1605** | **1.022** |
| len4k (4108) | 3017 | **3212** | **1.065** |
| len16k (16455) | 2899 | **3108** | **1.072** |

- Two reps per arm: these ratios are above the levers' own measurements (cut 1.126x with 4 reps, split 1.03x with 6). The main arm's
  r1 ran slower than its r2 in both tables (decode 103.4 vs 107.3, len16k prefill 2806 vs 2992 tok/s), the power-state spread seen
  before. Read the two levers' own results docs as the measured gain; this slot shows that together they keep it and change no output.
- The first `--spec` serve start of this build took 197 s (cold kernel admission on the k7 plan shapes), the `--spec off` start 4.5 s
  and the batch-4 start 72 s. Admission is cached per host fingerprint and kernel pack hash; #551's k5 build took 140 s the same way.
- Not covered: HIP (no default changed there), serve speed, sampled requests with drafting on (`INTARSIA_SPEC_SAMPLED_DFLASH=1`).
