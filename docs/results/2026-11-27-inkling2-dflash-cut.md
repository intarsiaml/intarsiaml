# Inkling-2-Mini on Metal: a confidence cut for DFlash2 drafts (lane ink-dflash-cut, 2026-11-27)

Branch `ink/dflash-cut` (worktree `.worktrees/ink-dflash-cut`), on main `930d19d4c` (#553 + #554 merged). Question: with the Metal default
DFlash2 k5 (#551, #553), can a per-pass cut on the drafts' confidence make decode faster, with output unchanged?

## Why (measured inputs, before any code)

- **A verify row costs about 2-2.5 ms on a ~19 ms fixed pass.** From #546's k sweep (`docs/results/2026-11-26-inkling2-metal-dflash2/raw/s2-sweep/`,
  one process per k, 19 prompts): ms per pass = a + b x rows, least squares over k3 / k4 / k5 / k7 per prompt: b = 1.6-3.1 ms (median
  ~2.1), a = 17-23 ms. The rows are MoE-bound: a 6-row pass reads the union of ~6 x 8 routed experts.
- **Oracle bound.** If each pass verified only the drafts that will be accepted, k5 would be 1.19x (geomean of 18 prompts, k7 cap
  1.28x); static k is flat (k3 / k4 / k5 within 0.006 in the sweep) because a shorter k loses the long accepted runs.
- **Draft phases on Metal** (traced, 616 passes, `2026-11-27-inkling2-metal-dflash2-ready/raw/C/`): flush 0.84, block 2.45, head 1.94,
  selproj 0.22, download 0.64, host select 0.52 ms; verify ~24.8 ms. The draft is paid in full whatever the cut keeps.
- **The selector matters.** Inkling's DFlash2 drafter has a candidate selector (rank 256, top 16, softcap 30 after correction). It takes a
  candidate other than the row's argmax on 38.6 % of rows; those are accepted 40.8 % of the time when reached, argmax picks 83.9 %.

## Simulation (slot T, endor exclusive #12838 / #12843)

Opt-in trace `INTARSIA_DFLASH_CONF_TRACE=1` (one line per draft, per row `draft:p_full:p_raw1:p_sel:margin:p16`), `generate --prompt-ids
speed.json` greedy at k7 and k5 (`raw/slotT.sh`, `raw/T/`). `raw/sim.py`: keep drafts while the running product of a signal stays >= tau
(the crossing draft dropped), at most cap; tokens = min(accepted, kept) + 1, cost a + b x (kept + 1) with the prompt's own sweep fit;
baseline static k5. Check: the k5 trace's own tokens per pass match min(A7, 5) + 1 from the k7 trace within ~0.1 on all 18 prompts.
k7 trace and its rerun (`dfl7b`) give the identical pass stream.

| signal (cap 7) | tau 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---:|---:|---:|---:|---:|
| `p_full` (whole-row softmax at the draft) | 1.117 | 1.122 | **1.123** | 1.118 | 1.111 |
| `p16` (softmax over the 16 candidates' raw logits at the draft) | 1.116 | 1.123 | **1.123** | 1.119 | 1.111 |
| `p_raw1` (the row's top-1 probability) | 1.025 | 1.056 | 1.081 | 1.094 | 1.098 |
| `p_sel` (softmax of the selector's corrected scores) | 1.054 | 1.091 | ... | | |

cap 7, `p16`, tau 0.3 by class: short 1.106, 1k 1.116, 4k 1.148, 16k 1.149, prose 1.204, code 1.137 (mean verify rows 4.55 vs 6). cap 5,
tau 0.2-0.3: 1.063. `p16` is free (16 candidates already on the host); `p_full` needs a full-row exp pass, which in the trace build raised
host select from 0.52 to 2.7 ms a pass. So the rule uses `p16`. Full grid: `raw/sim-out-b.txt`.

## The change

`INTARSIA_DFLASH_CUT` (opt-in, off unless set; `crates/intarsia-core/src/model/dflash2.rs`): `Selector::pick_cut` / `select_cut` walk
`pick`'s rows (one shared `pick_one`, the same arithmetic), multiply `cand_prob` (= `p16`) and stop before the draft that takes the
product below the cut, at most `--spec-k`. The drafts kept are a prefix of the uncut drafts (unit test `cut_drafts_are_a_prefix_of_the_picks`,
40 seeds x 2 caps x k 1/3/5/7 x cuts 0.05-1.0, and `select_cut` == `pick_cut`). On Metal's host path a row's logits are downloaded only
when the walk reaches it. HIP's device-candidate path takes the cut too (unmeasured there).

## Plan (fixed before slot A)

Binary `intarsia-ct` md5 `77051af71927dadbdd9c50dcc3cc265f` (commit `057f9d6d1`). `raw/slotA.sh`, endor exclusive:

1. **Speed, ABBA 4 + 4**: `dfl5` (`--spec dflash --spec-draft <DFlash2> --spec-k 5`, the Metal default) vs `cut7` (`--spec-k 7` +
   `INTARSIA_DFLASH_CUT=0.3`), order dfl5 cut7 cut7 dfl5 dfl5 cut7 cut7 dfl5, speed.json (19 prompts, 256 tokens, EOS honoured), 45 s GPU
   idle before each process, macmon.
2. **Rule** (`raw/rule.py`, #546's unchanged): cells short / 1k / prose / chat / 4k / 16k, ratio per (prompt, rep), cell = geomean over
   reps, 95 % CI t-interval over the 4 reps. **GO: mean of the six cells >= 1.03, no cell with a CI upper bound < 0.99, ids identical in
   every process.** GO means an opt-in PR plus the question to the user whether to make it the Metal default (`--spec-k 7` + cut 0.3 in
   the recipe's `[tuning.metal]`); below the bar: opt-in or drop, by the numbers.
3. **Ids gate**: `off` / `dfl5` / `cut7` / `cut5` (k5 + cut 0.3) on g3.json (code, len4k, len16k), 256 tokens, `--ignore-eos`: md5 of
   every prompt's ids equal across the four and to the goldens (code `c69d1438ab3e715feb93a0614ebddcfe`, len4k
   `9e85ba1117e4a6a6cc12c5c917532e7f`, len16k `89994b876223335763b2004d0d281157`).

Not covered by this slot: serve, batched decode (`--max-batch` > 1 does not draft with DFlash2 on Metal), sampled requests (DFlash2 drafts
greedy requests only on Metal), HIP.
