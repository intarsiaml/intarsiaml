# Inkling-2-Mini on Metal: a confidence cut for DFlash2 drafts (lane ink-dflash-cut, 2026-11-27)

Status: **GO by the pre-registered rule: k7 + `INTARSIA_DFLASH_CUT=0.3` is 1.126x the Metal default (DFlash2 k5), every cell 1.08-1.16,
ids identical in every process. Ready gates (slot R) PASS: sampled ids, peak memory, serve multi-turn.** Opt-in, PR #556; no
default changed (making it the Metal default is the user's call).

Turn it on: `INTARSIA_DFLASH_CUT=0.3 intarsia generate|serve ... --backend metal --spec dflash --spec-draft <Inkling-2-Mini-DFlash2> --spec-k 7`.
Off: unset the variable (or `0`).

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
host select from 0.52 to 2.7 ms a pass. So the rule uses `p16`. Full grid: `raw/sim-out-b.txt`, reproduced from the repo by
`python3 raw/sim.py raw/T/dfl7b.err docs/results/2026-11-26-inkling2-metal-dflash2/raw/s2-sweep` (identical output; add `raw/T/dfl5.err` for
the k5 check).

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

## Results (slot A, endor exclusive fleetq #12850, 21:23:20-21:39:03)

Binary md5 `77051af71927dadbdd9c50dcc3cc265f`, git `b93823f54`; raw `raw/A/` (`<arm>@<rep>.err/.out/.t`, `macmon.jsonl`, `rule-out.txt`,
`cells-tps.txt`), log `raw/slotA-12850.log`.

**Rule (`raw/rule.py`, unchanged from #546): GO.**

| cell | ratio cut7 / dfl5 | 95 % CI | per rep (r1 r2 r3 r4) |
|---|---:|---|---|
| short | **1.081** | [0.847-1.379] | 0.861 1.201 1.161 1.136 |
| 1k | **1.162** | [0.969-1.393] | 0.997 1.306 1.212 1.154 |
| prose | **1.154** | [1.006-1.323] | 1.015 1.225 1.194 1.194 |
| chat | **1.080** | [0.838-1.393] | 0.859 1.243 1.137 1.123 |
| 4k | **1.136** | [1.014-1.274] | 1.028 1.214 1.141 1.171 |
| 16k | **1.148** | [1.048-1.257] | 1.055 1.184 1.195 1.161 |

Mean of the six cells 1.126 (the simulation said 1.123); no CI upper bound < 0.99; ids identical across all 8 processes, every prompt.

- **Tokens per pass as simulated:** dfl5 1215 passes, 6042 proposed, 3042 accepted (3.50 tokens / pass); cut7 1195 passes, 4244 proposed
  (3.55 drafts, 4.55 verify rows a pass), 3030 accepted (3.54 tokens / pass).
- **Rep 1 is a power-state mismatch** (`macmon.jsonl`, GPU busy samples per process): dfl5@r1 ran in endor's burst state (median 1479 MHz,
  42.8 W GPU), cut7@r1 right after it throttled (1091 MHz, 21.0 W); 0.966 that rep. Reps 3 and 4, at matched clocks (1180-1232 MHz,
  25.6-27.4 W): cut7 103.7 / 105.0 tok/s vs dfl5 88.9 / 91.4, 1.149-1.167. The verdict does not depend on rep 1 (r2-r4 alone: every
  cell > 1.12).
- **Absolute tok/s** (median of 4 reps, geomean over a cell's prompts; this slot ran mostly in endor's ~1200 MHz state, so below #546's
  absolute numbers):

| cell | dfl5 | cut7 |
|---|---:|---:|
| short | 93.4 | 107.3 |
| 1k | 72.6 | 85.8 |
| prose | 92.4 | 110.4 |
| chat | 91.6 | 103.5 |
| 4k | 93.7 | 107.2 |
| 16k | 69.2 | 79.9 |

- **Ids gate PASS** (g3.json, 256 tokens, `--ignore-eos`): off / dfl5 / cut7 / cut5 all code `c69d1438ab3e715feb93a0614ebddcfe`, len4k
  `9e85ba1117e4a6a6cc12c5c917532e7f`, len16k `89994b876223335763b2004d0d281157` (the goldens). cut7 193 passes / 3.93 tokens a pass,
  cut5 210 / 3.61, dfl5 203 / 3.76.
- **Per-row verify cost** (slot T traces, `raw/ana_d.py`): verify + rollback median 28.3 ms at 6 rows (k5), 30.9 ms at 8 rows (k7), so
  ~1.3 ms a row. Whole passes at matched clocks (reps 3 / 4): cut7 34.4 / 33.9 ms, dfl5 39.5 / 38.4 ms, ~4.5 ms less for 1.42 fewer verify
  rows, more than the verify rows alone; the rest (rollback, the host's per-row download and selection, the draft at k7) is not split
  out. A diagnostic slot D (#12853, traced k7 arms) was cancelled before it started once macmon explained rep 1.

## Ready gates (slot R, endor exclusive fleetq #12862, 21:50:07-21:57:52, binary md5 `77051af71927dadbdd9c50dcc3cc265f`, git `e8f2cbe1f`)

Script `raw/slotR.sh`, log `raw/slotR-12862.log`, outputs `raw/R/`.

| Gate | What ran | Result |
|---|---|---|
| R1 sampled | `generate` seed 7, 32 tokens, `--ignore-eos`, T1.0 (w100, 100 prompts) and T0.6 (w60, 60 prompts), off vs cut7 | **PASS**: 160 / 160 prompts' ids equal; cut7 0 verify passes (sampled requests do not draft with DFlash2 on Metal, as designed: "DFlash-family drafting off (default ...)"), so the cut never acts on them |
| R2 memory | `generate --max-seq 4194304 --kv-fraction 0.25`, 64 tokens of c1, dfl5 / cut7 / dfl5 / cut7 | **PASS**: context 615458 tokens in all four; peak footprint 48.97 / 48.82 / 48.82 / 48.82 GB (the first run's +0.15 GB is the first run, not the arm) |
| R3 serve | `serve --max-batch 1`, cut7 vs `--spec off`, 2 chats x 3 turns + 2 x 3 token-id extensions, 128 tokens each | **PASS**: 12 / 12 turns' ids equal, `cached_tokens` equal on every turn (0 / 30 / 47, 0 / 27 / 43, 0 / 93 / 256, 0 / 300 / 468) |

- Serve prints no per-request spec counts, so R3 shows the replies and the prefix cache are unchanged with the cut configured
  ("speculative decoding: Dflash, up to 7 drafts per verify pass" at load), not how many passes drafted.
- The cut7 server took 191.5 s to load, the off server 4.4 s. The time is kernel admission on the spec plan's shapes ("42 kernel(s)
  checked, 170 case(s) run, 0 cached"): results are cached per host fingerprint and kernel pack hash (`crates/intarsia-core/src/plan.rs`),
  so it is the first serve start of a new build with `--spec`, and a k5 server on this build would run it too (not measured here).

## Not covered

Serve speed, batched decode (`--max-batch` > 1 does not draft with DFlash2 on Metal), sampled requests with drafting turned on
(`INTARSIA_SPEC_SAMPLED_DFLASH=1`; off by default on Metal), HIP.
