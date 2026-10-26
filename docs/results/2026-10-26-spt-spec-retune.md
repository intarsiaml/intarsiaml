# Speculation retune on the MXFP4 multi-row kernel, Qwen3.8-27B on HIP, lane spt, 2026-10-26

Lane spt (coordinator 13:30, user's pick), after #175 (`1f979fad`, the tp=2 MTP default) and #176 (mxr's int8 multi-row MXFP4 GEMV,
verify8 now 1.42x a plain step at tp=1, was 1.98x). Question: with verify cheaper, is a longer draft better?

- **MTP**: k3 vs k2 at tp=2. Rule (coordinator, within the user's 11:07 decision "draft length 2 or 3, chosen from measurements"):
  k3 replaces k2 as `mtp_default_tp_k` only if its pooled mean is better, **no category is more than 2% worse than k2**, and every
  category is >= 0.99x of off.
- **DFlash2**: k7 and k5 at tp=2, record-only (defaults for DFlash do not change).

Setup: tatooine, 2x R9700 (gfx1201, ROCm 7.2), `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (`5233554c`), tree `b7bf17b02` (main `1f979fad` +
lane scripts only), `serve --tp 2 --max-batch 1`, localeval `quality --judge none`, 512 forced tokens (ignore_eos + min_tokens),
temperature 0, the 7 category files of lane dfh (`scripts/dfh/cats/`: categories-rcp recall/code/prose, held-out
`tools/dc/heldout_prompts.jsonl` prose/chat/code/recall), tables by `tools/dfh/cat_table.py` (checks every row for exactly 512 tokens
and every arm for the same rows). Scripts: `scripts/spt/` (copies of `scripts/dfh/` with spt names; the ids gate's off arm passes
`--spec off` explicitly, since no `--spec` at tp=2 is now the MTP default; `cat-window.sh` takes `ARMSEQ`, an explicit arm order per
round). Windows: at most two radiance-down windows under the 45-min cap (coordinator 13:41).

## 1. Window A: MTP k3 vs k2 at tp=2 -> keep k2

Window A (#5771, 13:34:57-14:12:54 radiance down; `raw/a1/`), arms `mtp2 mtp3; mtp3 off mtp2; mtp2 mtp3` (one off arm in the middle
as the 0.99x anchor: off measured 53.2 tok/s with < 0.4% spread in lane dfh's w8).

**ids**: `--spec mtp --spec-k 3` vs `--spec off` at tp=2, lane dfh's 18 prompts (`scripts/dfh/ids-prompts`), 256 greedy ids:
**18 / 18 identical** (`raw/a1/ids/`). Acceptance (`raw/a1/ids/acceptance.txt`): P(accepted >= 1, 2, 3) 0.78-0.82, 0.56-0.59,
0.39-0.45; 2.75-2.83 tokens per pass (k2: 2.38-2.44); a pass 31-34 ms (k2: 31-34 ms).

**Speed** (`raw/a1/cat-table.txt`), 42 runs, 0 problems; round 3's k3 arm was skipped by the window's time guard (494 s left), so k3 has
two rounds, k2 three, off one:

| category | off | k2 | k3 | k3 vs k2 |
|---|---|---|---|---|
| ho-chat | 53.3 | 85.6 (1.61x) | 86.0 (1.62x) | +0.5% |
| ho-code | 53.2 | 102.8 (1.93x) | 111.8 (2.10x) | +8.8% |
| ho-prose | 53.2 | 76.0 (1.43x) | 72.6 (1.37x) | **-4.5%** |
| ho-recall | 53.2 | 89.4 (1.68x) | 88.9 (1.67x) | -0.6% |
| rcp-code | 53.2 | 100.2 (1.88x) | 109.2 (2.05x) | +9.0% |
| rcp-prose | 53.2 | 77.4 (1.45x) | 73.7 (1.39x) | **-4.8%** |
| rcp-recall | 53.2 | 90.7 (1.70x) | 92.3 (1.74x) | +1.8% |

Pooled mean over the 7 categories: k2 88.9, k3 90.6 tok/s (+2.0%); every category >= 0.99x of off. But prose is 4.5% and 4.8% worse
than k2 (limit 2%), on both prompt sets; the gap is about 10 times the round-to-round spread (< 0.5%). **Decision: keep
`mtp_default_tp_k = 2`.** The third draft is accepted 39-45% of the time, which pays on code (3.4 tokens per pass) and not on prose
(2.25 tokens per pass for a 4-row verify instead of 3).

## 2. Window B: DFlash2 k7 and k5 at tp=2 (record-only) -> no change

Window B (#5779, 14:14:57-14:50:17 radiance down; `raw/b1/`), arms `off dfl7 dfl5; dfl5 dfl7 off` (x2, record-only per the
coordinator), 42 runs, 0 problems (`raw/b1/cat-table.txt`). Beside window A's MTP k2 (the tp=2 default):

| category | off | DFlash2 k7 | DFlash2 k5 | MTP k2 (A) | best |
|---|---|---|---|---|---|
| ho-chat | 53.2 | 63.7 (1.20x, 3.22 t/p) | 64.2 (1.21x, 3.05 t/p) | 85.6 | MTP k2 |
| ho-code | 53.2 | 105.7 (1.99x, 5.32 t/p) | 95.8 (1.80x, 4.54 t/p) | 102.8 | DFlash2 k7 (+2.8%) |
| ho-prose | 53.2 | 49.5 (**0.93x**, 2.50 t/p) | 51.9 (0.98x, 2.47 t/p) | 76.0 | MTP k2 |
| ho-recall | 53.2 | 71.2 (1.34x, 3.60 t/p) | 70.8 (1.33x, 3.37 t/p) | 89.4 | MTP k2 |
| rcp-code | 53.2 | 103.3 (1.94x, 5.20 t/p) | 95.0 (1.79x, 4.50 t/p) | 100.2 | DFlash2 k7 (+3.1%) |
| rcp-prose | 53.2 | 50.4 (**0.95x**, 2.54 t/p) | 52.8 (0.99x, 2.51 t/p) | 77.4 | MTP k2 |
| rcp-recall | 53.2 | 76.7 (1.44x, 3.89 t/p) | 75.9 (1.43x, 3.61 t/p) | 90.7 | MTP k2 |

#176 made DFlash2 cheaper (k7 at tp=2 was 0.75-1.61x in lane dfh's window 7, now 0.93-1.99x; prose +23%), but it still loses to off
on prose and to MTP k2 everywhere except code (+3%). k5 trades code for prose (prose 0.98-0.99x). DFlash2 does not beat the MTP
default in every category, so nothing changes: DFlash2 stays opt-in. The per-pass cost is the draft (about 24 ms at tp=2, run
whole on rank 0): see lane dfh's section 16 for what would close that.

## 3. Outcome

No default changes. The 27B on HIP at `--tp 2` keeps `--spec mtp --spec-k 2` (1.43-1.93x off in every category); `--spec-k 3` is a
documented opt-in for code-heavy traffic (+9% on code, -4.5% on prose). DFlash2 stays opt-in (`--spec dflash`). Windows used: 2 of 2.

## 4. Follow-ups

- Adaptive draft length (per-request k from running acceptance: k3 on high-acceptance text like code, k2 on prose) — would capture
  code +9% without the prose -4.5%; untried.
- Today, `--spec mtp --spec-k 3` (pass both: a `--spec-k` without `--spec` turns the tp=2 default off) is +9% for code-heavy use
  (ho-code 111.8 vs 102.8, rcp-code 109.2 vs 100.2 tok/s), at -4.5% on prose.
