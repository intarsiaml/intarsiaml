# Inkling-2 HIP: `INTARSIA_HIP_INK_PF2` on tiny shapes (lane ink-hip-pf2-tiny, one R9700, 2026-11-26)

Branch `ink/hip-pf2-tiny` from origin/main `b7450223e`. Tatooine, card 0. Follows lane ink-hip-p12 window 7
(`docs/results/2026-11-26-inkling2-hip-p12.md` on `ink/hip-p12`, PR #533): on main's own test binary,
`inkling2_tiny_g1b_teacher_forced_bf16` fails on HIP with the default `INTARSIA_HIP_INK_PF2` (mask 151) and passes with `=0`; the
failing errors differ run to run (`v00.attn_out` rel err 0.81 - 0.86).

Status: in progress. Culprit found (PF2 bit 1, the rel-table pass), cause found (d_rel above the rel extent), fixed on this branch;
gates in window 2.

## Result so far

- **Culprit: PF2 bit 1** (the prologue's rel tail as its own pass, `ink_rel_table_k`, lane ink-hip-pf2). Window B1, main's own test
  binary: every mask with bit 1 fails, every mask without it passes, and bit 1 alone fails (table below).
- **Cause: an out-of-row write when d_rel > E.** With bit 1 the prologue stages each head's `r` (dr values) in the first dr entries
  of that head's own rel row (`F_R_ROW`, `kernels/hip/ink_attn.hip:326`), row stride E. The tiny fixture has d_rel 16 and E 12 on the
  full layers (`rel_extent`) and 8 on the SWA layers (`sliding_window_size`), so entries E .. 15 of head h land in entries 0 .. 15-E of
  head h+1's row (written by another block of the same launch: a race, hence run-to-run varying errors) and, for the last head of
  the last row, past the end of the [T, H, E] table (16 - E bf16 into whatever the arena holds next). `ink_rel_table_k` then reads r
  back with the same stride. The launcher checked `dr <= RT_DR` and `E % 4 == 0` but not `dr <= E`.
- **Trigger, exactly:** PF2 bit 1 set (default), more than 16 call rows (`p.t > MAXROWS`), d_rel <= 16, E % 4 == 0, E <= 1024,
  and **d_rel > E**. Row count, position, partial chunk and verify length only gate bit 1 on or off; the overflow needs d_rel > E,
  which depends on the model's config alone.
- **The real model cannot hit it.** Inkling-2-Mini has d_rel 16 and E = 1024 on every layer (window 1024 on SWA layers, rel_extent
  1024 on full ones), so d_rel <= E for every call: prompts of 1..64 rows, the last partial chunk, verify passes of 2..6 rows (these
  are <= 16 rows and never take bit 1) and any batched-prefill packing (more rows, same d_rel and E). No report to the coordinator
  before the fix was needed (brief item 4).
- **Fix** (`kernels/hip/ink_attn.hip:1175`): the rel pass also needs `p.dr <= p.e`; other shapes keep the prologue's own rel tail,
  the form before bit 1 (bitwise the same values by construction: the rel pass is that tail's chain moved). Shapes of the real
  model take the same path as on main. Bit 128 (the OCC prologue) is gated by the rel pass, so it follows.
- **Admission case** (`crates/intarsia-core/src/inkling2/cases.rs`): InkKvPrologue gets (H 6, Hk 2, Dh 16, dr 16, E 12), the tiny
  fixture's d_rel / extent; its T = 40 rows take bit 1. Window 2 runs it on main's kernel (expected FAIL) and the fix.

## Window B1: bisect (fleetq #12222; radiance down 16:56:14 - 16:56:39, 0:25)

Log `raw/windowB1.log`, stage `raw/stage_b1.sh`, binary: main's `inkling2` test binary (`b7450223e`, md5 dde0c854), card 0.

| `INTARSIA_HIP_INK_PF2` | bits | g1b | `v00.attn_out` rel err |
|---|---|---|---|
| 151 (default) | 1 2 4 16 128 | FAIL, FAIL | 0.836, 0.848 |
| 0 | none | ok, ok | |
| 150 | 2 4 16 128 | ok, ok | |
| 149 | 1 4 16 128 | FAIL, FAIL | 0.845, 0.852 |
| 147 | 1 2 16 128 | FAIL, FAIL | 0.849, 0.846 |
| 135 | 1 2 4 128 | FAIL, FAIL | 0.840, 0.860 |
| 23 | 1 2 4 16 | FAIL, FAIL | 0.844, 0.832 |
| 1 | 1 | FAIL | 0.840 |
| 2 / 4 / 16 / 128 | one each | ok | |

(budget 1.801e-2)

## Briefs received

1. 2026-11-26 ~16:58 EST (coordinator, to lane ink-hip-p12, in substance; recorded here because #533 is frozen): run the PF2 bisect
   and go further; top priority. A default-on path with run-to-run varying errors is an accuracy bug against the user's
   exact-output goal, even if real-model md5 matches today. Radiance for this brief <= 8 min total, every window < 5 min.
   1. Bisect, one window < 30 s: g1b on main's binary with masks 150 / 149 / 147 / 135 / 23 (151 minus one bit each), each twice so
      a flaky pass shows.
   2. Root cause on the culprit form: why tiny shapes break it (uninitialised / stale scratch read, tile or bounds assumption such
      as rows < tile or D / heads under a minimum, or a race). Compute-sanitizer-style checks where HIP has them, else code reading
      plus a poisoned-scratch run (scratch filled with NaN before the call). State exactly which shape conditions trigger it, and
      whether any real Inkling-2-Mini call can meet them (prompt lengths 1..64 rows, the last partial chunk, verify passes of 2..6
      rows, batched prefill packing that lane ink-hip-bpf is building, which may create new row counts).
   3. Fix on a new branch `ink/hip-pf2-tiny` from current origin/main in `.worktrees/ink-hip-pf2-tiny`: the form correct for every
      shape, or the pre-PF2 form for the bad shapes if a correct one is not cheap. Gates: g1b passes 3 times in a row on HIP;
      real-model greedy md5 = main at short / 4k / 16k, 64 / 256 tokens, spec off and default; a poisoned-scratch real-model run
      gives the same md5; tiny suites on HIP >= 15/17 (only the 2 screened-head tests failing); admission for the touched op;
      prefill 4k / 16k within 0.5 % of main (single runs if they match, ABBA if not). Own PR, not merged.
   4. If step 2 shows a real-model call can hit it: report to the coordinator at once, before the fix, with the shapes (the bit
      may be turned off by default meanwhile).
   - #533: the coordinator merges it on green CI; do not change it further.

## Fleetq jobs and radiance windows

| Job | What | Radiance down |
|---|---|---|
| #12222 | B1: PF2 bisect on g1b (main's binary) | 16:56:14 - 16:56:39, 0:25 |
| #12225 | build main3 cli (`b7450223e`) in lane ink-hip-p12's dir (cpu) | none |
| #12226 | build pf2a (case only), pf2b (case + fix), test bins (cpu) | none |
