# Inkling-2 HIP: `INTARSIA_HIP_INK_PF2` on tiny shapes (lane ink-hip-pf2-tiny, one R9700, 2026-11-26)

Branch `ink/hip-pf2-tiny` from origin/main `b7450223e`. Tatooine, card 0. Follows lane ink-hip-p12 window 7
(`docs/results/2026-11-26-inkling2-hip-p12.md` on `ink/hip-p12`, PR #533): on main's own test binary,
`inkling2_tiny_g1b_teacher_forced_bf16` fails on HIP with the default `INTARSIA_HIP_INK_PF2` (mask 151) and passes with `=0`; the
failing errors differ run to run (`v00.attn_out` rel err 0.81 - 0.86).

Status: in progress.

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
