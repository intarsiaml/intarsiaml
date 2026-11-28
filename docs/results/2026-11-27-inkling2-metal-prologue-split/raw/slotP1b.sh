#!/bin/bash
# endor EXCLUSIVE fleetq slot, lane ink-prologue2: P1 again with 4096-row chunks added (INTARSIA_INK_LONG_CHUNK=1's pass size).
set -u
: "${FLEETQ_JOB:?run under fleetq}"
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
W=/opt/code/intarsiaml/.worktrees/ink-prologue2; S=$W/.scratch; R=$W/docs/results/2026-11-27-inkling2-metal-prologue-split/raw
echo "slot P1b start $(date +%T) test exe md5 $(md5 -q $S/bin/t-ps) git $(git -C $W rev-parse --short HEAD)"
$S/bin/t-ps --test-threads=1 --nocapture > $R/P/p1b.txt 2>&1
echo "p1b rc $? | $(grep -aE '^test result' $R/P/p1b.txt | tail -1) | $(grep -aE 'bitwise$' $R/P/p1b.txt | wc -l | tr -d ' ') cases bitwise | $(grep -aE 'panicked|differs' $R/P/p1b.txt | head -2 | cut -c1-300)"
echo "slot P1b end $(date +%T)"
