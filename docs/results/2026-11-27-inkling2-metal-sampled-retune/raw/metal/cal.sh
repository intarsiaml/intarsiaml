#!/bin/bash
# endor EXCLUSIVE fleetq slot, lane ink-metal-sampled2, calibration. Seeds 11 13 17 19 only (never a gate seed).
# 1. plain-step overhead of the policy mode: off vs "plainonly" (drafts once per prompt, then plain steps for the rest), ABBA, seed 13, T 1.0
# 2. traced fixed k3 and traced current-recipe policy, seed 13, T 1.0 (per-kind step times)
# 3. `drafts` traces (every step a draft call at k3 + a plain step): seeds 11 13 17 19 x T 1.0 / 0.7, back to back (no timing use)
set -u
: "${FLEETQ_JOB:?run under fleetq}"
. /opt/code/intarsiaml/.worktrees/ink-metal-sampled2/docs/results/2026-11-27-inkling2-metal-sampled-retune/raw/metal/common.sh
O=$S/metal/cal-${1:-$(date +%m%d-%H%M)}; mkdir -p $O/ab $O/tr $O/drafts
T0=$(date +%s)
guard
echo "start $(date +%T) out $O | $(md5 -q $BIN)"
macmon pipe -i 250 > $O/macmon.jsonl 2>/dev/null &
PO="drop=-100,hold=1000000,hold_max=1000000,probe=1"
run() { # name outdir seed temp envassignments...
  local n=$1 od=$2 sd=$3 t=$4; shift 4
  local f=$od/$n; stamp > $f.t
  env "$@" perl -e 'alarm shift; exec @ARGV' 300 $BIN $G --seed $sd --temperature $t > $f.out 2> $f.err
  local rc=$?; stamp >> $f.t
  echo "$n rc $rc $(date +%T) | $(grep -aE '^decode overall' $f.err) | $(grep -aE '^spec: verify' $f.err | cut -c1-110)"
}
i=0
for a in off po po off; do
  i=$((i+1)); sleep $IDLE
  case $a in off) run $a-$i $O/ab 13 1.0 FOO=1 ;; po) run $a-$i $O/ab 13 1.0 INTARSIA_SPEC_SAMPLED_DFLASH=1 INTARSIA_SPEC_SAMPLED_ADAPT=$PO ;; esac
done
sleep $IDLE
run fixtr $O/tr 13 1.0 INTARSIA_SPEC_SAMPLED_DFLASH=1 INTARSIA_SPEC_SAMPLED_ADAPT=0 INTARSIA_SPEC_TRACE=1
sleep $IDLE
run poltr $O/tr 13 1.0 INTARSIA_SPEC_SAMPLED_DFLASH=1 INTARSIA_SPEC_TRACE=1
for sd in 11 13 17 19; do for t in 1.0 0.7; do
  [ $(( $(date +%s) - T0 )) -ge ${BUDGET_S:-1000} ] && { echo "budget reached before drafts s$sd T$t"; break 2; }
  run s$sd-t$t $O/drafts $sd $t INTARSIA_SPEC_SAMPLED_DFLASH=1 INTARSIA_SPEC_SAMPLED_ADAPT=drafts INTARSIA_SPEC_TRACE=1
done; done
echo "end $(date +%T) ($(( $(date +%s) - T0 )) s)"
