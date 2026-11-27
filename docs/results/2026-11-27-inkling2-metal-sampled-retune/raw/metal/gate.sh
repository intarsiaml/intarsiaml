#!/bin/bash
# endor EXCLUSIVE fleetq slot (<= 20 min), lane ink-metal-sampled2: the gate speed A/B at ONE temperature on the gate seeds 7 23 29.
# usage: gate.sh TAG T SUFFIX   (T 1.0 suffix 1 | T 0.7 suffix 07). Env: POLSTR (the policy string), BIN_POL (default = BIN), SEEDS.
# arms: off (sampled drafting unset), fix (=1, ADAPT=0), pol (=1, ADAPT=$POLSTR); Latin square per seed; 45 s idle before every timed process.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
. /opt/code/intarsiaml/.worktrees/ink-metal-sampled2/docs/results/2026-11-27-inkling2-metal-sampled-retune/raw/metal/common.sh
TAG=$1; T=$2; SUF=$3; SEEDS=${SEEDS:-"7 23 29"}; BIN_POL=${BIN_POL:-$BIN}
: "${POLSTR:?POLSTR}"
O=$S/metal/gate-$TAG; mkdir -p $O/speed
T0=$(date +%s); guard
echo "start $(date +%T) out $O | off/fix $(md5 -q $BIN) pol $(md5 -q $BIN_POL) | pol string: $POLSTR"
macmon pipe -i 250 > $O/speed/macmon-$SUF.jsonl 2>/dev/null &
arm() { # arm seed
  local a=$1 sd=$2 f=$O/speed/$1$SUF-s$2 b=$BIN; shift 2
  local pre="FOO=1"
  case $a in fix) pre="INTARSIA_SPEC_SAMPLED_DFLASH=1 INTARSIA_SPEC_SAMPLED_ADAPT=0" ;; pol) pre="INTARSIA_SPEC_SAMPLED_DFLASH=1 INTARSIA_SPEC_SAMPLED_ADAPT=$POLSTR"; b=$BIN_POL ;; esac
  sleep $IDLE
  stamp > $f.t
  env $pre perl -e 'alarm shift; exec @ARGV' 300 $b $G --seed $sd --temperature $T > $f.out 2> $f.err
  local rc=$?
  stamp >> $f.t
  echo "$a$SUF s$sd rc $rc $(date +%T) | $(grep -aE '^decode overall' $f.err) | $(grep -aE '^spec: verify' $f.err | cut -c1-100)"
}
for sd in $SEEDS; do
  case $sd in 7) order="off fix pol" ;; 23) order="fix pol off" ;; 29) order="pol off fix" ;; *) order="off fix pol" ;; esac
  for a in $order; do
    [ $(( $(date +%s) - T0 )) -ge ${BUDGET_S:-1130} ] && { echo "budget reached before $a s$sd"; break 2; }
    arm $a $sd
  done
done
echo "end $(date +%T) ($(( $(date +%s) - T0 )) s)"
python3 $W/docs/results/2026-11-27-inkling2-metal-sampled-retune/raw/speed4.py $O/speed --macmon > $O/speed/speed-$SUF.txt 2>&1
cat $O/speed/speed-$SUF.txt
