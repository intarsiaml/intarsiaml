#!/bin/zsh
# ONE fleetq slot (--quiet gpu,cpu), the whole prioritised ABAB sequence inside it; stops starting new arms after 05:20 EST (hard stop 06:00).
# A = .scratch/intarsia.def (baseline), B = .scratch/intarsia.rt (this lane); serve --max-batch MB --spec off --max-seq 18000.
# P1 mb8 16k (conc 8,16)  P2 mb8 4k (conc 8,16)  P3 mb4 4k+16k (conc 4)
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/m27-hs-rowtable; cd $W
O=docs/results/2026-11-03-m27-hs-rowtable/raw/speed; S=docs/results/2026-11-03-m27-hs-rowtable/raw/scripts/conc-arm.sh
n=0
for cfg in "16384 8 8,16" "4096 8 8,16" "4096_16384 4 4"; do
  set -- ${=cfg}; sizes=${1//_/ }; mb=$2; conc=$3
  for t in def rt def rt; do
    n=$((n+1))
    if [ $(date +%H%M) -ge 0520 ] && [ $(date +%H%M) -lt 1200 ]; then echo "past 05:20: skipping $t mb$mb sizes $sizes (#$n)"; continue; fi
    echo "$(date +%T) start $t mb$mb sizes=$sizes conc=$conc #$n"
    SIZES="$sizes" GEN=512 zsh $S $t-mb$mb-$n $PWD/.scratch/intarsia.$t $O $mb $conc
    echo "$(date +%T) done  $t mb$mb #$n exit $?"
  done
done
