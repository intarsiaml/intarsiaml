#!/bin/bash
N=/opt/code/intarsiaml/.worktrees/nxp/.nxp-out
sub() { local label=$1 exp=$2 mem=$3 ld=$4; shift 4
  (nohup fleetq run --host endor $ld --mem $mem --expect $exp --label "$label" -- "$@" > $N/q-$(echo $label | tr -c 'A-Za-z0-9' _).log 2>&1 &); sleep 2; }
sub "NXP G1 decode-64 base/fp8/cand" 20 45 "--load gpu,cpu" $N/acc64.sh $N/g1-64 "base=$N/intarsia-base" "fp8=$N/intarsia-nax" "cand=$N/intarsia-cand"
(BIN=$N/intarsia-cand nohup fleetq run --host endor --quiet gpu,cpu --mem 45 --expect 60 --label "NXP G2 MMLU n=1425 cand (35B)" -- zsh $N/mmlu-arm.sh intarsia35 $N/mmlu-cand > $N/mmlu-cand.log 2>&1 &); sleep 2
sub "NXP G3 gates cand (gb gc long rec dense)" 45 45 "--load gpu,cpu" $N/gates.sh $N/g3c gb gc long rec dense
sub "NXP G3 ids base/cand (off mtp ngram c512)" 30 45 "--load gpu,cpu" $N/ids2.sh $N/ids2 short-0 4k-0 4k-1 16k-0
sub "NXP kernels test (full) + 4k decode-256" 15 45 "--load gpu,cpu" $N/kt.sh
sub "NXP G4b speed A/B extra 2 rounds" 12 45 "--quiet gpu,cpu" $N/speed-ab.sh $N/g4b 2 "cand=$N/intarsia-cand" "base=$N/intarsia-base"
sub "NXP G5 plan show other models" 8 30 "--load cpu,disk" $N/plans2.sh
