#!/usr/bin/env bash
# Lane ag1: HIP no-op gate, ON tatooine in a cpu slot (no weights are read): `plan show --json` of the lane binary (ag1-intarsia) against the base
# (ag1-intarsia-base) for the configurations the lane's change must not touch: 8B tp1, 35B tp1 / tp2, 27B tp2, each default / --spec off / --spec mtp k2;
# and the 27B tp1 (the one a default flip would change). Prints SAME / DIFF per configuration.
#   fleetq run --host tatooine --load cpu --expect 10 --label "ag1 hip no-op" -- ssh tatooine 'bash ~/intarsiaml-ag1/scripts/ag1/hip-noop.sh <out dir>'
set -uo pipefail
export O=${1:?out dir}; mkdir -p "$O"
source "$HOME/intarsiaml-ag1/scripts/ag1/lib.sh"
for cfg in "m8-tp1:$M8:" "m35-tp1:$M35:" "m35-tp2:$M35:$TP2" "m27-tp2:$M27:$TP2" "m27-tp1:$M27:"; do
  tag=${cfg%%:*}; rest=${cfg#*:}; m=${rest%%:*}; ex=${rest#*:}
  gpus=0; [[ -n $ex ]] && gpus=0,1   # tp2 needs both cards
  for sp in default off mtp2; do
    [[ $tag == m8-* && $sp == mtp2 ]] && continue   # the 8B has no MTP head
    case $sp in default) sa="";; off) sa="--spec off";; mtp2) sa="--spec mtp --spec-k 2";; esac
    for side in base lane; do
      bin=ag1-intarsia; [[ $side == base ]] && bin=ag1-intarsia-base
      # shellcheck disable=SC2086
      BIN=$bin run dev $gpus 300 "plan-$tag-$sp-$side" plan show --backend hip --model "$m" --mem-budget 29 --max-seq 4096 --json $ex $sa
    done
    if [[ -s "$O/plan-$tag-$sp-lane.out" ]] && cmp -s "$O/plan-$tag-$sp-base.out" "$O/plan-$tag-$sp-lane.out"; then log "SAME plan $tag $sp ($(wc -c < "$O/plan-$tag-$sp-lane.out") bytes)"
    else log "DIFF plan $tag $sp (lane $(wc -c < "$O/plan-$tag-$sp-lane.out") bytes, base $(wc -c < "$O/plan-$tag-$sp-base.out") bytes)"; fi
  done
done
log "hip no-op done"
