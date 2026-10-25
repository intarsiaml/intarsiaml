#!/usr/bin/env bash
# Lane sp, ON tatooine inside a fleetq job that takes radiance down: the baseline and the equality gate on real 35B ids.
#   baseline.sh <sizes> [model dir name]  : sp-base (main) spec off / mtp eager, then sp (branch) mtp (captured) / INTARSIA_SPEC_GRAPH=0 / off
# Card 0 runs the main tree, card 1 the branch tree in parallel (ids.sh pins card 0 via HIP_VISIBLE_DEVICES, overridden below).
set -uo pipefail
sizes=${1:-128,1024,4096,16384}
trap 'bash $HOME/sp-win.sh restore' EXIT
systemctl --user stop radiance; sleep 3
S=$HOME/intarsiaml-sp/scripts/sp
run() { # run <gpu> <tree> <tag> [generate args / env via E]
  g=$1; shift
  HIP_VISIBLE_DEVICES=$g bash $S/ids.sh "$@"
}
( run 0 sp-base base-off $sizes --spec off
  run 0 sp-base base-mtp2 $sizes --spec mtp --spec-k 2 ) > $HOME/sp-ids/b0.log 2>&1 &
( run 1 sp br-mtp2 $sizes --spec mtp --spec-k 2
  INTARSIA_SPEC_GRAPH=0 run 1 sp br-mtp2-eager $sizes --spec mtp --spec-k 2
  run 1 sp br-off $sizes --spec off ) > $HOME/sp-ids/b1.log 2>&1 &
wait
cat $HOME/sp-ids/b0.log $HOME/sp-ids/b1.log
