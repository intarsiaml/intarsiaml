#!/usr/bin/env bash
# Speed stage (lane ink-hip-spec): cross-process A/B of DSpark on HIP. Arms: off (--spec off), gate (--spec dflash k3, gate 0.8 = the
# non-Metal default rule), cut (--spec dflash k3, INTARSIA_DSPARK_TAU=0 INTARSIA_DSPARK_CUT=0.5 = Metal's default rule). One process per
# arm and rep over speed.json (the 12 greedy golden prompts = short, then the 1021 / 4108 / 16455-id prompts), 128 tokens, greedy,
# no --ignore-eos. Reps alternate the arm order (off gate cut / cut gate off). REPS="1 2" picks the reps of this window.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-spec
S=$REPO/.scratch
OUT=$REPO/logs/speed
mkdir -p "$OUT"
M=/models/Inkling-2-Mini
B=/work/.scratch/bin/intarsia-${BR:-spec1}
G="generate -m $M --backend hip --max-seq 20480 --profile-every 0 --show-ids --max-tokens 128 --prompt-ids /work/.scratch/prompts/speed.json"
D="--spec dflash --spec-draft $M --spec-k 3"
now() { date '+%F %T'; }
arm() {  # name rep
  local f=$OUT/$1-r$2
  case $1 in
    off) bash "$S/run.sh" $B $G --spec off > $f.out 2> $f.err ;;
    gate) bash "$S/run.sh" $B $G $D > $f.out 2> $f.err ;;
    cut) bash "$S/run.sh" env INTARSIA_DSPARK_TAU=0 INTARSIA_DSPARK_CUT=0.5 $B $G $D > $f.out 2> $f.err ;;
  esac
  echo "[stage] $1 r$2 rc $? $(now) | $(grep -aE '^decode overall' $f.err | head -1) | $(grep -aE '^spec: verify' $f.err | head -1)"
}
for rep in ${REPS:-1 2}; do
  if [ $((rep % 2)) = 1 ]; then order="off gate cut"; else order="cut gate off"; fi
  for a in $order; do arm $a $rep; done
done
echo "[stage] done $(now)"
