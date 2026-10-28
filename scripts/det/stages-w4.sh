#!/usr/bin/env bash
# Lane det w4 (ON tatooine, radiance down): catch the rare divergence with the layer tap on. w1+w2: 230 plain --spec off runs
# bitwise equal; 1 of 30 fresh --spec mtp processes parted at token 20 (off margin there 0.124, so a sizeable one-step glitch,
# not a last-bit difference). Tapped fresh processes, alternating mtp and off, so a diverging run says which pass and layer
# first differ from the others (and whether rank 0 and rank 1 disagree there). No --trace-margins: it turns speculation off.
set -uo pipefail
export O=${1:?out dir}
source "$(dirname "$0")/lib.sh"
clear w4 || exit 2
TP2="--tp 2 --devices 0,1"
for i in $(seq 1 45); do
  TAP=1 g mtptap$i 0,1 short-5 1 $TP2 --spec mtp --spec-k 2
  TAP=1 g offtap$i 0,1 short-5 1 $TP2 --spec off
done
log "w4 done"
