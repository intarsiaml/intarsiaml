#!/usr/bin/env bash
# Lane det w2 (ON tatooine, radiance down): soak for the rare plain-decode divergence vat saw (1 fresh process of ~6 on short-5).
# w1 found 72 runs in 11 processes bitwise equal, so this window counts fresh processes: (A) vat's own sequence, a plain --spec off
# process then an --spec mtp process (the next plain process starts on the memory that one left), 30 times; (B) 40 plain processes
# with the layer tap; (C) one process, 60 runs with the tap.
set -uo pipefail
export O=${1:?out dir}
source "$(dirname "$0")/lib.sh"
clear w2 || exit 2
TP2="--tp 2 --devices 0,1"
for i in $(seq 1 30); do
  g soakA-ids$i 0,1 short-5 1 $TP2 --spec off
  g soakA-mtp$i 0,1 short-5 1 $TP2 --spec mtp --spec-k 2
done
for i in $(seq 1 40); do TAP=1 g soakB-tap$i 0,1 short-5 1 $TP2 --spec off; done
TAP=1 g soakC-tap 0,1 short-5 60 $TP2 --spec off
log "w2 done"
