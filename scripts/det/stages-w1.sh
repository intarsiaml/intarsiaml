#!/usr/bin/env bash
# Lane det w1 (ON tatooine, radiance down): is plain greedy decode (--spec off) of Qwen3.8-27B MXFP4 repeatable run to run?
# Prompt short-5 (vat's flaky one), 256 ids per run, several runs per process (--repeat) and several processes.
# Arms: the real path (on-device argmax, no tap); the layer tap + logits hashes (rank 0 vs rank 1 inside a run, and run vs run);
# tp=1 on one card; the all-reduce drain, poison and peer-sync knobs.
set -uo pipefail
export O=${1:?out dir}
source "$(dirname "$0")/lib.sh"
clear w1 || exit 2
TP2="--tp 2 --devices 0,1"
g base-ids 0,1 short-5 10 $TP2 --spec off
for i in 1 2 3; do TAP=1 g base-tap$i 0,1 short-5 6 $TP2 --spec off --trace-margins; done
TAP=1 g tp1-tap 0 short-5 6 --spec off --trace-margins
TAP=1 E="INTARSIA_HIP_AR_DRAIN=1" g drain1-tap 0,1 short-5 6 $TP2 --spec off --trace-margins
TAP=1 E="INTARSIA_HIP_POISON=ff" g poisonff-tap 0,1 short-5 4 $TP2 --spec off --trace-margins
TAP=1 E="INTARSIA_HIP_POISON=00" g poison00-tap 0,1 short-5 4 $TP2 --spec off --trace-margins
E="INTARSIA_TP_PEER_SYNC=1" g peersync-ids 0,1 short-5 10 $TP2 --spec off
TAP=1 E="INTARSIA_SOLO_GRAPHS=0" g eager-tap 0,1 short-5 4 $TP2 --spec off --trace-margins
for p in short-0 short-1 short-2; do g ids-$p 0,1 $p 6 $TP2 --spec off; done
log "w1 done"
