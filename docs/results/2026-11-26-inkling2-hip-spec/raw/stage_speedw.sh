#!/usr/bin/env bash
# Speed window (lane ink-hip-spec): stage_speed.sh on both cards at once, one rep per card (pairs stay within a card: a rep runs its
# three arms back to back on one card). C0REP / C1REP name the reps; give both cards reps of the same parity (odd: off gate cut,
# even: cut gate off) so the two cards run the same arm at the same time and the host load is the same for every arm of a pair.
set -uo pipefail
S=$HOME/intarsiaml-ink-hip-spec/.scratch
echo "[speedw] start $(date +%T)"
HIP_VISIBLE_DEVICES=0 REPS="${C0REP:-1}" bash $S/stage_speed.sh > $S/../logs/speedw-c0-r${C0REP:-1}.log 2>&1 &
HIP_VISIBLE_DEVICES=1 INK_CONTAINER=ink-hip-spec-run2 REPS="${C1REP:-3}" bash $S/stage_speed.sh > $S/../logs/speedw-c1-r${C1REP:-3}.log 2>&1 &
wait
cat $S/../logs/speedw-c0-r${C0REP:-1}.log $S/../logs/speedw-c1-r${C1REP:-3}.log
echo "[speedw] done $(date +%T)"
