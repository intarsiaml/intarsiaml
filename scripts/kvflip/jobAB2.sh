#!/usr/bin/env bash
# kvflip tp2 gate, second round (more reps; no ids): BAAB after jobAB.sh's ABBA, 5 reps per cell per window.
#   FLEETQ_OWNER=claude-kvflip fleetq run --host spark1,spark2 --quiet gpu,cpu --mem spark1=45,spark2=45 --expect 40 --label "kvflip tp2 BAAB" -- bash scripts/kvflip/jobAB2.sh
D="$(dirname "${BASH_SOURCE[0]}")"
ON="INTARSIA_CUDA_ATTN_KV_ONCE=1"
C="0.7:4096,32768,65536:5 0:4096,32768,65536:5"
bash "$D/tp2.sh" b3 "$ON" $C
bash "$D/tp2.sh" a3 - $C
bash "$D/tp2.sh" a4 - $C
bash "$D/tp2.sh" b4 "$ON" $C
