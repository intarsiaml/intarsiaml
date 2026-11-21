#!/usr/bin/env bash
# samphost gate: ABBA windows (none, fast, fast, none) on one binary; sampled T=0.7 at 4k / 32k / 64k (8 reps per window), greedy (3 reps),
# seeded replies (two seeds, 4k / 32k / 64k, 700 tokens) for the byte-for-byte comparison.
#   FLEETQ_OWNER=claude-samphost fleetq run --host spark2 --quiet gpu,cpu --mem 55 --expect 120 --label "samphost ABBA" -- bash scripts/samphost/jobA.sh [first-window-index]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
SC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
F=INTARSIA_SAMPLED_VERIFY_FAST=1
export SEEDS=11,23
# re-export the profile captures whose decode-window table failed (nsys_tables.py quoting bug, fixed), before any timing window
REPO="$(git -C "$SC" rev-parse --show-toplevel)"; RAW=$REPO/docs/results/2026-11-20-samphost/raw
for c in n0:nsys-n0-T0.7-4096 n0:nsys-n0-T0-4096 n0:nsys-n0-T0.7-65536 n1:nsys-n1-T0.7-4096; do
  w=${c%%:*}; n=${c#*:}
  [[ -f $RAW/nsys-$w/$n-decode-window.json ]] && continue
  ssh -n -o BatchMode=yes spark2 "cd ~/intarsiaml-samphost/out/nsys-$w && nsys export --type sqlite --force-overwrite true -o $n.sqlite $n.nsys-rep > /dev/null 2>&1; python3 ../nsys_tables.py $n.sqlite --json $n-decode-window.json > $n-tables.txt 2>&1; rm -f $n.sqlite"
  scp -q -o BatchMode=yes "spark2:intarsiaml-samphost/out/nsys-$w/$n-decode-window.json" "spark2:intarsiaml-samphost/out/nsys-$w/$n-tables.txt" "$RAW/nsys-$w/" || echo "re-export $n failed"
done
W=(a1 - a2 "$F" a3 "$F" a4 -)
for ((i=${1:-0}*2; i<${#W[@]}; i+=2)); do
  bash "$SC/window.sh" "${W[i]}" "" "${W[i+1]}" seeded:4096,32768,65536 0.7:4096,32768,65536:8 0:4096,32768,65536:3 || echo "window ${W[i]} exit $?"
done
