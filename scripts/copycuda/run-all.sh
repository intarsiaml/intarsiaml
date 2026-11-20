#!/usr/bin/env bash
# copycuda driver (runs on endor): after the ids gate, the two ABBA halves (2 rounds each, pooled as 4 rounds) and the nsys slow-mode run, each in its own fleetq slot on spark1.
cd "$(dirname "$0")/../.." || exit 1
export FLEETQ_OWNER=claude-0b386a17-copycuda
RAW=docs/results/2026-11-18-copycuda/raw
CELLS="chat-short code-gen reason agent-edit agent-batch copy3k json-xform"
scp -q scripts/copycuda/nsys2.sh scripts/copycuda/ab.sh spark1:intarsiaml-copycuda/src/scripts/copycuda/
for h in 1 2; do
  fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 45 --label "copycuda: A/B half $h (2 rounds ABBA)" -- ssh spark1 "W=ab$h ROUNDS=2 REPS=3 CELLS='$CELLS' ARMS='base:X=1 n20:INTARSIA_SPEC_COPY=20' bash ~/intarsiaml-copycuda/src/scripts/copycuda/ab.sh" > $RAW/ab$h-run.out 2>&1
done
fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 60 --label "copycuda: nsys slow-mode" -- ssh spark1 "W=ns1 EXTRA='agent-edit copy3k' ARMS='b1:X=1 c1:INTARSIA_SPEC_COPY=20+X=1 c2:INTARSIA_SPEC_COPY=20+X=1 b2:X=1' bash ~/intarsiaml-copycuda/src/scripts/copycuda/nsys2.sh" > $RAW/ns1-run.out 2>&1
echo "all done"
