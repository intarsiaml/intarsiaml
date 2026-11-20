#!/bin/zsh
# waits for the build slot (#10431) to finish, then takes the gates slot
cd ${0:A:h:h:h}; export FLEETQ_OWNER=claude-copyflipc
RAW=docs/results/2026-11-20-copyflipc/raw
until grep -qE "fleetq: #[0-9]+ (done|failed)" $RAW/build-run.out; do sleep 15; done
grep -q "fleetq: #[0-9]* done (exit 0)" $RAW/build-run.out || { echo "build failed"; exit 1; }
fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 100 --label "copyflipc: gates (ids + KV-once ABBA)" -- bash scripts/copyflipc/gates.sh > $RAW/gates-run.out 2>&1
echo "driver done"
