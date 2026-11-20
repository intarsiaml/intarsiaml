#!/usr/bin/env bash
# copyflipc hardware gates on spark1, inside ONE fleetq slot (--quiet gpu,cpu): (a)+(b) ids gate, then (c) ABBA with KV-once in both arms.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 100 --label "copyflipc: gates" -- bash scripts/copyflipc/gates.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
cd "$(dirname "$0")/../.."
RAW=docs/results/2026-11-20-copyflipc/raw; mkdir -p $RAW
L='~/intarsiaml-copyflipc'
# the lane's scripts are re-synced (they may be newer than the build), and the first launches of the new binary run under the lane's
# free-memory watchdog (kills this lane's processes below 15 GB available), also in the ids phase
scp -q -o BatchMode=yes scripts/copyflipc/ab.sh scripts/copyflipc/ids.sh scripts/copyflipc/wd.sh spark1:intarsiaml-copyflipc/src/scripts/copyflipc/
ssh -n -o BatchMode=yes spark1 'mkdir -p ~/intarsiaml-copyflipc/out; nohup bash ~/intarsiaml-copyflipc/src/scripts/copyflipc/wd.sh > ~/intarsiaml-copyflipc/out/wd-ids.log 2>&1 < /dev/null & echo $! > ~/intarsiaml-copyflipc/out/wd-ids.pid'
trap 'ssh -n -o BatchMode=yes spark1 "kill \$(cat ~/intarsiaml-copyflipc/out/wd-ids.pid) 2>/dev/null"' EXIT
echo "== ids gate $(date +%T)"
ssh -o BatchMode=yes spark1 "ARMS='off zero def' bash -s copy3k code-gen json-xform" < scripts/copyflipc/ids.sh 2>&1 | tee $RAW/ids-run.out
echo "== A/B $(date +%T)"
ssh -n -o BatchMode=yes spark1 "W=ab1 PORT=8411 ROUNDS=4 REPS=3 CELLS='code-gen copy3k' ARMS='def:INTARSIA_CUDA_ATTN_KV_ONCE=1 zero:INTARSIA_CUDA_ATTN_KV_ONCE=1+INTARSIA_SPEC_COPY=0' bash ~/intarsiaml-copyflipc/src/scripts/copyflipc/ab.sh" 2>&1 | tee $RAW/ab1-run.out
echo "== fetch $(date +%T)"
mkdir -p $RAW/ab1 $RAW/ids
scp -q -o BatchMode=yes -r spark1:intarsiaml-copyflipc/out/ab1/. $RAW/ab1/ 2>&1 | tail -2
scp -q -o BatchMode=yes -r spark1:intarsiaml-copyflipc/out/ids/. $RAW/ids/ 2>&1 | tail -2
echo "gates done"
