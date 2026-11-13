#!/usr/bin/env bash
# fp8-decode: one queued serve-run on spark1: q.sh ARM EXPECT_MIN SERVE_ENV SERVE_ARGS REPS_ARGS [BIN]
cd "$(dirname "$0")/../.."
O=$PWD/docs/results/2026-11-13-fp8-decode/raw
ARM=$1 OUT=$O BIN=${6:-'~/intarsia-fp8-decode/bin/intarsia'} SERVE_ENV=$3 SERVE_ARGS=$4 REPS_ARGS=$5 MAXB=${MAXB:-2} FLEETQ_OWNER=claude-fp8-decode \
  fleetq run --host spark1 --mem 35 --quiet "${QUIET:-gpu}" --expect "$2" --label "fp8-decode: $1" -- bash scripts/fp8-decode/serve-run.sh > "$O/run-$1.log" 2>&1
echo "exit $? $1"
