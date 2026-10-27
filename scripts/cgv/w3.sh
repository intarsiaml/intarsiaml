#!/usr/bin/env bash
# Lane cgv window W3 driver (Mac side, inside a held fleetq slot: FLEETQ_JOB=<id> scripts/cgv/w3.sh): decode profiles, plan show, GPU tests, ids, mutants.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
D=$(cd "$(dirname "$0")" && pwd); RAW=$D/../../docs/results/2026-10-27-cgv-cuda-mxfp4-gemv/raw; mkdir -p "$RAW/plan" "$RAW/gates" "$RAW/ids" "$RAW/mutants"
echo "== profiles $(date +%T)"
"$D/profs.sh" main-off:main:--spec_off main-on:main dev-off:dev:--spec_off dev-on:dev
echo "== plan show $(date +%T)"
ssh spark1 "FLEETQ_JOB=$FLEETQ_JOB bash -s" < "$D/plan.sh" | tee "$RAW/plan/plan.txt"
scp -q 'spark1:intarsiaml-cgv/out/plan/*' "$RAW/plan/" 2>/dev/null
echo "== gpu tests $(date +%T)"
ssh spark1 "FLEETQ_JOB=$FLEETQ_JOB bash -s" < "$D/gpu-tests.sh" | tee "$RAW/gates/gpu-tests.txt"
scp -q 'spark1:intarsiaml-cgv/out/gates/*' "$RAW/gates/" 2>/dev/null
echo "== ids $(date +%T)"
ssh spark1 "FLEETQ_JOB=$FLEETQ_JOB bash -s short-0 short-1 mid-0 4k-0" < "$D/ids.sh" | tee "$RAW/ids/ids.txt"
scp -q 'spark1:intarsiaml-cgv/out/ids/*' "$RAW/ids/" 2>/dev/null
echo "== mutants $(date +%T)"
ssh spark1 "FLEETQ_JOB=$FLEETQ_JOB python3 - dev" < "$D/mutants.py" | tee "$RAW/mutants/mutants.txt"
echo "== W3 done $(date +%T)"
