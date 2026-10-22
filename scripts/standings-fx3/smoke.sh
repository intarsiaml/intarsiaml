#!/usr/bin/env bash
# Smoke: arms once at sizes 128,1024 (1 rep), conc 1,4; non-exclusive, so numbers are NOT results. Debugs start commands/flags.
#   MODEL=8b ARMS="intarsia llamacpp" fleetq run --host spark2 --mem 24 --expect 15 --label ... -- scripts/standings-fx3/smoke.sh
set -u
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HOST=${HOST:-spark2} MACHINE=${MACHINE:-spark-fa00} SIZES=128,1024 REPS=1 CONC=1,4 PHASES=speed,conc
OUT=$S/../../docs/results/2026-10-22-fx3-spark-standings/raw/smoke PASS=smoke-${MODEL}-${ARMS// /_} "$S/window.sh"
