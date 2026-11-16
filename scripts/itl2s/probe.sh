#!/usr/bin/env bash
# itl2s: post-EOS probe on the default server (see eos_probe.py). Run under fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 8
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
LOGNAME_=probe; source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
arm_setup F.a
if start_serve F.a "$ENVS" $ARGS; then
  rsh "python3 $L/scripts/itl2s/eos_probe.py http://localhost:$PORT/v1 guidance $L/out/eos-probe.json" 2>&1 | tee -a "$LOG"
  rsh "cat $L/out/eos-probe.json" > "$OUT/eos-probe.json"
  finish_serve F.a
fi
