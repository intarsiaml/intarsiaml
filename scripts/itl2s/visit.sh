#!/usr/bin/env bash
# itl2s: for each visit in ARMS (e.g. "F.a OLD.a OLD.b F.b"): fresh server, the CELLS ("en:0:default code:0:default ..."), logs + results back.
# Run under: fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 30 --label "itl2s: ..." -- env ARMS="..." CELLS="..." scripts/itl2s/visit.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
LOGNAME_=${LOGNAME_:-visit}; source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
log "start; vLLM: $(curl -s -o /dev/null -w '%{http_code}' http://$HN:$VPORT/v1/models) running $(vllm_busy); earlyoom: $(rsh 'systemctl is-active earlyoom'); $(rsh 'free -g | sed -n 2p')"
for v in ${ARMS:?ARMS}; do
  arm_setup "$v" || { log "unknown arm $v"; continue; }
  if start_serve "$v" "$ENVS" $ARGS; then
    log "vLLM requests running before $v: $(vllm_busy)"
    curl -s "http://$HN:$PORT/v1/intarsia/plan" > "$OUT/plan-$v.json"
    run_cells "$v"
    log "vLLM requests running after $v: $(vllm_busy)"
    finish_serve "$v"
  fi
done
