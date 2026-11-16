#!/usr/bin/env bash
# dsubset: generate the COLLECT prompts on the 35B (CUDA, spark1) with a plain server (no drafting, --max-batch 8) and keep the
# generated ids (raw/collect.jsonl). BIN: any recent intarsia build (generation only). Run under:
#   fleetq run --host spark1 --quiet gpu --mem 50 --expect 40 --label "dsubset: collect" -- scripts/dsubset/collect.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
LOGNAME_=collect; source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RAW=$REPO/docs/results/2026-11-15-dsubset/raw
log "collect start; vLLM: $(curl -s -o /dev/null -w '%{http_code}' http://$HN:$VPORT/v1/models); earlyoom: $(rsh 'systemctl is-active earlyoom'); $(rsh 'free -g | sed -n 2p')"
if start_serve collect "X=1" --max-batch 8; then
  python3 "$S/collect.py" "$URL" guidance "$RAW/collect.jsonl" --conc 8 --max-tokens 1024 2>&1 | tee -a "$LOG"
  finish_serve collect
fi
