#!/usr/bin/env bash
# mtp-sampled paired speed A/B at temperature 1.0, prefix store off, spark1's vLLM up (idle is checked, not enforced).
# Run under: fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 50 --label "mtp-sampled: speed A/B" -- scripts/mtp-sampled/ab.sh
#   A  INTARSIA_BATCHED_MTP_SAMPLED=0  (main: sampled requests take plain steps)
#   B  sampled drafting on             (this lane)
# both with the shipped defaults otherwise (MTP k2 while one sequence decodes), ABBA per --max-batch 2 and 4.
#   N2 / V2 (after the A/B): INTARSIA_BATCHED_MTP=2 without / with INTARSIA_BATCHED_VERIFY=1 at --max-batch 2, 2 streams:
#   whether sampled drafting pays while two sequences decode.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
LOGNAME_=ab; source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
log "ab start; vLLM: $(curl -s -o /dev/null -w '%{http_code}' http://$HN:$VPORT/v1/models); earlyoom: $(rsh 'systemctl is-active earlyoom'); $(rsh 'free -g | sed -n 2p')"
visit() { # visit <name> <streams> "<envs>" <args...>
  local name=$1 streams=$2; shift 2
  if start_serve "$name" "$@"; then
    log "vLLM requests running before $name: $(vllm_busy)"
    ( cd /opt/code/deepeval && uv run python "$S/speed.py" --url "$URL" --model guidance --arm "$name" --streams "$streams" --out "$OUT" --metrics "http://$HN:$PORT/metrics" ) 2>&1 | tee -a "$LOG"
    log "vLLM requests running after $name: $(vllm_busy)"
    finish_serve "$name"
  fi
}
A="INTARSIA_BATCHED_MTP_SAMPLED=0"; B="INTARSIA_BATCHED_MTP_SAMPLED=1"
for part in ${PARTS:-mb2 mb4 extra}; do
  case $part in
    mb2) visit mb2-A1 1,2 "$A" --max-batch 2; visit mb2-B1 1,2 "$B" --max-batch 2; visit mb2-B2 1,2 "$B" --max-batch 2; visit mb2-A2 1,2 "$A" --max-batch 2 ;;
    mb4) visit mb4-A1 1,2,4 "$A" --max-batch 4; visit mb4-B1 1,2,4 "$B" --max-batch 4; visit mb4-B2 1,2,4 "$B" --max-batch 4; visit mb4-A2 1,2,4 "$A" --max-batch 4 ;;
    extra)
      visit mb2-N2 2 "INTARSIA_BATCHED_MTP=2" --max-batch 2
      visit mb2-V2 2 "INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1" --max-batch 2 ;;
  esac
done
python3 "$S/ab_table.py" "$OUT" 2>&1 | tee -a "$LOG" "$OUT/ab-table.md"
