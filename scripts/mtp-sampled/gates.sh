#!/usr/bin/env bash
# mtp-sampled CUDA gate: seeded sampled ids with drafting on equal drafting off, B = 1, 2, 4, and acceptance at T 0.6 / 1.0.
# Run under: fleetq run --host spark1 --quiet gpu --mem 50 --expect 40 --label "mtp-sampled: ids gate" -- scripts/mtp-sampled/gates.sh
#   ref      --spec off                                   (no drafting at all: the reference)
#   off      MTP at B <= 4, INTARSIA_BATCHED_MTP_SAMPLED=0 (greedy requests draft, sampled ones take plain steps: main)
#   on       MTP at B <= 4, sampled drafting on, solo verify passes
#   onv      the same with INTARSIA_BATCHED_VERIFY=1       (one ragged verify pass, logits rows for the sampled sequences)
#   default  shipped defaults (--max-batch 4: MTP while one sequence decodes)
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
LOGNAME_=gates; source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
log "gates start; vLLM: $(curl -s -o /dev/null -w '%{http_code}' http://$HN:$VPORT/v1/models); earlyoom: $(rsh 'systemctl is-active earlyoom'); $(rsh 'free -g | sed -n 2p')"
arm() { # arm <name> "<envs>" <args...>
  local name=$1
  if start_serve "$@"; then
    python3 "$S/ids.py" "$URL" guidance "$OUT/ids-$name.json" "http://$HN:$PORT/metrics" 2>&1 | tee -a "$LOG"
    finish_serve "$name"
  fi
}
ARMS=${ARMS:-ref off on onv default}
for a in $ARMS; do
  case $a in
    ref)     arm ref "MTP_SAMPLED_ARM=1" --max-batch 4 --spec off ;;
    off)     arm off "INTARSIA_BATCHED_MTP=4 INTARSIA_BATCHED_MTP_SAMPLED=0" --max-batch 4 --spec-off-at 5 ;;
    on)      arm on "INTARSIA_BATCHED_MTP=4" --max-batch 4 --spec-off-at 5 ;;
    onv)     arm onv "INTARSIA_BATCHED_MTP=4 INTARSIA_BATCHED_VERIFY=1" --max-batch 4 --spec-off-at 5 ;;
    default) arm default "MTP_SAMPLED_ARM=1" --max-batch 4 ;;
  esac
done
python3 "$S/ids_compare.py" "$OUT" $ARMS 2>&1 | tee -a "$LOG" "$OUT/ids-compare.txt"
