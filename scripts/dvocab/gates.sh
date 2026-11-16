#!/usr/bin/env bash
# dvocab CUDA gates of INTARSIA_VERIFY_TOPK on spark2 (one fleetq slot, the lane's binary ~/intarsia-dvocab/bin/intarsia-topk):
#   plan:  `plan show --json` of the 35B, main's binary vs the lane's, knob unset (and set)
#   ids:   seeded ids (ids.py: temperature 0.6 / 1.0 at 1, 2, 4 concurrent, top_k 0, mixed greedy + sampled), --max-batch 2:
#            ref  --spec off (no drafting)
#            vd   INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1 --mtp-draft-vocab 65536 (logits rows)
#            vdt  vd + INTARSIA_VERIFY_TOPK=1 (top-k rows)
# Run under: fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 40 --label "dvocab: gates" -- scripts/dvocab/gates.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
export OUT=${OUT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/docs/results/2026-11-14-dvocab/raw/gates}
export BIN='~/intarsia-dvocab/bin/intarsia-topk'
LOGNAME_=gates; source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
log "gates start; vLLM: $(curl -s -o /dev/null -w '%{http_code}' http://$HN:$VPORT/v1/models); earlyoom: $(rsh 'systemctl is-active earlyoom'); $(rsh 'free -g | sed -n 2p')"
DV="--spec mtp --spec-k 2 --mtp-draft-vocab 65536"
for part in ${PARTS:-plan ids}; do
  case $part in
    plan)
      for b in intarsia intarsia-topk; do
        rsh "~/intarsia-dvocab/bin/$b plan show --backend cuda --model $M35 --max-seq 8192 --mem-budget 40 --json > ~/intarsia-dvocab/out/plan-$b.json 2> ~/intarsia-dvocab/out/plan-$b.err; echo exit \$?"
      done
      rsh "INTARSIA_VERIFY_TOPK=1 ~/intarsia-dvocab/bin/intarsia-topk plan show --backend cuda --model $M35 --max-seq 8192 --mem-budget 40 --json > ~/intarsia-dvocab/out/plan-topk-on.json 2> /dev/null; echo exit \$?"
      log "plan show json, main's binary vs the lane's (knob unset): $(rsh "cmp ~/intarsia-dvocab/out/plan-intarsia.json ~/intarsia-dvocab/out/plan-intarsia-topk.json && echo IDENTICAL; wc -c < ~/intarsia-dvocab/out/plan-intarsia.json" | tr '\n' ' ')"
      log "plan show json, the lane's binary, knob set vs unset: $(rsh "cmp ~/intarsia-dvocab/out/plan-intarsia-topk.json ~/intarsia-dvocab/out/plan-topk-on.json && echo IDENTICAL" | tr '\n' ' ')"
      mkdir -p "$OUT/plan"; scp -q -o BatchMode=yes "$HOST:intarsia-dvocab/out/plan-*" "$OUT/plan/"
      ;;
    ids)
      for a in ${ARMS:-ref vd vdt}; do
        case $a in
          ref) e="X=1"; args="--spec off" ;;
          vd)  e="INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1"; args="$DV" ;;
          vdt) e="INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1 INTARSIA_VERIFY_TOPK=1"; args="$DV" ;;
        esac
        if start_serve "ids-$a" "$e" --max-batch 2 $args; then
          python3 "$S/ids.py" "$URL" guidance "$OUT/ids-$a.json" "http://$HN:$PORT/metrics" 2>&1 | tee -a "$LOG"
          finish_serve "ids-$a"
        fi
      done
      python3 "$S/ids_compare.py" "$OUT" ${ARMS:-ref vd vdt} 2>&1 | tee -a "$LOG" "$OUT/ids-compare.txt"
      ;;
  esac
done
