#!/usr/bin/env bash
# flipprep CUDA gates on spark2 (one fleetq slot). PARTS (default "plan ids tools"):
#   plan   `plan show --json` of the 35B on CUDA: main's binary vs PR 1's (must be byte-identical) and vs the flip's (reported)
#   ids    ids.py against servers R (flip binary, --spec off), F (flip defaults), A (main defaults), --max-batch 2: greedy and seeded T = 1.0, B = 1, 2
#   tools  tools.py (24 chat requests with a `tools` array) against A, F and F0 (full head): greedy 1 stream, T = 1.0 1 stream, greedy 2 streams
# Run under: fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 60 --label "flipprep: gates" -- scripts/flipprep/gates.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
OUT=${OUT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/docs/results/2026-11-15-flipprep/raw/gates}
LOGNAME_=gates; source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"; source "$S/arms.sh"
log "gates start; vLLM: $(curl -s -o /dev/null -w '%{http_code}' http://$HN:$VPORT/v1/models); earlyoom: $(rsh 'systemctl is-active earlyoom'); $(rsh 'free -g | sed -n 2p')"
for part in ${PARTS:-plan ids tools}; do
  case $part in
    plan)
      for b in main p1 flip; do
        rsh "$B/intarsia-$b plan show --backend cuda --model $M35 --max-seq 8192 --mem-budget 40 --json > ~/intarsia-flipprep/out/plan-$b.json 2> ~/intarsia-flipprep/out/plan-$b.err; echo plan $b exit \$?" | tee -a "$LOG"
      done
      log "plan show json, main vs PR 1 (35B, cuda): $(rsh "cmp ~/intarsia-flipprep/out/plan-main.json ~/intarsia-flipprep/out/plan-p1.json && echo IDENTICAL; wc -c < ~/intarsia-flipprep/out/plan-main.json" | tr '\n' ' ')"
      log "plan show json, main vs the flip (35B, cuda): $(rsh "cmp ~/intarsia-flipprep/out/plan-main.json ~/intarsia-flipprep/out/plan-flip.json && echo IDENTICAL" | tr '\n' ' ')"
      mkdir -p "$OUT/plan"; scp -q -o BatchMode=yes "$HOST:intarsia-flipprep/out/plan-*" "$OUT/plan/"
      ;;
    ids)
      for a in ${ARMS:-R F A}; do
        arm_setup $a
        if start_serve "ids-$a" "$ENVS" --max-batch 2 $ARGS; then
          python3 "$S/ids.py" "$URL" guidance "$OUT/ids-$a.json" "http://$HN:$PORT/metrics" 2>&1 | tee -a "$LOG"
          finish_serve "ids-$a"
        fi
      done
      python3 "$S/ids_compare.py" "$OUT" ${ARMS:-R F A} 2>&1 | tee -a "$LOG" "$OUT/ids-compare.txt"
      ;;
    tools)
      for a in ${TARMS:-A F F0}; do
        arm_setup $a
        if start_serve "tools-$a" "$ENVS" --max-batch 2 $ARGS; then
          for cell in "0 1" "1.0 1" "0 2"; do
            set -- $cell
            python3 "$S/tools.py" --url "$URL" --model guidance --arm "$a" --out "$OUT/tools" --metrics "http://$HN:$PORT/metrics" --temperature "$1" --streams "$2" 2>&1 | tee -a "$LOG"
          done
          finish_serve "tools-$a"
        fi
      done
      ;;
  esac
done
