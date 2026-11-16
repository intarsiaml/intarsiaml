#!/usr/bin/env bash
# flipprep coupling gate (coordinator, lane mx6): with the flip's recipe on top of mx6's plan-probe fix, the batched verify pass REALLY runs (equal ids prove nothing):
#   load line `batched verify: the plan covers passes of up to 2 sequences (6 rows)`, no `batched verify off`, `intarsia_batch_verify_passes_total{seqs="2"}` > 0 in /metrics.
# Arms: C = flip + mx6 (defaults), CO = C with INTARSIA_BATCHED_VERIFY=0 (the revert switch: no batched pass, the log says so, the counter stays 0), R = --spec off reference.
# Also ids.py (B = 1, 2, greedy + T = 1.0) against R. Run under:
#   fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 30 --label "flipprep: proof the batched verify pass runs (flip + mx6)" -- scripts/flipprep/proof.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
OUT=${OUT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/docs/results/2026-11-15-flipprep/raw/proof}; mkdir -p "$OUT"
LOGNAME_=proof; source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"; source "$S/arms.sh"
for a in ${ARMS:-C CO R}; do
  arm_setup $a; [[ $a == R ]] && BIN=$B/intarsia-comb
  if start_serve "proof-$a" "$ENVS" --max-batch 2 $ARGS; then
    python3 "$S/ids.py" "$URL" guidance "$OUT/ids-$a.json" "http://$HN:$PORT/metrics" 2>&1 | tee -a "$LOG"
    curl -s "http://$HN:$PORT/metrics" | grep -a "batch_verify" | tee -a "$LOG" "$OUT/metrics-verify-$a.txt"
    log "$a: plan line: $(rsh "grep -a 'plan covers' ~/intarsia-flipprep/out/serve-proof-$a.log | cut -c1-200")"
    log "$a: 'batched verify off' lines (must be 0 for C): $(rsh "grep -ac 'batched verify off' ~/intarsia-flipprep/out/serve-proof-$a.log")"
    finish_serve "proof-$a"
  fi
done
python3 "$S/ids_compare.py" "$OUT" R $(for a in ${ARMS:-C CO R}; do [ "$a" != R ] && echo "$a"; done) 2>&1 | tee -a "$LOG" "$OUT/ids-compare.txt"
