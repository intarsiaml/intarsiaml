#!/usr/bin/env bash
# Lane PF27, FROM THE MAC (worktree root) as the command of one fleetq GPU job: refresh scripts/pf27 on tatooine, check the binaries,
# run w1.sh there (it stops radiance and restores it, waiting until :8080 answers, before it exits), copy the logs back.
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance-rank0,radiance-rank1 --quiet gpu,cpu --mem 30 --expect 70 \
#     --label "pf27 W1 ..." -- env W=w1 SEC="P T ..." scripts/pf27/win.sh
set -uo pipefail
: "${W:?W=<window name>}"
SSH="ssh -o BatchMode=yes tatooine"
OUT=docs/results/2026-10-28-pf27-hip-27b-prefill/raw/$W; mkdir -p "$OUT"
scp -q scripts/pf27/*.sh scripts/pf27/*.py tatooine:intarsiaml-pf27/scripts/pf27/
for f in pf27-base-cli pf27-dev-cli pf27-bin/mxfp4_a8-base pf27-bin/mxfp4_a8-dev; do
  $SSH "test -x ~/intarsiaml-pf27/$f" < /dev/null || { echo "MISSING $f: not starting the window"; exit 3; }
done
$SSH "O=~/intarsiaml-pf27/pf27-out/$W SEC='${SEC:-}' ${WENV:-} bash ~/intarsiaml-pf27/scripts/pf27/w1.sh" < /dev/null 2>&1 | tee "$OUT/w.log"
rc=${PIPESTATUS[0]}
scp -q -r "tatooine:intarsiaml-pf27/pf27-out/$W/." "$OUT/" < /dev/null
$SSH "curl -s -m 5 http://localhost:8080/v1/models" < /dev/null | head -c 120; echo
exit $rc
