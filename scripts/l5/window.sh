#!/usr/bin/env bash
# Lane l5: ONE radiance-down window on tatooine card 1 (tatooine-r9700-2), started from the Mac inside a fleetq slot. Stops radiance, runs the
# stage script ON tatooine from its own snapshot dir (timeout CAP s), restores radiance in an EXIT trap that runs on tatooine (so it also runs
# if the Mac side dies), then fetches the logs into docs/results/2026-10-25-l5-moe-fp8-wmma/raw/<WIN>/.
#   NAME=<snapshot> WIN=<tag> STAGES=scripts/l5/stages-bench.sh [CAP=900] \
#   FLEETQ_OWNER=claude-f31ccf91-l5 fleetq run --host tatooine-r9700-2 --takes-down radiance-rank1 --quiet gpu,cpu --mem 4 --expect 12 \
#       --label "l5: ..." -- scripts/l5/window.sh
# Every command inside the window runs with HIP_VISIBLE_DEVICES=1 (card 1); the stage script is given the snapshot dir, the output dir and the card.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}" "${NAME:?NAME=<snapshot>}" "${WIN:?WIN=<tag>}" "${STAGES:?STAGES=<script>}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
RAW=$REPO/docs/results/2026-10-25-l5-moe-fp8-wmma/raw/$WIN
mkdir -p "$RAW"
CAP=${CAP:-900}
ssh -o BatchMode=yes tatooine 'mkdir -p ~/intarsiaml-l5-logs'
scp -q "$REPO/$STAGES" "tatooine:intarsiaml-l5-logs/stages-$WIN.sh"
ssh -o BatchMode=yes tatooine "FLEETQ_JOB=$FLEETQ_JOB NAME=$NAME WIN=$WIN CAP=$CAP EXTRA='${EXTRA:-}' bash -s" <<'EOS' 2>&1 | tee "$RAW/window.log"
set -uo pipefail
O=$HOME/intarsiaml-l5-logs/$WIN; rm -rf "$O"; mkdir -p "$O"
now() { date '+%F %T'; }
restore() {
  echo "[window] starting radiance $(now)"
  systemctl --user start radiance
  for i in $(seq 1 36); do
    if curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\['; then echo "[window] radiance answers $(now) (after $((i * 5)) s)"; return 0; fi
    sleep 5
  done
  echo "[window] RADIANCE DID NOT ANSWER within 180 s $(now)"; return 1
}
trap restore EXIT
echo "[window] job $FLEETQ_JOB win $WIN snapshot $NAME; stopping radiance $(now)"
systemctl --user stop radiance
sleep 3
echo "[window] radiance stopped $(now)"
timeout "$CAP" env NAME="$NAME" O="$O" G=1 EXTRA="$EXTRA" bash $HOME/intarsiaml-l5-logs/stages-$WIN.sh
echo "[window] stages done $(now) (exit $?)"
EOS
rc=${PIPESTATUS[0]}
scp -q -r "tatooine:intarsiaml-l5-logs/$WIN/*" "$RAW/" 2>/dev/null
exit $rc
