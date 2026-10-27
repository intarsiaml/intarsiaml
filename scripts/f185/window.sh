#!/usr/bin/env bash
# Lane f185 (from the Mac; adapted from scripts/dfh/window.sh): ONE radiance-down window on tatooine running stages-ab.sh, radiance restored in an EXIT trap
# on tatooine and awaited at /v1/models. Output fetched to docs/results/2026-10-27-mpf-macro-prefill-mtp/raw/f185/<W>/.
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu --expect 35 --mem 30 --label "f185: HIP checks" -- env W=ab scripts/f185/window.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}" "${W:?W=<window name>}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$S/../.." && pwd)"
RAW=$REPO/docs/results/2026-10-27-mpf-macro-prefill-mtp/raw/f185; mkdir -p "$RAW/$W"
CAP=${CAP:-1500}
ssh -o BatchMode=yes tatooine 'mkdir -p ~/intarsiaml-f185/stages' < /dev/null
scp -q "$S"/stages-*.sh tatooine:intarsiaml-f185/stages/
ssh -o BatchMode=yes tatooine "FLEETQ_JOB=$FLEETQ_JOB W=$W CAP=$CAP STAGES_SEL='${STAGES_SEL:-a b}' bash -s" <<'EOS' 2>&1 | tee "$RAW/$W/window.log"
set -uo pipefail
O=$HOME/intarsiaml-f185/out/$W; rm -rf "$O"; mkdir -p "$O"
now() { date '+%F %T'; }
restore() {
  docker ps -q --filter ancestor=intarsia-hip-dev:rocm7.2.4-rust1.90.0 | xargs -r docker rm -f > /dev/null 2>&1
  echo "[window] starting radiance $(now)"
  systemctl --user start radiance < /dev/null
  for i in $(seq 1 48); do
    if curl -s -m 3 http://localhost:8080/v1/models < /dev/null | grep -q '"data":\['; then echo "[window] radiance answers $(now) (after $((i * 5)) s)"; return 0; fi
    sleep 5
  done
  echo "[window] RADIANCE DID NOT ANSWER within 240 s $(now)"; return 1
}
trap restore EXIT
echo "[window] job $FLEETQ_JOB $W; stopping radiance $(now)"
systemctl --user stop radiance < /dev/null
sleep 2
echo "[window] radiance stopped $(now)"
timeout "$CAP" bash $HOME/intarsiaml-f185/stages/stages-ab.sh "$O" < /dev/null
echo "[window] stages done $(now) (exit $?)"
EOS
rc=${PIPESTATUS[0]}
rsync -a "tatooine:intarsiaml-f185/out/$W/" "$RAW/$W/" 2>/dev/null
echo "$(date +%F) fleetq #$FLEETQ_JOB $W: $(grep -m1 'radiance stopped' "$RAW/$W/window.log") / $(grep -m1 'radiance answers' "$RAW/$W/window.log")" >> "$RAW/downtime.log"
exit $rc
