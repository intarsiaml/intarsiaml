#!/usr/bin/env bash
# Lane mx: ONE radiance-down window on tatooine, from the Mac (after T2's window.sh). Stops radiance, runs
# scripts/mx/stages-<W>.sh ON tatooine (timeout CAP s), restores radiance in an EXIT trap on tatooine and waits until it
# answers, then fetches ~/mx/out/<W>/ into docs/results/2026-10-25-mx-hip-27b-prefill/raw/<W>/ (csv traces gzipped).
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu --expect 40 --label "mx <W>" -- \
#     env W=w1 scripts/mx/window.sh
# Lane cap: 45 min per window, so CAP (stages, default 2520 s) + restore (<= 180 s) stays under it.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}" "${W:?W=<window name>}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
RAW=$REPO/docs/results/2026-10-25-mx-hip-27b-prefill/raw
mkdir -p "$RAW/$W"
CAP=${CAP:-2520}
ssh -o BatchMode=yes tatooine 'mkdir -p ~/mx' < /dev/null
shopt -s nullglob
scp -q "$S"/lib.sh "$S"/build.sh "$S"/stages-*.sh "$S"/*.py "$S"/*.hip tatooine:mx/ < /dev/null
ssh -o BatchMode=yes tatooine "FLEETQ_JOB=$FLEETQ_JOB W=$W CAP=$CAP bash -s" <<'EOS' 2>&1 | tee "$RAW/$W/window.log"
set -uo pipefail
O=$HOME/mx/out/$W; rm -rf "$O"; mkdir -p "$O"
now() { date '+%F %T'; }
restore() {
  docker ps -q --filter ancestor=intarsia-hip-dev:rocm7.2.4-rust1.90.0 | xargs -r docker rm -f >/dev/null 2>&1
  docker ps -q --filter ancestor=intarsia-hip-prof:rocm7.2.4-rust1.90.0 | xargs -r docker rm -f >/dev/null 2>&1
  echo "[window] starting radiance $(now)"
  systemctl --user start radiance < /dev/null
  for i in $(seq 1 40); do
    if curl -s -m 3 http://localhost:8080/v1/models < /dev/null | grep -q '"data":\['; then echo "[window] radiance answers $(now) (after $((i * 5)) s); $(systemctl --user is-active radiance)"; return 0; fi
    sleep 5
  done
  echo "[window] RADIANCE DID NOT ANSWER within 200 s $(now); $(systemctl --user is-active radiance)"; return 1
}
trap restore EXIT
echo "[window] job $FLEETQ_JOB $W; stopping radiance $(now)"
systemctl --user stop radiance < /dev/null
sleep 3
echo "[window] radiance stopped $(now)"
BUDGET=$((CAP - 60)) timeout "$CAP" bash $HOME/mx/stages-$W.sh "$O" < /dev/null
rc=$?; echo "[window] stages done $(now), exit $rc"
EOS
rsync -a --exclude '*.safetensors' tatooine:mx/out/$W/ "$RAW/$W/" < /dev/null
gzip -f "$RAW/$W"/*.csv 2>/dev/null
echo "$(date '+%F %T') job $FLEETQ_JOB window $W: $(grep -E 'stopping radiance|radiance answers|DID NOT' "$RAW/$W/window.log" | tr '\n' ' ')" >> "$RAW/downtime.log"
