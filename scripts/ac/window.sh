#!/usr/bin/env bash
# Lane ac: ONE radiance-down window on tatooine, from the Mac (after scripts/35b-tp2-hip/window.sh). Stops radiance, runs
# scripts/ac/stages.sh ON tatooine (timeout CAP s), restores radiance in an EXIT trap (on tatooine, so it runs even if the Mac
# side dies), then fetches ~/intarsiaml-ac/ac-out + the stage logs into docs/results/2026-10-25-ac-27b-accuracy-gate/raw/.
#   fleetq run --host tatooine-r9700 --takes-down radiance --mem 30 --expect 15 --label "ac: ..." -- env PC=5241377 scripts/ac/window.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}" "${PC:?PC=<head sha7>}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
RAW=$REPO/docs/results/2026-10-25-ac-27b-accuracy-gate/raw
mkdir -p "$RAW"
CAP=${CAP:-900}
scp -q "$S/stages.sh" tatooine:ac-logs/stages.sh
ssh -o BatchMode=yes tatooine "FLEETQ_JOB=$FLEETQ_JOB PC=$PC CAP=$CAP bash -s" <<'EOS' 2>&1 | tee "$RAW/window.log"
set -uo pipefail
O=$HOME/ac-logs/out; rm -rf "$O"; mkdir -p "$O"
now() { date '+%F %T'; }
restore() {
  for c in $(docker ps --no-trunc --filter ancestor=intarsia-hip-dev:rocm7.2.4-rust1.90.0 --format '{{.ID}} {{.Command}}' | grep '/work/ac-intarsia' | cut -d' ' -f1); do docker rm -f "$c" >/dev/null 2>&1; done
  echo "[window] starting radiance $(now)"
  systemctl --user start radiance
  for i in $(seq 1 36); do
    if curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\['; then echo "[window] radiance answers $(now) (after $((i * 5)) s)"; return 0; fi
    sleep 5
  done
  echo "[window] RADIANCE DID NOT ANSWER within 180 s $(now)"; return 1
}
trap restore EXIT
echo "[window] job $FLEETQ_JOB; stopping radiance $(now)"
systemctl --user stop radiance
sleep 2
echo "[window] radiance stopped $(now)"
timeout "$CAP" bash $HOME/ac-logs/stages.sh "$O"
echo "[window] stages done $(now) (exit $?)"
EOS
rc=${PIPESTATUS[0]}
scp -q -r "tatooine:ac-logs/out/*" "$RAW/" 2>/dev/null
scp -q "tatooine:intarsiaml-ac/ac-out/*" "$RAW/" 2>/dev/null
exit $rc
