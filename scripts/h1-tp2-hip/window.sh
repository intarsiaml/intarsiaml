#!/usr/bin/env bash
# Lane H1: ONE radiance-down window on tatooine, from the Mac (after G2b's window.sh). Stops radiance, runs
# scripts/h1-tp2-hip/stages-<W>.sh ON tatooine (timeout CAP s), restores radiance in an EXIT trap on tatooine (so it runs
# even if the Mac side dies) and waits until it answers, then fetches ~/h1/out/<W>/ into
# docs/results/2026-10-21-h1-tp2-hip/raw/<W>/ (dumps excluded) and appends the window to raw/downtime.log.
#   fleetq run --host tatooine --exclusive --takes-down radiance --mem 30 --label "h1 <W>" -- env W=w1 scripts/h1-tp2-hip/window.sh
# Lane cap (lane-rules.md): at most 20 min of radiance downtime per window, so CAP (stages) + restore (<= 180 s) <= 20 min.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}" "${W:?W=<window name>}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
RAW=$REPO/docs/results/2026-10-21-h1-tp2-hip/raw
mkdir -p "$RAW/$W"
CAP=${CAP:-960}
ssh -o BatchMode=yes tatooine 'mkdir -p ~/h1' < /dev/null
scp -q "$S"/lib.sh "$S"/stages-*.sh tatooine:h1/
ssh -o BatchMode=yes tatooine "FLEETQ_JOB=$FLEETQ_JOB W=$W CAP=$CAP bash -s" <<'EOS' 2>&1 | tee "$RAW/$W/window.log"
set -uo pipefail
O=$HOME/h1/out/$W; rm -rf "$O"; mkdir -p "$O"
now() { date '+%F %T'; }
restore() {
  docker ps -q --filter name=h1- | xargs -r docker rm -f >/dev/null 2>&1
  echo "[window] starting radiance $(now)"
  systemctl --user start radiance < /dev/null
  for i in $(seq 1 36); do
    if curl -s -m 3 http://localhost:8080/v1/models < /dev/null | grep -q '"data":\['; then echo "[window] radiance answers $(now) (after $((i * 5)) s)"; return 0; fi
    sleep 5
  done
  echo "[window] RADIANCE DID NOT ANSWER within 180 s $(now)"; return 1
}
trap restore EXIT
echo "[window] job $FLEETQ_JOB $W; stopping radiance $(now)"
systemctl --user stop radiance < /dev/null
sleep 2
echo "[window] radiance stopped $(now)"
timeout "$CAP" bash $HOME/h1/stages-$W.sh "$O" < /dev/null
echo "[window] stages done $(now) (exit $?)"
EOS
rc=${PIPESTATUS[0]}
rsync -a --exclude '*.safetensors' "tatooine:h1/out/$W/" "$RAW/$W/" 2>/dev/null
# the accuracy dumps (not committed) for the offline KL tables
DUMPS=${DUMPS:-/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/h1/dumps}
mkdir -p "$DUMPS/$W" && rsync -a --include '*.safetensors' --exclude '*' "tatooine:h1/out/$W/" "$DUMPS/$W/" 2>/dev/null
start=$(grep -m1 'radiance stopped' "$RAW/$W/window.log" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}')
back=$(grep -m1 'radiance answers' "$RAW/$W/window.log" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
echo "$(date +%F) fleetq #$FLEETQ_JOB $W radiance stopped $start answers ${back:-NEVER} (window.log in raw/$W)" >> "$RAW/downtime.log"
exit $rc
