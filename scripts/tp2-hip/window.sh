#!/usr/bin/env bash
# Lane tp2: ONE radiance-down window on tatooine, from the Mac (after G2b's window.sh). Stops radiance, runs
# scripts/tp2-hip/stages-<W>.sh ON tatooine (timeout CAP s), restores radiance in an EXIT trap on tatooine (so it runs
# even if the Mac side dies) and waits until it answers, then fetches ~/tp2/out/<W>/ into
# docs/results/2026-10-25-tp2-hip/raw/<W>/ (dumps excluded) and appends the window to raw/downtime.log.
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --expect 40 --mem 60 --label "tp2 <W>" -- env W=w1 scripts/tp2-hip/window.sh
#   (both cards, ONE job; the radiance unit serves TP=2 over both)
# XENV="D=stack B=base" passes extra variables to the stages script.
# Lane cap (lane tp2 brief, 2026-10-25): at most 45 min of radiance downtime per window, so CAP (stages) + restore (<= 240 s) <= 45 min.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}" "${W:?W=<window name>}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
RAW=$REPO/docs/results/2026-10-25-tp2-hip/raw
mkdir -p "$RAW/$W"
CAP=${CAP:-2400}
ssh -o BatchMode=yes tatooine 'mkdir -p ~/tp2' < /dev/null
scp -q "$S"/lib.sh "$S"/stages-*.sh tatooine:tp2/
ssh -o BatchMode=yes tatooine "FLEETQ_JOB=$FLEETQ_JOB W=$W CAP=$CAP ${XENV:-} bash -s" <<'EOS' 2>&1 | tee "$RAW/$W/window.log"
set -uo pipefail
O=$HOME/tp2/out/$W; rm -rf "$O"; mkdir -p "$O"
now() { date '+%F %T'; }
restore() {
  docker ps -q --filter name=tp2- | xargs -r docker rm -f >/dev/null 2>&1
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
timeout "$CAP" bash $HOME/tp2/stages-$W.sh "$O" < /dev/null
echo "[window] stages done $(now) (exit $?)"
EOS
rc=${PIPESTATUS[0]}
rsync -a --exclude '*.safetensors' "tatooine:tp2/out/$W/" "$RAW/$W/" 2>/dev/null
# the accuracy dumps (not committed) for the offline KL tables
DUMPS=${DUMPS:-/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/tp2/dumps}
mkdir -p "$DUMPS/$W" && rsync -a --include '*.safetensors' --exclude '*' "tatooine:tp2/out/$W/" "$DUMPS/$W/" 2>/dev/null
start=$(grep -m1 'radiance stopped' "$RAW/$W/window.log" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}')
back=$(grep -m1 'radiance answers' "$RAW/$W/window.log" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
echo "$(date +%F) fleetq #$FLEETQ_JOB $W radiance stopped $start answers ${back:-NEVER} (window.log in raw/$W)" >> "$RAW/downtime.log"
exit $rc
