#!/usr/bin/env bash
# Lane vat (copy of scripts/vat/window.sh): ONE radiance-down window on tatooine, from the Mac (after G2b's window.sh). Stops radiance, runs
# scripts/vat/stages-<W>.sh ON tatooine (timeout CAP s), restores radiance in an EXIT trap on tatooine (so it runs
# even if the Mac side dies) and waits until it answers, then fetches ~/intarsiaml-vat/.vat/out/<W>/ into
# docs/results/2026-10-28-vat-verify-attn-rows/raw/<W>/ (dumps excluded) and appends the window to raw/downtime.log.
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu --expect 40 --mem 30 --label "vat <W>" -- env W=w1 scripts/vat/window.sh
#   (--quiet gpu,cpu: a cpu-load build cannot share the box with a window; XENV="D=batch SNAP=batch,mut-x" runs from per-window snapshots of the trees)
#   (both cards, ONE job; the radiance unit serves TP=2 over both)
# XENV="D=stack B=base" passes extra variables to the stages script.
# Lane cap (as lane tp2): at most 45 min of radiance downtime per window, so CAP (stages) + restore (<= 240 s) <= 45 min.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}" "${W:?W=<window name>}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
RAW=$REPO/docs/results/2026-10-28-vat-verify-attn-rows/raw
mkdir -p "$RAW/$W"
CAP=${CAP:-2400}
ssh -o BatchMode=yes tatooine 'mkdir -p ~/intarsiaml-vat/.vat' < /dev/null
scp -q "$S"/lib.sh "$S"/stages-*.sh tatooine:intarsiaml-vat/.vat/
ssh -o BatchMode=yes tatooine "FLEETQ_JOB=$FLEETQ_JOB W=$W CAP=$CAP ${XENV:-} bash -s" <<'EOS' 2>&1 | tee "$RAW/$W/window.log"
set -uo pipefail
O=$HOME/intarsiaml-vat/.vat/out/$W; rm -rf "$O"; mkdir -p "$O"
now() { date '+%F %T'; }
restore() {
  docker ps -q --filter name=vat- | xargs -r docker rm -f >/dev/null 2>&1
  echo "[window] starting radiance $(now)"
  systemctl --user start radiance < /dev/null
  for i in $(seq 1 48); do
    if curl -s -m 3 http://localhost:8080/v1/models < /dev/null | grep -q '"data":\['; then echo "[window] radiance answers $(now) (after $((i * 5)) s)"; return 0; fi
    sleep 5
  done
  echo "[window] RADIANCE DID NOT ANSWER within 240 s $(now)"; return 1
}
# A window runs from its own snapshot of the trees (SNAP=batch,mut-batch-peer-no-table in XENV): no other job (a sync, a rebuild) can write to
# ~/intarsiaml-vat-run-<W>-<tree> (w13 lost its tree to a sync). lib.sh's T() reads RUN. Removed when the window ends.
if [ -n "${SNAP:-}" ]; then
  for t in ${SNAP//,/ }; do
    rm -rf "$HOME/intarsiaml-vat"
    cp -a "$HOME/intarsiaml-vat-$t" "$HOME/intarsiaml-vat" || { echo "[window] SNAPSHOT OF $t FAILED"; exit 3; }
  done
  export RUN=$W
  echo "[window] running from snapshots $HOME/intarsiaml-vat-run-$W-{${SNAP}} $(now)"
fi
trap 'rm -rf "$HOME"/intarsiaml-vat-run-"$W"-*; restore' EXIT
echo "[window] job $FLEETQ_JOB $W; stopping radiance $(now)"
systemctl --user stop radiance < /dev/null
sleep 2
echo "[window] radiance stopped $(now)"
timeout "$CAP" bash $HOME/intarsiaml-vat/.vat/stages-${STAGES:-$W}.sh "$O" < /dev/null
echo "[window] stages done $(now) (exit $?)"
EOS
rc=${PIPESTATUS[0]}
rsync -a --exclude '*.safetensors' "tatooine:intarsiaml-vat/.vat/out/$W/" "$RAW/$W/" 2>/dev/null
start=$(grep -m1 'radiance stopped' "$RAW/$W/window.log" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}')
back=$(grep -m1 'radiance answers' "$RAW/$W/window.log" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
echo "$(date +%F) fleetq #$FLEETQ_JOB $W radiance stopped $start answers ${back:-NEVER} (window.log in raw/$W)" >> "$RAW/downtime.log"
exit $rc
