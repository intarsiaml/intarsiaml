#!/usr/bin/env bash
# Lane det (after scripts/vat/window.sh): ONE radiance-down window on tatooine, from the Mac. Stops radiance, runs
# scripts/det/stages-<W>.sh ON tatooine (timeout CAP s), restores radiance in an EXIT trap on tatooine (so it runs even if the Mac
# side dies) and waits until it answers, then fetches ~/intarsiaml/.worktrees/det/.det/out/<W>/ into
# docs/results/2026-10-28-det-tp2-nondeterminism/raw/<W>/ and appends the window to raw/downtime.log.
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu --expect 40 --mem 30 --label "det <W>" -- env W=w1 scripts/det/window.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}" "${W:?W=<window name>}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
RAW=$REPO/docs/results/2026-10-28-det-tp2-nondeterminism/raw
mkdir -p "$RAW/$W"
CAP=${CAP:-2400}
ssh -o BatchMode=yes tatooine 'mkdir -p ~/intarsiaml/.worktrees/det/.det' < /dev/null
scp -q "$S"/lib.sh "$S"/stages-*.sh tatooine:intarsiaml/.worktrees/det/.det/
ssh -o BatchMode=yes tatooine "FLEETQ_JOB=$FLEETQ_JOB W=$W CAP=$CAP ${XENV:-} bash -s" <<'EOS' 2>&1 | tee "$RAW/$W/window.log"
set -uo pipefail
T=$HOME/intarsiaml/.worktrees/det
O=$T/.det/out/$W; rm -rf "$O"; mkdir -p "$O"
now() { date '+%F %T'; }
restore() {
  echo "[window] starting radiance $(now)"
  systemctl --user start radiance < /dev/null
  for i in $(seq 1 48); do
    if curl -s -m 3 http://localhost:8080/v1/models < /dev/null | grep -q '"data":\['; then echo "[window] radiance answers $(now) (after $((i * 5)) s)"; return 0; fi
    sleep 5
  done
  echo "[window] RADIANCE DID NOT ANSWER within 240 s $(now)"; return 1
}
# no binary from the build that ran before this window (it failed or has not run): no downtime
[[ -x $T/det-intarsia && $T/det-intarsia -nt $T/Cargo.toml ]] || { echo "[window] NO FRESH $T/det-intarsia: radiance left up"; exit 3; }
trap restore EXIT
echo "[window] job $FLEETQ_JOB $W; stopping radiance $(now)"
systemctl --user stop radiance < /dev/null
sleep 2
echo "[window] radiance stopped $(now)"
timeout "$CAP" bash "$T/.det/stages-${STAGES:-$W}.sh" "$O" < /dev/null
echo "[window] stages done $(now) (exit $?)"
EOS
rc=${PIPESTATUS[0]}
rsync -a "tatooine:intarsiaml/.worktrees/det/.det/out/$W/" "$RAW/$W/" 2>/dev/null
start=$(grep -m1 'radiance stopped' "$RAW/$W/window.log" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}')
back=$(grep -m1 'radiance answers' "$RAW/$W/window.log" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
echo "$(date +%F) fleetq #$FLEETQ_JOB $W radiance stopped $start answers ${back:-NEVER} (window.log in raw/$W)" >> "$RAW/downtime.log"
exit $rc
