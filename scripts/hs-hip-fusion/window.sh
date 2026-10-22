#!/usr/bin/env bash
# Lane HS: ONE radiance-down window on tatooine, from the Mac (after H1's window.sh). Ships this lane's scripts and tools to
# tatooine:~/hs, stops radiance, runs scripts/hs-hip-fusion/stages-<W>.sh ON tatooine (timeout CAP s), restores radiance in an
# EXIT trap on tatooine (so it runs even if the Mac side dies) and waits until it answers, then fetches ~/hs/out/<W>/ into
# docs/results/2026-10-21-hs-hip-fusion-soak/raw/<W>/ and appends the window to raw/downtime.log.
#   fleetq run --host tatooine --exclusive --takes-down radiance --mem 30 --label "hs <W>" -- env W=soak scripts/hs-hip-fusion/window.sh
# Lane cap (lane-rules.md): at most 20 min of radiance downtime per window, so CAP (stages) + restore (<= 180 s) <= 20 min.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}" "${W:?W=<window name>}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
RAW=$REPO/docs/results/2026-10-21-hs-hip-fusion-soak/raw
mkdir -p "$RAW/$W"
CAP=${CAP:-1030}
ssh -o BatchMode=yes tatooine 'mkdir -p ~/hs/ref ~/hs/tools' < /dev/null
scp -q "$S"/lib.sh "$S"/stages-*.sh tatooine:hs/
scp -q "$REPO"/tools/hs_soak.py "$REPO"/tools/p2prime_soak.py "$REPO"/tools/cbidentity.py "$REPO"/tools/cbsoak.py tatooine:hs/tools/
scp -q -r "$REPO"/tools/prompts tatooine:hs/tools/
scp -q "$REPO"/scripts/hip-decode/prompt4k.txt tatooine:hs/prompt4k.txt
# H1's window-7 reference outputs (committed on the PR #59 head): fusion-on / graphs-only tp=2 ids, tp=1 ids and plan
W7="$REPO"/docs/results/2026-10-21-h1-tp2-hip/raw/w7
scp -q "$W7"/gen-head-35b-tp2gf-short.ids "$W7"/gen-head-35b-tp2gf-4k.ids "$W7"/gen-head-35b-tp2g-short.ids "$W7"/gen-head-35b-tp2g-4k.ids \
  "$W7"/gen-head-8b-tp2gf-short.ids "$W7"/gen-head-8b-tp2g-short.ids "$W7"/gen-head-35b-short.ids "$W7"/plan-head-35b.out tatooine:hs/ref/
ssh -o BatchMode=yes tatooine "FLEETQ_JOB=$FLEETQ_JOB W=$W CAP=$CAP bash -s" <<'EOS' 2>&1 | tee "$RAW/$W/window.log"
set -uo pipefail
O=$HOME/hs/out/$W; rm -rf "$O"; mkdir -p "$O"
now() { date '+%F %T'; }
restore() {
  docker ps -aq --filter name=hs- | xargs -r docker rm -f >/dev/null 2>&1
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
timeout "$CAP" bash $HOME/hs/stages-$W.sh "$O" < /dev/null
echo "[window] stages done $(now) (exit $?)"
EOS
rc=${PIPESTATUS[0]}
rsync -a --exclude '*.safetensors' "tatooine:hs/out/$W/" "$RAW/$W/" 2>/dev/null
start=$(grep -m1 'radiance stopped' "$RAW/$W/window.log" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}')
back=$(grep -m1 'radiance answers' "$RAW/$W/window.log" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
echo "$(date +%F) fleetq #$FLEETQ_JOB $W radiance stopped $start answers ${back:-NEVER} (window.log in raw/$W)" >> "$RAW/downtime.log"
exit $rc
