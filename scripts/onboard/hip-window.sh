#!/usr/bin/env bash
# Lane `on`: ONE radiance-down window on tatooine, from the Mac. Runs (or resumes) `intarsia onboard` on one R9700 in the HIP
# dev image under `timeout CAP`, restores radiance in an EXIT trap on tatooine, fetches the run directory into
# $RAW/<RUN>/ and appends the window to $RAW/downtime.log. Resumable: run it again with the same RUN until onboard exits 0/1.
#   fleetq run --host tatooine --exclusive --takes-down radiance --mem 30 --label "on: HIP onboard window" -- env RUN=35b-hip scripts/onboard/hip-window.sh
# Env: RUN (out dir name), CAP (seconds of onboard, default 1000; restore adds <= 180 s, so the window stays under 20 min),
#      MODEL (default /models/Qwen3.6-35B-A3B-NVFP4), ONBOARD_ARGS (extra onboard args, before the `--`), QUALIFY_ARGS.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}" "${RUN:?RUN=<name>}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
RAW=${RAW:-$REPO/docs/results/2026-10-22-on-onboard/raw/hip}
mkdir -p "$RAW/$RUN"
CAP=${CAP:-1000}
MODEL=${MODEL:-/models/Qwen3.6-35B-A3B-NVFP4}
ssh -o BatchMode=yes tatooine "FLEETQ_JOB=$FLEETQ_JOB RUN=$RUN CAP=$CAP MODEL=$MODEL ONBOARD_ARGS='${ONBOARD_ARGS:-}' QUALIFY_ARGS='${QUALIFY_ARGS:-}' bash -s" <<'EOS' 2>&1 | tee -a "$RAW/$RUN/window.log"
set -uo pipefail
now() { date '+%F %T'; }
restore() {
  docker ps -q --filter ancestor=intarsia-hip-dev:rocm7.2.4-rust1.90.0 | xargs -r docker rm -f >/dev/null 2>&1
  echo "[window] starting radiance $(now)"
  systemctl --user start radiance
  for i in $(seq 1 36); do
    if curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\['; then echo "[window] radiance answers $(now) (after $((i * 5)) s)"; return 0; fi
    sleep 5
  done
  echo "[window] RADIANCE DID NOT ANSWER within 180 s $(now)"; return 1
}
trap restore EXIT
echo "[window] job $FLEETQ_JOB $RUN; stopping radiance $(now)"
systemctl --user stop radiance
sleep 2
echo "[window] radiance stopped $(now)"
export INTARSIA_REPO=$HOME/intarsiaml-on INTARSIA_TARGET_VOL=intarsia-cargo-target-on
mkdir -p $INTARSIA_REPO/onboard-runs
O=/work/onboard-runs/$RUN
echo "[window] onboard $MODEL -> $O (cap ${CAP}s) $(now)"
HIP_VISIBLE_DEVICES=0 timeout "$CAP" $INTARSIA_REPO/deploy/tatooine/dev.sh /work/on-intarsia onboard --model "$MODEL" --backend hip \
  --mem-budget 29 --out "$O/report" $ONBOARD_ARGS -- --no-unim $QUALIFY_ARGS < /dev/null > $INTARSIA_REPO/onboard-runs/$RUN.driver.log 2>&1
rc=$?
echo "[window] onboard exit $rc $(now)"
tail -25 $INTARSIA_REPO/onboard-runs/$RUN.driver.log
exit $rc
EOS
rc=${PIPESTATUS[0]}
mkdir -p "$RAW/$RUN"
rsync -a "tatooine:intarsiaml-on/onboard-runs/$RUN/" "$RAW/$RUN/" 2>/dev/null
rsync -a "tatooine:intarsiaml-on/onboard-runs/$RUN.driver.log" "$RAW/$RUN.driver.log" 2>/dev/null
start=$(grep 'radiance stopped' "$RAW/$RUN/window.log" | tail -1 | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}')
back=$(grep 'radiance answers' "$RAW/$RUN/window.log" | tail -1 | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
echo "$(date +%F) fleetq #$FLEETQ_JOB $RUN radiance stopped $start answers ${back:-NEVER} (window.log in raw/hip/$RUN)" >> "$RAW/downtime.log"
exit $rc
