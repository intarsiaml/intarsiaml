#!/usr/bin/env bash
# Lane wht6-8b (ON tatooine, under nohup inside a fleetq lease that takes radiance down): stop radiance, run one window
# script, and ALWAYS start radiance again when it ends or dies; prints when radiance answers.
#   FLEETQ_JOB=<id> O=<out dir> nohup scripts/wht6-8b/window.sh kernels-window.sh > log 2>&1 &
: "${FLEETQ_JOB:?run inside a fleetq lease}"
restore() {
  docker rm -f wht6-8b-serve >/dev/null 2>&1
  docker ps -q --filter volume=intarsia-cargo-target-wht6-8b | xargs -r docker kill >/dev/null 2>&1
  echo "[window] starting radiance $(date '+%F %T')"
  systemctl --user start radiance < /dev/null
  for i in $(seq 1 48); do
    if curl -s -m 3 http://localhost:8080/v1/models < /dev/null | grep -q '"data":\['; then echo "[window] radiance answers $(date '+%F %T') (after $((i * 5)) s)"; return 0; fi
    sleep 5
  done
  echo "[window] RADIANCE DID NOT ANSWER within 240 s $(date '+%F %T')"
}
trap restore EXIT
echo "[window] stopping radiance $(date '+%F %T')"
systemctl --user stop radiance < /dev/null
sleep 3
s=$1; shift
timeout "${CAP:-2400}" "$(dirname "$0")/$s" "$@"
echo "[window] $s exit $? $(date '+%F %T')"
