#!/usr/bin/env bash
# Lane rocm102-bisect (ON tatooine, nohup, inside a fleetq lease that takes radiance down): stop radiance, run stages.sh, and
# ALWAYS kill this lane's containers and start radiance again when it ends or dies (scripts/rocm102/window.sh's shape).
: "${FLEETQ_JOB:?run inside a fleetq lease}"
. "$(dirname "$0")/lib.sh"
restore() {
  killall_lane
  echo "[window] starting radiance $(date '+%F %T')"
  systemctl --user start radiance < /dev/null
  for i in $(seq 1 48); do
    if curl -s -m 3 http://localhost:8080/v1/models < /dev/null | grep -q '"data":\['; then echo "[window] radiance answers $(date '+%F %T') (after $((i * 5)) s)"; return 0; fi
    sleep 5
  done
  echo "[window] RADIANCE DID NOT ANSWER within 240 s $(date '+%F %T')"
}
trap restore EXIT
echo "[window] stopping radiance $(date '+%F %T'); logs $O"
systemctl --user stop radiance < /dev/null
sleep 3
O=$O timeout "${CAP:-2400}" "$(dirname "$0")/stages.sh"
echo "[window] stages exit $? $(date '+%F %T')"
