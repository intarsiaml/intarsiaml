#!/usr/bin/env bash
# Lane gdn-default (ON tatooine, nohup, inside a fleetq lease that takes radiance down): stop radiance, run stages.sh, and ALWAYS
# start radiance again when it ends or dies.   FLEETQ_JOB=<id> nohup scripts/gdn-default/window.sh > log 2>&1 &
: "${FLEETQ_JOB:?run inside a fleetq lease}"
restore() {
  docker ps -q --filter ancestor=intarsia-hip-dev:rocm7.2.4-rust1.90.0 | while read -r c; do
    docker inspect --format '{{.Args}}' "$c" | grep -q gdn-default && docker kill "$c" >/dev/null
  done
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
timeout "${CAP:-4200}" "$(dirname "$0")/stages.sh"
echo "[window] stages exit $? $(date '+%F %T')"
