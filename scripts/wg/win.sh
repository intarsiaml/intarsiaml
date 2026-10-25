#!/usr/bin/env bash
# Lane wg, ON tatooine inside a fleetq job that takes radiance down (--takes-down radiance). Runs one command with radiance stopped and
# ALWAYS restores radiance afterwards (trap), waiting until it answers.   win.sh <command...>   e.g.  win.sh bash ~/wg-bench.sh run
set -uo pipefail
now() { date '+%F %T'; }
restore() {
  echo "[win] starting radiance $(now)"
  systemctl --user start radiance
  for i in $(seq 1 36); do
    if curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\['; then
      echo "[win] radiance answers $(now) (after $((i * 5)) s); $(systemctl --user is-active radiance)"; return 0
    fi
    sleep 5
  done
  echo "[win] RADIANCE DID NOT ANSWER within 180 s $(now); $(systemctl --user is-active radiance)"; return 1
}
trap restore EXIT
echo "[win] stopping radiance $(now)"
systemctl --user stop radiance
"$@"
