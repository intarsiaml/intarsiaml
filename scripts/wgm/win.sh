#!/usr/bin/env bash
# Lane wg merge gates, ON tatooine inside a fleetq job that takes radiance down: stop radiance, run the command as a background job we `wait`
# for (so TERM/INT fire the trap at once), ALWAYS start radiance again (EXIT trap) and wait until it answers.
set -uo pipefail
now() { date '+%F %T'; }
CHILD=
restore() {
  trap - EXIT TERM INT
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
trap 'echo "[win] caught TERM/INT $(now): restoring"; [ -n "$CHILD" ] && kill "$CHILD" 2>/dev/null; exit 143' TERM INT
echo "[win] stopping radiance $(now)"
systemctl --user stop radiance
sleep 3
"$@" <&0 &
CHILD=$!
wait "$CHILD"; rc=$?; CHILD=
exit $rc
