#!/usr/bin/env bash
# Lane l7, ON tatooine, inside a fleetq slot that takes radiance down: stop / restore radiance.
#   win.sh stop      systemctl --user stop radiance
#   win.sh restore   start radiance, wait until it answers (<= 3 min), print its state (idempotent)
cmd=${1:?}; shift
case "$cmd" in
  stop) systemctl --user stop radiance; sleep 3; echo "[l7] radiance $(systemctl --user is-active radiance)" ;;
  restore)
    systemctl --user start radiance
    for i in $(seq 1 36); do curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\[' && break; sleep 5; done
    echo "[l7] radiance $(systemctl --user is-active radiance) after $((i * 5)) s" ;;
esac
