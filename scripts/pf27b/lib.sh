#!/usr/bin/env bash
# Lane PF27b shared helpers, ON tatooine (sourced) inside a fleetq job. Every GPU command runs in the dev image through deploy/tatooine/dev.sh.
# Layout: ~/intarsiaml-pf27b is the branch tree (rsynced from the Mac), ~/intarsiaml-pf27b/pf27b-base-tree is main's tree;
# binaries live in ~/intarsiaml-pf27b (/work inside the container): pf27b-<tag>-cli, pf27b-bin/<test>-<tag>; outputs in pf27b-out/<window>.
R=$HOME/intarsiaml-pf27b; V=intarsia-cargo-pf27b; D=$R/deploy/tatooine/dev.sh
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
H=/data/models/radiance/hf-cache/hub
now() { date '+%F %T'; }
# our containers still running (a `timeout` kills the docker client, not the container): they would hold the cards
killdev() { docker ps -q --filter name=pf27b-run- | xargs -r docker kill > /dev/null 2>&1; }
restore() {
  killdev; docker rm -f pf27b-serve > /dev/null 2>&1
  local t0; t0=$(date +%s); echo "[pf27b] starting radiance $(now)"
  systemctl --user start radiance < /dev/null
  # radiance answers on :8080 (:8000 does not answer even when it is healthy); wait up to 10 min, the slot is held until then
  for i in $(seq 1 120); do curl -s -m 3 http://localhost:8080/v1/models < /dev/null | grep -q '"data":\[' && break; sleep 5; done
  echo "[pf27b] RADIANCE-RESTORE $(systemctl --user is-active radiance) after $(( $(date +%s) - t0 )) s at $(now); models: $(curl -s -m 3 http://localhost:8080/v1/models < /dev/null | head -c 100)"
}
stop_radiance() { echo "[pf27b] stopping radiance $(now)"; systemctl --user stop radiance < /dev/null; sleep 3; echo "[pf27b] radiance $(systemctl --user is-active radiance) $(now)"; }
# gpu <timeout s> <cmd...>: dev image, both cards (G overrides), the lane's tree + target volume
gpu() { local t=$1 nm=pf27b-run-$RANDOM$RANDOM; shift
  INTARSIA_CONTAINER_NAME=$nm INTARSIA_REPO=$R INTARSIA_TARGET_VOL=$V INTARSIA_MODELS=${MODELS:-$H} HIP_VISIBLE_DEVICES=${G:-0,1} timeout -s KILL "$t" $D "$@" < /dev/null
  local rc=$?; if [[ $rc -eq 137 ]]; then echo "[pf27b] timeout after $t s: killing the container" >&2; docker kill $nm > /dev/null 2>&1; fi; return $rc
}
gpup() { local t=$1 nm=pf27b-run-$RANDOM$RANDOM; shift
  INTARSIA_CONTAINER_NAME=$nm INTARSIA_IMAGE=intarsia-hip-prof:rocm7.2.4-rust1.90.0 INTARSIA_REPO=$R INTARSIA_TARGET_VOL=$V INTARSIA_MODELS=${MODELS:-$H} HIP_VISIBLE_DEVICES=${G:-0,1} timeout -s KILL "$t" $D "$@" < /dev/null
  local rc=$?; if [[ $rc -eq 137 ]]; then docker kill $nm > /dev/null 2>&1; fi; return $rc
}
