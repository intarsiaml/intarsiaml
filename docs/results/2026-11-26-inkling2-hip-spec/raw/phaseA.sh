#!/usr/bin/env bash
# ON tatooine inside a fleetq --load cpu job (no GPU): seed the lane's cargo volume from the image's warm default volume (read-only
# copy, only when the lane volume is new and the source is under 40 GB), then build main's binary (intarsia-main) from
# .scratch/main-src. Env BRANCH=<name>: also build the branch binary with the test targets in HTESTS / MTESTS.
set -uo pipefail
LANE="$HOME/intarsiaml-ink-hip-spec"
IMAGE=intarsia-hip-dev:rocm7.2.4-rust1.90.0
V=intarsia-cargo-target-ink-hip-spec
SRCV=intarsia-cargo-target-67e66c624a8f
cd "$LANE"; mkdir -p logs
if ! docker volume inspect $V >/dev/null 2>&1; then
  INTARSIA_TARGET_VOL=$V INTARSIA_REPO=$LANE bash deploy/tatooine/dev.sh true < /dev/null   # creates the volume, world-writable
  kb=$(docker run --rm -v $SRCV:/s:ro "$IMAGE" sh -c 'du -sk /s | cut -f1')
  echo "[phaseA] seed source $((kb / 1024 / 1024)) GB $(date +%T)"
  if [ "${kb:-999999999}" -lt 41943040 ]; then
    docker run --rm --user "$(id -u):$(id -g)" -v $SRCV:/s:ro -v $V:/d "$IMAGE" sh -c 'cp -a /s/. /d/' < /dev/null
    echo "[phaseA] seeded exit $? $(date +%T)"
  fi
fi
if [ -z "${SKIP_MAIN:-}" ]; then
  echo "[phaseA] build main $(date +%T)"
  bash "$LANE/.scratch/build.sh" main .scratch/main-src || exit $?
fi
if [ -n "${BRANCH:-}" ]; then
  echo "[phaseA] build branch $BRANCH $(date +%T)"
  bash "$LANE/.scratch/build.sh" "$BRANCH" || exit $?
fi
echo "[phaseA] done $(date +%T)"
