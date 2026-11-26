#!/usr/bin/env bash
# ON tatooine inside a fleetq --load cpu job (no GPU): create the lane's cargo volume, seeded read-only from lane ink-hip-batch's
# volume (it last built main 3226af2b9) when that is under 40 GB, then build the branch's binary as intarsia-$NAME and the test
# targets in MTESTS / HTESTS (build.sh).
set -uo pipefail
LANE="$HOME/intarsiaml-ink-hip-verify"
IMAGE=intarsia-hip-dev:rocm7.2.4-rust1.90.0
V=intarsia-cargo-target-ink-hip-verify
SRCV=${SRCV:-intarsia-cargo-target-ink-hip-batch}
cd "$LANE"; mkdir -p logs
if ! docker volume inspect $V >/dev/null 2>&1; then
  INTARSIA_TARGET_VOL=$V INTARSIA_REPO=$LANE bash deploy/tatooine/dev.sh true < /dev/null
  kb=$(docker run --rm -v $SRCV:/s:ro "$IMAGE" sh -c 'du -sk /s | cut -f1' < /dev/null)
  echo "[setup] seed source $SRCV $((kb / 1024 / 1024)) GB $(date +%T)"
  if [ "${kb:-999999999}" -lt 41943040 ]; then
    docker run --rm --user "$(id -u):$(id -g)" -v $SRCV:/s:ro -v $V:/d "$IMAGE" sh -c 'cp -a /s/. /d/' < /dev/null
    echo "[setup] seeded exit $? $(date +%T)"
  fi
fi
echo "[setup] build ${NAME:-main} $(date +%T)"
bash "$LANE/.scratch/build.sh" "${NAME:-main}" || exit $?
echo "[setup] done $(date +%T)"
