#!/usr/bin/env bash
# ON tatooine inside a fleetq --load cpu job (Phase A, no GPU): seed the lane's cargo volume from the image's warm default volume
# (read-only copy, only when the lane volume is new and the source is under 40 GB), build main's binary (intarsia-main) from
# .scratch/main-src, then the branch's (intarsia-pf), the hip crate's test targets, its lib tests.
set -uo pipefail
LANE="$HOME/intarsiaml-ink-hip-pfflip"
IMAGE=intarsia-hip-dev:rocm7.2.4-rust1.90.0
V=intarsia-cargo-target-ink-hip-pfflip
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
echo "[phaseA] build main $(date +%T)"
bash "$LANE/.scratch/build.sh" main .scratch/main-src || exit $?
echo "[phaseA] build branch $(date +%T)"
TESTS="ink_moe ink_attn" LIBTESTS=1 bash "$LANE/.scratch/build.sh" pf || exit $?
echo "[phaseA] done $(date +%T)"
