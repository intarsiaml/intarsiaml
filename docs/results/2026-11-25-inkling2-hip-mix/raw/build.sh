#!/usr/bin/env bash
# run ON tatooine inside a fleetq slot (--load cpu): build the hip CLI + the mix tests in the lane's own target volume
set -uo pipefail
cd ~/intarsiaml-ink-hip-mix
export INTARSIA_REPO=$PWD INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-mix
if ! docker volume inspect "$INTARSIA_TARGET_VOL" >/dev/null 2>&1; then
  docker volume create "$INTARSIA_TARGET_VOL" >/dev/null
  docker run --rm --user 0 -v intarsia-cargo-target-ink-hip:/s:ro -v "$INTARSIA_TARGET_VOL":/d intarsia-hip-dev:rocm7.2.4-rust1.90.0 bash -c "cp -a /s/. /d/ && chmod 0777 /d" < /dev/null
fi
for what in "$@"; do
  deploy/tatooine/dev.sh bash -c "$what" < /dev/null || exit $?
done
