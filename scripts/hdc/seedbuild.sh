#!/usr/bin/env bash
# Lane HDC, ON tatooine inside a fleetq job (--load cpu): seed the lane's cargo target volume from LDF's once, then build the
# release CLI of the lane's tree as ~/intarsiaml-hdc/<name> (default hdc-base-cli).
#   bash ~/intarsiaml-hdc/scripts/hdc/seedbuild.sh [name]
set -uo pipefail
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
if ! docker volume inspect intarsia-cargo-hdc > /dev/null 2>&1; then
  docker volume create intarsia-cargo-hdc > /dev/null
  docker run --rm --user 0 -v intarsia-cargo-ldf:/a:ro -v intarsia-cargo-hdc:/b "$IMG" sh -c 'cp -a /a/. /b/ && chmod 0777 /b' && echo "seeded volume"
fi
bash "$HOME/intarsiaml-hdc/scripts/hdc/build.sh" "${1:-hdc-base-cli}"
