#!/usr/bin/env bash
# Lane LDF, ON tatooine inside a fleetq job (--load cpu): seed the lane's cargo target volume from GLR's once, then build the
# release CLI and the mxfp4_a8 test binary of the lane's tree under the given tag (keeps ldf-<tag>-cli and ldf-bin/mxfp4_a8-<tag>).
#   bash ~/intarsiaml-ldf/scripts/ldf/seedbuild.sh <tag>
set -uo pipefail
tag=${1:-dev}
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
if ! docker volume inspect intarsia-cargo-ldf > /dev/null 2>&1; then
  docker volume create intarsia-cargo-ldf > /dev/null
  docker run --rm --user 0 -v intarsia-cargo-glr:/a:ro -v intarsia-cargo-ldf:/b "$IMG" sh -c 'cp -a /a/. /b/ && chmod 0777 /b' && echo "seeded volume"
fi
bash "$HOME/intarsiaml-ldf/scripts/ldf/build.sh" "ldf-$tag-cli"
bash "$HOME/intarsiaml-ldf/scripts/ldf/tbuild.sh" "$tag"
