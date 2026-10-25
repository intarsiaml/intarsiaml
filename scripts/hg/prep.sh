#!/usr/bin/env bash
# Lane hg, on tatooine inside a fleetq job (--load cpu,disk, radiance stays up): copy the hr3 cargo target volumes and
# build the dev and base serve binaries in the HIP dev container.
set -euo pipefail
cd ~
for pair in "hr3 hg" "hr3base hgbase"; do
  set -- $pair
  docker volume inspect intarsia-cargo-target-$2 >/dev/null 2>&1 && continue
  docker volume create intarsia-cargo-target-$2 >/dev/null
  docker run --rm -v intarsia-cargo-target-$1:/s:ro -v intarsia-cargo-target-$2:/d alpine sh -c 'cp -a /s/. /d/ && chmod 0777 /d'
done
for t in hg hg-base; do
  vol=intarsia-cargo-target-${t/-base/base}
  echo "== build $t ($vol)"
  INTARSIA_REPO=$HOME/intarsiaml-$t INTARSIA_TARGET_VOL=$vol \
    ~/intarsiaml-hg/deploy/tatooine/dev.sh cargo build --release --locked --features hip -p intarsia-cli 2>&1 | tail -4
done
echo PREP DONE
