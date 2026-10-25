#!/usr/bin/env bash
# Lane wg merge gates, ON tatooine inside a `fleetq run --load cpu` job (radiance stays up): seed two cargo target volumes from the lane's
# existing ones, then build the dev tree (~/intarsiaml-wgm, the merged head) and the base tree (~/intarsiaml-wgm-base, main 0c142a33):
# the hip CLI, and for dev the test binaries. usage: prep.sh [vols|dev|base|all]
set -uo pipefail
D=$HOME/intarsiaml-wgm/deploy/tatooine/dev.sh
what=${1:-all}
vols() {
  for pair in "wg wgm" "wgbase wgmbase"; do
    set -- $pair
    docker volume inspect intarsia-cargo-target-$2 >/dev/null 2>&1 && continue
    docker volume create intarsia-cargo-target-$2 >/dev/null
    docker run --rm -v intarsia-cargo-target-$1:/s:ro -v intarsia-cargo-target-$2:/d alpine sh -c 'cp -a /s/. /d/ && chmod 0777 /d'
  done
}
[[ $what == all || $what == vols ]] && vols
if [[ $what == all || $what == dev ]]; then
  echo "== build dev $(date +%T)"
  INTARSIA_REPO=$HOME/intarsiaml-wgm INTARSIA_TARGET_VOL=intarsia-cargo-target-wgm $D cargo build --release --locked --features hip -p intarsia-cli 2>&1 | tail -3
  INTARSIA_REPO=$HOME/intarsiaml-wgm INTARSIA_TARGET_VOL=intarsia-cargo-target-wgm $D cargo test --release --locked -p intarsia-backend-hip -p intarsia-models --no-run 2>&1 | tail -3
fi
if [[ $what == all || $what == base ]]; then
  echo "== build base $(date +%T)"
  INTARSIA_REPO=$HOME/intarsiaml-wgm-base INTARSIA_TARGET_VOL=intarsia-cargo-target-wgmbase $D cargo build --release --locked --features hip -p intarsia-cli 2>&1 | tail -3
fi
echo "PREP DONE $(date +%T)"
