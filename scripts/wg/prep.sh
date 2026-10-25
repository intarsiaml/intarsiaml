#!/usr/bin/env bash
# Lane wg, ON tatooine inside a fleetq job (--load cpu,disk; radiance stays up): copy the hg cargo target volumes, then build
# the base (origin/main) serve binary in the HIP dev container.  usage: prep.sh [volumes|base|dev]...
set -euo pipefail
cd ~
vols() {
  for pair in "hg wg" "hgbase wgbase"; do
    set -- $pair
    docker volume inspect intarsia-cargo-target-$2 >/dev/null 2>&1 && continue
    docker volume create intarsia-cargo-target-$2 >/dev/null
    docker run --rm -v intarsia-cargo-target-$1:/s:ro -v intarsia-cargo-target-$2:/d alpine sh -c 'cp -a /s/. /d/ && chmod 0777 /d'
  done
}
build() {  # tree volume
  INTARSIA_REPO=$HOME/intarsiaml-$1 INTARSIA_TARGET_VOL=intarsia-cargo-target-$2 \
    ~/intarsiaml-wg/deploy/tatooine/dev.sh cargo build --release --locked --features hip -p intarsia-cli 2>&1 | tail -4
}
for step in "$@"; do
  case $step in
    volumes) vols ;;
    base) build wg-base wgbase ;;
    dev) build wg wg ;;
    tests) INTARSIA_REPO=$HOME/intarsiaml-wg INTARSIA_TARGET_VOL=intarsia-cargo-target-wg ~/intarsiaml-wg/deploy/tatooine/dev.sh \
             cargo test --release --locked -p intarsia-backend-hip -p intarsia-models --no-run 2>&1 | tail -15 ;;
    bench) bash ~/wg-bench.sh build 2>&1 | tail -20 ;;
  esac
done
echo PREP DONE
