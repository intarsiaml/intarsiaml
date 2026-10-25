#!/usr/bin/env bash
# Lane l5, ON tatooine inside a --load cpu fleetq job: build the release `intarsia` binary of the branch snapshot (~/intarsiaml-l5-$NAME,
# volume intarsia-cargo-target-l5-dev) and/or of origin/main (~/intarsiaml-l5-base, volume intarsia-cargo-target-l5-base). Env: NAME, WHAT=dev|base|both.
set -euo pipefail
cd ~
WHAT=${WHAT:-both}
[ "$WHAT" = base ] || SRC_DEV=$HOME/intarsiaml-l5-${NAME:?}
mkvol() { # mkvol <new> : copy the hg volume once so the deps are not rebuilt from scratch
  docker volume inspect "$1" >/dev/null 2>&1 && return
  docker volume create "$1" >/dev/null
  docker run --rm -v intarsia-cargo-target-hg:/s:ro -v "$1":/d alpine sh -c 'cp -a /s/. /d/ && chmod 0777 /d'
}
build() { # build <tree> <vol> [cargo args...]
  local tree=$1 vol=$2; shift 2
  echo "== build $tree ($vol)"
  INTARSIA_REPO=$tree INTARSIA_TARGET_VOL=$vol ${DEVSH:-$tree/scripts/l5/dev.sh} cargo build --release --locked --features hip -p intarsia-cli "$@" 2>&1 < /dev/null | tail -5
}
if [ "$WHAT" != base ]; then mkvol intarsia-cargo-target-l5-dev; build "$SRC_DEV" intarsia-cargo-target-l5-dev; fi
if [ "$WHAT" != dev ]; then mkvol intarsia-cargo-target-l5-base; DEVSH=$HOME/intarsiaml-l5-base/deploy/tatooine/dev.sh build "$HOME/intarsiaml-l5-base" intarsia-cargo-target-l5-base; fi
if [ "$WHAT" != base ] && [ "${TESTS:-1}" = 1 ]; then
  echo "== host-only knob test + build of the device test (no GPU)"
  dv() { INTARSIA_REPO=$SRC_DEV INTARSIA_TARGET_VOL=intarsia-cargo-target-l5-dev $SRC_DEV/scripts/l5/dev.sh "$@" 2>&1 < /dev/null; }
  dv cargo test --release -p intarsia-backend-hip --test moe_f8_knob -- --nocapture | grep -aE "^test |^test result|error|panicked" | head -20
  dv cargo test --release -p intarsia-backend-hip --test moe_f8 --no-run | tail -3
  dv cargo test --release -p intarsia-models --no-run | tail -3
fi
echo PREP DONE
