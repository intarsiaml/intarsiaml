#!/usr/bin/env bash
# Lane l5, ON tatooine inside a --load cpu fleetq job: build the release `intarsia` binary of the branch snapshot (~/intarsiaml-l5-$NAME,
# volume intarsia-cargo-target-l5-zdev) and/or of origin/main (~/intarsiaml-l5-zbase, volume intarsia-cargo-target-l5-zbase). Env: NAME, WHAT=dev|base|both.
set -euo pipefail
mkdir -p ~/intarsiaml-l5-logs
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
  INTARSIA_REPO=$tree INTARSIA_TARGET_VOL=$vol ${DEVSH:-$tree/scripts/l5r7/dev.sh} cargo build --release --locked --features hip -p intarsia-cli "$@" > ~/intarsiaml-l5-logs/prep-build-$(basename $tree).log 2>&1 < /dev/null && rc=0 || rc=$?; tail -4 ~/intarsiaml-l5-logs/prep-build-$(basename $tree).log; echo "build rc=$rc"; [ $rc = 0 ]
}
if [ "$WHAT" != base ]; then mkvol intarsia-cargo-target-l5-zdev; build "$SRC_DEV" intarsia-cargo-target-l5-zdev; fi
if [ "$WHAT" != dev ]; then mkvol intarsia-cargo-target-l5-zbase; DEVSH=$HOME/intarsiaml-l5-zbase/deploy/tatooine/dev.sh build "$HOME/intarsiaml-l5-zbase" intarsia-cargo-target-l5-zbase; fi
if [ "$WHAT" != base ] && [ "${TESTS:-1}" = 1 ]; then
  echo "== build of the device tests (no GPU)"
  dv() { INTARSIA_REPO=$SRC_DEV INTARSIA_TARGET_VOL=intarsia-cargo-target-l5-zdev $SRC_DEV/scripts/l5r7/dev.sh "$@" 2>&1 < /dev/null; }
  dv cargo test --release -p intarsia-backend-hip --test moe_pairs --test quantmoe --no-run | tail -4
  dv cargo test --release -p intarsia-models --test batch_cuda_real --no-run | tail -3
fi
echo PREP DONE
