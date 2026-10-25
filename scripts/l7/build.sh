#!/usr/bin/env bash
# Lane l7, ON tatooine inside a fleetq job (--load cpu): build the dev tree (~/intarsiaml-l7-hip) and the base tree (~/intarsiaml-l7-base)
# into bind-mounted cargo target dirs on /data (the root disk is nearly full; ~/intarsiaml-l7-target-* are symlinks to them): the
# intarsia binary of both, plus the dev tests the gate windows run (compiled, not run). Re-runnable: incremental.
#   build.sh [dev|base|both]
set -uo pipefail
what=${1:-both}
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
mkvol() { # <name> <dir> <link>
  mkdir -p "$2" && chmod 0777 "$2"
  [ -e "$3" ] || ln -s "$2" "$3"
  docker volume inspect "$1" >/dev/null 2>&1 || docker volume create --driver local -o type=none -o o=bind -o device="$2" "$1" >/dev/null
}
mkvol intarsia-cargo-l7-dev /data/intarsiaml-l7-target-dev "$HOME/intarsiaml-l7-target-dev"
mkvol intarsia-cargo-l7-base /data/intarsiaml-l7-target-base "$HOME/intarsiaml-l7-target-base"
# own crates.io cache, seeded once from the shared one (read-only), so no download is needed
if ! docker volume inspect intarsia-cargo-l7-registry >/dev/null 2>&1; then
  docker volume create intarsia-cargo-l7-registry >/dev/null
  docker run --rm --user 0 -v intarsia-cargo-target-registry:/src:ro -v intarsia-cargo-l7-registry:/dst "$IMG" bash -c 'cp -a /src/. /dst/ && chmod -R a+rwX /dst' 2>&1 | tail -2
fi
export INTARSIA_REGISTRY_VOL=intarsia-cargo-l7-registry
D=$HOME/intarsiaml-l7-hip/deploy/tatooine/dev.sh
if [ "$what" != base ]; then
  echo "== syntax check of the new translation unit"
  out=$($HOME/intarsiaml-l7-hip/scripts/l7/syntax.sh); echo "$out"
  echo "$out" | grep -q "rc=[1-9]" && { echo "SYNTAX CHECK FAILED, no build"; exit 1; }
fi
if [ "$what" != base ]; then
  echo "== dev: intarsia"
  INTARSIA_REPO=$HOME/intarsiaml-l7-hip INTARSIA_TARGET_VOL=intarsia-cargo-l7-dev $D cargo build --release --locked --features hip -p intarsia-cli 2>&1 < /dev/null | tail -3
  echo "== dev: tests (compile only)"
  INTARSIA_REPO=$HOME/intarsiaml-l7-hip INTARSIA_TARGET_VOL=intarsia-cargo-l7-dev $D cargo test --release --no-run -p intarsia-backend-hip --test moe_block --test rows_exact --test rows_registry 2>&1 < /dev/null | tail -4
  INTARSIA_REPO=$HOME/intarsiaml-l7-hip INTARSIA_TARGET_VOL=intarsia-cargo-l7-dev $D cargo test --release --no-run -p intarsia-models --test batch_cuda_real --test batch_recurrent 2>&1 < /dev/null | tail -4
fi
if [ "$what" != dev ]; then
  echo "== base: intarsia"
  INTARSIA_REPO=$HOME/intarsiaml-l7-base INTARSIA_TARGET_VOL=intarsia-cargo-l7-base $D cargo build --release --locked --features hip -p intarsia-cli 2>&1 < /dev/null | tail -3
fi
df -h /data | tail -1; echo BUILD DONE
