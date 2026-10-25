#!/usr/bin/env bash
# Lane mr, ON tatooine inside a fleetq job (--load cpu): build the dev tree (~/mr-hip) and the base tree (~/mr-hip/.base) into
# bind-mounted cargo target dirs on /data (the root disk is full): the intarsia binary of both, plus the dev tests the gate
# window runs (compiled, not run). Re-runnable: incremental.
#   build.sh [dev|base|both]
set -uo pipefail
what=${1:-both}
mkvol() { # <name> <dir>
  mkdir -p "$2" && chmod 0777 "$2"
  docker volume inspect "$1" >/dev/null 2>&1 || docker volume create --driver local -o type=none -o o=bind -o device="$2" "$1" >/dev/null
}
mkvol intarsia-cargo-target-mr /data/mr-target-dev
mkvol intarsia-cargo-target-mrbase /data/mr-target-base
D=$HOME/mr-hip/deploy/tatooine/dev.sh
if [ "$what" != base ]; then
  echo "== dev: intarsia"
  INTARSIA_REPO=$HOME/mr-hip INTARSIA_TARGET_VOL=intarsia-cargo-target-mr $D cargo build --release --locked --features hip -p intarsia-cli 2>&1 < /dev/null | tail -3
  echo "== dev: tests (compile only)"
  INTARSIA_REPO=$HOME/mr-hip INTARSIA_TARGET_VOL=intarsia-cargo-target-mr $D cargo test --release --no-run -p intarsia-backend-hip --test rows_exact --test rows_registry 2>&1 < /dev/null | tail -4
  INTARSIA_REPO=$HOME/mr-hip INTARSIA_TARGET_VOL=intarsia-cargo-target-mr $D cargo test --release --no-run -p intarsia-models --test batch_cuda_real 2>&1 < /dev/null | tail -3
fi
if [ "$what" != dev ]; then
  echo "== base: intarsia"
  INTARSIA_REPO=$HOME/mr-hip/.base INTARSIA_TARGET_VOL=intarsia-cargo-target-mrbase $D cargo build --release --locked --features hip -p intarsia-cli 2>&1 < /dev/null | tail -3
  echo "== base: tests (compile only, for the 27B batching check)"
  INTARSIA_REPO=$HOME/mr-hip/.base INTARSIA_TARGET_VOL=intarsia-cargo-target-mrbase $D cargo test --release --no-run -p intarsia-models --test batch_cuda_real 2>&1 < /dev/null | tail -3
fi
df -h /data | tail -1; echo BUILD DONE
