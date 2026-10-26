#!/usr/bin/env bash
# Lane lb, ON tatooine inside a fleetq job (--load cpu): build the dev and base trees (~/intarsiaml-lb/{dev,base}) into bind-mounted cargo
# target dirs on /data (the root disk is nearly full; ~/intarsiaml-lb/target-* are symlinks to them). One target per tree: the
# intarsia binary, the HIP crate tests and the models tests share it (compiled, not run). Re-runnable: incremental.
#   build.sh [dev|base|both] [tests|both]     (tests: also compile the GPU tests of the dev tree; "both": of the base tree too)
set -uo pipefail
what=${1:-both}; tests=${2:-}
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
LB=$HOME/intarsiaml-lb
mkvol() { # <name> <dir> <link>
  mkdir -p "$2" && chmod 0777 "$2"
  [ -e "$3" ] || ln -s "$2" "$3"
  docker volume inspect "$1" >/dev/null 2>&1 || docker volume create --driver local -o type=none -o o=bind -o device="$2" "$1" >/dev/null
}
mkvol intarsia-cargo-lb-dev /data/intarsiaml-lb-target-dev "$LB/target-dev"
mkvol intarsia-cargo-lb-base /data/intarsiaml-lb-target-base "$LB/target-base"
if ! docker volume inspect intarsia-cargo-lb-registry >/dev/null 2>&1; then
  docker volume create intarsia-cargo-lb-registry >/dev/null
  docker run --rm --user 0 -v intarsia-cargo-target-registry:/src:ro -v intarsia-cargo-lb-registry:/dst "$IMG" bash -c 'cp -a /src/. /dst/ && chmod -R a+rwX /dst' 2>&1 | tail -2
fi
export INTARSIA_REGISTRY_VOL=intarsia-cargo-lb-registry
for t in dev base; do
  [ "$what" = both ] || [ "$what" = $t ] || continue
  D=$LB/$t/deploy/tatooine/dev.sh
  export INTARSIA_REPO=$LB/$t INTARSIA_TARGET_VOL=intarsia-cargo-lb-$t
  echo "== $t: intarsia $(date +%T)"
  $D cargo build --release --locked --features hip -p intarsia-cli 2>&1 < /dev/null | tail -3
  if [ -n "$tests" ] && { [ $t = dev ] || [ "$tests" = both ]; }; then
    echo "== $t: tests (compile only) $(date +%T)"
    $D cargo test --release --no-run -p intarsia-backend-hip 2>&1 < /dev/null | tail -4
    $D cargo test --release --no-run -p intarsia-models --test batch_cuda_real --test batch_recurrent 2>&1 < /dev/null | tail -4
  fi
done
df -h /data | tail -1; echo "BUILD DONE $(date +%T)"
