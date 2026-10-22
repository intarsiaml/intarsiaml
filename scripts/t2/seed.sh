#!/usr/bin/env bash
# Lane T2 (ON tatooine, CPU slot): give each new t2 target volume a copy of P1's (intarsia-cargo-target-p1), so the first
# build is incremental instead of recompiling every HIP kernel. Then build.sh with the TREES / HIPTEST / TEST it gets.
set -uo pipefail
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
for n in ${SEED:-head base}; do
  v=intarsia-cargo-target-t2-$n
  if ! docker volume inspect "$v" >/dev/null 2>&1; then
    docker volume create "$v" >/dev/null
    docker run --rm -v intarsia-cargo-target-p1:/from:ro -v "$v":/to --user 0 "$IMG" bash -c 'cp -a /from/. /to/ && chmod 0777 /to' < /dev/null
    echo "seeded $v from intarsia-cargo-target-p1 $(date +%T)"
  fi
done
bash "$HOME/intarsiaml-t2-head/scripts/t2/build.sh"
