#!/usr/bin/env bash
# Lane LDF, ON tatooine inside a fleetq job (--load cpu): build the mxfp4_a8 test binary of the lane's tree and keep a copy
# ~/intarsiaml-ldf/ldf-bin/mxfp4_a8-<tag> (self-contained; run it in the dev container without cargo inside a GPU window).
#   bash ~/intarsiaml-ldf/scripts/ldf/tbuild.sh <tag>
set -uo pipefail
export INTARSIA_REPO=$HOME/intarsiaml-ldf INTARSIA_TARGET_VOL=intarsia-cargo-ldf
cd "$INTARSIA_REPO"; mkdir -p ldf-bin
touch kernels/hip/*.hip crates/intarsia-backend-hip/build.rs crates/intarsia-backend-hip/src/*.rs
tag=${1:-dev}
deploy/tatooine/dev.sh bash -c "
  exe=\$(cargo test --release --no-run -p intarsia-backend-hip --test mxfp4_a8 2>&1 < /dev/null | tee /work/ldf-bin/build-$tag.log | grep -aoE '/target/release/deps/mxfp4_a8-[0-9a-f]+' | tail -1)
  [ -n \"\$exe\" ] || { echo BUILD-FAILED; grep -aE '^error' -A8 /work/ldf-bin/build-$tag.log | head -40; exit 3; }
  cp \$exe /work/ldf-bin/mxfp4_a8-$tag && sha256sum /work/ldf-bin/mxfp4_a8-$tag | cut -c1-16" < /dev/null
