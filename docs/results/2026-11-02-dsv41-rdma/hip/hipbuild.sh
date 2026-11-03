#!/usr/bin/env bash
# ON tatooine, radiance up (CPU only): build the HIP CLI of two trees into ONE target volume (main first, then head: the second
# build only recompiles what differs), copy each CLI to <tree>/rdma-intarsia.
set -uo pipefail
mkdir -p ~/rdma-logs
export INTARSIA_TARGET_VOL=intarsia-cargo-target-rdma
for n in main head; do
  export INTARSIA_REPO=$HOME/intarsiaml-rdma-$n
  D=$INTARSIA_REPO/deploy/tatooine/dev.sh
  echo "== $n $(date +%T)"
  $D env INTARSIA_BUILD_STAMP="rdma-$n" CARGO_BUILD_JOBS=${JOBS:-12} cargo build --release -p intarsia-cli --features hip > ~/rdma-logs/build-$n.log 2>&1 < /dev/null
  echo "build $n exit $? $(date +%T)"; grep -E '^(error|warning: unused)|Finished' ~/rdma-logs/build-$n.log | tail -3
  $D cp /target/release/intarsia /work/rdma-intarsia < /dev/null
done
ls -la ~/intarsiaml-rdma-*/rdma-intarsia; docker system df | grep Volumes
