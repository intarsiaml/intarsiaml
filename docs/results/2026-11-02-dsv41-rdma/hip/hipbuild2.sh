#!/usr/bin/env bash
# ON tatooine, radiance up (CPU only): rebuild the head tree after touching every source (git archive gives files the commit time,
# older than the artifacts of the main build in the shared target volume: cargo kept the stale intarsia-server)
set -uo pipefail
export INTARSIA_TARGET_VOL=intarsia-cargo-target-rdma INTARSIA_REPO=$HOME/intarsiaml-rdma-head
find $INTARSIA_REPO/crates $INTARSIA_REPO/kernels -type f -exec touch {} +
D=$INTARSIA_REPO/deploy/tatooine/dev.sh
echo "== head $(date +%T)"
$D env INTARSIA_BUILD_STAMP="rdma-head" CARGO_BUILD_JOBS=12 cargo build --release -p intarsia-cli --features hip > ~/rdma-logs/build-head.log 2>&1 < /dev/null
echo "build head exit $? $(date +%T)"; grep -E '^(error|warning: unused)|Finished' ~/rdma-logs/build-head.log | tail -3
$D cp /target/release/intarsia /work/rdma-intarsia < /dev/null
ls -la ~/intarsiaml-rdma-*/rdma-intarsia; md5sum ~/intarsiaml-rdma-*/rdma-intarsia
