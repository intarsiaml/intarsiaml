#!/usr/bin/env bash
# Lane gdn-chunk: per-launch time of the chunked delta rule against the sequential kernels across T (tests/gdn_timing.rs), for
# the min-T threshold. ON tatooine, inside a fleetq lease with radiance down; the test binary is built beforehand
# (`cargo test --release -p intarsia-backend-hip --test gdn_timing --no-run`, a CPU slot).
set -u
cd "$(dirname "$0")/../.." && . scripts/gdn-chunk/lib.sh
HIP_VISIBLE_DEVICES=0 timeout 600 "$TREE/deploy/tatooine/dev.sh" cargo test --release -p intarsia-backend-hip --test gdn_timing -- --nocapture \
  gdn_chunk_vs_sequential_timing > "$O/timing.out" 2> "$O/timing.err" < /dev/null
log "timing exit $?"
