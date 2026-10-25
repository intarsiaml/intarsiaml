#!/usr/bin/env bash
# Lane hg: HIP tests of the packed-prefill tree (hgpk) on card 0, ON tatooine inside a fleetq job that has radiance down. Then radiance
# is NOT restored (the A/B runs in the same window); restore is the caller's last step.
set -uo pipefail
O=$HOME/hg-logs/gates; mkdir -p "$O"
D="env INTARSIA_REPO=$HOME/intarsiaml-hgpk INTARSIA_TARGET_VOL=intarsia-cargo-target-hgpk HIP_VISIBLE_DEVICES=0 $HOME/intarsiaml-hgpk/deploy/tatooine/dev.sh"
M35=/models/Qwen3.6-35B-A3B-NVFP4
T="env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35"
f() { grep -aE "packed|^test |^test result|FAILED|panicked|^error|slots:"; }
echo "== batch_recurrent (hip), packed test included"
$D $T cargo test --release -p intarsia-models --test batch_recurrent -- --test-threads=1 --nocapture < /dev/null 2>&1 | f
echo "== packed_rows"
$D cargo test --release -p intarsia-backend-hip --test packed_rows -- --test-threads=1 --nocapture < /dev/null 2>&1 | grep -aE "rows identical|segment of|^test |^test result|FAILED|panicked|differ"
echo "== 35B packed prefill equals the local engine (ignored)"
$D $T cargo test --release -p intarsia-models --test batch_cuda_real -- --ignored real_35b_packed_prefill_equals_the_local_engine --test-threads=1 --nocapture < /dev/null 2>&1 | f
echo "== 35B batch_cuda_real (non-ignored) on the packed tree, knob unset"
$D $T cargo test --release -p intarsia-models --test batch_cuda_real -- --test-threads=1 --nocapture < /dev/null 2>&1 | f
echo "== done"
