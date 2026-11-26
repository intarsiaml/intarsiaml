#!/usr/bin/env bash
# CPU-only build of the lane's test binary (no GPU use): ON tatooine inside a fleetq --load cpu job.
set -uo pipefail
cd "$HOME/intarsiaml-ink-hip-moe"
mkdir -p logs
INK_CONTAINER=ink-hip-moe-build bash .scratch/run.sh cargo test --release -j8 -p intarsia-backend-hip --test ink_moe --no-run > logs/build.log 2>&1
rc=$?
grep -E "^(error|warning: unused)|error:|ink_moe\.hip|Finished|Executable" logs/build.log | head -60
echo "build exit $rc"
exit $rc
