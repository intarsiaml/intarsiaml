#!/usr/bin/env bash
# CPU-only build of the intarsia binary (hip) and the lane's test binary: ON tatooine inside a fleetq --load cpu job.
set -uo pipefail
cd "$HOME/intarsiaml-ink-hip-moe"
mkdir -p logs
INK_CONTAINER=ink-hip-moe-build bash .scratch/run.sh cargo build --release -j12 --features hip -p intarsia-cli > logs/build-cli.log 2>&1
rc=$?
tail -3 logs/build-cli.log
echo "cli build exit $rc"
[ $rc -ne 0 ] && exit $rc
INK_CONTAINER=ink-hip-moe-build bash .scratch/run.sh cargo test --release -j12 -p intarsia-backend-hip --test ink_moe --no-run > logs/build.log 2>&1
rc=$?
tail -2 logs/build.log
echo "test build exit $rc"
exit $rc
