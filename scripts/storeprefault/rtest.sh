#!/usr/bin/env bash
# storeprefault: run the prefix store unit tests ON spark2 (Linux: the MADV_HUGEPAGE branch compiles and runs there), in the synced source,
# with the same free-memory watchdog as the build. Run under fleetq --load cpu, never beside a server:
#   ssh spark2 bash ~/intarsiaml-storeprefault/out/rtest.sh
set -uo pipefail
L=$HOME/intarsiaml-storeprefault; R=$L/src
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
CARGO_BUILD_JOBS=3 setsid bash -c "cd $R && exec cargo test --release -p intarsia-core --lib prefix_store" > $L/out/rtest.log 2>&1 < /dev/null &
PID=$!
while kill -0 $PID 2>/dev/null; do
  a=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  if [ "$a" -lt 15 ]; then echo "WATCHDOG: avail $a GB < 15, killing"; kill -TERM -$PID; sleep 5; kill -KILL -$PID 2>/dev/null; exit 3; fi
  sleep 10
done
wait $PID; echo "cargo test exit $?"
grep -E '^test |test result' $L/out/rtest.log
