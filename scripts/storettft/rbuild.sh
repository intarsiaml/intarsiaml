#!/usr/bin/env bash
# storettft: runs ON spark2. Builds the CUDA CLI in ~/intarsiaml-storettft/src with its own free-memory watchdog (started first,
# on this side so a dropped ssh cannot stop it). Kills cargo's process group by PGID if MemAvailable < 15 GB.
set -uo pipefail
L=$HOME/intarsiaml-storettft; R=$L/src
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
mkdir -p $L/out $L/bin
rm -f $L/out/build-mem.txt
INTARSIA_BUILD_STAMP="$1" CARGO_BUILD_JOBS=3 MAX_JOBS=3 FLASHINFER_NVCC_THREADS=1 setsid bash -c "cd $R && exec cargo build --release --locked --features cuda -p intarsia-cli" > $L/out/build.log 2>&1 < /dev/null &
PID=$!
echo "cargo pid $PID $(date +%T)"
while kill -0 $PID 2>/dev/null; do
  a=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  echo "$(date +%T) avail ${a} GB" >> $L/out/build-mem.txt
  if [ "$a" -lt 15 ]; then echo "WATCHDOG: avail $a GB < 15, killing build"; kill -TERM -$PID 2>/dev/null; pkill -TERM -P $PID; sleep 5; kill -KILL -$PID 2>/dev/null; exit 3; fi
  sleep 20
done
wait $PID; echo "cargo exit $?"
tail -4 $L/out/build.log
cp $R/target/release/intarsia $L/bin/intarsia && $L/bin/intarsia --version | head -1
echo "build end $(date +%T)"
