#!/usr/bin/env bash
# lane mtp-store: the CUDA gates on spark1 (the 35B, vLLM stays up): `mtp_store_cuda` (store-on MTP-on ids = store-on MTP-off ids,
# single and batched, acceptance after a restore) and, with ALSO=1, the existing `prefix_store_cuda` identity gates.
#   fleetq run --host spark1 --quiet gpu --mem 45 --expect 25 --label "mtp-store: cuda gate" -- bash scripts/mtp-store/gate.sh
# The output is copied to docs/results/2026-11-13-mtp-store/raw/. A watchdog kills the test under 10 GB MemAvailable.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=$REPO/docs/results/2026-11-13-mtp-store/raw; mkdir -p "$OUT"
L='~/intarsia-mtp-store'; R="$L/src"; H=spark1; NAME=${NAME:-gate}
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$H" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
TESTS="--test mtp_store_cuda"; FILTER="mtp_on_store"
[[ "${ALSO:-0}" == 1 ]] && { TESTS="--test prefix_store_cuda"; FILTER="prefix_store_"; }
echo "$(date +%T) start; earlyoom $(rsh 'systemctl is-active earlyoom'); $(rsh 'free -g | sed -n 2p')"
rsh "cd $R && nohup env INTARSIA_TEST_REAL_MODEL=1 CARGO_BUILD_JOBS=4 MAX_JOBS=4 bash -c 'cargo test --release --locked -p intarsia-models $TESTS -- --nocapture --test-threads=1 $FILTER; echo GATE-EXIT \$?' > $L/out/$NAME.log 2>&1 < /dev/null &"
sleep 3
min=999999
while ! rsh "grep -q GATE-EXIT $L/out/$NAME.log"; do
  a=$(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo"); (( a < min )) && min=$a
  if [[ "$a" -lt 10240 ]]; then rsh "pkill -9 -f 'deps/(mtp_store_cuda|prefix_store_cuda)'"; echo "KILLED: MemAvailable ${a} MB"; break; fi
  sleep 5
done
rsh "cat $L/out/$NAME.log" > "$OUT/$NAME.log"
echo "$(date +%T) end; min MemAvailable ${min} MB; $(rsh 'free -g | sed -n 2p')" | tee -a "$OUT/$NAME.log"
grep -E "MTP-STORE-CUDA|PREFIX-STORE-CUDA|^test |test result|GATE-EXIT|panicked|KILLED" "$OUT/$NAME.log" | cut -c1-400
grep -q "GATE-EXIT 0" "$OUT/$NAME.log"
