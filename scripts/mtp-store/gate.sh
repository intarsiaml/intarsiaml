#!/usr/bin/env bash
# lane mtp-store: the CUDA gates on spark1 (the 35B, vLLM stays up): `mtp_store_cuda` (store-on MTP-on ids = store-on MTP-off ids,
# single and batched, acceptance after a restore) and, with ALSO=1, the existing `prefix_store_cuda` identity gates.
#   fleetq run --host spark1 --quiet gpu --mem 45 --expect 25 --label "mtp-store: cuda gate" -- bash scripts/mtp-store/gate.sh
# The output is copied to docs/results/2026-11-13-mtp-store/raw/. A watchdog on the box (memwatch.sh) kills the test under 10 GB
# MemAvailable. The ssh that runs the test stays open for its whole run (one call; under 4 min per test file).
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
# the watchdog first (it waits for the test binary), then the test: the gate does not depend on when the launching ssh returns
scp -q -o BatchMode=yes "$REPO/scripts/mtp-store/memwatch.sh" "$H:intarsia-mtp-store/out/" || { echo "script copy failed"; exit 1; }
rsh "nohup bash $L/out/memwatch.sh 'dep[s]/(mtp_store_cuda|prefix_store_cuda)-' $L/out/mem-$NAME.txt > /dev/null 2>&1 < /dev/null & echo watchdog \$!"
rsh "cd $R && INTARSIA_TEST_REAL_MODEL=1 CARGO_BUILD_JOBS=4 MAX_JOBS=4 cargo test --release --locked -p intarsia-models $TESTS -- --nocapture --test-threads=1 $FILTER > $L/out/$NAME.log 2>&1; echo GATE-EXIT \$? >> $L/out/$NAME.log"
rsh "cat $L/out/$NAME.log" > "$OUT/$NAME.log"
rsh "cat $L/out/mem-$NAME.txt" > "$OUT/mem-$NAME.txt"
echo "$(date +%T) end; min MemAvailable $(sort -k2 -n "$OUT/mem-$NAME.txt" | head -1) MB over $(wc -l < "$OUT/mem-$NAME.txt" | tr -d ' ') samples, KILLED $(grep -c KILLED "$OUT/mem-$NAME.txt"); $(rsh 'free -g | sed -n 2p')" | tee -a "$OUT/$NAME.log"
grep -E "MTP-STORE-CUDA|PREFIX-STORE-CUDA|^test |test result|GATE-EXIT|panicked|KILLED" "$OUT/$NAME.log" | cut -c1-400
grep -q "GATE-EXIT 0" "$OUT/$NAME.log"
