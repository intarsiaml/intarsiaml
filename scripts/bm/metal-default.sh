#!/usr/bin/env bash
# Lane bm, N=1 default: Metal gates on endor inside one fleetq slot (--exclusive):
#   fleetq run --host endor --exclusive --mem 40 --expect 45 --label "bm-default: Metal no-op 8B/27B + 35B ids" -- scripts/bm/metal-default.sh
# (1) no-op vs main 196e768b6 for Qwen3-8B and the Qwen3.8-27B MXFP4 (noop.sh's "35" slot holds the 27B here): plan show text + json and 256
#     greedy ids, short and ~3.5k prompts. (2) 35B: `--spec mtp --max-batch 8` with no INTARSIA_BATCHED_MTP (recipe default N=1) vs the solo
#     spec-off server, 6 concurrent prompts; the startup line names N=1.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
DEV="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
S=$DEV/.scratch/metal; O=$DEV/docs/results/2026-11-05-batched-mtp/raw/default-metal; mkdir -p $S $O
M8=$HOME/models/Qwen3-8B; M27=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4; M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4
rm -rf $S/base309-src; mkdir -p $S/base309-src && git -C $DEV archive 196e768b6 | tar -x -C $S/base309-src
unset INTARSIA_BATCHED_MTP
for t in base dev; do
  if [ $t = base ]; then T=$S/base309-src; else T=$DEV; fi
  echo "== build $t $(date +%T)"
  (cd $T && CARGO_TARGET_DIR=$S/target-$t cargo build --release --features metal -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3)
  RUN="$S/target-$t/release/intarsia" bash $DEV/scripts/tp2-hip/noop/noop.sh run $O $t metal $M8 $M27 $DEV/scripts/tp2-hip/noop/prompts
done
bash $DEV/scripts/tp2-hip/noop/noop.sh cmp $O base dev | sed 's/ 35 / 27B /' | tee $O/noop.txt
B=$S/target-dev/release/intarsia
PORT=8196; SPID=""
stop() { [[ -n "$SPID" ]] && { kill $SPID 2>/dev/null; for i in $(seq 1 20); do kill -0 $SPID 2>/dev/null || break; sleep 1; done; kill -9 $SPID 2>/dev/null; SPID=""; }; }
trap stop EXIT
ids() { # ids <name> <conc> <args...>
  local name=$1 conc=$2; shift 2
  echo "== ids $name $(date +%T)"
  $B serve --backend metal --model $M35 --port $PORT --host 127.0.0.1 --served-model-name m --max-seq 8192 --profile-every 0 "$@" > $O/serve-$name.log 2>&1 < /dev/null &
  SPID=$!
  local up=0; for i in $(seq 1 150); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && { up=1; break; }; sleep 2; done
  if [[ $up == 1 ]]; then python3 $DEV/scripts/bm/ids.py http://127.0.0.1:$PORT/v1 m $O/ids-$name.json $conc
    curl -sf http://127.0.0.1:$PORT/metrics | grep -E '^intarsia_spec_(proposed|accepted)_tokens_total' | tee $O/ids-$name.metrics
  else echo "server $name did not come up"; tail -5 $O/serve-$name.log; fi
  grep -E "batched MTP" $O/serve-$name.log | head -2
  stop
}
ids solo-off 1 --max-batch 1 --spec off
ids mtp8-default 6 --max-batch 8 --spec mtp --spec-k 2
python3 - "$O/ids-solo-off.json" "$O/ids-mtp8-default.json" <<'PY' | tee -a $O/ids.txt
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
same = [x == y for x, y in zip(a, b)]
print(sys.argv[2].split('/')[-1], "identical per sequence:", same, "ALL" if all(same) and len(a) == len(b) else "MISMATCH")
PY
echo "== done $(date +%T)"
