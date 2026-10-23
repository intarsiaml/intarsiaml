#!/usr/bin/env bash
# One measurement pass on spark1, run on endor INSIDE an exclusive fleetq slot (vLLM stays up; nothing is taken down):
#   fleetq run --host spark1 --exclusive --mem 32 --expect 50 --label "..." -- \
#     env ARMS="off mtp1 mtp2 mtp3" PASS=p1 docs/results/2026-10-22-mtp-spec-decode/scripts/window.sh
# Each arm: start `intarsia serve` (tree ~/intarsia-deploy/mtp), localeval speed 128/1024/4096 x 3 reps x 128 tokens,
# scrape the server's spec counters, stop the server.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ARMS=${ARMS:-"off mtp1 mtp2 mtp3"}; PASS=${PASS:-p1}; REF=${REF:-?}; SIZES=${SIZES:-128,1024,4096}; REPS=${REPS:-3}
OUT=${OUT:-$REPO/docs/results/2026-10-22-mtp-spec-decode/raw/speed}; mkdir -p "$OUT"
HOST=${HOST:-spark1}; MACHINE=${MACHINE:-spark-4e77}; PORT=8191; NAME=intarsia-35b
BIN='$HOME/intarsia-deploy/mtp/target/release/intarsia'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
stop() { rsh 'p=$(cat $HOME/mtp-run/server.pid 2>/dev/null); [ -n "$p" ] && kill $p 2>/dev/null; for i in $(seq 1 40); do kill -0 $p 2>/dev/null || break; sleep 1; done; kill -9 $p 2>/dev/null; rm -f $HOME/mtp-run/server.pid' || true; }
trap stop EXIT INT TERM
rsh 'mkdir -p $HOME/mtp-run'
for arm in $ARMS; do
  # composer lanes: a speculative run says "spec" in its label; the off arm is plain decode
  case $arm in off) SPEC="--spec off"; LABEL="mtp-lane-plain-off-$PASS";; mtp*) SPEC="--spec mtp --spec-k ${arm#mtp}"; LABEL="spec-mtp-k${arm#mtp}-$PASS";; *) echo "bad arm $arm"; exit 2;; esac
  rsh "M=\$(ls -d \$HOME/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1); nohup $BIN serve --backend cuda --model \$M --host 0.0.0.0 --port $PORT --served-model-name $NAME --max-seq 20480 --mem-budget 30 --profile-every 0 --no-supervisor --no-ui $SPEC > \$HOME/mtp-run/serve-$PASS-$arm.log 2>&1 < /dev/null & echo \$! > \$HOME/mtp-run/server.pid"
  ok=0; for i in $(seq 1 60); do curl -sf -m 5 "http://$MACHINE:$PORT/v1/models" > /dev/null 2>&1 && { ok=1; break; }; sleep 5; done
  if [[ $ok != 1 ]]; then echo "$arm: server did not come up"; rsh "tail -20 \$HOME/mtp-run/serve-$PASS-$arm.log"; stop; continue; fi
  (cd /opt/code/deepeval && FLEETQ_JOB=$FLEETQ_JOB uv run localeval speed --url "http://$MACHINE:$PORT/v1" --model $NAME --thinking off \
     --sizes "$SIZES" --gen 128 --reps "$REPS" --label "$LABEL" --recipe "intarsia serve --backend cuda --max-seq 20480 --mem-budget 30 --profile-every 0 $SPEC (lane-mtp $REF)" 2>&1) | tee "$OUT/localeval-$PASS-$arm.txt" | grep -E "^\|.*[0-9]|run id|result|unanimis" | head -20
  curl -s -m 5 "http://$MACHINE:$PORT/metrics" | grep -E "^intarsia_spec" | tee "$OUT/metrics-$PASS-$arm.txt"
  stop
  if [[ $arm == mtp* ]] && ! grep -qE '^intarsia_spec_steps_total [1-9]' "$OUT/metrics-$PASS-$arm.txt"; then
    echo "$arm: no verify passes ran (intarsia_spec_steps_total 0): the arm measured plain decode; aborting the pass"
    rsh "cat \$HOME/mtp-run/serve-$PASS-$arm.log" > "$OUT/serve-$PASS-$arm.log" 2>&1
    exit 3
  fi
  rsh "cat \$HOME/mtp-run/serve-$PASS-$arm.log" > "$OUT/serve-$PASS-$arm.log" 2>&1
done
