#!/usr/bin/env bash
# Lane msd: fixed-seed served-token identity, base vs dev, both `serve --max-batch 16` (scripts/htk/equal.py: 7 sampling settings x 1/4/16 streams,
# seeds 100+i, 64 tokens each, token ids compared). Inside a fleetq slot on spark1 (not a timing run):
#   fleetq run --host spark1 --mem 35 --expect 15 --label "msd: served-token identity" -- scripts/moe16b/ident.sh <bin-a> <bin-b> [serve args]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
A=$1; B=$2; shift 2; XA=${*:---max-batch 16}
HOST=spark1; HN=spark-4e77; PORT=8191
RAW=/opt/code/intarsiaml/.worktrees/moe16b/docs/results/2026-10-28-moe16b-step-overlap/raw/ident; mkdir -p $RAW
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
SPID=""
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
stop_server() { [[ -n "$SPID" ]] && { rsh "kill $SPID 2>/dev/null; for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null; sleep 2"; SPID=""; }; }
trap 'stop_server' EXIT INT TERM
for bin in $A $B; do
  SPID=$(rsh "mkdir -p ~/intarsiaml-moe16b/out; nohup ~/intarsiaml-moe16b/bin/$bin serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name intarsia --max-seq 8192 --mem-budget 40 --profile-every 0 $XA > ~/intarsiaml-moe16b/out/ident-$bin.log 2>&1 < /dev/null & echo \$!")
  up=0; for i in $(seq 1 150); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; sleep 2; done
  [[ $up == 1 ]] || { echo "server $bin did not come up"; stop_server; exit 1; }
  python3 scripts/htk/equal.py run --url http://$HN:$PORT/v1 --out $RAW/$bin.json > $RAW/$bin.run.log 2>&1
  tail -2 $RAW/$bin.run.log
  stop_server
done
python3 scripts/htk/equal.py compare $RAW/$A.json $RAW/$B.json | tee $RAW/compare.log | tail -3
