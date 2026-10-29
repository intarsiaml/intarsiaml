#!/usr/bin/env bash
# Lane srv16 (from scripts/moe16b/ident.sh): fixed-seed served-token identity of two arms, both `serve --max-batch 16`:
# scripts/htk/equal.py (7 sampling settings x 1/4/16 streams, 64 tokens) and scripts/srv16/ident_long.py (300..1100-token prompts,
# 4 and 16 at once, so prompts prefill together). Inside a fleetq slot on spark1 (not a timing run):
#   fleetq run --host spark1 --mem 35 --expect 15 --label "srv16: served-token identity" -- scripts/srv16/ident.sh <arm-a> <arm-b> [serve args]
# arm = label:bin[:ENV=V,ENV=V]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
A=$1; B=$2; shift 2; XA=${*:---max-batch 16}
HOST=spark1; HN=spark-4e77; PORT=8191
RAW=/opt/code/intarsiaml/.worktrees/srv16/docs/results/2026-10-28-srv16-serve-gap/raw/${IDENT_OUT:-ident}; mkdir -p $RAW
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$H/../.."
SPID=""
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
stop_server() { [[ -n "$SPID" ]] && { rsh "kill $SPID 2>/dev/null; for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null; sleep 2"; SPID=""; }; }
trap 'stop_server' EXIT INT TERM
for spec in $A $B; do
  IFS=: read -r lab bin envs <<< "$spec"; envs=${envs//,/ }
  SPID=$(rsh "mkdir -p ~/intarsiaml-srv16/out; nohup env $envs ~/intarsiaml-srv16/bin/$bin serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name intarsia --max-seq 8192 --mem-budget 40 --profile-every 0 $XA > ~/intarsiaml-srv16/out/ident-$lab.log 2>&1 < /dev/null & echo \$!")
  up=0; for i in $(seq 1 150); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; sleep 2; done
  [[ $up == 1 ]] || { echo "server $lab did not come up"; stop_server; exit 1; }
  python3 "$REPO/scripts/htk/equal.py" run --url http://$HN:$PORT/v1 --out $RAW/$lab.json > $RAW/$lab.run.log 2>&1
  python3 "$H/ident_long.py" run --url http://$HN:$PORT/v1 --corpus "$REPO/tools/prompts/corpus_docs.txt" --out $RAW/$lab-long.json > $RAW/$lab-long.run.log 2>&1
  tail -1 $RAW/$lab.run.log; tail -1 $RAW/$lab-long.run.log
  curl -s "http://$HN:$PORT/metrics" | grep -E "^intarsia_batch_(iterations|decode_steps|prefill|mixed|step_ewma)" > $RAW/$lab.metrics.txt
  cat $RAW/$lab.metrics.txt
  stop_server
done
la=${A%%:*}; lb=${B%%:*}; rc=0
python3 "$REPO/scripts/htk/equal.py" compare $RAW/$la.json $RAW/$lb.json > $RAW/compare.log; rc=$?
python3 "$H/ident_long.py" compare $RAW/$la-long.json $RAW/$lb-long.json > $RAW/compare-long.log; rc=$((rc + $?))
tail -1 $RAW/compare.log; tail -1 $RAW/compare-long.log
exit $rc
