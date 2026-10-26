#!/usr/bin/env bash
# Lane sk2: 256 greedy ids through `intarsia serve --max-batch N` (batched scheduler), per arm; the first arm is the reference.
#   fleetq run --host spark2 --mem 45 --expect 25 --label "sk2: ids" -- scripts/sk/ids-serve.sh <outname> <arm> ...   arm = name:tree[:ENV=V,ENV=V]
# Each arm: a fresh server (--max-seq 20480 --mem-budget 45 --max-batch ${SLOTS:-4}), ids-serve.py alone per size, then all sizes at once.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
OUTN=$1; shift; ARMS=("$@"); HOST=${HOST:-spark2}; HN=${HN:-spark-fa00}; PORT=${PORT:-8191}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW=/opt/code/intarsiaml/.worktrees/sk2/docs/results/2026-10-25-sk2-serve-macro/raw/$OUTN; mkdir -p "$RAW"
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
SPID=""
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
stop_server() { [[ -n "$SPID" ]] && { rsh "kill $SPID 2>/dev/null; for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null; sleep 2"; SPID=""; }; }
trap 'stop_server' EXIT INT TERM
for a in "${ARMS[@]}"; do
  IFS=: read -r name tree envs <<< "$a"; envs=${envs//,/ }
  mkdir -p /dev/null 2>/dev/null; rsh "mkdir -p ~/intarsiaml-sk-$tree/out"
  SPID=$(rsh "nohup env $envs ~/intarsiaml-sk-$tree/target/acc/release/intarsia serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name intarsia-35b --max-seq 20480 --mem-budget 45 --profile-every 0 --max-batch ${SLOTS:-4} > ~/intarsiaml-sk-$tree/out/ids-serve-$name.log 2>&1 < /dev/null & echo \$!")
  up=0; for i in $(seq 1 150); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; sleep 2; done
  [[ $up == 1 ]] || { echo "server $name did not come up"; rsh "tail -5 ~/intarsiaml-sk-$tree/out/ids-serve-$name.log"; stop_server; continue; }
  python3 "$HERE/ids-serve.py" "http://$HN:$PORT" "$RAW/$name" 2>&1 | tee "$RAW/$name.log"
  stop_server
done
ref=${ARMS[0]%%:*}
for a in "${ARMS[@]:1}"; do n=${a%%:*}; for tag in alone conc; do for f in "$RAW/$ref-$tag-"*.ids; do c=${f##*-}
  if cmp -s "$f" "$RAW/$n-$tag-$c" && [[ -s $f ]]; then echo "IDS $tag ${c%.ids} $ref vs $n: identical"; else echo "IDS $tag ${c%.ids} $ref vs $n: DIFFER"; fi
done; done; done
