#!/bin/bash
# fleetq --load gpu job: start each arm, two greedy requests, grab metrics, stop. A/C: mlx-serve; Bc/Dc: intarsia at --max-batch 16 (boot check only).
. /opt/code/intarsiaml/.worktrees/mlx-serve-bench/arms.sh
O=$B/logs/smoke3; mkdir -p $O
P='Write a short story about a lighthouse keeper who finds a message in a bottle. It'
for arm in A B C D; do
  echo "=== arm $arm: $(cmd_$arm)"
  start_server $arm $O/server_$arm.log || { echo "arm $arm failed to start"; continue; }
  port=$(port_of $arm)
  m=$(model_id_of $port); echo "model id: $m"
  body=$(python3 -c "import json,sys;print(json.dumps({'model':sys.argv[2],'messages':[{'role':'user','content':sys.argv[1]}],'temperature':0,'max_tokens':64,'stream':False}))" "$P" "$m")
  curl -s -m 120 http://127.0.0.1:$port/v1/chat/completions -H 'Content-Type: application/json' -d "$body" > $O/resp_$arm.json
  curl -s -m 120 http://127.0.0.1:$port/v1/chat/completions -H 'Content-Type: application/json' -d "$body" > $O/resp2_$arm.json
  craw=$(python3 -c "import json,sys;print(json.dumps({'model':sys.argv[2],'prompt':sys.argv[1],'temperature':0,'max_tokens':40,'stream':False}))" "Once upon a time, in a lighthouse on a lonely rock, the keeper found a bottle with a message inside. The message said" "$m")
  curl -s -m 120 http://127.0.0.1:$port/v1/completions -H 'Content-Type: application/json' -d "$craw" > $O/raw_$arm.json
  curl -s -m 5 http://127.0.0.1:$port/metrics > $O/metrics_$arm.txt
  curl -s -m 5 http://127.0.0.1:$port/metrics.json > $O/metrics_$arm.json
  curl -s -m 5 http://127.0.0.1:$port/v1/models > $O/models_$arm.json
  ( for s in 1 2 3; do ps -o rss= -p $SRV_PID; sleep 1; done ) | sort -n | tail -1 | awk '{printf "rss_mb %d\n",$1/1024}'
  cp ~/.mlx-serve/logs/mlx-serve-$port.log $O/mlxlog_$arm.log 2>/dev/null
  stop_server
done
echo SMOKE_DONE
