#!/bin/bash
# Shared launch lines. Source me. Every server is started via start_server and stopped (whole tree) by stop_server / EXIT / TERM.
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
B=/opt/code/intarsiaml/.worktrees/mlx-serve-bench
MLX=/opt/code/intarsiaml/.worktrees/mlx-serve-src/zig-out/bin/mlx-serve
INT=$B/intarsia/target/release/intarsia
PM=18101   # mlx-serve
PI=18102   # intarsia
SRV_PID=""
term_tree() { local p=$1 c; for c in $(pgrep -P $p 2>/dev/null); do term_tree $c; done; /bin/kill -TERM $p 2>/dev/null; }
stop_server() {
  if [ -n "$SRV_PID" ]; then
    term_tree $SRV_PID
    for i in $(seq 1 60); do /bin/kill -0 $SRV_PID 2>/dev/null || break; sleep 1; done
    /bin/kill -9 $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null
  fi
  SRV_PID=""
}
trap 'stop_server' EXIT
trap 'stop_server; for c in $(pgrep -P $$); do term_tree $c; done; exit 143' TERM INT
# arm command lines (exact strings recorded in RESULTS.md)
cmd_A() { echo "$MLX --model $CK --serve --host 127.0.0.1 --port $PM --ctx-size 20480"; }                       # draft head on (default)
cmd_C() { echo "$MLX --model $B/model-nodraft --serve --host 127.0.0.1 --port $PM --ctx-size 20480 --no-pld"; }          # draft head absent
cmd_B() { echo "$INT serve --model $CK --backend metal --max-batch 1 --max-seq 20480 --port $PI --served-model-name inkling-2-mini --spec dflash --spec-draft $CK --spec-k 3"; }
cmd_D() { echo "$INT serve --model $CK --backend metal --max-batch 1 --max-seq 20480 --port $PI --served-model-name inkling-2-mini --spec off"; }
cmd_Bc() { echo "$INT serve --model $CK --backend metal --max-batch 16 --max-seq 8192 --port $PI --served-model-name inkling-2-mini --spec dflash --spec-draft $CK --spec-k 3"; }
cmd_Dc() { echo "$INT serve --model $CK --backend metal --max-batch 16 --max-seq 8192 --port $PI --served-model-name inkling-2-mini --spec off"; }
port_of() { case $1 in A|C) echo $PM;; *) echo $PI;; esac; }
model_id_of() { curl -s -m 5 http://127.0.0.1:$1/v1/models 2>/dev/null | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["data"][0]["id"])
except Exception: print("inkling-2-mini")'; }
ready_probe() {  # mlx-serve answers /v1/models with 200 before its weights are in; a 1-token chat call succeeds only when it can serve
  local r; r=$(curl -s -m 20 http://127.0.0.1:$1/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"$(model_id_of $1)\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"temperature\":0}" 2>/dev/null)
  case "$r" in *'"choices"'*) return 0;; *) return 1;; esac
}
start_server() {  # $1 = arm key (A B C D Bc Dc), $2 = log file
  local arm=$1 log=$2 port; port=$(port_of ${arm:0:1})
  stop_server
  if curl -s -m 2 http://127.0.0.1:$port/v1/models >/dev/null 2>&1; then echo "port $port already serving; abort"; return 1; fi
  echo "$(cmd_$arm)" > $log.cmd
  $(cmd_$arm) > $log 2>&1 &
  SRV_PID=$!
  for i in $(seq 1 240); do
    /bin/kill -0 $SRV_PID 2>/dev/null || { echo "server died"; tail -20 $log; return 1; }
    ready_probe $port && { echo "ready after ~$((i*2))s"; return 0; }
    sleep 2
  done
  echo "not ready in 6 min"; return 1
}
