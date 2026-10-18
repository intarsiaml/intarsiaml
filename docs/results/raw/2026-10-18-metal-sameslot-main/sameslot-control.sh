#!/bin/zsh
# Metal same-slot re-measurement of origin/main 5eabb1a7: intarsia -> mlx-lm -> llama.cpp -> intarsia (ABBA) in ONE exclusive fleetq slot.
# Derived from docs/results/raw/2026-10-15-metal-final/sameslot.sh (same server commands, ports, localeval args, GPU-idle/zoom
# recording; paths adapted). intarsia = origin/main 5eabb1a7 built from a detached worktree, serve defaults (Metal chunk 2048 from recipes/qwen3.toml).
B=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad
BIN=$B/intarsia-main
CANDBIN=$B/final/intarsia-cand
WT=/opt/code/intarsiaml
OUT=$WT/docs/results/raw/2026-10-18-metal-sameslot-main
MODEL=/Users/josef/models/Qwen3-8B
GGUF=/Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $OUT
J=$(fleetq acquire --host endor --exclusive --mem 40 --expect 60 --label "intarsiaml main 5eabb1a7 vs 10-15 CAND control same-slot" --wait --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "fleetq job $J" | tee $OUT/fleetq-job-control.txt
SPID=""
cleanup() { [[ -n "$SPID" ]] && kill $SPID 2>/dev/null; fleetq release $J; }
trap cleanup EXIT
export FLEETQ_JOB=$J

{
  echo "date: $(date -Iseconds)"; echo "fleetq job: $FLEETQ_JOB"
  echo "main head: 5eabb1a7490243d1da888e785269be40d4d2eba6 (binary built from a detached worktree; md5 verified below)"
  echo "main binary: $(ls -l $BIN)"; echo "main md5: $(md5 -q $BIN)"; echo "cand binary: $(ls -l $CANDBIN)"; echo "cand md5: $(md5 -q $CANDBIN)"
  /Users/josef/.local/share/uv/tools/mlx-lm/bin/python -c "import importlib.metadata as m; print({p: m.version(p) for p in ['mlx','mlx-lm','mlx-metal']})"
  /opt/homebrew/bin/llama-server --version 2>&1 | head -3
  echo "localeval: $(grep -m1 '^version' /opt/code/deepeval/pyproject.toml)"
  echo "deepeval head: $(git -C /opt/code/deepeval rev-parse HEAD)"
  sw_vers | tr '\n' ' '; echo
} > $OUT/versions-control.txt 2>&1

gpu_check() { # $1 = label
  local f=$OUT/$1-gpu-idle.txt; : > $f
  local tries=0 med
  while true; do
    echo "== sample set $(date -Iseconds)" >> $f
    local vals=()
    for i in 1 2 3; do
      v=$(ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*' | head -1)
      echo "$v" >> $f; vals+=(${v##*=}); sleep 1
    done
    med=${${(on)vals}[2]}
    echo "median=$med" >> $f
    echo "zoom: $(pgrep -fl zoom.us | tr '\n' ';')" >> $f
    if (( med <= 10 )) || (( tries >= 12 )); then break; fi
    echo "idle util $med > 10, waiting 10s (try $tries)" >> $f; tries=$((tries+1)); sleep 10
  done
  echo "FINAL median=$med tries=$tries" >> $f
  cat $f
}

wait_port_closed() { # $1 port
  for i in {1..30}; do lsof -iTCP:$1 -sTCP:LISTEN -nP >/dev/null 2>&1 || return 0; sleep 1; done; return 1
}

run_engine() { # label port health-path modelid cmd...
  local label=$1 port=$2 hp=$3 mid=$4; shift 4
  echo "##### $label  $(date -Iseconds)"
  "$@" > $OUT/$label-server.out 2> $OUT/$label-server.err &
  SPID=$!
  local up=0
  for i in {1..180}; do
    curl -sf http://127.0.0.1:$port$hp >/dev/null 2>&1 && { up=1; break; }
    kill -0 $SPID 2>/dev/null || break
    sleep 1
  done
  [[ $up == 1 ]] || { echo "SERVER FAILED $label"; SPID=""; return 1; }
  echo "server up after ${i}s"
  echo "chunk line: $(grep -h 'prefill chunk' $OUT/$label-server.err | head -1)"
  gpu_check $label
  macmon pipe -i 1000 > $OUT/$label-macmon.jsonl 2>/dev/null & MPID=$!
  ( cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$port/v1 --model $mid --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label $label ) > $OUT/$label-localeval.out 2> $OUT/$label-localeval.err
  echo "localeval exit $?"
  kill $MPID 2>/dev/null; wait $MPID 2>/dev/null
  tail -30 $OUT/$label-localeval.out
  kill $SPID 2>/dev/null
  for i in {1..30}; do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  kill -0 $SPID 2>/dev/null && { kill -9 $SPID; sleep 1; }
  wait $SPID 2>/dev/null
  wait_port_closed $port && echo "port $port closed" || echo "PORT $port STILL OPEN"
  kill -0 $SPID 2>/dev/null && echo "PID $SPID STILL ALIVE" || echo "pid $SPID gone"
  SPID=""
  sleep 5
}

MAIN=($BIN serve --model $MODEL --backend metal --device 0 --host 127.0.0.1 --port 8093 --served-model-name intarsia-metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui)
CAND=($CANDBIN serve --model $MODEL --backend metal --device 0 --host 127.0.0.1 --port 8093 --served-model-name intarsia-metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui)

run_engine ctl-cand-1 8093 /health intarsia-metal $CAND
run_engine ctl-main-1 8093 /health intarsia-metal $MAIN
run_engine ctl-main-2 8093 /health intarsia-metal $MAIN
run_engine ctl-cand-2 8093 /health intarsia-metal $CAND
echo ALLDONE
