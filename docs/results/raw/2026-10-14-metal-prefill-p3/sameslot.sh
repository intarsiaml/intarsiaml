#!/bin/zsh
# Metal P3 final same-slot measurement: intarsia (P3 build) -> mlx-lm -> llama.cpp -> intarsia, one exclusive fleetq slot.
# Copy of docs/results/raw/2026-10-14-metal-phase-b/sameslot.sh (same server commands, ports, localeval args, GPU-idle/zoom
# recording); only the output dir, the intarsia binary and the version/chunk records differ.
# Runs inside one exclusive fleetq slot (the job id comes from the environment, FLEETQ_JOB).
OUT=/opt/code/intarsiaml/.claude/worktrees/agent-af7ec7dd89422c6a9/docs/results/raw/2026-10-14-metal-prefill-p3/sameslot
WT=/opt/code/intarsiaml/.claude/worktrees/agent-af7ec7dd89422c6a9
BIN=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/p3/intarsia-p3
MODEL=/Users/josef/models/Qwen3-8B
GGUF=/Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf
mkdir -p $OUT
SPID=""
cleanup() { [[ -n "$SPID" ]] && kill $SPID 2>/dev/null; }
trap cleanup EXIT

{
  echo "date: $(date -Iseconds)"; echo "fleetq job: $FLEETQ_JOB"
  echo "worktree HEAD: $(git -C $WT rev-parse HEAD)"
  echo "intarsia binary: $(ls -l $BIN)"
  /Users/josef/.local/share/uv/tools/mlx-lm/bin/python -c "import importlib.metadata as m; print({p: m.version(p) for p in ['mlx','mlx-lm','mlx-metal']})"
  /opt/homebrew/bin/llama-server --version 2>&1 | head -3
  sw_vers | tr '\n' ' '; echo
} > $OUT/versions.txt 2>&1

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
  echo "##### $label"
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
  ( cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$port/v1 --model $mid --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label $label ) > $OUT/$label-localeval.out 2> $OUT/$label-localeval.err
  echo "localeval exit $?"
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

INT=($BIN serve --model $MODEL --backend metal --device 0 --host 127.0.0.1 --port 8093 --served-model-name intarsia-metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui)
MLX=(/Users/josef/.local/bin/mlx_lm.server --model $MODEL --port 8092 --chat-template-args '{"enable_thinking":false}' --prompt-cache-size 1)
LLA=(/opt/homebrew/bin/llama-server -m $GGUF -ngl 999 -fa on -c 8192 --cache-ram 0 --port 8091)

run_engine sameslot-intarsia-1 8093 /health intarsia-metal $INT
run_engine sameslot-mlx-lm 8092 /v1/models $MODEL $MLX
run_engine sameslot-llamacpp 8091 /health $GGUF $LLA
run_engine sameslot-intarsia-2 8093 /health intarsia-metal $INT
echo ALLDONE
