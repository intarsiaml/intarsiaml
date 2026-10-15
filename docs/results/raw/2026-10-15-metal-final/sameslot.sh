#!/bin/zsh
# Metal FINAL same-slot measurement: SAFE -> CAND -> mlx-lm -> llama.cpp -> CAND -> SAFE in ONE exclusive fleetq slot.
# Derived from docs/results/raw/2026-10-14-metal-prefill-p3/sameslot.sh (same server commands, ports, localeval args,
# GPU-idle/zoom recording). SAFE = spike/metal a2102bf (chunk 256), CAND = metal-candidate2 (SAFE + metal-p3-chunk, chunk 2048).
B=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/final
SAFEBIN=$B/intarsia-safe
CANDBIN=$B/intarsia-cand
WT=/opt/code/intarsiaml-final-cand
OUT=$WT/docs/results/raw/2026-10-15-metal-final/sameslot
MODEL=/Users/josef/models/Qwen3-8B
GGUF=/Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $OUT
J=$(fleetq acquire --host endor --exclusive --mem 40 --expect 60 --label "metal final same-slot" --wait --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "fleetq job $J" | tee $OUT/fleetq-job.txt
SPID=""
cleanup() { [[ -n "$SPID" ]] && kill $SPID 2>/dev/null; fleetq release $J; }
trap cleanup EXIT
export FLEETQ_JOB=$J

{
  echo "date: $(date -Iseconds)"; echo "fleetq job: $FLEETQ_JOB"
  echo "cand head: $(git -C $WT rev-parse HEAD)"; echo "safe head: $(git -C /opt/code/intarsiaml-final-safe rev-parse HEAD)"
  echo "safe binary: $(ls -l $SAFEBIN)"; echo "cand binary: $(ls -l $CANDBIN)"
  /Users/josef/.local/share/uv/tools/mlx-lm/bin/python -c "import importlib.metadata as m; print({p: m.version(p) for p in ['mlx','mlx-lm','mlx-metal']})"
  /opt/homebrew/bin/llama-server --version 2>&1 | head -3
  echo "localeval: $(grep -m1 '^version' /opt/code/deepeval/pyproject.toml)"
  echo "deepeval head: $(git -C /opt/code/deepeval rev-parse HEAD)"
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

SAFE=($SAFEBIN serve --model $MODEL --backend metal --device 0 --host 127.0.0.1 --port 8093 --served-model-name intarsia-metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui)
CAND=($CANDBIN serve --model $MODEL --backend metal --device 0 --host 127.0.0.1 --port 8093 --served-model-name intarsia-metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui)
MLX=(/Users/josef/.local/bin/mlx_lm.server --model $MODEL --port 8092 --chat-template-args '{"enable_thinking":false}' --prompt-cache-size 1)
LLA=(/opt/homebrew/bin/llama-server -m $GGUF -ngl 999 -fa on -c 8192 --cache-ram 0 --port 8091)

run_engine final-safe-1 8093 /health intarsia-metal $SAFE
run_engine final-cand-1 8093 /health intarsia-metal $CAND
run_engine final-mlx-lm 8092 /v1/models $MODEL $MLX
run_engine final-llamacpp 8091 /health $GGUF $LLA
run_engine final-cand-2 8093 /health intarsia-metal $CAND
run_engine final-safe-2 8093 /health intarsia-metal $SAFE
echo ALLDONE
