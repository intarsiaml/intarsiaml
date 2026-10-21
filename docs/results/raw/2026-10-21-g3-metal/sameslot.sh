#!/bin/zsh
# G3 step 1: same-slot re-measurement of origin/main 4be7b7d4 (intarsia) vs mlx-lm vs llama.cpp, ABBA (intarsia, mlx-lm,
# llama.cpp, intarsia), sizes 128/1k/4k/16k, gen 256, 3 reps, localeval speed, macmon clock log at 250 ms.
# Derived from docs/results/raw/2026-10-18-metal-sameslot-main/sameslot.sh (same server flags, ports, localeval args),
# plus 16k (max-seq 20000 / llama.cpp -c 20480) and a doctor copy-bandwidth probe before each arm.
# Run inside the slot:  fleetq run --host endor --exclusive --mem 24 --label "..." -- zsh sameslot.sh <tag>
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
TAG=${1:-s1}
S=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad
BIN=${BIN:-$S/intarsia-base}
OUT=/opt/code/intarsiaml/.claude/worktrees/agent-a34836167f72ea687/docs/results/raw/2026-10-21-g3-metal/$TAG
MODEL=/Users/josef/models/Qwen3-8B
GGUF=/Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf
SIZES=${SIZES:-128,1024,4096,16384}
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $OUT
SPID=""; MPID=""
cleanup() { [[ -n "$SPID" ]] && kill $SPID 2>/dev/null; [[ -n "$MPID" ]] && kill $MPID 2>/dev/null; }
trap cleanup EXIT
{
  echo "date: $(date -Iseconds)"; echo "fleetq job: $FLEETQ_JOB"
  echo "binary: $(ls -l $BIN)"; echo "binary md5: $(md5 -q $BIN)"
  /Users/josef/.local/share/uv/tools/mlx-lm/bin/python -c "import importlib.metadata as m; print({p: m.version(p) for p in ['mlx','mlx-lm','mlx-metal']})"
  /opt/homebrew/bin/llama-server --version 2>&1 | head -3
  echo "localeval: $(grep -m1 '^version' /opt/code/deepeval/pyproject.toml)"; echo "deepeval head: $(git -C /opt/code/deepeval rev-parse HEAD)"
  sw_vers | tr '\n' ' '; echo
  echo "top cpu:"; top -l 1 -o cpu -n 6 -stats pid,command,cpu | tail -7
  echo "memory_pressure: $(memory_pressure | tail -1)"; sysctl vm.swapusage
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
  tail -1 $f
}

wait_port_closed() { for i in {1..30}; do lsof -iTCP:$1 -sTCP:LISTEN -nP >/dev/null 2>&1 || return 0; sleep 1; done; return 1; }

run_engine() { # label port health-path modelid cmd...
  local label=$1 port=$2 hp=$3 mid=$4; shift 4
  echo "##### $label  $(date -Iseconds)"
  $BIN doctor --backend metal --json --bw-mib 1024 > $OUT/$label-doctor.json 2>/dev/null
  "$@" > $OUT/$label-server.out 2> $OUT/$label-server.err &
  SPID=$!
  local up=0
  for i in {1..240}; do
    curl -sf http://127.0.0.1:$port$hp >/dev/null 2>&1 && { up=1; break; }
    kill -0 $SPID 2>/dev/null || break
    sleep 1
  done
  [[ $up == 1 ]] || { echo "SERVER FAILED $label"; SPID=""; return 1; }
  echo "server up after ${i}s; chunk line: $(grep -h 'prefill chunk' $OUT/$label-server.err | head -1)"
  gpu_check $label
  /Users/josef/bin/macmon pipe -i 250 > $OUT/$label-macmon.jsonl 2>/dev/null & MPID=$!
  ( cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$port/v1 --model $mid --sizes $SIZES --gen 256 --reps 3 --thinking off --label g3-$TAG-$label ) > $OUT/$label-localeval.out 2> $OUT/$label-localeval.err
  echo "localeval exit $?"
  kill $MPID 2>/dev/null; wait $MPID 2>/dev/null; MPID=""
  tail -12 $OUT/$label-localeval.out
  kill $SPID 2>/dev/null
  for i in {1..30}; do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  kill -0 $SPID 2>/dev/null && { kill -9 $SPID; sleep 1; }
  wait $SPID 2>/dev/null
  wait_port_closed $port && echo "port $port closed" || echo "PORT $port STILL OPEN"
  SPID=""
  sleep 5
}

MAIN=($BIN serve --model $MODEL --backend metal --device 0 --host 127.0.0.1 --port 8093 --served-model-name intarsia-metal --max-seq 20000 --profile-every 0 --no-supervisor --no-ui)
MLX=(/Users/josef/.local/bin/mlx_lm.server --model $MODEL --port 8092 --chat-template-args '{"enable_thinking":false}' --prompt-cache-size 1)
LLA=(/opt/homebrew/bin/llama-server -m $GGUF -ngl 999 -fa on -c 20480 --cache-ram 0 --port 8091)

run_engine main-1 8093 /health intarsia-metal $MAIN
run_engine mlx-lm 8092 /v1/models $MODEL $MLX
run_engine llamacpp 8091 /health $GGUF $LLA
run_engine main-2 8093 /health intarsia-metal $MAIN
echo ALLDONE $(date -Iseconds)
