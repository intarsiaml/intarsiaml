#!/bin/zsh
# Final clock-aware 4-engine same-slot: 3 rounds x 4 engines, rotated order, one exclusive fleetq slot, macmon 100 ms throughout.
B=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/final
SAFEBIN=$B/intarsia-safe
CANDBIN=$B/intarsia-cand
OUT=/opt/code/intarsiaml/docs/results/raw/2026-10-15-metal-final-clock
P=$OUT/passes
MODEL=/Users/josef/models/Qwen3-8B
GGUF=/Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf
export INTARSIA_CPU_FREE_MEM_GB=60
export PYTHONUNBUFFERED=1
mkdir -p $P
J=$(fleetq acquire --host endor --exclusive --mem 40 --expect 60 --label "metal final clock-aware 4-engine" --wait --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "fleetq job $J" | tee $OUT/fleetq-job.txt
SPID=""; MPID=""
cleanup() { [[ -n "$SPID" ]] && kill $SPID 2>/dev/null; [[ -n "$MPID" ]] && kill $MPID 2>/dev/null; fleetq release $J; }
trap cleanup EXIT
export FLEETQ_JOB=$J

{
  echo "date: $(date -Iseconds)"; echo "fleetq job: $FLEETQ_JOB"
  echo "safe head: $(git -C /opt/code/intarsiaml-final-safe rev-parse HEAD)"; echo "cand head: $(git -C /opt/code/intarsiaml-final-cand rev-parse HEAD)"
  echo "safe binary: $(ls -l $SAFEBIN)"; echo "cand binary: $(ls -l $CANDBIN)"
  shasum -a 256 $SAFEBIN $CANDBIN
  /Users/josef/.local/share/uv/tools/mlx-lm/bin/python -c "import importlib.metadata as m; print({p: m.version(p) for p in ['mlx','mlx-lm','mlx-metal']})"
  /opt/homebrew/bin/llama-server --version 2>&1 | head -3
  echo "localeval: $(grep -m1 '^version' /opt/code/deepeval/pyproject.toml)"; echo "deepeval head: $(git -C /opt/code/deepeval rev-parse HEAD)"
  sw_vers | tr '\n' ' '; echo; /Users/josef/bin/macmon --version 2>&1
} > $OUT/versions.txt 2>&1

/Users/josef/bin/macmon pipe -i 100 > $OUT/macmon.jsonl 2> $OUT/macmon.err &
MPID=$!

gpu_check() { # $1 = label
  local f=$P/$1-gpu-idle.txt; : > $f
  local tries=0 med
  echo "zoom before: $(pgrep -fl zoom.us | tr '\n' ';')" >> $f
  while true; do
    echo "== sample set $(date +%s)" >> $f
    local vals=()
    for i in 1 2 3; do
      v=$(ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*' | head -1)
      echo "$v" >> $f; vals+=(${v##*=}); sleep 1
    done
    med=${${(on)vals}[2]}
    echo "median=$med" >> $f
    if (( med <= 10 )) || (( tries >= 12 )); then break; fi
    echo "idle util $med > 10, waiting 10s (try $tries)" >> $f; tries=$((tries+1)); sleep 10
  done
  echo "FINAL median=$med tries=$tries" >> $f
  tail -3 $f
}

wait_port_closed() { for i in {1..30}; do lsof -iTCP:$1 -sTCP:LISTEN -nP >/dev/null 2>&1 || return 0; sleep 1; done; return 1; }

run_engine() { # label port health-path modelid cmd...
  local label=$1 port=$2 hp=$3 mid=$4; shift 4
  echo "##### $label  $(date -Iseconds)  epoch $(date +%s)"
  echo "$label server_start $(date +%s.%N)" >> $OUT/events.log
  "$@" > $P/$label-server.out 2> $P/$label-server.err &
  SPID=$!
  local up=0
  for i in {1..180}; do
    curl -sf http://127.0.0.1:$port$hp >/dev/null 2>&1 && { up=1; break; }
    kill -0 $SPID 2>/dev/null || break
    sleep 1
  done
  [[ $up == 1 ]] || { echo "SERVER FAILED $label"; echo "$label SERVER_FAILED" >> $OUT/events.log; SPID=""; return 1; }
  echo "server up after ${i}s"
  echo "chunk line: $(grep -h 'prefill chunk' $P/$label-server.err | head -1)"
  gpu_check $label
  echo "$label localeval_start $(date +%s.%N)" >> $OUT/events.log
  ( cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$port/v1 --model $mid --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label $label 2>&1 ) | python3 $OUT/ts.py > $P/$label-localeval.ts
  echo "$label localeval_end $(date +%s.%N)" >> $OUT/events.log
  grep -E "rep [0-9]/3|calibrated|marginal|run_id|Run |results/" $P/$label-localeval.ts | tail -14
  kill $SPID 2>/dev/null
  for i in {1..30}; do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  kill -0 $SPID 2>/dev/null && { kill -9 $SPID; sleep 1; }
  wait $SPID 2>/dev/null
  { wait_port_closed $port && echo "port $port closed" || echo "PORT $port STILL OPEN"; } | tee -a $P/$label-stop.txt
  { kill -0 $SPID 2>/dev/null && echo "PID $SPID STILL ALIVE" || echo "pid $SPID gone"; } | tee -a $P/$label-stop.txt
  SPID=""
  sleep 5
}

SAFE=($SAFEBIN serve --model $MODEL --backend metal --device 0 --host 127.0.0.1 --port 8093 --served-model-name intarsia-metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui)
CAND=($CANDBIN serve --model $MODEL --backend metal --device 0 --host 127.0.0.1 --port 8093 --served-model-name intarsia-metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui)
MLX=(/Users/josef/.local/bin/mlx_lm.server --model $MODEL --port 8092 --chat-template-args '{"enable_thinking":false}' --prompt-cache-size 1)
LLA=(/opt/homebrew/bin/llama-server -m $GGUF -ngl 999 -fa on -c 8192 --cache-ram 0 --port 8091)

run_one() { # engine round
  case $1 in
    safe)  run_engine clk-safe-r$2 8093 /health intarsia-metal $SAFE ;;
    cand)  run_engine clk-cand-r$2 8093 /health intarsia-metal $CAND ;;
    mlx)   run_engine clk-mlx-lm-r$2 8092 /v1/models $MODEL $MLX ;;
    llama) run_engine clk-llamacpp-r$2 8091 /health $GGUF $LLA ;;
  esac
}
for r in 1 2 3; do
  case $r in 1) order=(safe cand mlx llama);; 2) order=(mlx llama safe cand);; 3) order=(cand llama safe mlx);; esac
  { echo "== round $r start $(date +%s.%N)"; pmset -g therm; } >> $P/therm.txt
  echo "ROUND $r ${order[*]}"
  for e in $order; do run_one $e $r; done
  { echo "== round $r end $(date +%s.%N)"; pmset -g therm; } >> $P/therm.txt
done
echo "ALLDONE $(date -Iseconds)"
