#!/usr/bin/env bash
# Window script (runs on spark1 INSIDE scripts/cuda-decode/window.sh, vLLM down): nsys decode-step profile of one or more
# builds of the 35B (graph decode, 160 greedy tokens), summarised per step by nsys_steps.py.
#   w-profile.sh <name:tree[:ENV=VAL,ENV=VAL]> ...      trees under ~/intarsia-decfuse (base, dev); output ~/intarsia-decfuse/out/prof-<name>
#   PLAIN=1 also runs the build once without nsys (the engine's own decode figure)
set -u
P=$HOME/intarsia-decfuse
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 60 )) || { echo "ABORT: memory"; exit 1; }
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv
PROMPT='Write a long story about a robot who learns to paint.'
for spec in "$@"; do
  name=${spec%%:*}; rest=${spec#*:}; T=${rest%%:*}; envs=""; [[ "$rest" == *:* ]] && envs=${rest#*:}; envs=${envs//,/ }
  BIN=$P/$T/target/acc/release/intarsia
  O=$P/out/prof-$name; rm -rf $O; mkdir -p $O
  GEN="env $envs $BIN generate -m $M --backend cuda --mem-budget 29 --max-seq 4096 --chat --no-think --profile-every 0 --ignore-eos -p '$PROMPT' --max-tokens 160"
  if [ "${PLAIN:-0}" = 1 ]; then
    echo "== $name: plain run (graphs on) $(date +%T)"
    bash -c "$GEN" 2>&1 >/dev/null | grep -aE "^finish=|^steps=" | cut -c1-230
  fi
  echo "== $name: nsys $(date +%T)"
  nsys profile -t cuda --cuda-graph-trace=node --force-overwrite true -o $O/nsys bash -c "$GEN" > $O/nsys.out 2> $O/nsys.err
  grep -aE "^finish=|^steps=" $O/nsys.err | cut -c1-230
  nsys stats -r cuda_gpu_trace -f csv --force-export=true -o $O/trace $O/nsys.nsys-rep > $O/stats.log 2>&1 || tail -3 $O/stats.log
  python3 "$(dirname "$0")/nsys_steps.py" $O/trace_cuda_gpu_trace.csv --json $O/steps.json | tee $O/steps.txt
  rm -f $O/nsys.sqlite
done
echo "== done $(date +%T)"
