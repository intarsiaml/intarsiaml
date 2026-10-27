#!/usr/bin/env bash
# G4X: nsys decode-step profile of Gemma-4-26B-A4B-NVFP4 on spark2 (run INSIDE a fleetq slot, --quiet gpu,cpu, --mem >= 32).
#   profile.sh <name> <binary> <prompt-bytes> [ENV=VAL ...]      e.g. profile.sh base1k bins/base 4000 INTARSIA_X=1
# The prompt is the first <prompt-bytes> bytes of tools/prompts/corpus_docs.txt (about 4 bytes per token), 64 greedy tokens after it;
# --cuda-graph-trace=node shows the graph's kernels. Output replay-out/prof/<name>/ (steps.txt: per-kernel table per decode step).
set -u
R=$HOME/intarsiaml-g4x; M=$HOME/models/Gemma-4-26B-A4B-NVFP4
name=$1; bin=$2; bytes=$3; shift 3
[[ $bin == /* ]] || bin=$R/replay-out/$bin
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 45 )) || { echo "ABORT: memory"; exit 1; }
O=$R/replay-out/prof/$name; rm -rf $O; mkdir -p $O
P=$(head -c $bytes $R/tools/prompts/corpus_docs.txt)
env "$@" nsys profile -t cuda --cuda-graph-trace=node --force-overwrite true -o $O/nsys \
  $bin generate -m $M --backend cuda --mem-budget 30 --max-seq 20480 --profile-every 0 --ignore-eos --max-tokens 64 -p "$P" > $O/gen.out 2> $O/gen.err
grep -aE "^finish=|^steps=|prompt" $O/gen.err | cut -c1-230 | head -5
nsys stats -r cuda_gpu_trace -f csv --force-export=true -o $O/trace $O/nsys.nsys-rep > $O/stats.log 2>&1 || tail -3 $O/stats.log
python3 $R/scripts/cuda-decode/nsys_steps.py $O/trace_cuda_gpu_trace.csv --json $O/steps.json --top 40 | tee $O/steps.txt
rm -f $O/nsys.sqlite $O/nsys.nsys-rep
