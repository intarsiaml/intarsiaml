#!/usr/bin/env bash
# G4X accuracy gate of Gemma-4-26B-A4B-NVFP4 against the committed HF f32 golden (golden/gemma4-26b-a4b-nvfp4-fp32.safetensors,
# 10 prompts), one build per call (run INSIDE a fleetq slot, --quiet gpu,cpu, --mem >= 32). Never makes a golden (that is HF f32
# on the box; it hung spark2 once). Output replay-out/acc/<name>.log; compare the verdict lines of two builds.
#   acc.sh <name> <binary> [ENV=VAL ...]
set -u
R=$HOME/intarsiaml-g4x; M=$HOME/models/Gemma-4-26B-A4B-NVFP4
name=$1; bin=$2; shift 2
[[ $bin == /* ]] || bin=$R/replay-out/$bin
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 45 )) || { echo "ABORT: memory"; exit 1; }
mkdir -p $R/replay-out/acc
ENVS=("$@"); [ ${#ENVS[@]} -eq 0 ] && ENVS=(INTARSIA_X=0)
env "${ENVS[@]}" $bin accuracy --model $M --backend cuda --golden $R/golden/gemma4-26b-a4b-nvfp4-fp32.safetensors --mode both \
  --max-seq 8192 --mem-budget 30 --prefill-chunk 1024 --allow-dirty > $R/replay-out/acc/$name.log 2>&1
echo "$name exit $?"
grep -aE "top-1|top1|KL|PASS|FAIL|verdict" $R/replay-out/acc/$name.log | tail -n 12 | cut -c1-200
