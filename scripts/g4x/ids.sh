#!/usr/bin/env bash
# G4X: greedy token ids of Gemma-4-26B-A4B-NVFP4 for a fixed prompt set, one build per call (run INSIDE a fleetq slot,
# --quiet gpu,cpu, --mem >= 32), compared across builds with `diff -r`.
#   ids.sh <name> <binary> [ENV=VAL ...]       output replay-out/ids/<name>/<prompt>.ids
# Prompts: 3 short raw prompts, 1k / 4k / 16k tokens of tools/prompts/corpus_docs.txt (first 4000 / 16000 / 64000 bytes), plus the
# code corpus at 8k (first 32000 bytes of corpus_code.txt); 96 greedy tokens each (--ignore-eos).
set -u
R=$HOME/intarsiaml-g4x; M=$HOME/models/Gemma-4-26B-A4B-NVFP4
name=$1; bin=$2; shift 2
[[ $bin == /* ]] || bin=$R/replay-out/$bin
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 45 )) || { echo "ABORT: memory"; exit 1; }
O=$R/replay-out/ids/$name; rm -rf $O; mkdir -p $O
gen() { # label prompt
  local label=$1 prompt=$2
  env "${ENVS[@]}" $bin generate -m $M --backend cuda --mem-budget 30 --max-seq 20480 --profile-every 0 --ignore-eos --max-tokens 96 \
    --show-ids -p "$prompt" > $O/$label.out 2> $O/$label.err
  echo "$label rc=$? $(grep -a -m1 -E 'prompt_tokens|prompt tokens|tokens in' $O/$label.err | cut -c1-120)"
}
ENVS=("$@"); [ ${#ENVS[@]} -eq 0 ] && ENVS=(INTARSIA_X=0)
gen short1 "The capital of France is"
gen short2 "Write a Python function that returns the n-th Fibonacci number, with a docstring and two doctests."
gen short3 "Explain in three sentences why the sky is blue, and then why sunsets are red."
gen docs1k "$(head -c 4000 $R/tools/prompts/corpus_docs.txt)"
gen docs4k "$(head -c 16000 $R/tools/prompts/corpus_docs.txt)"
gen code8k "$(head -c 32000 $R/tools/prompts/corpus_code.txt)"
gen docs16k "$(head -c 64000 $R/tools/prompts/corpus_docs.txt)"
