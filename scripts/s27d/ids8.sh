#!/usr/bin/env bash
# Lane s27d: ON spark1. Greedy ids (256, --ignore-eos) + first-token top-20 logits (4 decimals) per arm, compared with the first arm.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 40 --label "s27d: ids8" -- ssh spark1 'MODELS="8 35" PROMPTS="short-0 4k-0 16k-0" bash -s' < scripts/s27d/ids8.sh
# ARMS: space-separated name:binary-tree[:ENV=V,ENV=V]  (default: base:base rope:rope rope0:rope:INTARSIA_CUDA_ROPE_MULTI=0)
# SRC: tree holding scripts/mpf/ids-prompts (default ~/intarsiaml-spf/rope). O: output dir.
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-s27d; H=/home/nvidia/.cache/huggingface/hub
SRC=${SRC:-$R/trees/dev}; O=${O:-$R/out/ids8}; mkdir -p "$O"
ARMS=${ARMS:-"base:base dev:dev"}
declare -A MODEL=([8]=$(ls -d $H/models--Qwen--Qwen3-8B/snapshots/* | head -1) [35]=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1) [g]=$(ls -d $H/models--nvidia--Gemma-4-26B-A4B-NVFP4/snapshots/* | head -1))
fails=0
for m in ${MODELS:-8 35}; do
  for n in ${PROMPTS:-short-0 4k-0 16k-0}; do
    case $n in short*) MS=4096 ;; 4k*) MS=8192 ;; *) MS=20480 ;; esac
    first=""
    for arm in $ARMS; do
      IFS=: read -r name tree envs <<< "$arm"; envs=${envs//,/ }
      b=$R/trees/$tree/target/release/intarsia
      env $envs $b generate -m ${MODEL[$m]} --backend cuda --mem-budget 40 --max-seq $MS --chat --no-think --temperature 0 --profile-every 0 --ignore-eos \
        --show-ids --show-logits 20 -p "$(cat $SRC/scripts/mpf/ids-prompts/$n.txt)" --max-tokens ${NTOK:-256} < /dev/null > $O/$m-$n-$name.out 2> $O/$m-$n-$name.err
      grep -a '^ids:' $O/$m-$n-$name.err > $O/$m-$n-$name.ids; grep -a '^top logits' $O/$m-$n-$name.err > $O/$m-$n-$name.logits
      pt=$(grep -aoE 'prefill [0-9]+ ms' $O/$m-$n-$name.err | head -1)
      if [[ -z $first ]]; then first=$name; echo "ref   $m $n $name: $(tr ',' '\n' < $O/$m-$n-$name.ids | wc -l) ids, $pt"; continue; fi
      if cmp -s $O/$m-$n-$first.ids $O/$m-$n-$name.ids && [ -s $O/$m-$n-$name.ids ] && cmp -s $O/$m-$n-$first.logits $O/$m-$n-$name.logits && [ -s $O/$m-$n-$name.logits ]; then
        echo "PASS  $m $n $first == $name: ids + top-20 logits identical, $pt"
      else echo "FAIL  $m $n $first vs $name (ids $(cmp -s $O/$m-$n-$first.ids $O/$m-$n-$name.ids && echo same || echo differ), logits $(cmp -s $O/$m-$n-$first.logits $O/$m-$n-$name.logits && echo same || echo differ))"; fails=$((fails+1)); fi
    done
  done
done
echo "ids fails=$fails"
