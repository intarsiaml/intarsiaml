#!/usr/bin/env bash
# Lane tp2: helpers for the stages scripts (ON tatooine, sourced). Trees: ~/intarsiaml-tp2-<name>, CLI at /work/tp2-intarsia
# inside the HIP dev image (deploy/tatooine/dev.sh mounts the tree at /work). Every run: stdout -> $O/<tag>.out,
# stderr -> $O/<tag>.err; dumps land in the tree's tp2-out/ and are moved to $O.
# Environment for a run: E="K=V K2=V2" (passed with `env` inside the container, so any variable reaches it).
O=${O:?out dir}
M35=/models/Qwen3.6-35B-A3B-NVFP4; M8=/models/Qwen3-8B; G=/models/golden
# SG: the short golden the reference noise was measured on (fnv cd6ef94b); the file without -meta is another vLLM run (b58845c3)
SG=$G/qwen36-35b-a3b-nvfp4-vllm-top20-meta.safetensors; LG=$G/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
NS=/work/golden/ref_noise/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors.json
NL=/work/golden/ref_noise/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors.json
SHORT='Write a long story about a robot who learns to paint.'
TP2="--tp 2 --devices 0,1"
T() { echo "$HOME/intarsiaml-tp2-${RUN:+run-$RUN-}$1"; }  # RUN=<window>: the window runs from its own snapshot (window.sh SNAP=)
log() { echo "$(date +%T) $*" | tee -a "$O/steps.log"; }
# run <tree> <gpus> <timeout> <tag> <intarsia args...>
run() {
  local tree=$1 gpus=$2 to=$3 tag=$4; shift 4
  mkdir -p "$(T "$tree")/tp2-out"
  # shellcheck disable=SC2086
  INTARSIA_REPO=$(T "$tree") HIP_VISIBLE_DEVICES=$gpus timeout "$to" "$(T "$tree")/deploy/tatooine/dev.sh" env ${E:-} /work/tp2-intarsia "$@" \
    > "$O/$tag.out" 2> "$O/$tag.err" < /dev/null
  local rc=$?
  log "$tag exit $rc${E:+ [$E]}: $(grep -ahE '^golden gate|self-golden gate|SELF-STATS|ref-noise|^Error|tensor-parallel rank 1: decode graph|graph replays=' "$O/$tag.out" "$O/$tag.err" | cut -c1-200 | tr '\n' '|')"
  return $rc
}
# gen <tree> <gpus> <tag> <model> <short|4k> [args]: 256 greedy ids -> $O/gen-<tag>.ids
gen() {
  local tree=$1 gpus=$2 tag=$3 m=$4 kind=$5; shift 5
  local p="$SHORT"; [[ $kind == 4k ]] && p="$(cat "$(T "$tree")/scripts/hip-decode/prompt4k.txt")"
  run "$tree" "$gpus" 240 "gen-$tag" generate -m "$m" --backend hip --mem-budget 29 --max-seq 8192 --chat --no-think \
    --profile-every 0 --ignore-eos --show-ids -p "$p" --max-tokens 256 "$@"
  grep -a '^ids:' "$O/gen-$tag.err" > "$O/gen-$tag.ids"
}
# acc <tree> <gpus> <tag> <model> <golden> [args]: accuracy --mode both, dump -> $O/acc-<tag>.safetensors
acc() {
  local tree=$1 gpus=$2 tag=$3 m=$4 g=$5; shift 5
  run "$tree" "$gpus" 420 "acc-$tag" accuracy --backend hip --model "$m" --golden "$g" --mode both --mem-budget 29 \
    --max-seq 17408 --write-self-golden /work/tp2-out/acc-$tag.safetensors "$@"
  mv "$(T "$tree")/tp2-out/acc-$tag.safetensors" "$O/" 2>/dev/null
}
# self <tree> <gpus> <tag> <dump in $O> [args]: change vs base (--self-golden), dump copied into the tree first
self() {
  local tree=$1 gpus=$2 tag=$3 d=$4; shift 4
  cp "$O/$d" "$(T "$tree")/tp2-out/$d"
  run "$tree" "$gpus" 420 "self-$tag" accuracy --backend hip --model $M35 --mem-budget 29 --max-seq 17408 \
    --self-golden /work/tp2-out/$d --mode both --allow-same-code --allow-config-mismatch "$@"
}
plan() {
  local tree=$1 gpus=$2 tag=$3 m=$4; shift 4
  run "$tree" "$gpus" 300 "plan-$tag" plan show --backend hip --model "$m" --mem-budget 29 --max-seq 4096 --json "$@"
}
same() { # same <a> <b> <what>: cmp two files in $O
  if cmp -s "$O/$1" "$O/$2"; then log "SAME $3 ($1 == $2)"; else log "DIFF $3 ($1 != $2)"; fi
}
