#!/usr/bin/env bash
# Lane G2b window 1 (ON tatooine, radiance down, called by window.sh): HIP tp=1 no-op of feat/35b-tp2 (dev) against
# origin/main (base), then a first 35B tp=2 smoke on both R9700s.
#   GPU 0: 35B base then dev; GPU 1 (in parallel): 8B base then dev. Same card for both trees of a model.
#   Per tree and model: plan show --json, accuracy --mode both on the short and long golden (with --write-self-golden dumps,
#   compared bitwise offline), 256 greedy ids on the short prompt and on scripts/hip-decode/prompt4k.txt.
#   Then (both cards): 8B tp=2 ids base vs dev (the existing HIP tp path), 35B tp=2 plan / ids / short accuracy on dev.
set -uo pipefail
O=${1:?out dir}
# trees: BASE_TREE / DEV_TREE (default base / dev = ~/intarsiaml-g2b-base, -dev); NOOP_ONLY=1 skips the tp=2 part
T() { if [[ $1 == base ]]; then echo "$HOME/intarsiaml-g2b-${BASE_TREE:-base}"; else echo "$HOME/intarsiaml-g2b-${DEV_TREE:-dev}"; fi; }
log() { echo "$(date +%T) $*" | tee -a "$O/steps.log"; }
M35=/models/Qwen3.6-35B-A3B-NVFP4; M8=/models/Qwen3-8B; G=/models/golden
SHORT='Write a long story about a robot who learns to paint.'
# run <tree> <gpus> <timeout> <out-prefix> <intarsia args...>: stdout -> <prefix>.out, stderr -> <prefix>.err
run() {
  local tree=$1 gpus=$2 to=$3 pfx=$4; shift 4
  mkdir -p "$(T "$tree")/g2b-out"
  INTARSIA_REPO=$(T "$tree") HIP_VISIBLE_DEVICES=$gpus timeout "$to" "$(T "$tree")/deploy/tatooine/dev.sh" /work/g2b-intarsia "$@" \
    > "$O/$pfx.out" 2> "$O/$pfx.err"
  local rc=$?
  log "$pfx exit $rc"
  return $rc
}
gen() { # gen <tree> <gpus> <tag> <prompt-kind> [args]
  local tree=$1 gpus=$2 tag=$3 kind=$4; shift 4
  local m=$M35; [[ $tag == *8b* ]] && m=$M8
  local p="$SHORT"; [[ $kind == 4k ]] && p="$(cat "$(T "$tree")/scripts/hip-decode/prompt4k.txt")"
  run "$tree" "$gpus" 600 "gen-$tag" generate -m $m --backend hip --mem-budget 29 --max-seq 8192 --chat --no-think \
    --profile-every 0 --ignore-eos --show-ids -p "$p" --max-tokens 256 "$@"
  grep -a '^ids:' "$O/gen-$tag.err" > "$O/gen-$tag.ids"
}
acc() { # acc <tree> <gpus> <tag> <model> <golden file> [args]
  local tree=$1 gpus=$2 tag=$3 m=$4 g=$5; shift 5
  run "$tree" "$gpus" 900 "acc-$tag" accuracy --backend hip --model $m --golden $G/$g --mode both --mem-budget 29 \
    --max-seq 17408 --write-self-golden /work/g2b-out/acc-$tag.safetensors "$@"
  mv "$(T "$tree")/g2b-out/acc-$tag.safetensors" "$O/" 2>/dev/null
}
plan() { # plan <tree> <gpus> <tag> <model> [args]
  local tree=$1 gpus=$2 tag=$3 m=$4; shift 4
  run "$tree" "$gpus" 300 "plan-$tag" plan show --backend hip --model $m --mem-budget 29 --max-seq 4096 --json "$@"
}

for t in base dev; do echo "$t: $(T $t)"; done | tee "$O/versions.txt"
rocm-smi --showmeminfo vram 2>/dev/null | grep -E 'GPU|Used' | head -6 >> "$O/versions.txt"

track35() {
  for t in base dev; do
    plan $t 0 $t-35b $M35
    acc $t 0 $t-35b-short $M35 qwen36-35b-a3b-nvfp4-vllm-top20.safetensors
    acc $t 0 $t-35b-long $M35 qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
    gen $t 0 $t-35b-short short
    gen $t 0 $t-35b-4k 4k
  done
}
track8() {
  for t in base dev; do
    plan $t 1 $t-8b $M8
    acc $t 1 $t-8b-short $M8 qwen3-8b.safetensors
    acc $t 1 $t-8b-long $M8 qwen3-8b-long.safetensors
    gen $t 1 $t-8b-short short
    gen $t 1 $t-8b-4k 4k
  done
}
log "no-op tracks start"
track35 & p35=$!
track8 & p8=$!
wait $p35 $p8
log "no-op tracks done"

# quick verdicts (the offline bitwise dump comparison runs on the Mac)
for m in 35b 8b; do
  cmp -s "$O/plan-base-$m.out" "$O/plan-dev-$m.out" && log "PLAN $m identical ($(wc -c < "$O/plan-dev-$m.out") bytes)" || log "PLAN $m DIFFERS"
  for k in short 4k; do
    cmp -s "$O/gen-base-$m-$k.ids" "$O/gen-dev-$m-$k.ids" && log "IDS $m $k identical ($(wc -w < "$O/gen-dev-$m-$k.ids") words)" || log "IDS $m $k DIFFER"
  done
done

[[ "${NOOP_ONLY:-0}" == 1 ]] && { log "no-op done"; exit 0; }
# tp=2 on both cards
gen base 0,1 base-8b-tp2-short short --tp 2 --devices 0,1
gen dev 0,1 dev-8b-tp2-short short --tp 2 --devices 0,1
cmp -s "$O/gen-base-8b-tp2-short.ids" "$O/gen-dev-8b-tp2-short.ids" && log "IDS 8b tp2 short base==dev" || log "IDS 8b tp2 short base!=dev"
plan dev 0,1 dev-35b-tp2 $M35 --tp 2 --devices 0,1
gen dev 0,1 dev-35b-tp2-short short --tp 2 --devices 0,1 && {
  gen dev 0,1 dev-35b-tp2-4k 4k --tp 2 --devices 0,1
  acc dev 0,1 dev-35b-tp2-short $M35 qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --tp 2 --devices 0,1
}
log "done"
