#!/usr/bin/env bash
# Lane G2b (ON tatooine, inside a radiance-down window; called by stages-w4.sh): tp=1 with fusion off (INTARSIA_NO_FUSION=1),
# dev tree 04306ee, one R9700. What each tp=2 rank runs minus the sharding (core turns fusion off at tp > 1): accuracy dumps
# on the short and long goldens for the offline KL, and 256 greedy ids at ~16k against window 2's tp=1 default ids.
set -uo pipefail
O=${1:?out dir}
log() { echo "$(date +%T) $*" | tee -a "$O/steps.log"; }
R=$HOME/intarsiaml-g2b-dev
M35=/models/Qwen3.6-35B-A3B-NVFP4; G=/models/golden
SG=$G/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors; LG=$G/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
# INTARSIA_NO_FUSION=1 inside the container (dev.sh forwards only INTARSIA_HIP_* / INTARSIA_TP_*)
run() { # run <timeout> <out-prefix> <intarsia args...>
  local to=$1 pfx=$2; shift 2
  INTARSIA_REPO=$R HIP_VISIBLE_DEVICES=0 timeout "$to" "$R/deploy/tatooine/dev.sh" env INTARSIA_NO_FUSION=1 /work/g2b-intarsia "$@" \
    > "$O/$pfx.out" 2> "$O/$pfx.err"
  local rc=$?; log "$pfx exit $rc: $(grep -ahE '^golden gate|^Error' "$O/$pfx.out" "$O/$pfx.err" | cut -c1-200 | tr '\n' '|')"
  return $rc
}
A="accuracy --backend hip --model $M35 --mem-budget 29 --max-seq 17408"
mkdir -p $R/g2b-out
run 600 acc-tp1nf-short $A --golden $SG --mode both --write-self-golden /work/g2b-out/tp1nf-short.safetensors
run 900 acc-tp1nf-long $A --golden $LG --mode both --write-self-golden /work/g2b-out/tp1nf-long.safetensors
mv $R/g2b-out/tp1nf-*.safetensors "$O/" 2>/dev/null
run 600 gen-tp1nf-16k generate -m $M35 --backend hip --mem-budget 29 --max-seq 20480 --chat --no-think --profile-every 0 \
  --ignore-eos --show-ids -p "$(for i in 1 2 3 4; do cat $R/scripts/hip-decode/prompt4k.txt; done)" --max-tokens 256
grep -a '^ids:' "$O/gen-tp1nf-16k.err" > "$O/gen-tp1nf-16k.ids"
cmp -s "$O/gen-tp1nf-16k.ids" $HOME/g2b/out/w2/gen-tp1-16k.ids && log "IDS 16k tp1 fusion-off == tp1 default" || log "IDS 16k tp1 fusion-off != tp1 default"
