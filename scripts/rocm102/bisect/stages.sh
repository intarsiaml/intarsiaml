#!/usr/bin/env bash
# Lane rocm102-bisect, a radiance-down window (run by window.sh; tp1 on card 0). Per CLI binary, in ONE container per image:
#   tap  `generate` of prompt.txt, 4 tokens, INTARSIA_DET_TAP: per-layer hashes of the first 64 rows (scripts/rocm102/bisect/tapcmp.py)
#   acc  the acc-oracle drift check by hand: `accuracy --self-golden` against the recorded 35B HIP tp1 dump (4379dd0, 7.2.4)
# STEPS: cross  each image runs both cargo-built CLIs and the relinked all-7.2.4 / all-10.2 kernel archives (compiler vs runtime)
#        swap   image $SWAPIMG (r10) runs mix/s-<tu> for every swap CLI present (7.2.4 kernels except that one TU from 10.2)
#        list   image $LISTIMG runs the CLIs named in $LIST ("tag=/work/path ...")
#        tp2    image r10 / r7: the tp=2 drift check (recorded tp2 dump) for $TP2 ("img:tag=/work/path ...")
set -u
. "$(dirname "$0")/lib.sh"
cd "$TREE"
STEPS=${STEPS:-"cross swap"}
has() { [[ " $STEPS " == *" $1 "* ]]; }
SG=/work/golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.hip.selfgolden.safetensors
SG2=/work/golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.hip-tp2.selfgolden.safetensors
# batch <image arm> <gpus> <timeout> <items...>: item = tag=/work/path/to/intarsia
batch() {
  local a=$1 gpus=$2 to=$3; shift 3
  local i="IMG_$a" v="VOL_$a" cn=rocm102-bis-$a-$RANDOM
  INTARSIA_REPO=$TREE INTARSIA_IMAGE=${!i} INTARSIA_TARGET_VOL=${!v} INTARSIA_CONTAINER_NAME=$cn HIP_VISIBLE_DEVICES=$gpus timeout "$to" "$D" bash -c '
    W=$1 M=$2 SG=$3 SG2=$4; shift 4
    P="$(cat /work/scripts/rocm102/bisect/prompt.txt)"
    for it in "$@"; do
      tag=${it%%=*} bin=${it#*=}; t0=$(date +%s)
      if [[ $tag == tp2-* ]]; then
        $bin accuracy --model $M --backend hip --device 0 --tp 2 --devices 0,1 --mem-budget 29 --self-golden $SG2 --mode prefill --decode-positions 16 --max-seq 4096 \
          --model-id Qwen3.6-35B-A3B-NVFP4 --allow-same-code --pre-change 4379dd0 > $W/$tag.acc.out 2> $W/$tag.acc.err; ra=$?; rg=-
      else
        rm -f $W/$tag.tap
        INTARSIA_DET_TAP=$W/$tag.tap $bin generate -m $M --backend hip --mem-budget 29 --max-seq 4096 --profile-every 0 --ignore-eos --show-ids \
          -p "$P" --max-tokens 4 > $W/$tag.gen.out 2> $W/$tag.gen.err; rg=$?
        $bin accuracy --model $M --backend hip --device 0 --mem-budget 29 --self-golden $SG --mode prefill --decode-positions 16 --max-seq 4096 \
          --model-id Qwen3.6-35B-A3B-NVFP4 --allow-same-code --pre-change 4379dd0 > $W/$tag.acc.out 2> $W/$tag.acc.err; ra=$?
      fi
      echo "$(date +%T) $tag gen=$rg acc=$ra $(( $(date +%s) - t0 ))s | $(grep -a "^ids" $W/$tag.gen.err $W/$tag.gen.out 2>/dev/null | head -1 | cut -c1-80) | $(grep -aE "mean KL\(21-way\)|top-1 flips|golden gate" $W/$tag.acc.out | tr -s " " | tr "\n" ";" | cut -c1-220)" >> $W/batch.log
    done' _ "$W" "$M35" "$SG" "$SG2" "$@" < /dev/null > "$O/batch-$a-$(date +%H%M%S).log" 2>&1
  log "batch $a exit $? ($# items)"
  docker rm -f "$cn" > /dev/null 2>&1   # a timeout kills only the docker client
}
log "stages start; $STAMP; STEPS=$STEPS"
if has cross; then
  batch r10 0 900 r10ctl=/work/.bisect/bins/r10/intarsia x7on10=/work/.bisect/bins/r7/intarsia mixnone-on10=/work/.bisect/mix/none/intarsia mixall-on10=/work/.bisect/mix/all/intarsia
  batch r7 0 900 r7ctl=/work/.bisect/bins/r7/intarsia x10on7=/work/.bisect/bins/r10/intarsia mixnone-on7=/work/.bisect/mix/none/intarsia mixall-on7=/work/.bisect/mix/all/intarsia
fi
if has swap; then
  items=(); for d in "$B"/mix/s-*/; do n=$(basename "$d"); [[ -x $d/intarsia ]] && items+=("$n=/work/.bisect/mix/$n/intarsia"); done
  batch "${SWAPIMG:-r10}" 0 1800 "${items[@]}"
fi
if has list; then batch "${LISTIMG:-r10}" 0 1800 $LIST; fi
if has tp2; then for it in $TP2; do batch "${it%%:*}" 0,1 600 "${it#*:}"; done; fi
cat "$O/batch.log" >> "$O/steps.log" 2>/dev/null
log "stages done"
