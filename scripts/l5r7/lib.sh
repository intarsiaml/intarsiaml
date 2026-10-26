#!/usr/bin/env bash
# Lane l5, ON tatooine, sourced by the stage scripts (inside scripts/l5r7/window.sh; env NAME = snapshot, O = out dir, G = card).
#   dv  <cmd...>         the branch build / snapshot (volume intarsia-cargo-target-l5-zdev, tree ~/intarsiaml-l5-$NAME), card $G
#   bs  <cmd...>         origin/main (volume intarsia-cargo-target-l5-zbase, tree ~/intarsiaml-l5-zbase), card $G
# Environment given to the function call (INTARSIA_HIP_MOE_FP8_WMMA=1 ...) reaches the container (dev.sh passes INTARSIA_(HIP|TP|BATCH|QWEN)_*).
M35=/models/Qwen3.6-35B-A3B-NVFP4
M27=/hf/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
GOLD=/models/golden
now() { date '+%F %T'; }
dv() { INTARSIA_REPO=$HOME/intarsiaml-l5-$NAME INTARSIA_TARGET_VOL=intarsia-cargo-target-l5-zdev HIP_VISIBLE_DEVICES=$G \
         $HOME/intarsiaml-l5-$NAME/scripts/l5r7/dev.sh "$@" < /dev/null; }
bs() { INTARSIA_REPO=$HOME/intarsiaml-l5-zbase INTARSIA_TARGET_VOL=intarsia-cargo-target-l5-zbase HIP_VISIBLE_DEVICES=$G \
         $HOME/intarsiaml-l5-$NAME/scripts/l5r7/dev.sh "$@" < /dev/null; }
# gen <fn: dv|bs> <tag> <model> <promptfile> <max-tokens> [extra generate flags]: greedy ids after a prompt -> $O/gen-<tag>.{err,ids}
gen() {
  local fn=$1 tag=$2 model=$3 pf=$4 mt=$5; shift 5
  $fn /target/release/intarsia generate -m "$model" --backend hip --max-seq 8192 "$@" --chat --no-think --temperature 0 --profile-every 0 --ignore-eos \
    --show-ids -p "$(cat "$pf")" --max-tokens "$mt" > /dev/null 2> "$O/gen-$tag.err"
  grep -a '^ids:' "$O/gen-$tag.err" > "$O/gen-$tag.ids"
  echo "[gen] $tag: $(tr ',' '\n' < "$O/gen-$tag.ids" | wc -l) ids, md5 $(md5sum < "$O/gen-$tag.ids" | cut -c1-8)  $(grep -a 'finish=' "$O/gen-$tag.err" | tail -1 | cut -c1-140)"
}

# wait_vram [cap_s=60] [limit_bytes=1000000000]: card 1 (HIP device 1 = 0000:07:00.0, radiance rank 1) must be free before a model load: poll the used VRAM
# (sysfs mem_info_vram_used, the number rocm-smi --showmeminfo vram prints) every 2 s until it is under the limit or the cap is reached; logs every reading.
# An OOM that follows a confirmed-free card is real.
wait_vram() {
  local cap=${1:-60} lim=${2:-1000000000} t0 u
  t0=$(date +%s)
  while :; do
    u=$(cat /sys/bus/pci/devices/0000:07:00.0/mem_info_vram_used 2>/dev/null || echo -1)
    echo "[vram] card1 used=$u B ($((u / 1048576)) MiB) after $(( $(date +%s) - t0 )) s $(now)"
    if [ "$u" -ge 0 ] && [ "$u" -lt "$lim" ]; then return 0; fi
    if [ $(( $(date +%s) - t0 )) -ge "$cap" ]; then echo "[vram] card1 NOT free after ${cap} s"; return 1; fi
    sleep 2
  done
}
