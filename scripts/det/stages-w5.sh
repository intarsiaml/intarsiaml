#!/usr/bin/env bash
# Lane det w5 (ON tatooine, radiance down): fresh-process rate of the early divergence, cheaply. Every event so far was in
# the first ~30 generated tokens of a fresh process, so: 48 tokens per process, all processes of an arm in ONE dev.sh container
# (a shell loop, no docker run per process). Arms: tp=2 x150, then tp=1 on card 0 and on card 1 at the same time x150 each.
set -uo pipefail
export O=${1:?out dir}
source "$(dirname "$0")/lib.sh"
clear w5 || exit 2
# loop <tag> <gpus> <n> [intarsia args]: n fresh processes in one container, stderr of each to $O/<tag>-<i>.err (ids line only kept)
loop() {
  local tag=$1 gpus=$2 n=$3; shift 3
  local o=${O/#$TREE//work}
  INTARSIA_REPO=$TREE HIP_VISIBLE_DEVICES=$gpus timeout 2400 "$TREE/deploy/tatooine/dev.sh" bash -c '
    tag=$1 n=$2 o=$3; shift 3
    for i in $(seq 1 "$n"); do
      /work/det-intarsia generate -m "'"$M27"'" --backend hip --mem-budget 29 --max-seq 20480 --chat --no-think --profile-every 0 \
        --ignore-eos --show-ids -p "$(cat /work/scripts/dfh/ids-prompts/short-5.txt)" --max-tokens 48 "$@" > /dev/null 2> "$o/$tag.tmp"
      echo "rc=$? $(grep -a "^ids" "$o/$tag.tmp")" >> "$o/$tag.ids"
    done; rm -f "$o/$tag.tmp"' _ "$tag" "$n" "$o" "$@" < /dev/null
  log "$tag: $(wc -l < "$O/$tag.ids") processes, $(cut -d' ' -f2- "$O/$tag.ids" | sort | uniq -c | sort -rn | awk '{print $1}' | tr '\n' ' ')distinct-count"
}
loop tp2 0,1 150 --tp 2 --devices 0,1 --spec off
loop tp1c0 0 150 --spec off &
loop tp1c1 1 150 --spec off &
wait
log "w5 done"
