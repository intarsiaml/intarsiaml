#!/usr/bin/env bash
# Lane sp, ON tatooine inside a fleetq slot that holds the cards (radiance down): 256 greedy ids of the natural-v1 prompts
# with and without --spec mtp, for the bitwise-identity gate.
#   ids.sh <tree> <tag> <sizes: 128,1024,...> [generate args...]     ids -> ~/sp-ids/<tag>-n<size>.ids
# Prompts: ~/intarsiaml-<tree>/sp-out/n<size>.txt (the natural-v1 files, scp'd there; not in git).
set -uo pipefail
tree=$1; tag=$2; sizes=$3; shift 3
OUT=$HOME/sp-ids; mkdir -p "$OUT"
case "$tree" in sp) vol=intarsia-cargo-target-sp ;; sp-base) vol=intarsia-cargo-target-spbase ;; *) echo "unknown tree $tree"; exit 2 ;; esac
for n in ${sizes//,/ }; do
  f=$HOME/intarsiaml-$tree/sp-out/n$n.txt
  INTARSIA_MODELS=${MODELS:-/data/models/intarsia} INTARSIA_REPO=$HOME/intarsiaml-$tree INTARSIA_TARGET_VOL=$vol HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0} \
    timeout 600 ~/intarsiaml-sp/deploy/tatooine/dev.sh bash -c \
    "env INTARSIA_SPEC_COSTGATE=${INTARSIA_SPEC_COSTGATE:-} INTARSIA_SPEC_GRAPH=${INTARSIA_SPEC_GRAPH:-1} INTARSIA_SPEC_TRACE=${INTARSIA_SPEC_TRACE:-0} /target/release/intarsia generate -m ${MODEL:-/models/Qwen3.6-35B-A3B-NVFP4} --backend hip --mem-budget 29 --max-seq 20480 --profile-every ${PROFILE_EVERY:-0} --ignore-eos --show-ids --max-tokens 256 $* -p \"\$(cat /work/sp-out/n$n.txt)\"" \
    > "$OUT/$tag-n$n.out" 2> "$OUT/$tag-n$n.err" < /dev/null
  grep -a '^ids:' "$OUT/$tag-n$n.err" > "$OUT/$tag-n$n.ids"
  echo "$tag n$n md5=$(md5sum < "$OUT/$tag-n$n.ids" | cut -c1-8) | $(grep -a 'finish=' "$OUT/$tag-n$n.err" | tail -1 | sed 's/.*| decode/decode/' | cut -c1-80) | $(grep -a '^spec:' "$OUT/$tag-n$n.err" | tail -1 | cut -c1-120) | $(grep -a '^steps=' "$OUT/$tag-n$n.err" | tail -1 | cut -c1-90)"
done
