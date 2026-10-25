#!/usr/bin/env bash
# Lane sp, ON endor (Metal), inside a fleetq job. Metal must not regress with the branch: MTP equality (spec off == spec mtp k=2,
# 256 greedy ids) and no-op (branch vs main at 2c2d9941, spec off and spec mtp) at 128 and 1k natural-v1 tokens.
#   metal-gates.sh build   --load cpu: build base and branch Metal binaries (targets warm-copied from hg-targets)
#   metal-gates.sh run     --quiet gpu,cpu: the ids runs and the comparison
set -uo pipefail
R=/opt/code/intarsiaml/.worktrees
BASE=$R/sp-base; DEV=/opt/code/wt-sp; T=$R/sp-targets
M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4
P=/opt/code/deepeval/tests/data/prompts/natural-v1
O=${OUT:-$DEV/docs/results/2026-10-25-sp-hip-spec/raw/metal}; mkdir -p "$O"
case "${1:-run}" in
  build)
    for t in base dev; do
      if [ $t = base ]; then S=$BASE; else S=$DEV; fi
      echo "== build $t"
      (cd $S && CARGO_TARGET_DIR=$T/$t-metal cargo build --release --features metal -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3)
    done ;;
  run)
    for t in base dev; do
      for n in 128 1024; do
        for a in "off:--spec off" "mtp2:--spec mtp --spec-k 2"; do
          tag=${a%%:*}; args=${a#*:}
          f=$O/$t-$tag-n$n
          $T/$t-metal/release/intarsia generate -m $M35 --backend metal --max-seq 4096 --profile-every 0 --ignore-eos --show-ids \
            --max-tokens 256 $args -p "$(cat $P/n$n.txt)" > $f.out 2> $f.err < /dev/null
          grep -a '^ids:' $f.err > $f.ids
          echo "$t $tag n$n md5=$(md5 -q $f.ids) | $(grep -a 'finish=' $f.err | tail -1 | sed 's/.*| decode/decode/' | cut -c1-60) | $(grep -a '^spec:' $f.err | tail -1 | cut -c1-110)"
        done
      done
    done | tee $O/summary.txt
    echo "== equality (spec off == mtp, per tree) and no-op (base == dev), by md5"
    for n in 128 1024; do
      for t in base dev; do
        cmp -s $O/$t-off-n$n.ids $O/$t-mtp2-n$n.ids && echo "n$n $t: spec off == mtp k=2" || echo "n$n $t: DIFFER off vs mtp"
      done
      for a in off mtp2; do
        cmp -s $O/base-$a-n$n.ids $O/dev-$a-n$n.ids && echo "n$n $a: base == dev" || echo "n$n $a: DIFFER base vs dev"
      done
    done ;;
esac
