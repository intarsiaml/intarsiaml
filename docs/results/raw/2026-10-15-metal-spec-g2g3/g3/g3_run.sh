#!/bin/zsh
# usage (inside a slot): g3_run.sh <bin> <outdir> <reps> <modes: "off on"|"off"> [rows-glob-regex]
# One `generate` per (row, rep, mode), greedy, --chat --no-think, 256 tokens, EOS honoured, --show-ids. Mode order alternates by rep
# (off,on / on,off / off,on) so clock drift does not favour one mode. Both modes use the same binary.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
[ -d "$DEVELOPER_DIR" ] || { echo "DEVELOPER_DIR missing" >&2; exit 3; }
BIN=$1; O=$2; REPS=$3; MODES=(${=4}); PAT=${5:-.}
D=/opt/code/intarsiaml-spec/docs/results/raw/2026-10-15-metal-spec-g2g3/g3
M=$HOME/models/Qwen3-8B
mkdir -p $O
for f in $D/prompts/*.txt; do
  row=${f:t:r}; echo $row | grep -Eq "$PAT" || continue
  for rep in $(seq 1 $REPS); do
    order=($MODES); (( rep % 2 == 0 )) && order=(${(Oa)MODES})
    for mode in $order; do
      [ $mode = on ] && SPEC=(--spec ngram) || SPEC=(--spec off)
      $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --show-ids --max-tokens 256 --chat --no-think $SPEC \
        --prompt "$(cat $f)" < /dev/null > $O/$mode-$row-$rep.out 2> $O/$mode-$row-$rep.err
      echo "$row $rep $mode $(grep '^finish=' $O/$mode-$row-$rep.err | sed 's/ | prefill.*| decode/ decode/')"
    done
  done
done
