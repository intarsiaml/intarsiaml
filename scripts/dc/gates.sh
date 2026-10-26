#!/usr/bin/env bash
# Lane dc gates on spark1 (CUDA). Sources: ~/intarsiaml-dc-src (this worktree rsynced), base = origin/main tree ~/intarsiaml-dc-base.
#   gates.sh buildbase   (fleetq --load cpu)  clone-less export of origin/main to ~/intarsiaml-dc-base and build its CLI
#   gates.sh g0          (fleetq --quiet gpu,cpu)  flags off: dev vs base plan show (json+text) and 256 greedy ids, 8B short and 4k
#   gates.sh g3 <ids...> (fleetq --quiet gpu,cpu)  dev --spec off vs --spec dflash greedy ids, same prompts; prints spec: stats
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
H=/home/nvidia/.cache/huggingface/hub
M8=$(ls -d $H/models--Qwen--Qwen3-8B/snapshots/* | head -1)
DR=$(ls -d $H/models--TD-Labs--Qwen3-8B-DCOCO/snapshots/* | head -1)
SRC=$HOME/intarsiaml-dc-src; BASE=$HOME/intarsiaml-dc-base; O=$HOME/intarsiaml-dc-out
BIN_DEV=$HOME/intarsiaml-dc-target/release/intarsia; BIN_BASE=$HOME/intarsiaml-dc-target-base/release/intarsia
mkdir -p $O
md() { md5sum "$1" | cut -c1-8; }
case "${1:?}" in
buildbase)
  rm -rf $BASE && mkdir -p $BASE && (cd /dev/null 2>/dev/null; true)
  # the exported tree is rsynced from endor beforehand (scripts/dc/spark1.sh basesync)
  cd $BASE && CARGO_TARGET_DIR=$HOME/intarsiaml-dc-target-base CARGO_BUILD_JOBS=8 cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3 ;;
builddev)
  cd $SRC && CARGO_TARGET_DIR=$HOME/intarsiaml-dc-target CARGO_BUILD_JOBS=8 cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3 ;;
g0)
  PD=$SRC/scripts/hg/prompts; fails=0
  for t in base dev; do
    if [ $t = base ]; then B=$BIN_BASE; else B=$BIN_DEV; fi
    for p in short 4k; do
      $B generate -m $M8 --backend cuda --mem-budget 30 --max-seq 8192 --chat --no-think --temperature 0 --profile-every 0 --ignore-eos --show-ids \
        -p "$(cat $PD/$p.txt)" --max-tokens 256 < /dev/null > /dev/null 2> $O/g0-$p-$t.err
      grep -a '^ids:' $O/g0-$p-$t.err > $O/g0-$p-$t.ids
    done
    $B plan show --backend cuda --model $M8 --max-seq 8192 --mem-budget 30 --json < /dev/null > $O/g0-plan-$t.json 2> /dev/null
    $B plan show --backend cuda --model $M8 --max-seq 8192 --mem-budget 30 < /dev/null > $O/g0-plan-$t.txt 2> /dev/null
  done
  for f in short.ids 4k.ids; do :; done
  for p in short 4k; do cmp -s $O/g0-$p-base.ids $O/g0-$p-dev.ids && [ -s $O/g0-$p-dev.ids ] && echo "PASS g0 ids $p identical ($(tr ',' '\n' < $O/g0-$p-dev.ids | wc -l) ids, md5 $(md $O/g0-$p-dev.ids))" || { echo "FAIL g0 ids $p"; fails=$((fails+1)); }; done
  for k in json txt; do cmp -s $O/g0-plan-base.$k $O/g0-plan-dev.$k && [ -s $O/g0-plan-dev.$k ] && echo "PASS g0 plan $k byte-identical ($(wc -c < $O/g0-plan-dev.$k) B)" || { echo "FAIL g0 plan $k"; fails=$((fails+1)); }; done
  echo "g0 fails=$fails" ;;
g3)
  shift; fails=0
  for id in "$@"; do
    f=$SRC/scripts/dc/prompts/$id.txt; [ -f "$f" ] || f=$SRC/scripts/hg/prompts/$id.txt
    for mode in off dflash; do
      if [ $mode = off ]; then A="--spec off"; else A="--spec dflash --spec-draft $DR"; fi
      env ${GATE:+INTARSIA_SPEC_GATE=$GATE} $BIN_DEV generate -m $M8 --backend cuda --mem-budget 30 --max-seq 8192 $A --chat --no-think --temperature 0 --profile-every 0 --ignore-eos --show-ids \
        -p "$(cat $f)" --max-tokens ${NTOK:-192} < /dev/null > /dev/null 2> $O/g3-$id-$mode.err
      grep -a '^ids:' $O/g3-$id-$mode.err > $O/g3-$id-$mode.ids
    done
    st=$(grep -a '^spec:' $O/g3-$id-dflash.err | tail -1 | cut -c1-160); fin=$(grep -a 'finish=' $O/g3-$id-dflash.err | tail -1 | sed 's/.*| decode/decode/' | cut -c1-60); fo=$(grep -a 'finish=' $O/g3-$id-off.err | tail -1 | sed 's/.*| decode/decode/' | cut -c1-60)
    if cmp -s $O/g3-$id-off.ids $O/g3-$id-dflash.ids && [ -s $O/g3-$id-off.ids ]; then echo "PASS g3 $id ids identical ($(tr ',' '\n' < $O/g3-$id-off.ids | wc -l) ids) | off: $fo | dflash: $fin | $st"; else echo "FAIL g3 $id ids differ | $st"; fails=$((fails+1)); fi
  done
  echo "g3 fails=$fails" ;;
esac
