#!/bin/zsh
# copyflipm 27B gate, inside a fleetq slot on endor (--quiet gpu,cpu): the Qwen3.8-27B (Metal, recipe MTP default, no --spec flag) gets the
# copy-draft default too. Greedy ids `--spec off` = default = INTARSIA_SPEC_COPY=0 = base (no copy), copy3k / json-xform / code-gen, one tok/s line each.
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=${0:A:h:h:h}; RAW=$W/docs/results/2026-11-18-copyflipm/raw/ids27; mkdir -p $RAW
BASE=$W/target-cf-base/intarsia-base; NEW=$W/target-cf/release/intarsia; CELLS=$W/target-cf/cells; M=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4
mkdir -p $CELLS; [ -f $CELLS/copy3k.txt ] || python3 $W/scripts/mcopy/prompts.py $CELLS > /dev/null
fails=0
for n in copy3k json-xform code-gen; do
  case $n in copy3k) MS=8192; NT=1000 ;; json-xform) MS=8192; NT=1000 ;; *) MS=4096; NT=300 ;; esac
  for arm in off def zero base; do
    case $arm in off) A=(--spec off); E=(X=1); B=$NEW ;; def) A=(); E=(X=1); B=$NEW ;; zero) A=(); E=(INTARSIA_SPEC_COPY=0); B=$NEW ;; base) A=(); E=(X=1); B=$BASE ;; esac
    f=$RAW/gen-$n-$arm; echo "-- $n $arm"
    env $E INTARSIA_SPEC_TRACE=1 $B generate -m $M --backend metal --max-seq $MS $A --chat --no-think --temperature 0 --profile-every 0 --show-ids -p "$(cat $CELLS/$n.txt)" --max-tokens $NT < /dev/null > $f.out 2> $f.err
    echo "exit $?"
    if [ $arm = def ]; then grep -aq "copy drafts on: a chain of up to 15 tokens.*default on Metal" $f.err && echo "PASS chain $n def: on, len 15, by default" || { echo "FAIL chain $n def: no default chain of 15"; fails=$((fails+1)); }
    else grep -aq "copy drafts on" $f.err && { echo "FAIL chain $n $arm: copy drafts on"; fails=$((fails+1)); } || echo "PASS chain $n $arm: copy drafts off"; fi
    grep -a '^ids:' $f.err > $f.ids; grep -a "spec: copy chain" $f.err | tail -1 | cut -c1-200; echo "tok/s $n $arm: $(grep -a 'tok/s' $f.err | tail -1 | cut -c1-200)"
  done
  if [ -s $RAW/gen-$n-off.ids ] && cmp -s $RAW/gen-$n-off.ids $RAW/gen-$n-def.ids && cmp -s $RAW/gen-$n-def.ids $RAW/gen-$n-zero.ids && cmp -s $RAW/gen-$n-def.ids $RAW/gen-$n-base.ids; then echo "PASS ids $n: $(tr ',' '\n' < $RAW/gen-$n-def.ids | wc -l | tr -d ' ') ids identical (off = default = =0 = base)"; else echo "FAIL ids $n"; fails=$((fails+1)); fi
done
echo "gate27 fails=$fails"
