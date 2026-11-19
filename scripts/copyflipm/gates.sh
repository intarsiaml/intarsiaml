#!/bin/zsh
# copyflipm gates, inside a fleetq slot on endor (--quiet gpu,cpu): greedy ids off = default = =0 = base, one sampled
# request default vs =0, one sanity tok/s line per arm. plan.sh is the plan gate, serve.sh the served gate.
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=${0:A:h:h:h}; RAW=$W/docs/results/2026-11-18-copyflipm/raw; mkdir -p $RAW/plan $RAW/ids
BASE=$W/target-cf-base/intarsia-base; NEW=$W/target-cf/release/intarsia; CELLS=$W/target-cf/cells; M=$HOME/models/Qwen3.6-35B-A3B-NVFP4
mkdir -p $CELLS; [ -f $CELLS/copy3k.txt ] || python3 $W/scripts/mcopy/prompts.py $CELLS > /dev/null
fails=0
for n in copy3k json-xform code-gen; do
  case $n in copy3k) MS=8192; NT=3800 ;; json-xform) MS=8192; NT=2600 ;; *) MS=4096; NT=500 ;; esac
  for arm in off def zero base; do
    case $arm in off) A=(--spec off); E=(X=1); B=$NEW ;; def) A=($MTP); E=(X=1); B=$NEW ;; zero) A=($MTP); E=(INTARSIA_SPEC_COPY=0); B=$NEW ;; base) A=($MTP); E=(X=1); B=$BASE ;; esac
    f=$RAW/ids/gen-$n-$arm; echo "-- $n $arm"
    env $E INTARSIA_SPEC_TRACE=1 $B generate -m $M --backend metal --max-seq $MS $A --chat --no-think --temperature 0 --profile-every 0 --show-ids -p "$(cat $CELLS/$n.txt)" --max-tokens $NT < /dev/null > $f.out 2> $f.err
    echo "exit $?"
    if [ $arm = def ]; then grep -aq "copy drafts on: a chain of up to 15 tokens.*default on Metal" $f.err && echo "PASS chain $n def: on, len 15, by default" || { echo "FAIL chain $n def: no default chain of 15"; fails=$((fails+1)); }
    else grep -aq "copy drafts on" $f.err && { echo "FAIL chain $n $arm: copy drafts on"; fails=$((fails+1)); } || echo "PASS chain $n $arm: copy drafts off"; fi
    grep -a '^ids:' $f.err > $f.ids; grep -a "spec: copy" $f.err | tail -2 | cut -c1-200; echo "tok/s $n $arm: $(grep -a 'tok/s' $f.err | tail -1 | cut -c1-200)"
  done
  if [ -s $RAW/ids/gen-$n-off.ids ] && cmp -s $RAW/ids/gen-$n-off.ids $RAW/ids/gen-$n-def.ids && cmp -s $RAW/ids/gen-$n-def.ids $RAW/ids/gen-$n-zero.ids && cmp -s $RAW/ids/gen-$n-def.ids $RAW/ids/gen-$n-base.ids; then echo "PASS ids $n: $(tr ',' '\n' < $RAW/ids/gen-$n-def.ids | wc -l | tr -d ' ') ids identical (off = default = =0 = base)"; else echo "FAIL ids $n"; fails=$((fails+1)); fi
done
# one sampled request (temperature 0.7, seed 7): chains never fire (greedy only), so default == =0 exactly
for arm in def zero; do
  E=(X=1); [ $arm = zero ] && E=(INTARSIA_SPEC_COPY=0); f=$RAW/ids/gen-sampled-$arm; echo "-- sampled $arm"
  env $E INTARSIA_SPEC_TRACE=1 $NEW generate -m $M --backend metal --max-seq 8192 $MTP --chat --no-think --temperature 0.7 --seed 7 --profile-every 0 --show-ids -p "$(cat $CELLS/json-xform.txt)" --max-tokens 600 < /dev/null > $f.out 2> $f.err
  echo "exit $?"; grep -a '^ids:' $f.err > $f.ids; echo "copy chain passes: $(grep -ac 'spec: copy pass' $f.err)"; echo "tok/s sampled $arm: $(grep -a 'tok/s' $f.err | tail -1 | cut -c1-200)"
done
if [ -s $RAW/ids/gen-sampled-def.ids ] && cmp -s $RAW/ids/gen-sampled-def.ids $RAW/ids/gen-sampled-zero.ids && cmp -s $RAW/ids/gen-sampled-def.out $RAW/ids/gen-sampled-zero.out; then echo "PASS sampled: ids and text identical default vs =0"; else echo "FAIL sampled"; fails=$((fails+1)); fi
grep -aq "spec: copy pass" $RAW/ids/gen-sampled-def.err && { echo "FAIL sampled: a copy chain ran"; fails=$((fails+1)); } || echo "PASS sampled: no copy chain ran"
echo "gate fails=$fails"
