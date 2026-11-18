#!/bin/zsh
# mlen15 gates, inside a fleetq slot on endor: (b) plan show base vs new; (a) greedy ids off = new default = VERIFY16=0 on three cells.
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=${0:A:h:h:h}; RAW=$W/docs/results/2026-11-18-mlen15/raw; mkdir -p $RAW/plan $RAW/ids
BASE=$W/target-len15-base/intarsia-base; NEW=$W/target-len15/intarsia-new; CELLS=$W/target-len15/cells; M=$HOME/models/Qwen3.6-35B-A3B-NVFP4
mkdir -p $CELLS; [ -f $CELLS/copy3k.txt ] || python3 $W/scripts/mcopy/prompts.py $CELLS > /dev/null
show() { env ${=3:-X=1} $2 plan show --model $M --backend metal --spec mtp --spec-k 2 --max-seq 8192 --json < /dev/null > $RAW/plan/$1.json 2> $RAW/plan/$1.err; echo "$1: exit $? $(wc -c < $RAW/plan/$1.json | tr -d ' ') bytes sha $(shasum -a 256 < $RAW/plan/$1.json | cut -c1-16)"; }
show base $BASE; show new $NEW
show base-c20 $BASE INTARSIA_SPEC_COPY=20; show base-c20-15 $BASE INTARSIA_SPEC_COPY=20,15
show new-c20 $NEW INTARSIA_SPEC_COPY=20; show new-c20-15 $NEW INTARSIA_SPEC_COPY=20,15; show new-c20-7 $NEW INTARSIA_SPEC_COPY=20,7; show base-c20-7 $BASE INTARSIA_SPEC_COPY=20,7
pf=0
chk() { if cmp -s $RAW/plan/$1.json $RAW/plan/$2.json; then [ "$3" = same ] && echo "PASS plan: $1 == $2" || { echo "FAIL plan: $1 == $2 (should differ)"; pf=$((pf+1)); }; else [ "$3" = diff ] && echo "PASS plan: $1 != $2 (as intended)" || { echo "FAIL plan: $1 != $2"; pf=$((pf+1)); }; fi; }
chk base new same; chk base-c20 new-c20 diff; chk base-c20-15 new-c20 same; chk base-c20-15 new-c20-15 same; chk base-c20-7 new-c20-7 same; chk base-c20 new-c20-7 same
echo "plan fails=$pf"
fails=0
for n in copy3k json-xform code-gen; do
  case $n in copy3k) MS=8192; NT=3800 ;; json-xform) MS=8192; NT=2600 ;; *) MS=4096; NT=500 ;; esac
  for arm in off def v0 base20; do
    case $arm in off) A=(--spec off); E=(X=1); B=$NEW ;; def) A=(--spec mtp --spec-k 2); E=(INTARSIA_SPEC_COPY=20); B=$NEW ;; v0) A=(--spec mtp --spec-k 2); E=(INTARSIA_SPEC_COPY=20 INTARSIA_METAL_VERIFY16=0); B=$NEW ;; base20) A=(--spec mtp --spec-k 2); E=(INTARSIA_SPEC_COPY=20); B=$BASE ;; esac
    f=$RAW/ids/gen-$n-$arm; echo "-- $n $arm"
    env $E INTARSIA_SPEC_TRACE=1 $B generate -m $M --backend metal --max-seq $MS $A --chat --no-think --temperature 0 --profile-every 0 --show-ids -p "$(cat $CELLS/$n.txt)" --max-tokens $NT < /dev/null > $f.out 2> $f.err
    echo "exit $?"; want=15; [ $arm = base20 ] && want=7; [ $arm = off ] && want=0
    if [ $want -gt 0 ]; then grep -aq "chain of up to $want tokens" $f.err && echo "PASS len $n $arm: chain of up to $want tokens" || { echo "FAIL len $n $arm: not a chain of up to $want tokens"; fails=$((fails+1)); }; fi
    grep -a '^ids:' $f.err > $f.ids; grep -a "spec: copy" $f.err | tail -2 | cut -c1-200; grep -a "tok/s" $f.err | tail -1 | cut -c1-200
  done
  if [ -s $RAW/ids/gen-$n-off.ids ] && cmp -s $RAW/ids/gen-$n-off.ids $RAW/ids/gen-$n-def.ids && cmp -s $RAW/ids/gen-$n-def.ids $RAW/ids/gen-$n-v0.ids && cmp -s $RAW/ids/gen-$n-def.ids $RAW/ids/gen-$n-base20.ids; then echo "PASS ids $n: $(tr ',' '\n' < $RAW/ids/gen-$n-def.ids | wc -l | tr -d ' ') ids identical (off = default = VERIFY16=0 = base len7)"; else echo "FAIL ids $n"; fails=$((fails+1)); fi
done
echo "gate fails=$fails plan_fails=$pf"
