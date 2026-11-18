#!/bin/zsh
# mgdnv-flip gates, inside a fleetq slot on endor: (b) plan show base vs new (unset, =0); (a) greedy ids, new default vs =0 vs --spec off.
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=${0:A:h:h:h}; RAW=$W/docs/results/2026-11-18-mgdnv-flip/raw; mkdir -p $RAW/plan $RAW/ids
BASE=$W/target-flip-base/intarsia-base; NEW=$W/target-flip/intarsia-new; CELLS=$W/target-flip/cells; M=$HOME/models/Qwen3.6-35B-A3B-NVFP4
mkdir -p $CELLS; [ -f $CELLS/copy3k.txt ] || python3 $W/scripts/mcopy/prompts.py $CELLS > /dev/null
show() { env ${=3:-X=1} $2 plan show --model $M --backend metal --spec mtp --spec-k 2 --max-seq 8192 --json < /dev/null > $RAW/plan/$1.json 2> $RAW/plan/$1.err; echo "$1: exit $? $(wc -c < $RAW/plan/$1.json | tr -d ' ') bytes sha $(shasum -a 256 < $RAW/plan/$1.json | cut -c1-16)"; }
show base $BASE; show new $NEW; show new0 $NEW INTARSIA_METAL_GDN_VERIFY_FUSE=0
cmp $RAW/plan/base.json $RAW/plan/new.json && echo "plan: base == new default (byte-identical)"
fails=0
for n in chat-short code-gen copy3k; do
  case $n in copy3k) MS=8192; NT=3800 ;; *) MS=4096; NT=500 ;; esac
  for arm in off def zero; do
    case $arm in off) A=(--spec off); E=(X=1) ;; def) A=(--spec mtp --spec-k 2); E=(X=1) ;; zero) A=(--spec mtp --spec-k 2); E=(INTARSIA_METAL_GDN_VERIFY_FUSE=0) ;; esac
    f=$RAW/ids/gen-$n-$arm; echo "-- $n $arm"
    env $E $NEW generate -m $M --backend metal --max-seq $MS $A --chat --no-think --temperature 0 --profile-every 0 --show-ids -p "$(cat $CELLS/$n.txt)" --max-tokens $NT < /dev/null > $f.out 2> $f.err
    echo "exit $?"; grep -a '^ids:' $f.err > $f.ids; grep -a "tok/s" $f.err | tail -1 | cut -c1-200
  done
  if [ -s $RAW/ids/gen-$n-off.ids ] && cmp -s $RAW/ids/gen-$n-off.ids $RAW/ids/gen-$n-def.ids && cmp -s $RAW/ids/gen-$n-def.ids $RAW/ids/gen-$n-zero.ids; then echo "PASS ids $n: $(tr ',' '\n' < $RAW/ids/gen-$n-def.ids | wc -l | tr -d ' ') ids identical (off = default = 0)"; else echo "FAIL ids $n"; fails=$((fails+1)); fi
done
echo "gate fails=$fails"
