#!/bin/zsh
# mgdnv27 ids gate: `generate` on Metal, MODEL=$M27|$M35 (default 27B), at the model's DEFAULT recipe (no --spec flag), knob off / on, against `--spec off`.
#   arms: off = --spec off; def = recipe default; def+on = default + INTARSIA_METAL_GDN_VERIFY_FUSE=1; copy / copy+on = INTARSIA_SPEC_COPY=20,15 (16-row chains) without / with the knob.
#   SAMPLED=1: temperature 0.7 top_p 0.95 --seed ${SEED:-42}; each +on arm is compared with its knob-off twin only.
source ${0:A:h}/lib.sh
BIN=${BIN:-$NEW_BIN}; MODEL=${MODEL:-$M27}; O=${O:-$RAW/ids}; mkdir -p $O; fails=0; TP=${TAGP:-gen}
for n in "$@"; do
  P=$CELLS_DIR/$n.txt; case $n in copy3k) MS=8192; NT=3800 ;; json-xform) MS=8192; NT=2600 ;; *) MS=4096; NT=500 ;; esac
  [ -n "${NTOK:-}" ] && NT=$NTOK
  for arm in ${=ARMS:-off def def+on}; do
    E=(X=1); base=${arm%+on}; case $base in off) A=(--spec off) ;; def) A=() ;; copy) A=(); E=(INTARSIA_SPEC_COPY=20,15 INTARSIA_SPEC_TRACE=1) ;; esac
    [[ $arm == *+on ]] && E+=(INTARSIA_METAL_GDN_VERIFY_FUSE=1)
    T=(--temperature 0); [ -n "${SAMPLED:-}" ] && T=(--temperature 0.7 --top-p 0.95 --seed ${SEED:-42})
    f=$O/$TP-$n-${arm//[=,+]/_}; echo "-- $n $arm $(cool)"
    env $E $BIN generate -m $MODEL --backend metal --max-seq $MS $A --chat --no-think $T --profile-every 0 --show-ids \
      -p "$(cat $P)" --max-tokens $NT < /dev/null > $f.out 2> $f.err
    echo "exit $? thermal $(th)"; grep -a '^ids:' $f.err > $f.ids
    grep -a -i 'spec' $f.err | grep -av '^spec: copy token\|^spec: token' | tail -3 | cut -c1-200
  done
  for arm in ${=ARMS:-off def def+on}; do
    f=$O/$TP-$n-${arm//[=,+]/_}; ref=""
    [[ $arm == *+on ]] && ref=$O/$TP-$n-${${arm%+on}//[=,+]/_}.ids
    if [ -z "${SAMPLED:-}" ]; then
      [ "$arm" = off ] && continue
      if cmp -s $O/$TP-$n-off.ids $f.ids && [ -s $f.ids ] && { [ -z "$ref" ] || cmp -s $ref $f.ids; }; then echo "PASS ids $n $arm: $(tr ',' '\n' < $f.ids | wc -l | tr -d ' ') ids == off${ref:+ and knob-off twin}"; else echo "FAIL ids $n $arm"; fails=$((fails+1)); fi
    else
      [ -z "$ref" ] && continue
      if cmp -s $ref $f.ids && [ -s $f.ids ]; then echo "PASS sampled $n $arm: $(tr ',' '\n' < $f.ids | wc -l | tr -d ' ') ids == knob-off twin"; else echo "FAIL sampled $n $arm"; fails=$((fails+1)); fi
    fi
  done
done
echo "ids fails=$fails"
