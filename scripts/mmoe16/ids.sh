#!/bin/zsh
# mmoe16 gate (a): `generate` greedy ids on Metal (35B, tp 1), every arm against `--spec off` (and each `+on` arm against its knob-off twin).
#   fleetq run --host endor --quiet gpu,cpu --mem 40 --expect 30 --label "mmoe16: ids" -- env ARMS="off mtp copy=20" scripts/mmoe16/ids.sh copy3k json-xform ...
# Arms: off = --spec off; any arm + `+on` = the same with INTARSIA_METAL_VERIFY16=1 (compared against the arm without it AND against off); mtp = --spec mtp --spec-k 2; copy=<n>[,<len>] = --spec mtp --spec-k 2 + INTARSIA_SPEC_COPY (+ INTARSIA_SPEC_TRACE=1). BIN=<binary> (default new).
# Prompts: names of $CELLS_DIR/<name>.txt (scripts/mcopy/prompts.py).
source ${0:A:h}/lib.sh
BIN=${BIN:-$NEW_BIN}; O=${O:-$RAW/ids}; mkdir -p $O; fails=0
for n in "$@"; do
  P=$CELLS_DIR/$n.txt; case $n in copy3k) MS=8192; NT=3800 ;; json-xform) MS=8192; NT=2600 ;; *) MS=4096; NT=${NTOK:-500} ;; esac
  [ -n "${NTOK:-}" ] && NT=$NTOK
  for arm in ${=ARMS:-off copy=20}; do
    E=(X=1); base=${arm%+on}; case $base in off) A=(--spec off) ;; mtp) A=(--spec mtp --spec-k 2) ;; copy=*) A=(--spec mtp --spec-k 2); E=(INTARSIA_SPEC_COPY=${base#copy=} INTARSIA_SPEC_TRACE=1) ;; esac
    [[ $arm == *+on ]] && E+=(INTARSIA_METAL_VERIFY16=1)
    f=$O/gen-$n-${arm//[=,+]/_}; echo "-- $n $arm $(cool)"
    env $E $BIN generate -m $M --backend metal --max-seq $MS $A --chat --no-think --temperature 0 --profile-every 0 --show-ids \
      -p "$(cat $P)" --max-tokens $NT < /dev/null > $f.out 2> $f.err
    echo "exit $? thermal $(th)"; grep -a '^ids:' $f.err > $f.ids
  done
  for arm in ${=ARMS:-off copy=20}; do
    [ "$arm" = off ] && continue
    f=$O/gen-$n-${arm//[=,+]/_}; extra=""
    cp=$(grep -ac '^spec: copy token' $f.err); acc=$(grep -a '^spec: copy token' $f.err | sed 's/.*accepted \([0-9]*\) emits.*/\1/' | sort -n | uniq -c | tr '\n' ' ')
    [[ $arm == copy=* ]] && extra=" copy_passes=$cp accepted_hist[count acc]=$acc"
    st=$(grep -a '^spec:' $f.err | grep -av 'spec: copy token\|spec: token\|spec: mtp\|spec: cost' | tail -2 | cut -c1-200 | tr '\n' '|')
    ref=""; [[ $arm == *+on ]] && ref=$O/gen-$n-${${arm%+on}//[=,+]/_}.ids
    if cmp -s $O/gen-$n-off.ids $f.ids && [ -s $O/gen-$n-off.ids ] && { [ -z "$ref" ] || cmp -s $ref $f.ids; }; then echo "PASS ids $n $arm: $(tr ',' '\n' < $f.ids | wc -l | tr -d ' ') ids identical |$extra | $st"; else echo "FAIL ids $n $arm: differ/empty |$extra | $st"; fails=$((fails+1)); fi
  done
done
echo "ids fails=$fails"
