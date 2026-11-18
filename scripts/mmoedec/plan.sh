#!/bin/zsh
# mmoedec gate (b): `plan show --json` (Metal, 35B, --spec mtp --spec-k 2) of the base binary (twice) and the new one, INTARSIA_SPEC_COPY unset, must be byte-identical;
# with the knob set the new binary's plan is printed beside it.
#   fleetq run --host endor --quiet gpu,cpu --mem 40 --expect 15 --label "mmoedec: plan show" -- scripts/mmoedec/plan.sh
source ${0:A:h}/lib.sh
O=$RAW/plan; mkdir -p $O
show() { # <name> <binary> [ENV=val]
  env ${=${3:-X=1}//+/ } $2 plan show --model $M --backend metal --spec mtp --spec-k 2 --max-seq 8192 --json < /dev/null > $O/$1.json 2> $O/$1.err
  echo "$1: exit $? $(wc -c < $O/$1.json | tr -d ' ') bytes sha $(shasum -a 256 < $O/$1.json | cut -c1-16)"
}
show base $BASE_BIN; show new $NEW_BIN; show base2 $BASE_BIN; show new-on $NEW_BIN INTARSIA_METAL_MOE_BITDEC=1
cmp $O/base.json $O/base2.json && echo "base twice: byte-identical (deterministic)" || echo "base twice DIFFER"
cmp $O/base.json $O/new.json && echo "PASS plan show knob unset: base and new byte-identical" || { echo "FAIL plan show differs:"; diff <(python3 -m json.tool $O/base.json) <(python3 -m json.tool $O/new.json) | head -20; }
cmp $O/base.json $O/new-on.json && echo "PASS plan show knob ON: byte-identical to base" || echo "FAIL plan show differs with the knob on"
