#!/bin/zsh
# mmoe16 speed gate, one round (inside a fleetq slot): copy3k and json-xform, three arms, all with the fused GDN verify rows
# (INTARSIA_METAL_GDN_VERIFY_FUSE=1, lane mgdnv, opt-in on main) so the comparison is the best 16-row stack:
#   c7     INTARSIA_SPEC_COPY=20,7   (8-row chain passes, the default chain length)
#   c15off INTARSIA_SPEC_COPY=20,15  (16-row chain passes, this knob off)
#   c15on  INTARSIA_SPEC_COPY=20,15 + INTARSIA_METAL_VERIFY16=1
# ab.sh reverses the arm order on even rounds (ABBA across rounds). Raw: raw/ab. Tables: python3 scripts/mcopy/tables.py raw/ab c15off
#   for r in 1 2 3 4; do fleetq run --host endor --quiet gpu,cpu --mem 40 --expect 14 --label "mmoe16: A/B round $r" -- scripts/mmoe16/round.sh $r; done
r=${1:?round}; cd ${0:A:h:h:h}
G=INTARSIA_METAL_GDN_VERIFY_FUSE=1
env W=ab R0=$r ROUNDS=1 REPS=3 CELLS="copy3k json-xform" \
  ARMS="c7:$G+INTARSIA_SPEC_COPY=20,7 c15off:$G+INTARSIA_SPEC_COPY=20,15 c15on:$G+INTARSIA_SPEC_COPY=20,15+INTARSIA_METAL_VERIFY16=1" scripts/mmoe16/ab.sh
