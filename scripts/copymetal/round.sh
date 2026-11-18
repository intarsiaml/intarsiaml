#!/bin/zsh
# copymetal speed gate, one round (inside a fleetq slot): all seven cells, three arms of one binary (origin/main after #456):
#   base  INTARSIA_SPEC_COPY unset
#   c15   INTARSIA_SPEC_COPY=20     (bare = len 15 on Metal with VERIFY16 default-on)
#   c7    INTARSIA_SPEC_COPY=20,7
# ab.sh reverses the arm order on even rounds (ABBA across rounds). Tables: python3 scripts/mcopy/tables.py raw/ab base
r=${1:?round}; cd ${0:A:h:h:h}
env W=ab R0=$r ROUNDS=1 REPS=3 ARMS="base:X=1 c15:INTARSIA_SPEC_COPY=20 c7:INTARSIA_SPEC_COPY=20,7" scripts/copymetal/ab.sh
