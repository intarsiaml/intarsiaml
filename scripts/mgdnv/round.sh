#!/bin/zsh
# mgdnv gate (c), one round (inside a fleetq slot): group A = default MTP recipe, knob off / on, five cells; group B = INTARSIA_SPEC_COPY=20,15, knob off / on,
# copy3k and json-xform. ab.sh reverses the arm order on even rounds (ABBA across rounds). Raw: raw/abA, raw/abB.
#   scripts/mgdnv/run-ab.sh runs rounds 1..4, each in its own slot.
r=${1:?round}; cd ${0:A:h:h:h}
env W=abA R0=$r ROUNDS=1 REPS=3 CELLS="chat-short code-gen reason agent-edit copy3k" ARMS="knoff:X=1 knon:INTARSIA_METAL_GDN_VERIFY_FUSE=1" scripts/mgdnv/ab.sh
env W=abB R0=$r ROUNDS=1 REPS=3 CELLS="copy3k json-xform" ARMS="c15off:INTARSIA_SPEC_COPY=20,15 c15on:INTARSIA_SPEC_COPY=20,15+INTARSIA_METAL_GDN_VERIFY_FUSE=1" scripts/mgdnv/ab.sh
