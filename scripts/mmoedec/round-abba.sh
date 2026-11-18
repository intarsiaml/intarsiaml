#!/bin/zsh
# mmoedec speed gate, second design (inside a fleetq slot): rounds 1..10 (round.sh) showed an order effect (in the even rounds, knob first, every
# ratio was lower), so here ONE round runs each arm TWICE in the order A B B A (four server starts; odd round numbers keep ab.sh from reversing).
# round 11, 15, ...: off on on off; round 13, 17, ...: on off off on. Arm names carry the position: offA onB onC offD / onA offB offC onD.
#   chat-short code-gen reason agent-edit, 3 reps per server, both arms GDN verify fuse on, default recipe --spec mtp --spec-k 2.
r=${1:?round}; cd ${0:A:h:h:h}
G=INTARSIA_METAL_GDN_VERIFY_FUSE=1; K=INTARSIA_METAL_MOE_BITDEC=1
if [ $(( (r / 2) % 2 )) = 1 ]; then ARMS="offA:$G onB:$G+$K onC:$G+$K offD:$G"; else ARMS="onA:$G+$K offB:$G offC:$G onD:$G+$K"; fi
env W=ab R0=$r ROUNDS=1 REPS=3 CELLS="chat-short code-gen reason agent-edit" ARMS="$ARMS" scripts/mmoedec/ab.sh
