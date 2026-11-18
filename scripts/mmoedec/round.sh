#!/bin/zsh
# mmoedec speed gate, one round (inside a fleetq slot): chat-short, code-gen, reason, agent-edit, two arms, both with the fused GDN verify rows
# (INTARSIA_METAL_GDN_VERIFY_FUSE=1, opt-in on main), the default recipe `--spec mtp --spec-k 2` (ab.sh):
#   koff  knob unset
#   kon   INTARSIA_METAL_MOE_BITDEC=1
# ab.sh reverses the arm order on even rounds (ABBA across rounds). Raw: raw/ab. Tables: python3 scripts/mcopy/tables.py raw/ab koff
r=${1:?round}; cd ${0:A:h:h:h}
G=INTARSIA_METAL_GDN_VERIFY_FUSE=1
env W=ab R0=$r ROUNDS=1 REPS=3 CELLS="chat-short code-gen reason agent-edit" \
  ARMS="koff:$G kon:$G+INTARSIA_METAL_MOE_BITDEC=1" scripts/mmoedec/ab.sh
