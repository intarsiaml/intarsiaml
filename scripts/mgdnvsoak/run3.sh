#!/bin/zsh
# run 3: 27B, chat-template reasoning ON (no --no-think), greedy, recipe default (MTP k2): knob off (def) vs on (def+on), ids equal. Inside a fleetq slot.
source ${0:A:h}/lib.sh
THINK="" TWIN=1 ARMS="def def+on" NTOK=${NTOK:-700} MSEQ=4096 O=$RAW/think27 TAGP=think $S/ids.sh chat-short reason code-gen agent-edit
