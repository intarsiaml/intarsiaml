#!/usr/bin/env bash
# Lane det w6 (ON tatooine, radiance down): localise the corruption. w4 caught one fresh --spec mtp process (mtptap20) whose
# PREFILL (pass 0, 38 rows) already differed from the others: layer 0's last row equal, layer 1's last row different, rank 0
# equal to rank 1. The tap now hashes every row of a pass, so the first (layer, row) that differs names the op. Fresh
# processes, 8 tokens each (only the prefill and the first passes matter), tap on.
set -uo pipefail
export O=${1:?out dir}
source "$(dirname "$0")/lib.sh"
clear w6 || exit 2
TP2="--tp 2 --devices 0,1"
TAP=1 MT=8 loop mtp 0,1 70 $TP2 --spec mtp --spec-k 2
TAP=1 MT=8 loop off 0,1 70 $TP2 --spec off
log "w6 done"
