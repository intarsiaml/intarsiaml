#!/usr/bin/env bash
# samp64: 32k gate, ABBA, sampled T=0.7, 10 reps per window: none / 16384:1 / 16384:1,40000:0 / none (the 2 % tolerance needs more reps than the gate run's 5)
D="$(dirname "${BASH_SOURCE[0]}")"
export SAMP64_BIN='~/intarsiaml-samp64/bin/intarsia-fix'
C="0.7:32768:10"
bash "$D/window.sh" w16 "" - $C
bash "$D/window.sh" w17 "" "INTARSIA_SPEC_SAMPLED_CTX_K=16384:1" $C
bash "$D/window.sh" w18 "" "INTARSIA_SPEC_SAMPLED_CTX_K=16384:1,40000:0" $C
bash "$D/window.sh" w19 "" - $C
bash "$D/window.sh" w20 "" "INTARSIA_SPEC_SAMPLED_CTX_K=16384:1,40000:0" $C
bash "$D/window.sh" w21 "" "INTARSIA_SPEC_SAMPLED_CTX_K=16384:1" $C
bash "$D/window.sh" w22 "" - $C
