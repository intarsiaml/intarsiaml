#!/usr/bin/env bash
# samp64 final gate on the fixed binary: schedule S = 28000:1,56000:0 (k=1 above 28k, plain steps above 56k) against none, ABBA.
#   u1 none and u2 S: seeded (prompts straddling 28000 and 56000), sampled T=0.7 4k/32k/64k x 8 reps, greedy 4k/32k/64k x 5 reps; u3 S and u4 none: sampled only.
D="$(dirname "${BASH_SOURCE[0]}")"
export SAMP64_BIN='~/intarsiaml-samp64/bin/intarsia-fix'
SE="seeded:4096,27700,55700,68000"
SC="0.7:4096,32768,65536:8"
GR="0:4096,32768,65536:5"
S="INTARSIA_SPEC_SAMPLED_CTX_K=28000:1,56000:0"
bash "$D/window.sh" u1 "" - $SE $SC $GR
bash "$D/window.sh" u2 "" "$S" $SE $SC $GR
bash "$D/window.sh" u3 "" "$S" $SC
bash "$D/window.sh" u4 "" - $SC
