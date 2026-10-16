#!/bin/zsh
# usage: slot_b1_shed.sh OUTDIR     (inside a fleetq slot)
# 1. --max-batch 1 (default) same-slot A/B/A/B: baseline 28e5c5a vs head 6d99a6f, same seed/rate, 6 min each
# (shedding bursts moved to slot_soak.sh, B=8 slot)
W=/opt/code/intarsiaml-cb-soak2
BINS=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/bin
O=$1; export PORT=8107
for v in base head base head; do
  n=$((${n:-0}+1))
  INTARSIA_BIN=$BINS/$([ $v = base ] && echo intarsia-base28e5c5a || echo intarsia-head) SERVE_EXTRA="" \
    $W/tools/cbsoak_serve.sh $O/b1-$n-$v run --rate 0.3 --duration 360 --seed 31 --max-tokens-mix 16:2,48:3,96:1 --drain-s 90 --sample-s 5 --window-s 60
done
