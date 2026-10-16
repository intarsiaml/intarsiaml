#!/bin/zsh
# usage: slot_extra.sh OUTDIR   (inside a fleetq slot; sized to finish in about 25 min)
# 1. --max-batch 1 same-slot A/B, 6 min each: baseline 28e5c5a then head 6d99a6f (same seed, rate, mix)
# 2. --max-batch 4 scenario x5 (3 streams decoding + one 4k prompt), no soak
W=/opt/code/intarsiaml-cb-soak2
BINS=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/bin
O=$1; export PORT=8107
for v in base head; do
  INTARSIA_BIN=$BINS/$([ $v = base ] && echo intarsia-base28e5c5a || echo intarsia-head) SERVE_EXTRA="" \
    $W/tools/cbsoak_serve.sh $O/b1-$v run --rate 0.3 --duration 360 --seed 31 --max-tokens-mix 16:2,48:3,96:1 --drain-s 60 --sample-s 5 --window-s 60
done
INTARSIA_BIN=$BINS/intarsia-head SERVE_EXTRA="--max-batch 4" \
  $W/tools/cbsoak_serve.sh $O/b4-scenario scenario --streams 3 --reps 5 --solo-reps 3 --max-itl-prefill-p95-ms 150 --max-long-ttft-x-solo 2.5
