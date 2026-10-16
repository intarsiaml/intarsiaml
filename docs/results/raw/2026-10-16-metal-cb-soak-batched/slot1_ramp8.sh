#!/bin/zsh
# slot 1: B=8 ramp (low to high) to pick the soak rate; also a first real look at /metrics batch histogram + queue depth
W=/opt/code/intarsiaml-cb-soak2
export INTARSIA_BIN=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/bin/intarsia-head
export SERVE_EXTRA="--max-batch 8" PORT=8107
O=$1
C="--duration 100 --drain-s 60 --cancel-frac 0.1 --sample-s 3 --window-s 20"
$W/tools/cbsoak_serve.sh $O --rate 0.4 --seed 11 $=C --then --rate 0.7 --seed 12 $=C --then --rate 1.0 --seed 13 $=C --then --rate 1.4 --seed 14 $=C --then --rate 1.8 --seed 15 $=C
