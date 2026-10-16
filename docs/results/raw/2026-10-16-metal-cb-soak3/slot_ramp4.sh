#!/bin/zsh
# usage: slot_ramp4.sh OUT  (inside a fleetq slot). B=4 capacity ramp, 100 s segments, one server.
source ${0:A:h}/env.sh
export INTARSIA_BIN=$HEAD SERVE_EXTRA="--max-batch 4" PORT=8107
O=$1
C="--duration 100 --drain-s 60 --cancel-frac 0.1 --sample-s 3 --window-s 20"
$W/tools/cbsoak_serve.sh $O --rate 0.2 --seed 41 $=C --then --rate 0.3 --seed 42 $=C --then --rate 0.45 --seed 43 $=C --then --rate 0.6 --seed 44 $=C --then --rate 0.8 --seed 45 $=C
