#!/bin/zsh
# usage: slot_soak3.sh OUT B RATE STREAMS [DURATION_S]  (inside a fleetq slot). scenario x5 then Poisson soak with Q13 limits.
source ${0:A:h}/env.sh
export INTARSIA_BIN=$HEAD
O=$1; B=$2; RATE=$3; STREAMS=$4; DUR=${5:-1800}
export SERVE_EXTRA="--max-batch $B" PORT=8107
$W/tools/cbsoak_serve.sh $O scenario --streams $STREAMS --reps 5 --solo-reps 3 --max-itl-prefill-p95-ms 150 --max-steady-itl-p95-ms 60 --max-long-ttft-x-solo 2.5 \
  --then --rate $RATE --duration $DUR --seed 21 --cancel-frac 0.1 --drain-s 120 --sample-s 5 --window-s 60 \
  --max-ttft-p50-ms 1500 --max-ttft-p95-ms 5000 --max-ttft-p99-ms 8000 --max-itl-p95-ms 150 --max-itl-prefill-p95-ms 150 \
  --max-error-rate 0.001 --max-shed-rate 0.01 --max-mem-growth-mb 100 --max-device-mem-growth-mb 100
