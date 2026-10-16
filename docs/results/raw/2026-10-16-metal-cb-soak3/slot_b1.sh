#!/bin/zsh
# usage: slot_b1.sh OUT (inside a fleetq slot). B=1 A/B/A guard, 6 min each: base, head, head with keep-warm off, base. Same seed/rate/mix.
source ${0:A:h}/env.sh
O=$1; export PORT=8107
run_arm() { # name bin [env assignments via env]
  local name=$1 bin=$2; shift 2
  env "$@" INTARSIA_BIN=$bin SERVE_EXTRA="" $W/tools/cbsoak_serve.sh $O/b1-$name run --rate 0.3 --duration 360 --seed 31 --max-tokens-mix 16:2,48:3,96:1 --drain-s 60 --sample-s 5 --window-s 60
}
run_arm base1 $BASE X=1
run_arm head $HEAD X=1
run_arm headkwoff $HEAD INTARSIA_METAL_KEEPWARM=0
run_arm base2 $BASE X=1
${0:A:h}/end_sample.sh $O
