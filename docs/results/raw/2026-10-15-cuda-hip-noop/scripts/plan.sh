#!/bin/bash
# on spark1 inside fleetq job 2002: plan show on a 0.8 GB Qwen3-8B-shaped model (2 layers), spec off, A vs B
export CUDA_PATH=/usr/local/cuda
M=/tmp/cnc/tiny-qwen3
for T in A B A B; do
  n=$((n+1))
  /tmp/cnc/$T-target/release/intarsia plan show --model $M --backend cuda --max-seq 2048 --mem-budget 4 > /tmp/cnc/plan-$T-$n.txt 2> /tmp/cnc/plan-$T-$n.err; echo "$T $n exit $?"
  /tmp/cnc/$T-target/release/intarsia plan show --model $M --backend cuda --max-seq 2048 --mem-budget 4 --json > /tmp/cnc/plan-$T-$n.json 2>> /tmp/cnc/plan-$T-$n.err; echo "$T $n json exit $?"
done
md5sum /tmp/cnc/plan-*.txt /tmp/cnc/plan-*.json
