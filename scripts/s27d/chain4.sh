#!/usr/bin/env bash
# Lane s27d window 4, ON spark1: plan show base vs dev (8B, 35B, Gemma-4, 27B), 8B ids base vs dev, 27B ids off vs default, MXFP4 GEMV microbench.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 45 --label "s27d: w4 plan, ids, microbench" -- ssh spark1 'bash -s' < scripts/s27d/chain4.sh
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-s27d; S=$R/trees/dev/scripts/s27d; mkdir -p $R/out
echo "=== plan $(date +%T)"; bash $S/plan.sh 2>&1 | tail -12
echo "=== ids8 $(date +%T)"; MODELS="8" PROMPTS="short-0 short-1 4k-0" bash $S/ids8.sh 2>&1 | tail -8
echo "=== ids27 $(date +%T)"; ARMS="off auto" O=$R/out/ids27 bash $S/ids.sh ${IDS27:-short-0 short-1 short-2 4k-0} 2>&1 | tail -14
echo "=== microbench $(date +%T)"; cd $R/bb/scripts/s27d/mxfp4bench && nvcc -O3 -arch=sm_121 -I$R/bb/kernels/cuda -o bench bench.cu 2>&1 | grep -v "^$" | head -20 && ./bench 40
echo "=== done $(date +%T)"
