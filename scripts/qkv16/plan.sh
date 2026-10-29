#!/usr/bin/env bash
# Lane qkv16 (from msd): ON spark1 (plan show runs on the host, no GPU kernels): `plan show` json + text of base vs dev for 8B, 35B, Gemma-4; byte-compare.
#   fleetq run --host spark1 --load cpu --expect 10 --label "msd: plan show" -- ssh spark1 'bash -s' < scripts/qkv16/plan.sh
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-qkv16; H=/home/nvidia/.cache/huggingface/hub; O=$R/out/plan; mkdir -p $O; fails=0
for mm in 8:models--Qwen--Qwen3-8B 35:models--nvidia--Qwen3.6-35B-A3B-NVFP4 g:models--nvidia--Gemma-4-26B-A4B-NVFP4; do
  m=${mm%%:*}; M=$(ls -d $H/${mm#*:}/snapshots/* | head -1)
  for t in base dev; do
    b=$R/bin/$t
    $b plan show --backend cuda --model $M --max-seq 8192 --mem-budget 40 --json < /dev/null > $O/plan-$m-$t.json 2> /dev/null
    $b plan show --backend cuda --model $M --max-seq 8192 --mem-budget 40 < /dev/null > $O/plan-$m-$t.txt 2> /dev/null
  done
  for k in json txt; do
    if cmp -s $O/plan-$m-base.$k $O/plan-$m-dev.$k && [ -s $O/plan-$m-base.$k ]; then echo "PASS plan $m $k byte-identical ($(wc -c < $O/plan-$m-dev.$k) B, md5 $(md5sum < $O/plan-$m-dev.$k | cut -c1-8))"
    else echo "FAIL plan $m $k"; fails=$((fails+1)); fi
  done
done
echo "plan fails=$fails"
