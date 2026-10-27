#!/usr/bin/env bash
# Lane cmx: ON spark1 (plan show runs on the host, no GPU kernels): `plan show` json + text of base vs dev for 8B, 35B, Gemma-4; byte-compare.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 10 --label "cmx: plan show" -- ssh spark1 'bash -s' < scripts/cmx/plan.sh
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-cmx; H=/home/nvidia/.cache/huggingface/hub; O=${O:-$R/out/plan}; mkdir -p $O; fails=0
for mm in 8:models--Qwen--Qwen3-8B 35:models--nvidia--Qwen3.6-35B-A3B-NVFP4 g:models--nvidia--Gemma-4-26B-A4B-NVFP4 27:27B; do
  m=${mm%%:*}; M=$(ls -d $H/${mm#*:}/snapshots/* 2>/dev/null | head -1); [ $m = 27 ] && M=$HOME/intarsiaml-cmx/models/Qwen3.8-27B-MXFP4
  for t in base dev; do
    b=$R/trees/$t/target/release/intarsia
    $b plan show --backend cuda --model $M --max-seq 8192 --mem-budget 40 --json < /dev/null > $O/plan-$m-$t.json 2> /dev/null
    $b plan show --backend cuda --model $M --max-seq 8192 --mem-budget 40 < /dev/null > $O/plan-$m-$t.txt 2> /dev/null
  done
  for k in json txt; do
    if cmp -s $O/plan-$m-base.$k $O/plan-$m-dev.$k && [ -s $O/plan-$m-base.$k ]; then echo "PASS plan $m $k byte-identical ($(wc -c < $O/plan-$m-dev.$k) B, md5 $(md5sum < $O/plan-$m-dev.$k | cut -c1-8))"
    elif [ $m = 27 ]; then echo "INFO plan 27B $k differs from base, as intended (new matmul_mxfp4_mma / _gemv_rows picks; see the lines below)"
    else echo "FAIL plan $m $k"; fails=$((fails+1)); fi
  done
done
echo "== 27B matmul_mxfp4 picks (base, then dev)"
for t in base dev; do echo "-- $t"; grep -a "matmul_mxfp4" $O/plan-27-$t.txt | head -12 | cut -c1-200; done
echo "plan fails=$fails"
