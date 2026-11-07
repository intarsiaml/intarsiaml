#!/usr/bin/env bash
# Lane c27: CUDA no-op gate ON spark1 inside one fleetq slot (vLLM up, no model loaded): `plan show --json` of the 35B NVFP4 and the 8B, base (target-src) vs
# dev (target-dev), must be byte-identical; the 27B plan with and without INTARSIA_MTP_QUANT=fp8 (informational); `kernels test --quick` on dev.
#   fleetq run --host spark1 --load cpu --mem 20 --expect 25 --label "c27 cuda no-op" -- ssh spark1 'bash ~/intarsia-c27/src/scripts/c27/noop.sh'
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsia-c27; O=${O:-$R/out/noop}; mkdir -p $O
H=/home/nvidia/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
M8=$(ls -d $H/models--Qwen--Qwen3-8B/snapshots/* | head -1)
M27=$R/models/Qwen3.8-27B-Quark-AWQ-MXFP4
DEV=${DEV:-dev3}
for t in src $DEV; do
  B=$R/target-$t/release/intarsia
  for m in 35b:$M35 8b:$M8 27b:$M27; do
    $B plan show --backend cuda --model ${m#*:} --max-seq 8192 --mem-budget 30 --json < /dev/null > $O/plan-${m%%:*}-$t.json 2> $O/plan-${m%%:*}-$t.err
  done
  INTARSIA_MTP_QUANT=fp8 $B plan show --backend cuda --model $M27 --max-seq 8192 --mem-budget 30 --json < /dev/null > $O/plan-27b-$t-q.json 2> $O/plan-27b-$t-q.err
done
for m in 35b 8b 27b; do
  cmp -s $O/plan-$m-src.json $O/plan-$m-$DEV.json && [ -s $O/plan-$m-$DEV.json ] && echo "plan $m json: byte-identical ($(wc -c < $O/plan-$m-$DEV.json) B)" || echo "plan $m json: DIFFER or empty"
done
cmp -s $O/plan-27b-$DEV.json $O/plan-27b-$DEV-q.json && echo "plan 27b dev, MTP_QUANT=fp8 vs default: identical" || echo "plan 27b dev, MTP_QUANT=fp8 vs default: differ ($(diff <(tr ',' '\n' < $O/plan-27b-$DEV.json) <(tr ',' '\n' < $O/plan-27b-$DEV-q.json) | grep -c '^[<>]') lines)"
echo "== kernels test --quick ($DEV) $(date +%T)"
$R/target-$DEV/release/intarsia kernels test --quick --backend cuda < /dev/null > $O/kernels-quick.log 2>&1
echo "exit $? FAIL lines: $(grep -c FAIL $O/kernels-quick.log)"; tail -3 $O/kernels-quick.log
echo "== done $(date +%T)"
