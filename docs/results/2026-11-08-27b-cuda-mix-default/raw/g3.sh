#!/usr/bin/env bash
# lane mxd gate 3, on the box inside a fleetq cpu slot (vLLM up, no model loaded): `plan show --json` of the 35B NVFP4, the 8B and the 27B, base (origin/main)
# vs dev must be byte-identical for 35B / 8B; the 27B differs only by the MTP head quant lines (if at all); `kernels test --quick` on dev.
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsia-mxd; O=$R/out/g3; mkdir -p $O
H=$HOME/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
M8=$(ls -d $H/models--Qwen--Qwen3-8B/snapshots/* | head -1)
M27=$R/models/Qwen3.8-27B-Quark-AWQ-MXFP4
for t in main dev; do
  B=$R/target-$t/release/intarsia
  for m in 35b:$M35 8b:$M8 27b:$M27; do
    $B plan show --backend cuda --model ${m#*:} --max-seq 8192 --mem-budget 30 --json < /dev/null > $O/plan-${m%%:*}-$t.json 2> $O/plan-${m%%:*}-$t.err
  done
done
INTARSIA_MTP_QUANT=off $R/target-dev/release/intarsia plan show --backend cuda --model $M27 --max-seq 8192 --mem-budget 30 --json < /dev/null > $O/plan-27b-dev-off.json 2> $O/plan-27b-dev-off.err
for m in 35b 8b 27b; do
  cmp -s $O/plan-$m-main.json $O/plan-$m-dev.json && [ -s $O/plan-$m-dev.json ] && echo "plan $m json: main vs dev byte-identical ($(wc -c < $O/plan-$m-dev.json) B)" || echo "plan $m json: main vs dev DIFFER or empty ($(wc -c < $O/plan-$m-dev.json) B)"
done
cmp -s $O/plan-27b-main.json $O/plan-27b-dev-off.json && echo "plan 27b: main vs dev with INTARSIA_MTP_QUANT=off byte-identical" || echo "plan 27b: main vs dev with INTARSIA_MTP_QUANT=off DIFFER"
echo "== 27b main vs dev diff (json split at commas)"
diff <(tr ',' '\n' < $O/plan-27b-main.json) <(tr ',' '\n' < $O/plan-27b-dev.json) | head -30
echo "== kernels test --quick (dev) $(date +%T)"
$R/target-dev/release/intarsia kernels test --quick --backend cuda < /dev/null > $O/kernels-quick.log 2>&1
echo "exit $? FAIL lines: $(grep -c FAIL $O/kernels-quick.log)"; tail -3 $O/kernels-quick.log
echo "== done $(date +%T)"
