#!/usr/bin/env bash
# Lane scc: `plan show` of Gemma-4 (not in spark2's cache) base vs dev3 on spark1. The binaries were built on spark2 (same GB10 / aarch64)
# and are copied from a local directory ($BINS, default .scratch/bin). Inside a spark1 slot:
#   fleetq run --host spark1 --load cpu --expect 6 --label "scc: plan show Gemma-4" -- scripts/scc/plan-g.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
BINS=${BINS:-.scratch/bin}
ssh -n spark1 'mkdir -p ~/intarsiaml-scc/bin ~/intarsiaml-scc/out/plan'
scp -q $BINS/base $BINS/dev3 spark1:intarsiaml-scc/bin/
ssh spark1 'bash -s' <<'EOF'
export PATH=/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-scc; O=$R/out/plan; fails=0
M=$(ls -d /home/nvidia/.cache/huggingface/hub/models--nvidia--Gemma-4-26B-A4B-NVFP4/snapshots/* | head -1)
for t in base dev3; do
  $R/bin/$t plan show --backend cuda --model $M --max-seq 8192 --mem-budget 40 --json < /dev/null > $O/plan-g-$t.json 2> /dev/null
  $R/bin/$t plan show --backend cuda --model $M --max-seq 8192 --mem-budget 40 < /dev/null > $O/plan-g-$t.txt 2> /dev/null
done
for k in json txt; do
  if cmp -s $O/plan-g-base.$k $O/plan-g-dev3.$k && [ -s $O/plan-g-base.$k ]; then echo "PASS plan g $k byte-identical ($(wc -c < $O/plan-g-dev3.$k) B, md5 $(md5sum < $O/plan-g-dev3.$k | cut -c1-8))"
  else echo "FAIL plan g $k"; fails=$((fails+1)); fi
done
echo "plan g fails=$fails"
EOF
