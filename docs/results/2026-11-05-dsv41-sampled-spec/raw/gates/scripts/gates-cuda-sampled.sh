#!/bin/zsh
# CUDA no-op gate on spark1, one fleetq slot, run from endor. usage: gates-cuda-sampled.sh OUTDIR REFDIR
# Needs ~/intarsia-dsv41-sampled/bin/intarsia on spark1 (head of dsv41/sampled-spec). The base side is the stored outputs of the prefill3 lane
# (docs/results/2026-11-05-dsv41-prefill3/raw/gates/cuda, a main binary: plan show and kernels test do not depend on dsv41_engine.rs).
set -u
O=${1:A}; REF=${2:A}; mkdir -p $O
ssh spark1 'rm -rf ~/intarsia-dsv41-sampled/gates; mkdir -p ~/intarsia-dsv41-sampled/gates'
ssh spark1 'bash -s' > $O/spark.log 2>&1 <<'REMOTE'
T=$HOME/intarsia-dsv41-sampled; cd $T/gates
cat ../bin/STAMP; md5sum ../bin/intarsia; free -g | head -2
M35=$(ls -d $HOME/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1); M8=$HOME/models/Qwen3-8B
echo "35b=$M35 8b=$M8"
B=$T/bin/intarsia
for n in 35b 8b; do
  case $n in 35b) m=$M35;; 8b) m=$M8;; esac
  $B plan show --model $m --backend cuda > plan-$n-head.txt 2> plan-$n-head.err; echo "plan $n exit $?"
  $B plan show --model $m --backend cuda --json > plan-$n-head.json 2> plan-$n-head.jerr; echo "plan $n json exit $?"
done
free -g | awk 'NR==2{print "avail GB before kernels:", $7}'
$B kernels test --backend cuda --quick > kernels-head.log 2>&1; echo "kernels exit $?"
echo REMOTE_DONE
REMOTE
rsync -a spark1:intarsia-dsv41-sampled/gates/ $O/ 2>&1 | tail -2
cat $O/spark.log
for n in 35b 8b; do for e in txt json; do cmp -s $REF/plan-$n-base.$e $O/plan-$n-head.$e && echo "plan show $n $e: identical to main's ($(wc -c < $O/plan-$n-head.$e | tr -d ' ') bytes)" || echo "plan show $n $e: DIFFERS or missing"; done; done
echo "kernels head: $(grep -c FAIL $O/kernels-head.log) FAIL lines; $(tail -1 $O/kernels-head.log)"
diff <(grep -v -E ' in [0-9.]+s$' $REF/kernels-base.log) <(grep -v -E ' in [0-9.]+s$' $O/kernels-head.log) > $O/kernels-diff.txt && echo "kernels logs identical to main's apart from wall time" || { echo "kernels logs differ:"; head -20 $O/kernels-diff.txt; }
