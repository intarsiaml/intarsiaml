#!/bin/zsh
# CUDA gate on spark1, one fleetq slot, run from endor. usage: gates-cuda.sh OUTDIR
# Needs ~/intarsia-dsv41-attnwire/bin/{intarsia,intarsia.base} on spark1 (built by .scratch/build-both.sh: head = this branch, base = dbe5581ae).
# Runs plan show (35B, 8B; text and --json) and kernels test --backend cuda --quick with both, copies the logs back, compares.
# Same checks as the failpaths lane's gates-cuda.sh; no builds here, so no wait loop (that one's zsh-array bug under bash -s does not apply).
set -u
O=${1:A}; mkdir -p $O
ssh spark1 'rm -rf ~/intarsia-dsv41-attnwire/gates; mkdir -p ~/intarsia-dsv41-attnwire/gates'
ssh spark1 'bash -s' > $O/spark.log 2>&1 <<'REMOTE'
T=$HOME/intarsia-dsv41-attnwire; cd $T/gates
cat ../bin/STAMP; md5sum ../bin/intarsia ../bin/intarsia.base; free -g | head -2
M35=$(ls -d $HOME/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1); M8=$HOME/models/Qwen3-8B
echo "35b=$M35 8b=$M8"
for t in head base; do
  case $t in head) B=$T/bin/intarsia;; base) B=$T/bin/intarsia.base;; esac
  for n in 35b 8b; do
    case $n in 35b) m=$M35;; 8b) m=$M8;; esac
    $B plan show --model $m --backend cuda > plan-$n-$t.txt 2> plan-$n-$t.err; echo "plan $n $t exit $?"
    $B plan show --model $m --backend cuda --json > plan-$n-$t.json 2> plan-$n-$t.jerr; echo "plan $n $t json exit $?"
  done
  free -g | awk 'NR==2{print "avail GB before kernels:", $7}'
  $B kernels test --backend cuda --quick > kernels-$t.log 2>&1; echo "kernels $t exit $?"
done
echo REMOTE_DONE
REMOTE
rsync -a spark1:intarsia-dsv41-attnwire/gates/ $O/ 2>&1 | tail -2
cat $O/spark.log
for n in 35b 8b; do for e in txt json; do cmp -s $O/plan-$n-base.$e $O/plan-$n-head.$e && echo "plan show $n $e: identical ($(wc -c < $O/plan-$n-head.$e | tr -d ' ') bytes)" || echo "plan show $n $e: DIFFERS or missing"; done; done
for t in base head; do echo "kernels $t: $(grep -c FAIL $O/kernels-$t.log) FAIL lines; $(tail -1 $O/kernels-$t.log)"; done
diff <(grep -v -E ' in [0-9.]+s$' $O/kernels-base.log) <(grep -v -E ' in [0-9.]+s$' $O/kernels-head.log) > $O/kernels-diff.txt && echo "kernels logs identical apart from wall time" || { echo "kernels logs differ:"; head -20 $O/kernels-diff.txt; }
