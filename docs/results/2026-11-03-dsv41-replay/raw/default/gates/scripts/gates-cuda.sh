#!/bin/zsh
# CUDA gate on spark1, one fleetq slot. usage: gates-cuda.sh OUTDIR (local) ; run from the worktree root. Builds this branch's HEAD and origin/main side by side in ~/intarsia-replaygates
# on spark1 (3 cargo jobs each, MAX_JOBS=3), runs plan show (35B, 8B, text and --json) and kernels test --quick with both, copies the logs back, removes the tree.
set -u
O=${1:A}; mkdir -p $O; W=/opt/code/intarsiaml/.worktrees/replay-gates/new; cd $W
H=$(git rev-parse HEAD); BM=c39dcaeb3c97b0912c54de619b7d43dd624d40eb
echo "new $H base $BM"; echo "$H $BM" > $O/commits.txt
trap 'ssh spark1 "rm -rf ~/intarsia-replaygates"; echo cleaned' EXIT
ssh spark1 'rm -rf ~/intarsia-replaygates; mkdir -p ~/intarsia-replaygates/new ~/intarsia-replaygates/base ~/intarsia-replaygates/out'
git archive $H | ssh spark1 'tar -x -C ~/intarsia-replaygates/new'
git archive $BM | ssh spark1 'tar -x -C ~/intarsia-replaygates/base'
ssh spark1 'bash -s' > $O/spark.log 2>&1 <<'REMOTE'
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_PATH=/usr/local/cuda CARGO_BUILD_JOBS=3 MAX_JOBS=3 FLASHINFER_NVCC_THREADS=1
T=$HOME/intarsia-replaygates; cd $T
{ rustc --version; nvcc --version | tail -2; free -g | head -2; } > out/versions.txt 2>&1
PIDS=()
for t in new base; do ( cd $T/$t && CARGO_TARGET_DIR=$T/target-$t cargo build --release --locked -p intarsia-cli --features cuda > $T/out/build-$t.log 2>&1; echo "EXIT $?" >> $T/out/build-$t.log ) & PIDS+=($!); done
while [ -n "$(jobs -r)" ]; do free -g | awk 'NR==2{print $7}' >> $T/out/avail.log; sleep 20; done
touch $T/out/built
tail -2 out/build-new.log out/build-base.log
M35=$(ls -d $HOME/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/*); M8=$HOME/models/Qwen3-8B
for t in new base; do B=$T/target-$t/release/intarsia
  for m in $M35 $M8; do n=$(basename $(dirname $(dirname $m)) 2>/dev/null); case $m in *35B*) n=35b;; *) n=8b;; esac
    $B plan show --model $m --backend cuda > out/plan-$n-$t.txt 2> out/plan-$n-$t.err; echo "plan $n $t exit $?"
    $B plan show --model $m --backend cuda --json > out/plan-$n-$t.json 2> out/plan-$n-$t.jerr; echo "plan $n $t json exit $?"
  done
  free -g | awk 'NR==2{print "avail GB before kernels:", $7}'
  $B kernels test --backend cuda --quick > out/kernels-$t.log 2>&1; echo "kernels $t exit $?"
done
echo REMOTE_DONE
REMOTE
rsync -a spark1:intarsia-replaygates/out/ $O/ 2>&1 | tail -2
cat $O/spark.log
for n in 35b 8b; do for e in txt json; do cmp -s $O/plan-$n-base.$e $O/plan-$n-new.$e && echo "plan show $n $e: identical ($(wc -c < $O/plan-$n-new.$e | tr -d ' ') bytes)" || echo "plan show $n $e: DIFFERS or missing"; done; done
for t in base new; do echo "kernels $t: $(grep -c FAIL $O/kernels-$t.log) FAIL lines; $(tail -1 $O/kernels-$t.log)"; done
diff <(grep -v -E ' in [0-9.]+s$' $O/kernels-base.log) <(grep -v -E ' in [0-9.]+s$' $O/kernels-new.log) > $O/kernels-diff.txt && echo "kernels logs identical apart from wall time" || { echo "kernels logs differ:"; head -20 $O/kernels-diff.txt; }
