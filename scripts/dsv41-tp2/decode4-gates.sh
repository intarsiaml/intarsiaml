#!/usr/bin/env bash
# Lane dsv41/decode4: the CUDA gates of the HC_SIDE branch, one fleetq slot over spark1 and spark2 (--load gpu,cpu, vLLM stays up):
#   fleetq run --host spark1,spark2 --load gpu,cpu --mem spark1=20,spark2=30 --expect 25 --label "dsv41 decode4 cuda gates" -- scripts/dsv41-tp2/decode4-gates.sh OUTDIR
# 1) on spark2 (the build host): the new kernel test (the two halves of hc_sublayer equal the fused launch) and the real-shape test
#    against the cpu reference; 2) on spark1: plan show 35B / 8B (text and json) and kernels test --backend cuda --quick with the
#    branch binary, compared with the logs of the prefill3 lane's gates of main (docs/results/2026-11-05-dsv41-prefill3/raw/gates/cuda).
source "$(dirname "${BASH_SOURCE[0]}")/spark-dir.sh"
set -u
O=${1:?outdir}; mkdir -p "$O"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RP='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH; export MAX_JOBS=4 CARGO_BUILD_JOBS=4;'
ssh -n spark2 "$RP cd $SPARK_DIR/src && INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-cuda --test dsv41 -- hc_sublayer_halves real_shapes --nocapture 2>&1 | tail -25" > "$O/kernel-tests.log" 2>&1
cat "$O/kernel-tests.log"
ssh spark1 "mkdir -p $SPARK_DIR/gates" && ssh spark1 'bash -s' > "$O/spark.log" 2>&1 <<REMOTE
T=$SPARK_DIR; cd \$T/gates; cat ../bin/STAMP; free -g | head -2
M35=\$(ls -d \$HOME/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1); M8=\$HOME/models/Qwen3-8B
B=\$T/bin/intarsia
for n in 35b 8b; do case \$n in 35b) m=\$M35;; 8b) m=\$M8;; esac
  \$B plan show --model \$m --backend cuda > plan-\$n-head.txt 2> plan-\$n-head.err; echo "plan \$n exit \$?"
  \$B plan show --model \$m --backend cuda --json > plan-\$n-head.json 2> plan-\$n-head.jerr; echo "plan \$n json exit \$?"
done
\$B kernels test --backend cuda --quick > kernels-head.log 2>&1; echo "kernels exit \$?"
echo REMOTE_DONE
REMOTE
rsync -a "spark1:$SPARK_REL/gates/" "$O/" 2>&1 | tail -2
cat "$O/spark.log"
P="$REPO/docs/results/2026-11-05-dsv41-prefill3/raw/gates/cuda"
for n in 35b 8b; do for e in txt json; do cmp -s "$P/plan-$n-head.$e" "$O/plan-$n-head.$e" && echo "plan show $n $e: identical to main's ($(wc -c < "$O/plan-$n-head.$e" | tr -d ' ') bytes)" || echo "plan show $n $e: DIFFERS or missing"; done; done
echo "kernels: $(grep -c FAIL "$O/kernels-head.log") FAIL lines; $(tail -1 "$O/kernels-head.log")"
diff <(grep -v -E ' in [0-9.]+s$' "$P/kernels-head.log") <(grep -v -E ' in [0-9.]+s$' "$O/kernels-head.log") > "$O/kernels-diff.txt" && echo "kernels log identical to main's apart from wall time" || { echo "kernels logs differ:"; head -20 "$O/kernels-diff.txt"; }
