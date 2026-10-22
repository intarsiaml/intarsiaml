#!/usr/bin/env bash
# fx1 CUDA (spark1, non-exclusive, vLLM stays up; run under fleetq): build lane-fx1 for CUDA, then
#  q-0 qualify without a CUDA baseline -> MISSING-BASELINE (records the acc-short baseline, envelope, oracle dump);
#  install the recorded files in golden/baselines/ of the tree;
#  q-1 / q-1b clean -> PASS twice (the second shows the drift check reads 0 run to run on CUDA);
#  q-2 the x1.01 row-parallel scale mutant -> FAIL.
# usage (from the Mac): cuda-job.sh REV OUTDIR
set -u
: "${FLEETQ_JOB:?run under fleetq}"
REV=${1:?rev}; OUT=${2:?outdir}
H=spark1
W=/opt/code/wt-fx1
git -C $W archive "$REV" | ssh $H "rm -rf fx1/src && mkdir -p fx1/src fx1/mutants && tar -x -C fx1/src"
scp -q /opt/code/intarsiaml/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors $H:fx1/src/golden/
ssh -n -o BatchMode=yes $H "FLEETQ_JOB=$FLEETQ_JOB bash -s" <<'EOS'
set -u
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_PATH=/usr/local/cuda
cd ~/fx1/src
t0=$(date +%s)
CARGO_TARGET_DIR=$HOME/fx1/target nice cargo build --release -q -p intarsia-cli --features cuda -j 10 < /dev/null 2>&1 | grep -E "^error" | head
B=$HOME/fx1/target/release/intarsia
[ -x $B ] || { echo "BUILD FAILED"; exit 3; }
echo "build $(( $(date +%s) - t0 )) s"
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
[ -e ~/fx1/mutants/mut1.01/zz-scale-mutant.safetensors ] || python3 scripts/fx1-oracle/make_scale_mutant.py $M ~/fx1/mutants/mut1.01 1.01 < /dev/null
G=$PWD/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors
OR=$PWD/golden/oracle/qwen36-35b-a3b-nvfp4-cpu-oracle.safetensors
sha256sum $G
avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); echo "MemAvailable ${avail} GB"
[ "$avail" -ge 44 ] || { echo "ABORT: less than 44 GB available"; exit 3; }
O=$HOME/fx1/q; rm -rf $O; mkdir -p $O
export INTARSIA_REPO=$PWD
SK="--skip kernels --skip bench --skip acc-long --skip acc-4k --skip bench-long --skip ab --skip tp"
Q="qualify --backend cuda --golden $G --oracle-golden $OR --model-name Qwen3.6-35B-A3B-NVFP4 --mem-budget 40 --stage-timeout-min 20"
qual() { # name model [extra...]
  local n=$1 m=$2; shift 2
  t0=$(date +%s)
  timeout 1500 $B $Q --model $m --out $O/q-$n $SK "$@" < /dev/null > $O/q-$n.log 2>&1
  echo "qualify $n exit $? $(( $(date +%s) - t0 )) s"
  grep -E "\] [a-z0-9-]+: (PASS|FAIL|SKIP|ERROR|WARN|MISSING-BASELINE)" $O/q-$n.log | cut -c1-300
}
qual 0-nobaseline $M
Bd=$O/q-0-nobaseline/baselines
mkdir -p golden/baselines
cp $Bd/acc-short.baseline.json golden/baselines/qwen36-35b-a3b-nvfp4-vllm-top20.cuda.baseline.json
cp $Bd/envelope.json golden/baselines/qwen36-35b-a3b-nvfp4-vllm-top20.cuda.envelope.json
cp $Bd/qwen36-35b-a3b-nvfp4-cpu-oracle.cuda.selfgolden.safetensors golden/baselines/
echo "installed: $(ls golden/baselines | grep 35b | tr '\n' ' ')"
qual 1-clean $M --skip envelope
qual 1b-clean $M --skip envelope
qual 2-mut1.01 ~/fx1/mutants/mut1.01 --skip envelope
cp -r golden/baselines $O/installed-baselines
EOS
rc=$?
mkdir -p "$OUT"
rsync -a $H:fx1/q/ "$OUT"/
exit $rc
