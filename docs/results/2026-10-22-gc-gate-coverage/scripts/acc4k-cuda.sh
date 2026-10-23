#!/usr/bin/env bash
# gc lane: qualify acc-4k for the 35B on CUDA (spark1, non-exclusive, vLLM stays up). Run from the Mac under fleetq:
#   fleetq run --host spark1 --mem 40 --expect 25 --label "gc: CUDA acc-4k" -- acc4k-cuda.sh REV SHORT_GOLDEN OUTDIR
#  q-0: no baseline recorded for the 4k golden -> MISSING-BASELINE, records acc-4k.baseline.json
#  (the script installs it in golden/baselines/ of the tree), q-1: gated against it, 
set -u
: "${FLEETQ_JOB:?run under fleetq}"
REV=${1:?rev}; SHORT=${2:?short golden}; OUT=${3:?outdir}
H=spark1
W=/opt/code/wt-gc
git -C $W archive "$REV" | ssh $H "rm -rf gc/src && mkdir -p gc/src && tar -x -C gc/src"
scp -q "$SHORT" $H:gc/src/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors
# the long golden must be there too, else qualify takes the 4k golden for the long one and SKIPs acc-4k
scp -q "${LONG:?}" $H:gc/src/golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
ssh -o BatchMode=yes $H "FLEETQ_JOB=$FLEETQ_JOB bash -s" <<'EOS'
set -u
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
cd ~/gc/src
B=$HOME/intarsia-tp2/bin/gc-intarsia
[ -x $B ] || { echo "no binary $B"; exit 3; }
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
G=$PWD/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors
OR=$PWD/golden/oracle/qwen36-35b-a3b-nvfp4-cpu-oracle.safetensors
G4=$PWD/golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors
sha256sum $G4
avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); echo "MemAvailable ${avail} GB"
[ "$avail" -ge 34 ] || { echo "ABORT: less than 34 GB available"; exit 3; }
O=$HOME/gc/q; rm -rf $O; mkdir -p $O
export INTARSIA_REPO=$PWD
SK="--skip kernels --skip gate --skip acc-format --skip acc-short --skip acc-oracle --skip bench --skip acc-long --skip envelope --skip bench-long --skip ab --skip tp"
Q="qualify --backend cuda --golden $G --oracle-golden $OR --model-name Qwen3.6-35B-A3B-NVFP4 --mem-budget 33 --stage-timeout-min 25 --model $M"
qual() {
  local n=$1; shift
  local t0=$(date +%s)
  timeout 1700 $B $Q --out $O/q-$n $SK "$@" < /dev/null > $O/q-$n.log 2>&1
  echo "qualify $n exit $? $(( $(date +%s) - t0 )) s"
  grep -E "\] [a-z0-9-]+: (PASS|FAIL|SKIP|ERROR|WARN|MISSING-BASELINE)" $O/q-$n.log | cut -c1-400
}
qual 0-nobaseline
cp $O/q-0-nobaseline/baselines/acc-4k.baseline.json golden/baselines/qwen36-35b-a3b-nvfp4-vllm-4k-top20.cuda.baseline.json && echo "installed baseline"
qual 1-clean
cp -r golden/baselines $O/installed-baselines
EOS
rc=$?
mkdir -p "$OUT"
rsync -a $H:gc/q/ "$OUT"/
exit $rc
