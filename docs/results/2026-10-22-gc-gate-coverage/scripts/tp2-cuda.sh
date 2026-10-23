#!/usr/bin/env bash
# gc lane: record the CUDA tp=2 drift dump of the 35B and prove the tp stage on it. ONE fleetq job holds both Sparks
# (vLLM stays up; rank 0 = spark1, rank 1 = spark2's `intarsia tp-worker`, NCCL over ConnectX-7, as scripts/35b-tp2/xhost.sh):
#   fleetq run --host spark1,spark2 --mem spark1=20,spark2=20 --label "gc: CUDA tp=2 dump" -- tp2-cuda.sh REV SHORT_GOLDEN OUTDIR
# qualify cannot pass --tp-remote, so its `--bin` is tpshim.py (rank-0 side on spark1; it starts the matching worker on spark2).
#  q-0 no recorded dump -> MISSING-BASELINE, records the cuda-tp2 dump; the script installs it in golden/baselines/;
#  q-1 clean -> PASS; q-2 with the scale:1.01 test bug on the worker (rank 1) -> FAIL.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
REV=${1:?rev}; SHORT=${2:?short golden}; OUT=${3:?outdir}
W=/opt/code/wt-gc
D=$W/docs/results/2026-10-22-gc-gate-coverage/scripts
git -C $W archive "$REV" | ssh spark1 "rm -rf gc/src && mkdir -p gc/src && tar -x -C gc/src"
scp -q "$SHORT" spark1:gc/src/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors
scp -q $D/tpshim.py spark1:gc/tpshim.py
ssh -o BatchMode=yes spark1 "chmod +x ~/gc/tpshim.py; FLEETQ_JOB=$FLEETQ_JOB bash -s" <<'EOS'
set -u
cd ~/gc/src
B=$HOME/intarsia-tp2/bin/gc-intarsia
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
G=$PWD/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors
OR=$PWD/golden/oracle/qwen36-35b-a3b-nvfp4-cpu-oracle.safetensors
for h in spark1 spark2; do avail=$(ssh -n $h "awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo"); echo "$h MemAvailable ${avail} GB"; done
O=$HOME/gc/qtp; rm -rf $O; mkdir -p $O
export INTARSIA_REPO=$PWD
SK="--skip kernels --skip gate --skip acc-format --skip acc-short --skip acc-oracle --skip bench --skip acc-long --skip acc-4k --skip envelope --skip bench-long --skip ab"
Q="qualify --backend cuda --golden $G --oracle-golden $OR --model-name Qwen3.6-35B-A3B-NVFP4 --tp 2 --mem-budget 18 --bench-sizes 128 --reps 1 --gen 16 --stage-timeout-min 25 --model $M --bin $HOME/gc/tpshim.py"
qual() {
  local n=$1; shift
  local t0=$(date +%s)
  timeout 2400 $B $Q --out $O/q-$n $SK "$@" < /dev/null > $O/q-$n.log 2>&1
  echo "qualify $n exit $? $(( $(date +%s) - t0 )) s"
  grep -E "\] [a-z0-9-]+: (PASS|FAIL|SKIP|ERROR|WARN|MISSING-BASELINE)" $O/q-$n.log | cut -c1-500
}
qual 0-nobaseline
ls $O/q-0-nobaseline/baselines/
cp $O/q-0-nobaseline/baselines/*cuda-tp2.selfgolden.safetensors golden/baselines/ && echo "installed: $(ls golden/baselines | grep tp2)"
qual 1-clean
GC_WORKER_ENV=INTARSIA_TP_TEST_BUG=scale:1.01 qual 2-tp2bug1.01
cp -r golden/baselines $O/installed-baselines
EOS
rc=$?
mkdir -p "$OUT"
rsync -a spark1:gc/qtp/ "$OUT"/
rsync -a spark1:gc/worker-*.log "$OUT"/ 2>/dev/null
exit $rc
