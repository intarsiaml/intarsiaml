#!/usr/bin/env bash
# Lane c27: the Qwen3.8-27B on CUDA at tp=2 across the two Sparks (rank 0 = spark1 `generate --tp-remote`, rank 1 = spark2 `tp-worker`), vLLM left up.
# Runs HERE (the Mac) as the one command of ONE fleetq job that holds both Sparks:
#   fleetq run --host spark1,spark2 --quiet gpu,cpu --mem spark1=30,spark2=30 --expect 25 --label "c27 27B tp2" -- scripts/c27/tp2.sh
# Needs ~/intarsia-c27/target-src/release/intarsia and ~/intarsia-c27/models/Qwen3.8-27B-Quark-AWQ-MXFP4 on both boxes, prompts from tools/prompts/corpus_docs.txt in ~/intarsia-c27/src.
# Env: TREE (src), OUT (raw dir name), SIZES (1k 4k), ARMS ("off mtp": plain, then --spec mtp to record whether a remote rank takes it), NTOK (192)
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$S/../.." && pwd)"
TREE=${TREE:-src}; OUT=${OUT:-tp2-$(date +%H%M)}
L=$REPO/docs/results/2026-11-07-cuda27-decode/raw/$OUT; mkdir -p "$L"
R='$HOME/intarsia-c27'; BIN="$R/target-$TREE/release/intarsia"; M="$R/models/Qwen3.8-27B-Quark-AWQ-MXFP4"
WIP=10.100.152.3; WPORT=29556
NCCLENV="INTARSIA_NCCL_LIB=\$HOME/intarsia-nccl/libnccl.so.2 NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=WARN NCCL_MAX_NCHANNELS=8"
log() { echo "$(date +%T) $*" | tee -a "$L/steps.log"; }
r() { local h=$1; shift; ssh -n -o BatchMode=yes "$h" "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
XO='~/intarsia-c27/xout'
cleanup() {
  for h in spark1 spark2; do r $h "touch $R/stop-worker; pkill -TERM -f '[i]ntarsia-c27/target-$TREE/release/intarsia'; sleep 2; pkill -KILL -f '[i]ntarsia-c27/target-$TREE/release/intarsia'; true"; done
  rsync -a "spark1:intarsia-c27/xout/$OUT/" "$L/" 2>/dev/null; rsync -a "spark2:intarsia-c27/xout/$OUT/" "$L/spark2/" 2>/dev/null
  trap - EXIT INT TERM
}
trap cleanup EXIT INT TERM
for h in spark1 spark2; do r $h "mkdir -p $XO/$OUT; rm -f $R/stop-worker; echo $h free=\$(free -g | awk '/Mem:/ {print \$7}')GB"; done | tee -a "$L/steps.log"
FLAGS="-m $M --backend cuda --tp 2 --max-seq 8192 --mem-budget 24 --profile-every 0"
for arm in ${ARMS:-off mtp}; do for sz in ${SIZES:-1k 4k}; do
  case $sz in 1k) ch=3300;; 4k) ch=14035;; esac
  spec="--spec off"; [ $arm = mtp ] && spec="--spec mtp --spec-k 2"
  r spark2 "rm -f $R/stop-worker; nohup bash -c 'env $NCCLENV $BIN tp-worker --listen $WIP:$WPORT $FLAGS $spec > $XO/$OUT/worker-$sz-$arm.log 2>&1' > /dev/null 2>&1 < /dev/null &"
  sleep 3
  log "tp2 $arm $sz"
  r spark1 "env $NCCLENV timeout 600 $BIN generate $FLAGS $spec --tp-remote $WIP:$WPORT --chat --no-think --temperature 0 --ignore-eos --show-ids \
    -p \"\$(head -c $ch $R/src/tools/prompts/corpus_docs.txt)\" --max-tokens ${NTOK:-192} > $XO/$OUT/gen-$sz-$arm.out 2> $XO/$OUT/gen-$sz-$arm.err; echo rc=\$?; grep -a 'finish=' $XO/$OUT/gen-$sz-$arm.err | tail -1 | cut -c1-220; grep -a 'unsupported\|Error' $XO/$OUT/gen-$sz-$arm.err | head -2 | cut -c1-220; free -g | awk '/Mem:/ {print \"avail \" \$7}'" | tee -a "$L/steps.log"
  r spark2 "pkill -TERM -f '[t]p-worker --listen'; sleep 3; pkill -KILL -f '[t]p-worker --listen'; true"
done; done
log done
