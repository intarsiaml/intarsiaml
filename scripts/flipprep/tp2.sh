#!/usr/bin/env bash
# flipprep gate 1: the 35B at tp = 2 ACROSS the two Sparks (rank 0 = spark1 `--tp-remote`, rank 1 = spark2 `tp-worker`, NCCL over CX-7), flip binary,
# vLLM up on both. Two arms, each one `serve` (no --max-batch: batched MTP never runs at tp > 1) and ids.py (greedy and seeded T = 1.0, B = 1, 2):
#   R2  --spec off          (the ids reference)
#   F2  --spec mtp --spec-k 2   (no --mtp-draft-vocab: the recipe's 131072 applies; the rank-0 head shard is 124160 rows)
# ONE fleetq job over both boxes:
#   fleetq run --host spark1,spark2 --quiet gpu,cpu --mem spark1=40,spark2=40 --expect 45 --label "flipprep: tp2 gate" -- scripts/flipprep/tp2.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$S/../.." && pwd)"
L=${OUT:-$REPO/docs/results/2026-11-15-flipprep/raw/tp2}; mkdir -p "$L"
BIN='$HOME/intarsia-flipprep/bin/intarsia-flip'
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
WIP=10.100.152.3; WPORT=29555; PORT=8091; URL=http://spark-4e77:$PORT/v1
NCCLENV="INTARSIA_NCCL_LIB=\$HOME/intarsia-nccl/libnccl.so.2 NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=WARN"
log() { echo "$(date +%T) $*" | tee -a "$L/steps.log"; }
r() { local h=$1; shift; ssh -n -o BatchMode=yes "$h" "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
avail() { r $1 "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo"; }
kill_all() { for h in spark1 spark2; do r $h "touch ~/intarsia-flipprep/stop-worker; pkill -TERM -f '[i]ntarsia-flipprep/bin/'; for i in \$(seq 1 25); do pgrep -f '[i]ntarsia-flipprep/bin/' >/dev/null || break; sleep 1; done; pkill -KILL -f '[i]ntarsia-flipprep/bin/'; true"; done; }
GPID=""
guard() { # kill our processes on either box when MemAvailable < 10 GB
  while true; do for h in spark1 spark2; do a=$(avail $h 2>/dev/null || echo 99999); echo "$(date +%T) $h $a" >> "$L/mem.txt"; if [ "${a:-99999}" -lt 10240 ]; then echo "$(date +%T) KILLED $h $a" >> "$L/mem.txt"; kill_all; fi; done; sleep 5; done
}
cleanup() { trap - EXIT INT TERM; [[ -n "$GPID" ]] && kill $GPID 2>/dev/null; kill_all; for h in spark1 spark2; do rsync -a $h:intarsia-flipprep/tp2out/ "$L/$h/" 2>/dev/null; done; log end; }
trap cleanup EXIT; trap 'cleanup; exit 130' INT TERM
for h in spark1 spark2; do r $h "mkdir -p ~/intarsia-flipprep/tp2out; rm -f ~/intarsia-flipprep/stop-worker; echo $h: \$($BIN --version | head -1); free -g | sed -n 2p" | tee -a "$L/steps.log"; done
log "vLLM: spark1 $(curl -s -o /dev/null -w '%{http_code}' http://100.120.88.104:8000/v1/models) spark2 $(curl -s -o /dev/null -w '%{http_code}' http://100.96.95.89:8000/v1/models)"
guard & GPID=$!
for arm in ${ARMS:-R2 F2}; do
  case $arm in R2) SPEC="--spec off" ;; F2) SPEC="--spec mtp --spec-k 2" ;; F0) SPEC="--spec mtp --spec-k 2 --mtp-draft-vocab 0" ;; esac
  FL="-m $M35 --backend cuda --max-seq 8192 --mem-budget 30 --profile-every 0 --tp 2 $SPEC"
  log "== arm $arm: $FL"
  r spark2 "rm -f ~/intarsia-flipprep/stop-worker; nohup bash -c 'env $NCCLENV $BIN tp-worker --listen $WIP:$WPORT --serving $FL >> ~/intarsia-flipprep/tp2out/worker-$arm.log 2>&1' > /dev/null 2>&1 < /dev/null &"
  r spark1 "nohup env $NCCLENV $BIN serve $FL --tp-remote $WIP:$WPORT --port $PORT --host 0.0.0.0 --served-model-name guidance > ~/intarsia-flipprep/tp2out/serve-$arm.log 2>&1 < /dev/null &"
  up=0; for i in $(seq 1 120); do curl -sf -m 3 http://spark-4e77:$PORT/health >/dev/null && { up=1; break; }; sleep 3; done
  if [[ $up == 1 ]]; then
    log "arm $arm up; spark1 $(avail spark1) MB, spark2 $(avail spark2) MB available; $(r spark1 "grep -aE 'draft-vocab|vocab-parallel|not applicable|tp-remote|batched MTP|spec mtp' ~/intarsia-flipprep/tp2out/serve-$arm.log | cut -c1-220" | tr '\n' ';')"
    python3 "$S/ids.py" "$URL" guidance "$L/ids-$arm.json" "http://spark-4e77:$PORT/metrics" 2>&1 | tee -a "$L/steps.log"
  else
    log "arm $arm did not come up"; r spark1 "tail -15 ~/intarsia-flipprep/tp2out/serve-$arm.log" | tee -a "$L/steps.log"; r spark2 "tail -15 ~/intarsia-flipprep/tp2out/worker-$arm.log" | tee -a "$L/steps.log"
  fi
  kill_all
done
python3 "$S/ids_compare.py" "$L" ${ARMS:-R2 F2} 2>&1 | tee -a "$L/steps.log" "$L/ids-compare.txt"
