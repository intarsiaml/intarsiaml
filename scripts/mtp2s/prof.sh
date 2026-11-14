#!/usr/bin/env bash
# Lane mtp2s Step 1: nsys (node-level graph trace) of a served 35B decode window, one or more arms per call.
# Runs ON the Spark (ssh spark2 'bash -s ARM...' < prof.sh) inside a fleetq slot (--quiet gpu,cpu). vLLM stays up.
#   P2    default (batched_mtp = 1: no drafting at 2 streams), 2 streams
#   V2    INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1, k = 2, 2 streams
#   V2k1  the same at --spec-k 1
#   N2    INTARSIA_BATCHED_MTP=2 (solo verify passes), 2 streams
#   S1    default, 1 stream (the captured solo spec step)
#   P1    INTARSIA_BATCHED_MTP=0 (no drafting), 1 stream (plain decode)
#   VD    V2 + --mtp-draft-vocab 65536;  VDt / Vt: VD / V2 at temperature 1.0 (seeded)
# Extra env: GEN (tokens per stream in the profiled window, default 256), ENVPLUS (added to every arm's env).
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
L=$HOME/intarsia-mtp2s; O=$L/out/prof; mkdir -p $O
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
BIN=$L/bin/intarsia; PORT=8194; GEN=${GEN:-256}; PAT="^$L/bin/intarsia serve"
stop_srv() {
  [ -n "${SES:-}" ] && nsys stop --session=$SES >/dev/null 2>&1
  pkill -f "$PAT" 2>/dev/null
  for i in $(seq 1 20); do pgrep -f "$PAT" >/dev/null || break; sleep 1; done
  pkill -9 -f "$PAT" 2>/dev/null
  [ -n "${WP:-}" ] && kill $WP 2>/dev/null
  SES=""; WP=""; true
}
trap stop_srv EXIT
one_arm() {
  local ARM=$1 ENVX NS SA; export GEN_TEMP=0
  case $ARM in
    P2)   ENVX="X=1";                                              NS=2; SA="" ;;
    V2)   ENVX="INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1"; NS=2; SA="" ;;
    V2k1) ENVX="INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1"; NS=2; SA="--spec mtp --spec-k 1" ;;
    N2)   ENVX="INTARSIA_BATCHED_MTP=2";                           NS=2; SA="" ;;
    S1)   ENVX="X=1";                                              NS=1; SA="" ;;
    P1)   ENVX="INTARSIA_BATCHED_MTP=0";                           NS=1; SA="" ;;
    VD)   ENVX="INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1"; NS=2; SA="--spec mtp --spec-k 2 --mtp-draft-vocab 65536" ;;
    VDt)  ENVX="INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1"; NS=2; SA="--spec mtp --spec-k 2 --mtp-draft-vocab 65536"; export GEN_TEMP=1.0 ;;
    Vt)   ENVX="INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1"; NS=2; SA=""; export GEN_TEMP=1.0 ;;
    *) echo "unknown arm $ARM"; return 1 ;;
  esac
  SES=mtp2s$$$ARM
  rm -f $L/out/mem-prof-$ARM.txt
  env $ENVX ${ENVPLUS:-} nsys launch --session-new=$SES -t cuda,osrt --cuda-graph-trace=node \
    $BIN serve --backend cuda --model $M35 --model-id m --max-batch 2 $SA --profile-every 0 --mem-budget 40 --max-seq 8192 --no-supervisor --host 127.0.0.1 --port $PORT \
    > $O/serve-$ARM.log 2>&1 < /dev/null &
  sleep 3; local SP; SP=$(pgrep -f "$PAT" | head -1)
  nohup bash $L/memwatch.sh $SP prof-$ARM >/dev/null 2>&1 < /dev/null & WP=$!
  local i; for i in $(seq 1 100); do pgrep -f "$PAT" >/dev/null || { echo "serve died"; tail -20 $O/serve-$ARM.log; return 1; }; curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 3; done
  echo "== $ARM up $(date +%T) MemAvailable MB $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)"
  grep -a -E "batched MTP|spec" $O/serve-$ARM.log | head -4 | cut -c1-220
  python3 $L/src/scripts/mtp2s/gen2.py $PORT $NS 128 3 > $O/warm-$ARM.txt 2>&1   # captures graphs
  python3 $L/src/scripts/mtp2s/gen2.py $PORT $NS 128 3 >> $O/warm-$ARM.txt 2>&1
  curl -s http://127.0.0.1:$PORT/metrics | grep -E '^intarsia_(spec|batch|graph|decode)' | sed 's/{[^}]*}//' > $O/metrics0-$ARM.txt
  nsys start --session=$SES -o $O/$ARM --force-overwrite=true 2>&1 | tail -1
  python3 $L/src/scripts/mtp2s/gen2.py $PORT $NS $GEN 0 > $O/gen-$ARM.txt 2>&1
  nsys stop --session=$SES 2>&1 | tail -1
  curl -s http://127.0.0.1:$PORT/metrics | grep -E '^intarsia_(spec|batch|graph|decode)' | sed 's/{[^}]*}//' > $O/metrics1-$ARM.txt
  cat $O/gen-$ARM.txt
  echo "min MemAvailable MB: $(sort -k2 -n $L/out/mem-prof-$ARM.txt | head -1)"
  stop_srv
  nsys export --type sqlite --force-overwrite true -o $O/$ARM.sqlite $O/$ARM.nsys-rep >/dev/null 2>&1 && echo "exported $(du -h $O/$ARM.sqlite | cut -f1)"
}
for a in "$@"; do one_arm $a; done
