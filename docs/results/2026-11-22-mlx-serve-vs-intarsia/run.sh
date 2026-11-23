#!/bin/bash
# ONE fleetq quiet job (--quiet gpu,cpu): box check, then arms in the order of ORDER (arms.conf), each with
#   speed-plain (server with --max-seq 20480) and conc-plain-1k (intarsia: --max-batch 16 --max-seq 8192; mlx-serve: same server, default slots)
# through localeval (inherits FLEETQ_JOB, so it runs inside this slot). Servers are stopped by arms.sh traps.
. /opt/code/intarsiaml/.worktrees/mlx-serve-bench/arms.sh
. $B/arms.conf          # LANE_A.. , ORDER, STOP_AFTER_EPOCH, BOXCHECK_MIN
R=/opt/code/deepeval
L=$B/logs/run; mkdir -p $L
REV=b85c3cfa615fc413cc9af3379f491b963f3b586a
note() { echo "$(date +%H:%M:%S) $*" | tee -a $L/progress.log; }

model_id() {  # first id served on $1
  python3 - $1 <<'PY'
import json,sys,urllib.request
try:
    d=json.load(urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/v1/models",timeout=5))
    print(d["data"][0]["id"])
except Exception:
    print("inkling-2-mini")
PY
}

lecall() {  # label cells arm-key maxctx maxseqs lane [drafter]
  local label=$1 cells=$2 arm=$3 ctx=$4 seqs=$5 lane=$6 drafter=$7 port model
  port=$(port_of ${arm:0:1}); model=$(model_id $port)
  note "localeval $label cells=$cells model=$model lane=$lane"
  [ $port = $PM ] && { curl -s -m 5 http://127.0.0.1:$port/metrics > $L/$label.metrics_before.txt; curl -s -m 5 http://127.0.0.1:$port/metrics.json > $L/$label.metrics_before.json; }
  local dr=(); [ -n "$drafter" ] && dr=(--drafter "$drafter")
  ( cd $R && uv run localeval suite --url http://127.0.0.1:$port/v1 --model "$model" --tier adhoc --cells "$cells" \
      --thinking default --label "$label" --tokenizer-repo thinkingmachines/Inkling-2-Mini --tokenizer-revision $REV \
      --max-context $ctx --max-num-seqs $seqs --lane "$lane" "${dr[@]}" --recipe "$(cmd_$arm)" ) > $L/$label.out 2>&1 &
  wait $!; local rc=$?
  [ $port = $PM ] && { curl -s -m 5 http://127.0.0.1:$port/metrics > $L/$label.metrics_after.txt; curl -s -m 5 http://127.0.0.1:$port/metrics.json > $L/$label.metrics_after.json; }
  note "localeval $label exit $rc  $(grep -o 'run_id[^ ]*\|20261[0-9-]*-adhoc-[A-Za-z0-9._-]*' $L/$label.out | tail -1)"
}

boot() {  # arm-key tag ; starts the server and the memory sampler
  start_server $1 $L/server_$2.log >> $L/progress.log 2>&1 || return 1
  $B/memsample.sh $SRV_PID $L/mem_$2.txt &
}

box_check() {
  boot D boxcheck$1 || { note "box check: server failed" >&2; echo 0; return 1; }
  ( cd $R && uv run localeval speed --url http://127.0.0.1:$PI/v1 --model inkling-2-mini --sizes 128 --gen 128 --reps 2 --thinking default --no-unim --label mlxbench-boxcheck --recipe "$(cmd_D)" ) > $L/boxcheck$1.out 2>&1 &
  wait $!
  local d; d=$(ls -dt $R/results/*mlxbench-boxcheck* 2>/dev/null | head -1)
  python3 - "$d/result.json" > $L/boxcheck$1.val <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); print(r["speed"]["rows"][0]["decode_tok_s"]["median"])
PY
  stop_server
  tail -1 $L/boxcheck$1.val
}

run_arm() {  # pass arm
  local pass=$1 a=$2 lane drafter="" ctxc=20480
  if [ $(date +%s) -gt $STOP_AFTER_EPOCH ]; then note "pass $pass arm $a: past STOP_AFTER_EPOCH, not started"; return; fi
  eval lane=\$LANE_$a
  case $a in B) drafter=dspark;; A) drafter=${DRAFTER_A:-};; esac
  local seqs_speed=1
  note "=== pass $pass arm $a"
  boot $a ${pass}_$a || { note "arm $a failed to start"; return; }
  lecall mlxbench-$pass-$a-speed speed-plain $a 20480 1 "$lane" "$drafter"
  case $a in
    A|B|C)  lecall mlxbench-$pass-$a-conc conc-plain-1k $a 20480 1 "$lane" "$drafter"; stop_server ;;
    D)    stop_server
          boot ${a}c ${pass}_${a}c || { note "arm ${a}c (max-batch 16) failed to start: conc cell not run"; return; }
          lecall mlxbench-$pass-$a-conc conc-plain-1k ${a}c 8192 16 "$lane" "$drafter"; stop_server ;;
  esac
  if [ $a = A ] || [ $a = C ]; then cp ~/.mlx-serve/logs/mlx-serve-$PM.log $L/mlxlog_${pass}_$a.log 2>/dev/null; fi
  note "arm $a mem (max footprint sample; epoch rss_mb footprint_mb): $(sort -k3 -n $L/mem_${pass}_$a.txt | tail -1); conc server: $(sort -k3 -n $L/mem_${pass}_${a}c.txt 2>/dev/null | tail -1)"
}

S0=$(date +%s); EXPECT_MIN=${EXPECT_MIN:-400}
LIM=$((S0 + (EXPECT_MIN-55)*60)); [ $LIM -lt $STOP_AFTER_EPOCH ] && STOP_AFTER_EPOCH=$LIM
note "start; order=$ORDER stop_after=$STOP_AFTER_EPOCH ($(date -r $STOP_AFTER_EPOCH +%a\ %H:%M))"
v=$(box_check _pre); note "box check decode tok/s (intarsia spec off, 128-token prompt): $v (need >= $BOXCHECK_MIN)"
if ! python3 -c "import sys; sys.exit(0 if float('${v:-0}')>=$BOXCHECK_MIN else 1)"; then
  note "box slow; waiting 5 min and retrying once"
  sleep 150 & wait $!; sleep 150 & wait $!
  v=$(box_check _pre2); note "box check retry: $v"
  python3 -c "import sys; sys.exit(0 if float('${v:-0}')>=$BOXCHECK_MIN else 1)" || { note "box still slow: aborting without measuring"; exit 3; }
fi
n=0
for a in $ORDER; do n=$((n+1)); run_arm p1 $a; done
v=$(box_check _post); note "box check AFTER all arms (same measure as the first): $v"
note "ALL_DONE"
