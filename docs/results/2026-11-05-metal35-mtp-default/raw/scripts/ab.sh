#!/bin/zsh
# MTP default vs --spec off, same binary, served, ABAB. Inside a fleetq slot (FLEETQ_JOB set; localeval then runs inside it).
# arm "mtp" = no --spec flag at all (the recipe default must be announced on stderr), arm "off" = --spec off.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/flip-metal-mtp; BIN=${BIN:-$W/.scratch/intarsia.branch}; O=$W/.scratch/logs/${OUT:-ab}; mkdir -p $O
M=/Users/josef/models/Qwen3.6-35B-A3B-NVFP4; PORT=18491
ORDER=(${=ORDER:-mtp off mtp off})
n=0
for arm in $ORDER; do
  n=$((n+1)); tag=r$n-$arm
  case $arm in mtp) SP=();; off) SP=(--spec off);; esac
  CMD=($BIN serve --backend metal --model $M --served-model-name intarsia --max-seq 20000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT $SP)
  echo "## $tag $(date +%T)"
  /usr/bin/time -l -o $O/$tag-time.txt $CMD > $O/$tag-server.out 2> $O/$tag-server.err < /dev/null &
  TP=$!
  up=0; for i in {1..300}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $TP 2>/dev/null || break; sleep 1; done
  [ $up = 1 ] || { echo "SERVER FAILED $tag"; tail -3 $O/$tag-server.err; kill $TP; exit 1; }
  SPID=$(pgrep -P $TP | head -1); echo "server pid $SPID"
  ( while kill -0 $SPID 2>/dev/null; do footprint $SPID 2>/dev/null | grep -m1 'Footprint:' | sed "s/^/$(date +%T) /"; sleep 3; done ) > $O/$tag-footprint.txt &
  FP=$!
  grep -c "default of recipe qwen3_5_moe on metal" $O/$tag-server.err | sed "s/^/default-line count: /"
  (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia --sizes ${SIZES:-1024,4096,16384} --gen ${GEN:-256} --reps ${REPS:-3} \
     --thinking off --label "flipmtp-$tag" --recipe "${CMD[*]}" > $O/$tag-localeval.out 2> $O/$tag-localeval.err; echo "localeval speed exit $?")
  IDS=(); [ -z "${NOIDS:-}" ] && [ $n -le 2 ] && IDS=($O/$tag-ids.json 256)
  (cd /opt/code/deepeval && uv run python $W/.scratch/nat.py http://127.0.0.1:$PORT/v1 $O/$tag-nat.json ${NATSIZES:-1024,4096,16384} ${NATREPS:-3} $IDS > $O/$tag-nat.out 2> $O/$tag-nat.err; echo "nat exit $?")
  grep -a "spec:" $O/$tag-server.err | tail -2
  kill -INT $SPID; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill $FP 2>/dev/null; wait $TP 2>/dev/null
  grep -E "maximum resident|peak memory footprint" $O/$tag-time.txt
done
echo "## done $(date +%T)"
