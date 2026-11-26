#!/usr/bin/env bash
# ON tatooine (lane ink-hip-batch), called over ssh by the Mac driver (serve_window.sh) inside a held window (hold.sh):
#   serve_ctl.sh start <bin-name> <visit-name> <serve args...>   start `intarsia serve` in container ink-hip-batch-run, port 8131;
#                                                                 prints "IP <container ip>" once /health answers (<= 150 s)
#   serve_ctl.sh stop <visit-name>                                stop it; prints the visit's VRAM peak (sysfs, both cards, MiB)
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-batch
S=$REPO/.scratch
OUT=$REPO/logs/serve
mkdir -p "$OUT"
C=ink-hip-batch-run
cmd=$1; shift
case "$cmd" in
  start)
    bin=$1 name=$2; shift 2
    docker rm -f $C >/dev/null 2>&1
    date +%s.%N > "$OUT/$name.t0"
    nohup bash "$S/run.sh" /work/.scratch/bin/intarsia-$bin serve -m /models/Inkling-2-Mini --backend hip --host 0.0.0.0 --port 8131 \
      --model-id ink "$@" > "$OUT/$name.log" 2>&1 < /dev/null &
    for i in $(seq 1 150); do
      ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $C 2>/dev/null)
      if [ -n "$ip" ] && curl -sf -m 2 "http://$ip:8131/health" > /dev/null 2>&1; then
        # the forwarder: http://tatooine:18131 -> the container (killed by `stop`)
        [ -f "$OUT/fwd.pid" ] && kill "$(cat "$OUT/fwd.pid")" 2>/dev/null
        nohup python3 "$S/fwd.py" 18131 "$ip" 8131 > "$OUT/fwd.log" 2>&1 < /dev/null &
        echo $! > "$OUT/fwd.pid"
        sleep 1
        echo "IP $ip after ${i}s"; exit 0
      fi
      if [ "$i" -gt 8 ] && ! docker ps -q -f name=$C | grep -q .; then echo "EXITED"; tail -20 "$OUT/$name.log"; exit 1; fi
      sleep 1
    done
    echo "TIMEOUT"; tail -20 "$OUT/$name.log"; exit 1 ;;
  metrics)
    ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $C 2>/dev/null)
    curl -s -m 5 "http://$ip:8131/metrics" ;;
  stop)
    name=$1
    [ -f "$OUT/fwd.pid" ] && kill "$(cat "$OUT/fwd.pid")" 2>/dev/null; rm -f "$OUT/fwd.pid"
    docker rm -f $C >/dev/null 2>&1
    t0=$(cat "$OUT/$name.t0" 2>/dev/null || echo 0)
    awk -v t0="$t0" -v n="$name" '$1 >= t0 { if ($2 > m0) m0 = $2; if ($3 > m1) m1 = $3 } END { printf "VRAM peak %s: card0 %.0f MiB, card1 %.0f MiB\n", n, m0 / 1048576, m1 / 1048576 }' "$REPO/logs/vram.log"
    grep -aE 'continuous batching|prefill chunk|context|KV|kv cache|memory|budget|Error|error' "$OUT/$name.log" | head -14 ;;
esac
