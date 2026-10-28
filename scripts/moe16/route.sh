#!/usr/bin/env bash
# Lane moe16: distinct routed experts per batched decode call (Qwen3.6-35B-A3B NVFP4, spark1), one GPU window:
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 25 --label "moe16: route stats" -- scripts/moe16/route.sh
# Serves the dev CLI with INTARSIA_MOE_ROUTE_STATS=1 and INTARSIA_BATCH_GRAPHS=0 (eager batched steps, bitwise the graphs; the
# timings of this run mean nothing), then four loads, snapshotting the cumulative `moe-route` lines after each:
#   p1 burst.py 16 streams (the msd/msd2 profile load), p2 burst.py 8, p3 localeval speed --concurrency 8,16 (the gated benchmark's
#   prompts), p4 burst.py 16 with DISTINCT=1 (a different document slice per stream: a less-shared bound).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}; HN=${HN:-spark-4e77}; PORT=${PORT:-8191}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
RAW=$REPO/docs/results/2026-10-28-moe16-routed-experts/raw/route; mkdir -p "$RAW"
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
O=intarsiaml-moe16/out/route
rsh() { ssh -n -o BatchMode=yes "$HOST" "export FLEETQ_JOB=$FLEETQ_JOB PATH=/usr/local/cuda/bin:\$PATH; $*"; }
stop() { rsh "pkill -TERM -f 'intarsiaml-moe16/bin/dev serve' ; sleep 5; pkill -KILL -f 'intarsiaml-moe16/bin/dev serve'" || true; }
trap stop EXIT INT TERM
rsh "mkdir -p ~/$O"; scp -q "$HERE/burst.py" "$HOST:$O/"
rsh "nohup env INTARSIA_MOE_ROUTE_STATS=1 INTARSIA_BATCH_GRAPHS=0 ~/intarsiaml-moe16/bin/dev serve --backend cuda --model $M --port $PORT \
  --host 0.0.0.0 --served-model-name intarsia --max-seq 8192 --mem-budget 40 --profile-every 0 --max-batch 16 < /dev/null > ~/$O/serve.out 2> ~/$O/serve.err &"
up=0; for i in $(seq 1 150); do curl -sf "http://$HN:$PORT/health" > /dev/null && { up=1; break; }; sleep 2; done
[[ $up == 1 ]] || { echo "server not up"; rsh "tail -20 ~/$O/serve.err"; exit 1; }
snap() { rsh "grep -a '^moe-route' ~/$O/serve.err | awk '{t[\$2]=\$0} END {for (k in t) print t[k]}' | sort -t= -k2 -n" > "$RAW/snap-$1.txt"; echo "== $1"; cat "$RAW/snap-$1.txt"; }
rsh "python3 ~/$O/burst.py $PORT 2 32 1" | tee "$RAW/warm.txt"; snap p0
rsh "python3 ~/$O/burst.py $PORT 16 256 1" | tee "$RAW/p1-burst16.txt"; snap p1
rsh "python3 ~/$O/burst.py $PORT 8 256 1" | tee "$RAW/p2-burst8.txt"; snap p2
(cd /opt/code/deepeval && FLEETQ_JOB=$FLEETQ_JOB uv run localeval speed --url "http://$HN:$PORT/v1" --model intarsia --thinking off --sizes 1024 \
  --gen 256 --reps 1 --concurrency 8,16 --label moe16-route --no-unim) > "$RAW/p3-localeval.out" 2>&1; tail -8 "$RAW/p3-localeval.out"; snap p3
rsh "DISTINCT=1 python3 ~/$O/burst.py $PORT 16 256 1" | tee "$RAW/p4-burst16-distinct.txt"; snap p4
stop; trap - EXIT
rsh "grep -a '^moe-route' ~/$O/serve.err | wc -l; grep -aiE 'error|panic' ~/$O/serve.err | head -5"
scp -q "$HOST:$O/serve.err" "$RAW/serve.err"
