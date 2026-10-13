#!/usr/bin/env bash
# Start `intarsia serve` on spark1, wait until it is healthy, run a command against it, stop it.
# Run this INSIDE a fleetq lease on spark1 (`fleetq run --host spark1 --mem N -- ...` sets
# FLEETQ_JOB, which lets localeval carry on instead of queueing behind our own server):
#
#   fleetq run --host spark1 --exclusive --mem 24 --label "intarsiaml bench" -- \
#       deploy/localeval/serve-and-bench.sh --mem-budget 24 -- \
#       uv run --project /opt/code/deepeval localeval speed spark-intarsia --sizes 128,1024
#
# Arguments before `--` go to `intarsia serve` (default model: Qwen3-8B, cuda, port 8090, all
# interfaces); the command after `--` runs here. The server is always stopped on exit.
set -euo pipefail

HOST=${INTARSIA_HOST:-spark1}
PORT=${INTARSIA_PORT:-8090}
URL_HOST=${INTARSIA_URL_HOST:-spark-4e77}
BIN=${INTARSIA_BIN:-'~/intarsiaml/target/release/intarsia'}
MODEL=${INTARSIA_MODEL:-/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}

if [ -z "${FLEETQ_JOB:-}" ]; then
    echo "serve-and-bench: FLEETQ_JOB is not set; run this under 'fleetq run --host $HOST ...'" >&2
    exit 2
fi
serve_args=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do serve_args+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "serve-and-bench: no command after --" >&2; exit 2; }

PIDFILE=/tmp/intarsia-serve-$PORT.pid
stop() {
    ssh "$HOST" "if [ -f $PIDFILE ]; then kill \$(cat $PIDFILE) 2>/dev/null; rm -f $PIDFILE; fi" || true
}
trap stop EXIT INT TERM

ssh "$HOST" "nohup $BIN serve --backend cuda --model $MODEL --port $PORT --host 0.0.0.0 ${serve_args[*]:-} \
    > /tmp/intarsia-serve-$PORT.log 2>&1 < /dev/null & echo \$! > $PIDFILE"
echo "serve-and-bench: waiting for http://$URL_HOST:$PORT/health" >&2
for _ in $(seq 150); do
    curl -sf "http://$URL_HOST:$PORT/health" > /dev/null && break
    sleep 2
done
curl -sf "http://$URL_HOST:$PORT/health" > /dev/null || { echo "serve-and-bench: server did not come up" >&2; exit 1; }
"$@"
