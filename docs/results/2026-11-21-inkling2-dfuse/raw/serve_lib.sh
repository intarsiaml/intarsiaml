# Sourced by this lane's job scripts that run `intarsia serve` (coordinator 2026-11-22, after #10882 left an orphan server on
# 8099: `run $m $B serve ... &` put the server under a function subshell, so `kill $!` killed the subshell, not the server).
#   serve_start LOG ENV... -- start the server (env assignments first) on a free port of its own, in the foreground process
#                             tree of this job: SRV_PID is the server's supervisor itself; wait for /v1/models and check the
#                             listener is our supervisor's own child; sets SRV_URL. Returns 1 (and stops it) otherwise.
#   serve_stop            -- SIGTERM the supervisor (it stops its child), wait, then make sure nothing of ours listens.
# An EXIT trap stops a server that is still up when the job ends for any reason.
SRV_PID=""; SRV_PORT=""; SRV_URL=""
serve_stop() {
  [ -n "$SRV_PID" ] || return 0
  kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null
  for l in $(lsof -nP -iTCP:"$SRV_PORT" -sTCP:LISTEN -t 2>/dev/null); do
    if [ "$(ps -o ppid= -p "$l" | tr -d ' ')" = "$SRV_PID" ] || [ "$l" = "$SRV_PID" ]; then kill "$l" 2>/dev/null; fi
  done
  SRV_PID=""
}
trap 'serve_stop' EXIT
serve_start() {
  log=$1; shift
  SRV_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
  env "$@" $B serve --model $CK --backend metal --host 127.0.0.1 --port $SRV_PORT --max-batch 1 --max-seq 20480 --served-model-name inkling-2-mini > "$log" 2>&1 &
  SRV_PID=$!
  SRV_URL=http://127.0.0.1:$SRV_PORT/v1
  n=0
  until curl -sf $SRV_URL/models > /dev/null; do
    sleep 1; n=$((n + 1))
    if [ $n -gt 300 ] || ! kill -0 $SRV_PID 2>/dev/null; then echo "serve did not come up (see $log)"; serve_stop; return 1; fi
  done
  for l in $(lsof -nP -iTCP:$SRV_PORT -sTCP:LISTEN -t 2>/dev/null); do
    if [ "$l" != "$SRV_PID" ] && [ "$(ps -o ppid= -p "$l" | tr -d ' ')" != "$SRV_PID" ]; then
      echo "port $SRV_PORT listener $l is not our server ($SRV_PID)"; serve_stop; return 1
    fi
  done
  echo "serve up: pid $SRV_PID, $SRV_URL"
}
