# sourced: start_server <name> <port> <args...>  -> sets SPID; stop_server kills only the PID we started
# BIN: a private copy of the binary built from branch metal-cb-gates (70b15eb + scheduler trace hook 0bd5db7, 75ecb17)
S=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad
BIN=${BIN:-$S/bin/intarsia-gates2}
M=$HOME/models/Qwen3-8B
SPID=""
start_server() {
  local name=$1 port=$2; shift 2
  $BIN serve --model $M --backend metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui --port $port "$@" > $O/server-$name.log 2>&1 &
  SPID=$!
  for i in {1..180}; do
    curl -sf http://127.0.0.1:$port/health > /dev/null 2>&1 && { echo "server $name pid $SPID ready on $port"; return 0; }
    kill -0 $SPID 2>/dev/null || { echo "server $name died"; tail -5 $O/server-$name.log; return 1; }
    sleep 1
  done
  echo "server $name not ready"; return 1
}
stop_server() { [ -n "$SPID" ] && kill $SPID 2>/dev/null; [ -n "$SPID" ] && wait $SPID 2>/dev/null; SPID=""; }
