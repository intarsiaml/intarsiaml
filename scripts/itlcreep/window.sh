#!/usr/bin/env bash
# itlcreep lane: one server, two phases of soak2 traffic, client on the box (loopback) and on the Mac.
#   BOX=spark1 fleetq run --host spark1 --quiet gpu,cpu --mem 55 --expect 55 --label "itlcreep: window" -- bash scripts/itlcreep/window.sh [P1-min=22] [P2-min=20]
# Server: `serve --max-batch 2 --prefix-store on` (default 8 GiB host budget, MTP default on CUDA), the soak2 configuration, plus INTARSIA_CB_TRACE=1
# (scheduler events with ms wall-clock stamps to stderr; nothing is stored, no other behaviour change).
# Phase 1 (P1 min): the whole soak2 traffic (K=2, both workers) from a client ON THE BOX against localhost: seeds 401...
# Phase 2 (P2 min): worker 0's conversations from the Mac (over Tailscale/Wi-Fi, seeds 501...) and worker 1's from the box (localhost, seeds 601...)
#   at the same time, so the two vantage points see the same server in the same minutes (same K=2 load, conversations split between the two).
set -uo pipefail
P1=${1:-22}; P2=${2:-20}
WLOG=window
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RH='$HOME/intarsia-itlcreep'
[[ -f $OUT/replay-plan.json ]] || cp "$REPO/docs/results/2026-11-14-soak2/raw/replay-plan.json" "$OUT/replay-plan.json"
prep_box
SERVE_ENV="INTARSIA_CB_TRACE=1"
start_serve S --max-batch 2 --prefix-store on || exit 1
# Mac-side environment record: a ping per second to the box for the whole window (Tailscale path) and the Mac's load average per minute
( ping -i 1 -c $(( (P1+P2+10)*60 )) "$HN" 2>&1 | perl -MTime::HiRes=time -ne 'BEGIN{$|=1} printf "%.3f %s", time, $_' > "$OUT/ping.txt" ) &
( while true; do echo "$(date +%s) $(sysctl -n vm.loadavg)"; sleep 60; done > "$OUT/mac-load.txt" ) & MLOAD=$!
trap 'kill $MLOAD 2>/dev/null; pkill -f "ping -i 1 -c .* $HN" 2>/dev/null; stop_server; log "window end"' EXIT
# smoke: one short request, look at the trace format
curl -s --max-time 60 "$URL/chat/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",\"temperature\":0,\"max_tokens\":40,\"chat_template_kwargs\":{\"enable_thinking\":false},\"messages\":[{\"role\":\"user\",\"content\":\"[itlcreep $RANDOM] Say hello in ten words.\"}]}" | python3 -c "import sys,json; d=json.load(sys.stdin); print('smoke finish', d['choices'][0]['finish_reason'], d['usage']['completion_tokens'])" | tee -a "$OUT/window.log"
TR=$(rsh "grep -m6 '^cb-trace' $W/serve-S.log | cut -c1-200" | tr '\n' '|'); log "trace sample: $TR"
[[ -n $TR ]] || { log "ABORT: INTARSIA_CB_TRACE=1 produced no cb-trace lines: no server-side timestamps"; finish_serve S; exit 4; }
# box client smoke (imports, LOCALEVAL_SRC, python3 on the box): 2 conversations x 1 turn, 20 tokens, throwaway seed
rsh "mkdir -p $W/smoke; cd $RH/client && env LOCALEVAL_SRC=$RH/client python3 itlcreep/replay3.py --url http://127.0.0.1:$PORT/v1 --model $MODEL --arm X --seed 9001 --streams 2 --convs 2 --turns 1 --max-tokens 20 --plan $RH/client/replay-plan.json --trace $W/smoke/trace.json --out $W/smoke --ignore-busy 2>&1 | tail -4" | tee -a "$OUT/window.log"
rsh "test -s $W/smoke/X-interleaved-seed9001-k2.summary.json" || { log "ABORT: box client smoke wrote no summary"; finish_serve S; exit 5; }
clockoff() { # clockoff <label>: the Mac's clock minus the box's, bracketing one ssh round trip
  python3 - "$1" "$HOST" <<'PY' | tee -a "$OUT/clock.txt"
import subprocess, sys, time
t1 = time.time(); r = float(subprocess.check_output(["ssh", "-n", "-o", "BatchMode=yes", sys.argv[2], "date +%s.%N"]).decode()); t2 = time.time()
print(f"{sys.argv[1]} mac_mid={(t1+t2)/2:.3f} box={r:.3f} offset_mac_minus_box_s={(t1+t2)/2-r:+.3f} rtt_s={t2-t1:.3f}")
PY
}
clockoff start
sparkloop() { # sparkloop <tag> <seed0> <min> <outdir> <only|""> <ignore|""> -> prints the pid
  rsh "mkdir -p $4; nohup env TAG=$1 URL=http://127.0.0.1:$PORT/v1 SEED0=$2 MIN=$3 OUT=$4 PLAN=$RH/client/replay-plan.json SCRIPT=$RH/client/itlcreep/replay3.py PY=python3 LOCALEVAL_SRC=$RH/client ${5:+ONLY=$5} ${6:+IGNORE=$6} bash $RH/client/itlcreep/loop.sh > $4/loop.log 2>&1 < /dev/null & echo \$!"
}
waitloops() { # waitloops <spark pid or ""> <mac pid or "">: until both are gone; abort on a dead server
  while true; do
    local alive=0
    [[ -n "$1" ]] && rsh "kill -0 $1" 2>/dev/null && alive=1
    [[ -n "${2:-}" ]] && kill -0 "$2" 2>/dev/null && alive=1
    (( alive )) || break
    rsh "kill -0 $SPID" 2>/dev/null || { log "SERVER GONE"; return 1; }
    sleep 20
  done
}
# ---- phase 1: client on the box, whole soak2 traffic
log "phase 1: $P1 min, box-local client (K=2, both workers), seeds 401..; vLLM counters $(vcount)"
LP=$(sparkloop A 401 "$P1" "$W/p1" "" "")
waitloops "$LP" || { finish_serve S; exit 2; }
scrape "$OUT/serve-S.metrics.after-p1"
rsh "ls $W/p1/*.summary.json >/dev/null 2>&1" || { log "ABORT: phase 1 wrote no replay summary: $(rsh "tail -5 $W/p1/loop.log")"; finish_serve S; exit 6; }
log "phase 1 done; $(rsh "cat $W/p1/loop.log | tail -3" | tr '\n' '|')"
# ---- phase 2: Mac client (worker 0) and box client (worker 1) at the same time
clockoff phase2-start
log "phase 2: $P2 min, Mac client worker 0 (seeds 501..) + box client worker 1 (seeds 601..) concurrently"
LP=$(sparkloop B 601 "$P2" "$W/p2" 1 1)
( cd /opt/code/deepeval && TAG=M URL=$URL SEED0=501 MIN=$P2 OUT=$OUT/p2-mac PLAN=$OUT/replay-plan.json SCRIPT=$SC/replay3.py PY="uv run python" ONLY=0 IGNORE=1 bash "$SC/loop.sh" > "$OUT/loop-p2-mac.log" 2>&1 ) & MP=$!
waitloops "$LP" "$MP" || { finish_serve S; exit 2; }
scrape "$OUT/serve-S.metrics.after-p2"; clockoff phase2-end
log "phase 2 done; box: $(rsh "tail -2 $W/p2/loop.log" | tr '\n' '|') Mac: $(tail -2 "$OUT/loop-p2-mac.log" | tr '\n' '|')"
rsh "cd $W && tar czf - p1 p2" | tar xzf - -C "$OUT"
log "client results copied back; vLLM counters $(vcount)"
finish_serve S
