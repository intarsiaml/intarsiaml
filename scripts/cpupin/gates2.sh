#!/usr/bin/env bash
# cpupin gate (c) for the cells without a .txt (agent-edit, agent-batch: messages with a tool turn, which `generate -p` cannot take): served greedy replies of main's binary and the
# lane's must be equal (sha1 of the reply text per request, scripts/mcopy/cell.py). One server per binary, each cell 3 reps, plus the 5 other cells 1 rep as a cross-check of gate (b).
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 15 --label "cpupin: served sha vs main" -- ssh spark1 'bash ~/intarsiaml-cpupin/bin/gates2.sh'
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-cpupin; O=$R/out/gates2; mkdir -p $O; PORT=8297
H=/home/nvidia/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
nohup bash $R/bin/wd.sh > $O/wd.log 2>&1 < /dev/null &
WD=$!
pat() { echo "$R/bin/[a-z0-9-]+ serve.*--port $PORT"; }
stop() { pkill -f "$(pat)" 2>/dev/null; for i in $(seq 1 30); do pgrep -f "$(pat)" >/dev/null || break; sleep 1; done; }
trap 'stop; kill $WD 2>/dev/null' EXIT
for b in main new; do
  B=$R/bin/intarsia; [ $b = main ] && B=$R/bin/intarsia-main
  stop
  nohup $B serve --backend cuda --model $M35 --model-id m --max-batch 1 --max-seq 8192 --spec mtp --spec-k 2 --profile-every 0 --mem-budget 40 --no-supervisor --host 127.0.0.1 --port $PORT > $O/serve-$b.log 2>&1 < /dev/null &
  for i in $(seq 1 120); do pgrep -f "$(pat)" >/dev/null || { echo "serve died ($b)"; tail -5 $O/serve-$b.log; exit 1; }; curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 3; done
  echo "$b up $(date +%T) free: $(free -g | awk 'NR==2{print $7}') GB"
  python3 $R/src/scripts/mcopy/cell.py $PORT $R/cells/chat-short.json 1 warm > /dev/null 2>&1
  for c in agent-edit agent-batch copy3k json-xform chat-short reason code-gen; do
    reps=3; case $c in copy3k|json-xform|chat-short|reason|code-gen) reps=1 ;; esac
    python3 $R/src/scripts/mcopy/cell.py $PORT $R/cells/$c.json $reps g >> $O/$b.jsonl 2>&1
  done
done
python3 - $O <<'EOP'
import json, sys
d = sys.argv[1]
ld = lambda b: [json.loads(l) for l in open(f"{d}/{b}.jsonl") if l.startswith("{")]
m, n = ld("main"), ld("new"); fails = 0
for a, b in zip(m, n):
    ok = a["sha"] == b["sha"] and a["toks"] == b["toks"] and a["toks"] > 0
    fails += not ok
    print(("PASS" if ok else "FAIL"), "served", a["cell"], "rep", a["rep"], a["toks"], "tokens", a["sha"], b["sha"])
print("served sha gate: pairs", len(m), len(n), "fails", fails + abs(len(m) - len(n)))
EOP
