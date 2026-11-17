#!/bin/zsh
# Lane mchunkx: context for the findings: 35B decode step ms (cb-trace decode iteration p50) on main at --max-batch 16, 8 / 10 / 12 / 16 streams x 2 requests
# x 256 tokens (one visit, main binary, no verdict). Run under fleetq --quiet gpu,cpu. usage: step35.sh <outdir>
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mchunkx; O=$1; mkdir -p $O
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4; TXT=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-16384.txt; PORT=18693
INTARSIA_CB_TRACE=1 $W/.base/bin-main/intarsia serve --backend metal --model $M --served-model-name intarsia --max-seq 8192 --profile-every 0 --no-supervisor --no-ui \
  --host 127.0.0.1 --port $PORT --max-batch 16 > $O/server.out 2> $O/server.err < /dev/null &
TP=$!
for i in {1..300}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; kill -0 $TP 2>/dev/null || break; sleep 1; done
python3 $W/scripts/mchunkx/load.py --url http://127.0.0.1:$PORT/v1 --text $TXT --streams 2 --requests 1 --chars 430 --offset 50000 --gen 16 --out $O/warm.json > /dev/null
for n in 8 10 12 16; do
  echo "$(date +%s.%N) start s$n" >> $O/cells.txt
  python3 $W/scripts/mchunkx/load.py --url http://127.0.0.1:$PORT/v1 --text $TXT --streams $n --requests 2 --chars 430 --offset $((7919 + n*131)) --gen 256 --tag "s$n" --out $O/s$n.json | tee -a $O/cells.txt
  echo "$(date +%s.%N) end s$n" >> $O/cells.txt
done
kill -INT $TP; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null
python3 - <<PY | tee $O/step.txt
import re,statistics
win={}
for l in open("$O/cells.txt"):
    m=re.match(r"([\d.]+) (start|end) (s\d+)",l)
    if m: win.setdefault(m.group(3),[0,0])[m.group(2)=="end"]=float(m.group(1))
em=[]
for l in open("$O/server.err",errors="replace"):
    m=re.match(r"cb-trace-us t=(\d+) Emit \{(.*)\}",l)
    if m:
        kv=dict(re.findall(r"(\w+): (\d+)",m.group(2)))
        em.append((int(m.group(1))/1e6,(int(kv.get("prepare_us",0))+int(kv.get("verify_us",0))+int(kv.get("decode_us",0)))/1e3))
for tag,(t0,t1) in win.items():
    v=[e for t,e in em if t0<=t<=t1]
    if v: print(f"{tag}: decode iteration ms p50 {statistics.median(v):.1f} (n={len(v)})")
PY
gzip -f $O/server.err
