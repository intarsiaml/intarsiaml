#!/bin/zsh
# prefix reuse at chunk 2048 on a 4k prompt: warm (turn 2 reusing turn 1's cache) vs cold (fresh server) greedy text.
BIN=$1; O=$2; mkdir -p $O; PORT=18473
M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4; PF=/opt/code/intarsiaml/.worktrees/m27/docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-4096.txt
cat > $O/req.py <<'P'
import json,sys,urllib.request
port,pf,ext,out=sys.argv[1],sys.argv[2],sys.argv[3]=="1",sys.argv[4]
p=open(pf).read()
if ext: p+="\n\nQuestion: in one sentence, what is the text above about?\nAnswer:"
r=urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",json.dumps({"model":"intarsia","prompt":p,"max_tokens":48,"temperature":0}).encode(),{"Content-Type":"application/json"})
d=json.load(urllib.request.urlopen(r,timeout=900))
json.dump({"text":d["choices"][0]["text"],"usage":d.get("usage")},open(out,"w"))
print(out,d.get("usage"))
P
metric() { curl -s -m 3 http://127.0.0.1:$PORT/metrics | grep -E "^intarsia_tokens_(cached|prompt)_total"; }
start() { $BIN serve --backend metal --model $M --served-model-name intarsia --max-seq 8192 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT --spec off > $O/server-$1.out 2> $O/server-$1.err < /dev/null & SP=$!; for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 1; done }
stop() { kill $SP; sleep 3; kill -9 $SP 2>/dev/null; }
start A; python3 $O/req.py $PORT $PF 0 $O/A-turn1.json; metric > $O/A-metrics1.txt; python3 $O/req.py $PORT $PF 1 $O/A-turn2-warm.json; metric > $O/A-metrics2.txt; stop
start B; python3 $O/req.py $PORT $PF 1 $O/B-turn2-cold.json; metric > $O/B-metrics.txt; stop
cat $O/A-metrics1.txt $O/A-metrics2.txt $O/B-metrics.txt
python3 - <<P
import json
w=json.load(open("$O/A-turn2-warm.json"));c=json.load(open("$O/B-turn2-cold.json"))
print("warm:",repr(w["text"])); print("cold:",repr(c["text"])); print("IDENTICAL" if w["text"]==c["text"] else "DIFFERENT")
P
