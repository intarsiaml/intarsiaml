#!/usr/bin/env bash
# Lane s27d: single-sample 16k-context decode, base vs dev (the localeval warmup would double the 4.3 min prefill). Inside a fleetq quiet slot:
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 25 --label "s27d: 16k decode base vs dev" -- scripts/s27d/dec16k.sh [arm ...]   (arm = name:tree, default base:base dev:dev)
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
HOST=spark1; HN=spark-4e77; PORT=8191
OUT=/opt/code/intarsiaml/.worktrees/s27d/docs/results/2026-10-27-s27d-cuda-27b-decode/raw/dec16k; mkdir -p $OUT
M=/home/nvidia/intarsiaml-s27d/models/Qwen3.8-27B-MXFP4
ARMS=("$@"); [ ${#ARMS[@]} = 0 ] && ARMS=(base:base dev:dev)
# a server left over from an interrupted run of this script holds the port
ssh -n -o BatchMode=yes $HOST "pkill -f 'release/intarsia serve.*--port $PORT'; sleep 4" || true
for a in "${ARMS[@]}"; do
  IFS=: read -r name tree <<< "$a"
  SPID=$(ssh -n -o BatchMode=yes $HOST "mkdir -p ~/intarsiaml-s27d/trees/$tree/out; nohup ~/intarsiaml-s27d/trees/$tree/target/release/intarsia serve --backend cuda --model $M --port $PORT --host 0.0.0.0 --served-model-name m --max-seq 20480 --mem-budget 45 --profile-every 0 > ~/intarsiaml-s27d/trees/$tree/out/dec16k-$name.log 2>&1 < /dev/null & echo \$!")
  for i in $(seq 1 150); do curl -sf http://$HN:$PORT/health >/dev/null && break; sleep 2; done
  python3 - $HN:$PORT $name <<'PY' 2>&1 | tee $OUT/$name.txt
import json, sys, time, urllib.request
host, name = sys.argv[1], sys.argv[2]
text = open('/opt/code/intarsiaml/tools/prompts/corpus_docs.txt').read()[:54900]
body = {'model': 'm', 'prompt': 'nonce %d ' % int(time.time()) + text, 'max_tokens': 128, 'temperature': 0, 'stream': True, 'ignore_eos': True}
req = urllib.request.Request(f'http://{host}/v1/completions', json.dumps(body).encode(), {'content-type': 'application/json'})
t0 = time.time(); ts = []; usage = None
with urllib.request.urlopen(req, timeout=900) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith('data:') or line.endswith('[DONE]'): continue
        j = json.loads(line[5:])
        if j.get('usage'): usage = j['usage']
        if j.get('choices') and j['choices'][0].get('text'): ts.append(time.time())
n = len(ts)
print(f'{name}: prompt chars {len(text)} (~16.6k tokens) ttft {ts[0]-t0:.1f} s, decode tok/s (127 tokens after the first chunk) {127/(ts[-1]-ts[0]):.2f}, chunks {n}')
PY
  ssh -n $HOST "kill $SPID 2>/dev/null; sleep 5; kill -9 $SPID 2>/dev/null"
done
