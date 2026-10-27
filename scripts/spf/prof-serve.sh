#!/usr/bin/env bash
# Lane spf: steady-state per-request kernel profile: `intarsia serve` under nsys, N identical prompts of CHARS chars of
# tools/prompts/corpus_docs.txt (max_tokens 1); the report is split into requests at idle gaps and the LAST (warm) request ranked.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 45 --expect 10 --label "spf: ..." -- scripts/spf/prof-serve.sh <tree> <outname> [ENV=V ...]
# env: XARGS (extra serve args), CHARS (14035 ~ 4090 tokens), N (4), CHUNK (default: recipe), MODEL (default: the 8B), SN (served name)
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
TREE=$1; OUTN=$2; shift 2
HOST=${HOST:-spark1}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW=/opt/code/intarsiaml/.worktrees/spf/docs/results/2026-10-27-spf-cuda-prefill/raw/$OUTN; mkdir -p "$RAW"
scp -q "$HERE/rank_req.py" "$HOST:intarsiaml-spf/$TREE/out/rank_req.py" 2>/dev/null || { ssh -n $HOST "mkdir -p ~/intarsiaml-spf/$TREE/out"; scp -q "$HERE/rank_req.py" "$HOST:intarsiaml-spf/$TREE/out/rank_req.py"; }
ssh -n -o BatchMode=yes "$HOST" "export FLEETQ_JOB=$FLEETQ_JOB PATH=/usr/local/cuda/bin:\$PATH; set -u
P=\$HOME/intarsiaml-spf/$TREE; B=\$P/target/release/intarsia; O=\$P/out/$OUTN; rm -rf \$O; mkdir -p \$O
M=${MODEL:-\$(ls -d /home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/* | head -1)}
head -c ${CHARS:-14035} \$P/tools/prompts/corpus_docs.txt > \$O/prompt.txt
env $* nsys profile -t cuda --force-overwrite true -o \$O/nsys \$B serve --backend cuda --model \$M --port 8197 --host 127.0.0.1 --served-model-name m --max-seq 20480 --mem-budget 40 --profile-every 0 ${XARGS:-} ${CHUNK:+--prefill-chunk $CHUNK} --no-supervisor < /dev/null > \$O/serve.out 2> \$O/serve.err &
for i in \$(seq 1 200); do curl -sf http://127.0.0.1:8197/health > /dev/null && break; sleep 2; done
python3 - \$O/prompt.txt ${N:-4} <<'PY'
import json, sys, time, urllib.request
p = open(sys.argv[1]).read(); n = int(sys.argv[2])
for i in range(n):
    t0 = time.time()
    req = urllib.request.Request('http://127.0.0.1:8197/v1/completions', json.dumps({'model': 'm', 'prompt': str(i) + ' ' + p, 'max_tokens': 1, 'temperature': 0}).encode(), {'content-type': 'application/json'})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    print('req', i, 'prompt_tokens', r['usage']['prompt_tokens'], 'wall %.0f ms' % ((time.time() - t0) * 1000), flush=True)
    time.sleep(1.5)
PY
pkill -TERM -f '[r]elease/intarsia serve --backend cuda --model .* --port 8197' ; sleep 8
for i in \$(seq 1 60); do pgrep -f '[n]sys profile' > /dev/null || break; sleep 2; done
nsys stats -r cuda_gpu_trace -f csv --force-export=true -o \$O/trace \$O/nsys.nsys-rep < /dev/null > \$O/stats.log 2>&1
python3 \$P/out/rank_req.py \$O/trace_cuda_gpu_trace.csv | tee \$O/rank.txt
rm -f \$O/nsys.sqlite \$O/nsys.nsys-rep.bak"
mkdir -p "$RAW"; scp -q "$HOST:intarsiaml-spf/$TREE/out/$OUTN/rank.txt" "$HOST:intarsiaml-spf/$TREE/out/$OUTN/serve.err" "$RAW/" 
