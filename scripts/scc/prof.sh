#!/usr/bin/env bash
# Lane scc: nsys profile of `intarsia serve --max-batch 16` under an N-stream burst (scripts/scc/burst.py), ranked by scripts/scc/rank.py.
#   fleetq run --host spark2 --quiet gpu,cpu --mem 45 --expect 12 --label "scc: ..." -- scripts/scc/prof.sh <bin> <outname> [ENV=V ...]
# env: N (streams, 16), GEN (256), XARGS (extra serve args, default --max-batch 16)
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
BIN=$1; OUTN=$2; shift 2
HOST=${HOST:-spark2}; HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW=/opt/code/intarsiaml/.worktrees/scc/docs/results/2026-10-27-scc-cuda-35b-concurrency/raw/$OUTN; mkdir -p "$RAW"
ssh -n $HOST "mkdir -p ~/intarsiaml-scc/out/$OUTN"
scp -q "$HERE/burst.py" "$HERE/rank.py" "$HOST:intarsiaml-scc/out/$OUTN/"
ssh -n -o BatchMode=yes "$HOST" "export FLEETQ_JOB=$FLEETQ_JOB PATH=/usr/local/cuda/bin:\$PATH; set -u
P=\$HOME/intarsiaml-scc; O=\$P/out/$OUTN; B=\$P/bin/$BIN
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
env $* nsys profile -t cuda --cuda-graph-trace=node --force-overwrite true -o \$O/nsys \$B serve --backend cuda --model \$M --port 8197 --host 127.0.0.1 --served-model-name m --max-seq 8192 --mem-budget 40 --profile-every 0 ${XARGS:---max-batch 16} --no-supervisor < /dev/null > \$O/serve.out 2> \$O/serve.err &
for i in \$(seq 1 200); do curl -sf http://127.0.0.1:8197/health > /dev/null && break; sleep 2; done
python3 \$O/burst.py 8197 2 32 1 | tee \$O/warm.txt
python3 \$O/burst.py 8197 ${N:-16} ${GEN:-256} 1 | tee \$O/burst.txt
pkill -TERM -f 'bin/[${BIN:0:1}]${BIN:1} serve --backend cuda --model .* --port 8197'; sleep 8
for i in \$(seq 1 90); do pgrep -f '[n]sys profile' > /dev/null || break; sleep 2; done
nsys stats -r cuda_gpu_trace -f csv --force-export=true -o \$O/trace \$O/nsys.nsys-rep < /dev/null > \$O/stats.log 2>&1
python3 \$O/rank.py \$O/trace_cuda_gpu_trace.csv ${RANK_ARGS:-5} | tee \$O/rank.txt
rm -f \$O/nsys.sqlite \$O/nsys.nsys-rep.bak"
scp -q "$HOST:intarsiaml-scc/out/$OUTN/rank.txt" "$HOST:intarsiaml-scc/out/$OUTN/burst.txt" "$HOST:intarsiaml-scc/out/$OUTN/warm.txt" "$HOST:intarsiaml-scc/out/$OUTN/serve.err" "$RAW/"
