#!/usr/bin/env bash
# Lane srv16: packing diagnostic (not a timing run): dev serve + INTARSIA_PACK_PREFILL=1 + INTARSIA_CB_TRACE=1, a warm-up and a
# 16-stream burst, the scheduler trace's Chunk events kept (which chunks share an iteration = one packed pass).
#   fleetq run --host spark1 --mem 45 --expect 6 --label "srv16: pack diag" -- scripts/srv16/diag.sh <bin> <outname> [ENV=V ...]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
BIN=$1; OUTN=$2; shift 2
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW=/opt/code/intarsiaml/.worktrees/srv16/docs/results/2026-10-28-srv16-serve-gap/raw/$OUTN; mkdir -p "$RAW"
ssh -n spark1 "mkdir -p ~/intarsiaml-srv16/out/$OUTN"; scp -q "$H/burst.py" "spark1:intarsiaml-srv16/out/$OUTN/"
ssh -n -o BatchMode=yes spark1 "export FLEETQ_JOB=$FLEETQ_JOB; set -u
P=\$HOME/intarsiaml-srv16; O=\$P/out/$OUTN
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
env INTARSIA_CB_TRACE=1 $* \$P/bin/$BIN serve --backend cuda --model \$M --port 8197 --host 127.0.0.1 --served-model-name m --max-seq 8192 --mem-budget 40 --profile-every 0 --max-batch 16 --no-supervisor < /dev/null > \$O/serve.out 2> \$O/serve.err &
SP=\$!
for i in \$(seq 1 200); do curl -sf http://127.0.0.1:8197/health > /dev/null && break; sleep 2; done
python3 \$O/burst.py 8197 2 8 1 warm | tee \$O/warm.txt
python3 \$O/burst.py 8197 16 16 1 | tee \$O/burst.txt
kill \$SP; wait \$SP 2>/dev/null
grep -E 'cb-trace.*(Admitted|PrefillStart|Chunk)' \$O/serve.err | sed 's/cb-trace t=[0-9]* //' > \$O/chunks.txt
grep -vE 'cb-trace' \$O/serve.err | head -30 > \$O/serve-head.txt"
for f in chunks.txt serve-head.txt burst.txt; do scp -q "spark1:intarsiaml-srv16/out/$OUTN/$f" "$RAW/"; done
