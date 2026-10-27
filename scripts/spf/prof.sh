#!/usr/bin/env bash
# Lane spf: per-op prefill profile (nsys kernel ranking + the engine's own op table) of a tree on spark1, one fleetq slot:
#   fleetq run --host spark1 --quiet gpu,cpu --mem 40 --expect 15 --label "sk: ..." -- scripts/spf/prof.sh <tree> <outname> [ENV=V ...]
# Prompts: first 14000 / 55000 chars of tools/prompts/corpus_docs.txt (about 4.1k / 16.3k tokens), generate --max-tokens 1.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
TREE=$1; OUTN=$2; shift 2
HOST=${HOST:-spark1}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW=/opt/code/intarsiaml/.worktrees/spf/docs/results/2026-10-27-spf-cuda-prefill/raw/$OUTN; mkdir -p "$RAW"
scp -q "$HERE/rank.py" "$HOST:intarsiaml-spf/$TREE/out/rank.py" 2>/dev/null || { ssh -n $HOST "mkdir -p ~/intarsiaml-spf/$TREE/out"; scp -q "$HERE/rank.py" "$HOST:intarsiaml-spf/$TREE/out/rank.py"; }
ssh -n -o BatchMode=yes "$HOST" "export FLEETQ_JOB=$FLEETQ_JOB PATH=/usr/local/cuda/bin:\$PATH; set -u; export CHARS='${CHARS:-13600 55000}' CHUNK=${CHUNK:-4096}
P=\$HOME/intarsiaml-spf/$TREE; B=\$P/target/release/intarsia; O=\$P/out/$OUTN; rm -rf \$O; mkdir -p \$O
M35=\$(ls -d /home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/* | head -1)
for chars in ${CHARS:-13600 55000}; do
  d=\$O/\$chars; mkdir -p \$d; head -c \$chars \$P/tools/prompts/corpus_docs.txt > \$d/prompt.txt
  env $* nsys profile -t cuda --force-overwrite true -o \$d/nsys \$B generate -m \$M35 --backend cuda --mem-budget 40 --max-seq 20480 --prefill-chunk ${CHUNK:-4096} --profile-every 0 --max-tokens 1 -p \"\$(cat \$d/prompt.txt)\" < /dev/null > \$d/nsys.out 2> \$d/nsys.err
  grep -aE 'finish=' \$d/nsys.err | head -1 | cut -c1-200
  nsys stats -r cuda_gpu_kern_sum,cuda_gpu_trace -f csv --force-export=true -o \$d/kern \$d/nsys.nsys-rep < /dev/null > \$d/stats.log 2>&1
  rm -f \$d/nsys.sqlite \$d/nsys.nsys-rep.bak
  python3 \$P/out/rank.py \$d/kern_cuda_gpu_kern_sum.csv \$d/kern_cuda_gpu_trace.csv > \$d/rank.txt 2>&1
  # engine's own per-op table (timed ops sync around every launch: relative only)
  env $* \$B generate -m \$M35 --backend cuda --mem-budget 40 --max-seq 20480 --prefill-chunk ${CHUNK:-4096} --profile-every 1 --op-table --max-tokens 1 -p \"\$(cat \$d/prompt.txt)\" < /dev/null > \$d/optable.out 2> \$d/optable.err
done"
mkdir -p "$RAW"; scp -q -r "$HOST:intarsiaml-spf/$TREE/out/$OUTN/*" "$RAW/" 
for c in ${CHARS:-13600 55000}; do rm -f "$RAW/$c/nsys.nsys-rep"; [[ ${KEEP:-0} == 1 ]] || rm -f "$RAW/$c/kern_cuda_gpu_trace.csv"; done
for c in ${CHARS:-13600 55000}; do head -30 "$RAW/$c/rank.txt"; done
