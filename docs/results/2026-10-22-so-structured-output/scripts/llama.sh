#!/bin/zsh
# so lane: llama.cpp (BF16 GGUF, Metal) constrained decode through its native /completion `json_schema`, same schema and
# prompt as run.sh. run.sh's /v1/completions arm ran to the 600-token cap, so its grammar did not apply there.
# fleetq run --host endor --exclusive --mem 24 --expect 6 --label "so: llama.cpp native json_schema check" -- zsh llama.sh <out dir>
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
OUT=${1:?out dir}
mkdir -p $OUT
H=${0:A:h}
GGUF=${GGUF:-/Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf}
SS=(uv run --script $H/ssbench.py)
/opt/homebrew/bin/llama-server -m $GGUF -ngl 999 -fa on -c 8192 --cache-ram 0 --port 8091 > $OUT/llama-server.out 2> $OUT/llama-server.err < /dev/null &
SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT
for i in {1..240}; do curl -sf http://127.0.0.1:8091/health > /dev/null 2>&1 && break; sleep 1; done
curl -sf http://127.0.0.1:8091/completion -H 'content-type: application/json' -d '{"prompt":"Hello","n_predict":8}' > /dev/null < /dev/null
${SS[@]} llama-native --url http://127.0.0.1:8091 --reps 4 --label llama-native --out $OUT/llama-native.jsonl < /dev/null 2>&1 | tee $OUT/job.log
llama-server --version 2>&1 | head -2 > $OUT/llama-version.txt
