#!/usr/bin/env zsh
# Resume for the Metal quality tier of docs/results/2026-10-21-standings.md. DO NOT QUEUE until the endor hold is lifted.
# Needs an intarsia Metal release binary (BIN), e.g. built from origin/main with
#   cargo build --release -p intarsia-cli --features metal   (CPU work: also under the hold).
# One slot, one server, two small gsm8k probes (default --structured, then --structured prompt) to find out whether the
# 0.0 / ~30 s per call symptom of attempt 2 comes from the structured-output path. Kills the server by PID on exit.
#   BIN=/path/to/intarsia fleetq run --host endor --exclusive --mem 40 --expect 15 \
#     --label "standings metal quality diag" -- env BIN=$BIN zsh docs/results/2026-10-21-standings/raw/metal/resume-quality-diag.sh
: "${BIN:?set BIN to the intarsia release binary}"
OUT=${OUT:-/tmp/standings-quality-diag}
mkdir -p "$OUT"
SPID=""
cleanup() { [[ -n "$SPID" ]] && kill $SPID 2>/dev/null; sleep 2; [[ -n "$SPID" ]] && kill -9 $SPID 2>/dev/null; }
trap cleanup EXIT
INTARSIA_CPU_FREE_MEM_GB=60 $BIN serve --model /Users/josef/models/Qwen3-8B --backend metal --device 0 --host 127.0.0.1 --port 8093 \
  --served-model-name intarsia-metal --max-seq 20480 --profile-every 0 --no-supervisor --no-ui >"$OUT/server.out" 2>"$OUT/server.err" &
SPID=$!
for i in {1..180}; do curl -sf http://127.0.0.1:8093/health >/dev/null 2>&1 && break; sleep 1; done
curl -sf http://127.0.0.1:8093/health >/dev/null 2>&1 || { echo "server did not come up"; exit 1; }
cd /opt/code/deepeval || exit 1
for st in default prompt; do
  extra=(); [[ $st == prompt ]] && extra=(--structured prompt)
  uv run localeval quality --url http://127.0.0.1:8093/v1 --model intarsia-metal --bench gsm8k --limit 10 --thinking off \
    "${extra[@]}" --label "diag-structured-$st" >"$OUT/diag-$st.out" 2>&1
  tr '\r' '\n' <"$OUT/diag-$st.out" | grep -E "GSM8K Accuracy|^\| gsm8k" | tail -2
done
