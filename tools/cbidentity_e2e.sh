#!/bin/zsh
# End-to-end check of tools/cbidentity.py against a real `intarsia serve` (Qwen3-8B BF16, Metal) on endor.
# Everything runs inside ONE exclusive fleetq slot: server start, smoke, reference, concurrent runs, perturbation
# self-test, server stop. The slot is released by trap.
#
#   tools/cbidentity_e2e.sh OUTDIR [SAMPLE]        (OUTDIR is created; SAMPLE defaults to 40 cases)
set -u
OUT=${1:?usage: cbidentity_e2e.sh OUTDIR [SAMPLE]}; OUT=${OUT:A}
SAMPLE=${2:-40}
T=${0:A:h}
BIN=${INTARSIA_BIN:-/opt/code/intarsiaml-spec/target/release/intarsia}
MODEL=${MODEL:-$HOME/models/Qwen3-8B}
PORT=${PORT:-8099}
URL=http://127.0.0.1:$PORT
SERVE_FLAGS=(--backend metal --max-seq 8192 --profile-every 0 --prefill-chunk 256 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT)
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
[ -d "$DEVELOPER_DIR" ] || { echo "DEVELOPER_DIR missing" >&2; exit 3; }
mkdir -p "$OUT"
LOG=$OUT/e2e.log
exec > >(tee -a "$LOG") 2>&1

if [ -n "${DRY_URL:-}" ]; then
  URL=$DRY_URL   # wiring check against `cbidentity.py mock`: no slot, no server
else
OUTS=$(fleetq acquire --host endor --exclusive --mem 24 --expect 40 --label "metal cb-identity harness e2e (qwen3-8b serve, $SAMPLE cases)" --wait 2>&1)
echo "$OUTS"
ID=$(echo "$OUTS" | grep -Eo '#?[0-9]+' | head -1 | tr -d '#')
[ -n "$ID" ] || { echo "no job id" >&2; exit 4; }
SPID=""
cleanup() {
  [ -n "$SPID" ] && kill "$SPID" 2>/dev/null && wait "$SPID" 2>/dev/null
  pkill -f "intarsia serve.*--port $PORT" 2>/dev/null
  fleetq release "$ID" >/dev/null 2>&1
  echo "released $ID $(date +%T); leftover serve on port $PORT: $(pgrep -fl "intarsia serve.*--port $PORT" | wc -l | tr -d ' ')"
}
trap cleanup EXIT INT TERM
STATE=$(fleetq show "$ID" 2>&1 | head -1)
echo "$STATE" | grep -q " running on endor" || { echo "job $ID is not running: $STATE" >&2; exit 5; }
echo "slot $ID RUNNING $(date +%T): $STATE"
export FLEETQ_JOB=$ID

echo "binary: $BIN ($(stat -f %Sm "$BIN")), model $MODEL"
FLEETQ_JOB=$ID "$BIN" serve --model "$MODEL" "${SERVE_FLAGS[@]}" > "$OUT/server.log" 2>&1 &
SPID=$!
for i in $(seq 1 120); do
  curl -sf "$URL/health" >/dev/null 2>&1 && break
  kill -0 "$SPID" 2>/dev/null || { echo "server died"; tail -20 "$OUT/server.log"; exit 6; }
  sleep 2
done
curl -sf "$URL/health" >/dev/null || { echo "server never became healthy"; exit 7; }
echo "server up after ~$((i*2))s; plan: $(curl -s $URL/v1/intarsia/plan | head -c 600)"

fi
CB=(uv run $T/cbidentity.py)
CASES=$T/prompts/cb_identity_cases.jsonl
COMMON=(--cases "$CASES" --url "$URL" --arrival-scale 0.25)

echo "=== smoke (4 cases) ==="
"${CB[@]}" run --mode solo "${COMMON[@]}" --sample 4 --out "$OUT/smoke-ref.jsonl" || { echo "smoke reference failed"; exit 8; }
"${CB[@]}" run --mode concurrent --b 2 --stream mixed "${COMMON[@]}" --sample 4 --out "$OUT/smoke-b2.jsonl" || { echo "smoke concurrent failed"; exit 9; }
"${CB[@]}" compare --cases "$CASES" --ref "$OUT/smoke-ref.jsonl" --test "$OUT/smoke-b2.jsonl" --report "$OUT/smoke-report.json"
echo "smoke compare rc=$? (informational)"

echo "=== reference ($SAMPLE cases, alone, sequential, flush before each) ==="
"${CB[@]}" run --mode solo "${COMMON[@]}" --sample "$SAMPLE" --out "$OUT/ref.jsonl" || { echo "reference failed"; exit 10; }

echo "=== concurrent runs ==="
TESTS=()
for cfg in "2 mixed" "8 mixed" "4 on" "4 off"; do
  set -- ${=cfg}
  "${CB[@]}" run --mode concurrent --b "$1" --stream "$2" "${COMMON[@]}" --sample "$SAMPLE" --out "$OUT/conc-b$1-$2.jsonl" && TESTS+=("$OUT/conc-b$1-$2.jsonl")
done

echo "=== compare (the gate) ==="
"${CB[@]}" compare --cases "$CASES" --ref "$OUT/ref.jsonl" --test "${TESTS[@]}" --report "$OUT/report.json"
echo "GATE COMPARE rc=$?"

echo "=== perturbation self-test: concurrent run with max_tokens capped at 12 ==="
"${CB[@]}" run --mode concurrent --b 4 --stream mixed "${COMMON[@]}" --sample "$SAMPLE" --cap-max-tokens 12 --out "$OUT/perturbed-cap12.jsonl"
"${CB[@]}" compare --cases "$CASES" --ref "$OUT/ref.jsonl" --test "$OUT/perturbed-cap12.jsonl" --report "$OUT/perturbed-report.json"
echo "PLAIN COMPARE OF PERTURBED RUN rc=$? (must be nonzero)"
"${CB[@]}" compare --cases "$CASES" --ref "$OUT/ref.jsonl" --test "$OUT/perturbed-cap12.jsonl" --expect-cap 12 --report "$OUT/perturbed-expect-report.json"
echo "EXPECT-CAP COMPARE rc=$? (must be 0)"
curl -s "$URL/metrics" | grep -E "^intarsia_(requests|cached|prompt|completion)" > "$OUT/metrics-end.txt"
echo "done $(date +%T)"
