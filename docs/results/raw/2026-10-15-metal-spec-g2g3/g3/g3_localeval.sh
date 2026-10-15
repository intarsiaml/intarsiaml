#!/bin/zsh
# G3 through localeval (the primary harness path): `localeval quality --dataset g3-realistic.jsonl --judge none --max-tokens 256 --thinking off`
# against a fresh `intarsia serve` per pass, order ABBA = spec off, on, on, off, one exclusive slot (FLEETQ_JOB set by slot.sh). Per-row decode tok/s
# and the text of every answer come out of result.json (cases[].speed, cases[].actual_output). One call per row, no ids.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
[ -d "$DEVELOPER_DIR" ] || { echo "DEVELOPER_DIR missing" >&2; exit 3; }
BIN=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/bin/intarsia-gate-stamped
OUT=/opt/code/intarsiaml-spec/docs/results/raw/2026-10-15-metal-spec-g2g3/g3/localeval
DS=/opt/code/intarsiaml-spec/docs/results/raw/2026-10-15-metal-spec-g2g3/g3/g3-realistic.jsonl
MODEL=/Users/josef/models/Qwen3-8B; PORT=8093
export INTARSIA_CPU_FREE_MEM_GB=60 PYTHONUNBUFFERED=1
SPID=""; trap '[[ -n "$SPID" ]] && kill $SPID 2>/dev/null' EXIT INT TERM
run_pass() { # label mode
  local label=$1 mode=$2
  echo "##### $label spec=$mode $(date -Iseconds)"
  $BIN serve --model $MODEL --backend metal --device 0 --host 127.0.0.1 --port $PORT --served-model-name intarsia-metal --max-seq 8192 \
     --profile-every 0 --no-supervisor --no-ui --spec $mode > $OUT/$label-server.out 2> $OUT/$label-server.err &
  SPID=$!
  local up=0
  for i in {1..180}; do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SPID 2>/dev/null || break; sleep 1; done
  [[ $up == 1 ]] || { echo "SERVER FAILED $label"; SPID=""; return 1; }
  sleep 5
  ( cd /opt/code/deepeval && uv run localeval quality --url http://127.0.0.1:$PORT/v1 --model intarsia-metal --dataset $DS --judge none --bench "" \
      --max-tokens 256 --thinking off --label $label 2>&1 ) > $OUT/$label-localeval.out
  grep -E "results/|logged to unanimis" $OUT/$label-localeval.out | tail -3 | cut -c1-200
  curl -s http://127.0.0.1:$PORT/metrics | grep -E '^intarsia_spec' > $OUT/$label-metrics-final.txt
  kill $SPID 2>/dev/null
  for i in {1..30}; do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  kill -0 $SPID 2>/dev/null && { kill -9 $SPID; sleep 1; }
  wait $SPID 2>/dev/null; SPID=""
  for i in {1..30}; do lsof -iTCP:$PORT -sTCP:LISTEN -nP >/dev/null 2>&1 || break; sleep 1; done
  sleep 5
}
run_pass l1-off off
run_pass l2-on ngram
run_pass l3-on ngram
run_pass l4-off off
echo ALLDONE
