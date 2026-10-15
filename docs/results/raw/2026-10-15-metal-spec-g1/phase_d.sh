#!/bin/zsh
# Phase D (inside one slot): golden_4k-file prompts and an EOS-inside-a-run case (identity, base/off/on4/on6), and a
# two-turn serve check where turn 2 reuses the KV prefix turn 1 left after a stop inside an accepted run.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
[ -d "$DEVELOPER_DIR" ] || { echo "DEVELOPER_DIR missing" >&2; exit 3; }
R=/opt/code/intarsiaml-spec/docs/results/raw/2026-10-15-metal-spec-g1
NEW=$(grep '^NEW=' $R/run_identity.sh | cut -d= -f2)
M=$HOME/models/Qwen3-8B
$R/run_identity.sh $R/identity-c $R/cases-c.txt base off on4 on6
mkdir -p $R/serve2
for v in off on4; do
  SPEC=(--spec off); [ $v = on4 ] && SPEC=(--spec ngram)
  INTARSIA_SPEC_TRACE=1 $NEW serve --model $M --backend metal --max-seq 8192 --port 8097 --no-ui $SPEC > $R/serve2/$v-serve.log 2>&1 &
  SP=$!
  for i in {1..300}; do curl -sf http://127.0.0.1:8097/health > /dev/null && break; sleep 1; done
  python3 $R/serve_check2.py 8097 $R/prompts/short.txt kettles $R/serve2/$v.json
  kill $SP; wait $SP 2>/dev/null
  pkill -f "intarsia-new.* serve .*--port 8097" 2>/dev/null; sleep 2
done
