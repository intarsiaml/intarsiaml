#!/bin/zsh
# usage: serve.sh <outdir> <variants...>   (inside an exclusive endor slot)
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
[ -d "$DEVELOPER_DIR" ] || { echo "DEVELOPER_DIR missing" >&2; exit 3; }
O=$1; shift
HERE=${0:A:h}
BIN_DIR=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/bin
M=$HOME/models/Qwen3-8B
mkdir -p $O
for v in "$@"; do
  case $v in
    base) BIN=$BIN_DIR/intarsia-base-56ec2d5; SPEC=();;
    off) BIN=$BIN_DIR/intarsia-new-${NEW_REV:?}; SPEC=(--max-batch 1 --spec off);;
    on4) BIN=$BIN_DIR/intarsia-new-${NEW_REV:?}; SPEC=(--max-batch 1 --spec ngram);;
  esac
  INTARSIA_SPEC_TRACE=1 $BIN serve --model $M --backend metal --max-seq 8192 --port 8097 --no-ui $SPEC > $O/$v-serve.log 2>&1 &
  SP=$!
  for i in {1..300}; do curl -sf http://127.0.0.1:8097/health > /dev/null && break; sleep 1; done
  python3 $HERE/serve_check.py 8097 $HERE/prompts $O/$v.json
  kill $SP; wait $SP 2>/dev/null
  pkill -f "intarsia-.* serve .*--port 8097" 2>/dev/null; sleep 3
done
