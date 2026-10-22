#!/usr/bin/env bash
# Lane FX5c: the 35B reuse gate on Metal (endor) with the oracle-relative continue rule, long golden, chunks 1024 and 2048
# (fx10's command: scripts/fx10/reuse.sh), plus a mutation run (warm logits x1.2, then x1.01) at chunk 1024 on long_docs_4k.
#   fleetq run --host endor --mem 30 --expect 40 --label "fx5c: Metal 35B reuse gate, oracle-relative continue (non-exclusive, numerics only)" -- scripts/fx5c/metal-validate.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
RAW=${RAW:-$REPO/docs/results/2026-10-22-fx5c-oracle-gate/raw/metal}
mkdir -p "$RAW"
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/opt/code/target-fx5c}
cargo build --release -p intarsia-cli --features metal 2>&1 | tail -3 | tee "$RAW/00-build.out"
BIN="$CARGO_TARGET_DIR/release/intarsia"
G=/opt/code/intarsiaml/golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
export INTARSIA_REUSE_ORACLE_DIR=$REPO/golden/oracle
run() { # label, extra args...
  local label=$1; shift
  "$BIN" accuracy --model $HOME/models/Qwen3.6-35B-A3B-NVFP4 --backend metal --golden "$G" --mode reuse --max-seq ${MAXSEQ:-16384} --mem-budget 30 --allow-dirty "$@" > "$RAW/$label.log" 2>&1 < /dev/null
  echo "exit $?" | tee -a "$RAW/$label.log"
  grep -a -e 'continue judged' -e 'prefix-reuse' -e '^long_\|^chat_' "$RAW/$label.log" | cut -c1-260 | tail -25
}
if [[ "${QUICK:-0}" == 1 ]]; then   # only the case the oracle-relative rule judges, at both chunks (the other cases are unchanged code)
  for c in 1024 2048; do echo "=== reuse long_docs_4k, chunk $c"; MAXSEQ=8192 run reuse-q-c$c --prefill-chunk $c --reuse-prompt long_docs_4k; done
  exit 0
fi
for c in 1024 2048; do echo "=== reuse long golden, chunk $c"; run reuse-c$c --prefill-chunk $c --reuse-require-all; done
echo "=== mutation: warm logits x1.2, long_docs_4k, chunk 1024"
INTARSIA_REUSE_TEST_BUG=scale:1.2 run mut-x1.2 --prefill-chunk 1024 --reuse-prompt long_docs_4k
echo "=== mutation: warm logits x1.01"
INTARSIA_REUSE_TEST_BUG=scale:1.01 run mut-x1.01 --prefill-chunk 1024 --reuse-prompt long_docs_4k
echo "=== fallback: --no-reuse-oracle (absolute rule), chunk 1024, long_docs_4k"
run noorc --prefill-chunk 1024 --reuse-prompt long_docs_4k --no-reuse-oracle
