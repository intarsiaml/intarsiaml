#!/bin/zsh
# Phase B (inside one slot): mid-run identity cases, clean speed check, serve streaming check, crate tests, kernels test.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
[ -d "$DEVELOPER_DIR" ] || { echo "DEVELOPER_DIR missing" >&2; exit 3; }
R=/opt/code/intarsiaml-spec/docs/results/raw/2026-10-15-metal-spec-g1
NEW=$(ls /private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/bin/intarsia-new-* | tail -1)
M=$HOME/models/Qwen3-8B
cd /opt/code/intarsiaml-spec
$R/run_identity.sh $R/identity-b $R/cases-b.txt base off on4 on6
# speed: single runs, no trace
mkdir -p $R/speed
for rep in 1 2; do
  for c in grounded open; do
    for v in off on4; do
      SPEC=(--spec off); [ $v = on4 ] && SPEC=(--spec ngram)
      $NEW generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 256 --chat --no-think $SPEC --prompt "$(cat $R/prompts/$c.txt)" < /dev/null > $R/speed/$v-$c-$rep.out 2> $R/speed/$v-$c-$rep.err
      echo "speed $v $c $rep: $(grep '^finish=' $R/speed/$v-$c-$rep.err)"
    done
  done
done
# serve: streaming + non-streaming, spec off vs on
mkdir -p $R/serve
for v in off on4; do
  SPEC=(--spec off); [ $v = on4 ] && SPEC=(--spec ngram)
  $NEW serve --model $M --backend metal --max-seq 8192 --port 8097 --no-ui $SPEC > $R/serve/$v-serve.log 2>&1 &
  SP=$!
  for i in {1..300}; do curl -sf http://127.0.0.1:8097/health > /dev/null && break; sleep 1; done
  python3 $R/serve_check.py 8097 $R/prompts/grounded.txt $R/serve/$v.json
  curl -s http://127.0.0.1:8097/metrics > $R/serve/$v-metrics.txt
  kill $SP; wait $SP 2>/dev/null
  pkill -f "intarsia-new.* serve .*--port 8097" 2>/dev/null; sleep 2
done
# tests
cargo test -p intarsia-backend-metal --release > $R/cargo-test-metal.log 2>&1; echo "cargo test metal exit $?"
$NEW kernels test --backend metal -m $M > $R/kernels-test-metal.log 2>&1; echo "kernels test exit $?"
