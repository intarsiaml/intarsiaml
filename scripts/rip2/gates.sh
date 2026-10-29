#!/usr/bin/env bash
# Lane rip2 (copied from scripts/rip) GPU job (spark2, --quiet gpu,cpu): the correctness gates.
#  1. 8B model/engine gates: the probe (new cold reference, every split / reuse equal), the engine reuse + busy-chunk test, BI8's
#     batched == solo and engine == cold reference; decode-written rows are not prefill rows (bitwise); an ordinary short pass fits
#  2. served greedy vs the cold reference: solo 1/4/16 x 3 reps, --max-batch 16 1/4/8/16 x 3 reps (reps on one server); equal.py
#     21 cells solo vs --max-batch 16
#  3. shared paths: rows_exact, row_invariance, batch_cuda_real on the 35B, `plan show` base vs fix (8B / 35B / Gemma-4), the 35B
#     21-cell served identity base vs fix at --max-batch 16
#   OUT=... fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 90 --label "rip2: gates" -- scripts/rip2/gates.sh
# PARTS="model served shared" picks parts.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H=spark2; HN=spark-fa00; PORT=8197
D=$(cd "$(dirname "$0")" && pwd); OUT=${OUT:?}; mkdir -p "$OUT"
HUB=/home/nvidia/.cache/huggingface/hub
M8=$HUB/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
M35=$HUB/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
MG=/home/nvidia/models/Gemma-4-26B-A4B-NVFP4
B=/home/nvidia/intarsiaml-rip2/bin
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
# env VAR=... must follow the cd (a prefix before `cd` never reaches cargo): CT VARS... -p ...
CT="cd ~/intarsiaml-rip2/src && env INTARSIA_REQUIRE_GPU=1"
SPID=""
rsh() { ssh -n -o BatchMode=yes $H "$@"; }
stop_server() { [[ -n "$SPID" ]] && { rsh "kill $SPID 2>/dev/null; for i in \$(seq 1 15); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null; sleep 2"; SPID=""; }; }
trap stop_server EXIT INT TERM
start() { # name bin model args
  # rip2: row-invariant prefill is the CUDA default for dense models (no knob set); the base binary has it off, and the 35B /
  # Gemma-4 switch it off at load
  SPID=$(rsh "nohup env ${SERVE_ENV:-} $B/$2 serve --backend cuda --model $3 --port $PORT --host 0.0.0.0 --served-model-name intarsia --max-seq 8192 --mem-budget 40 --profile-every 0 $4 > ~/intarsiaml-rip2/out/serve-$1.log 2>&1 < /dev/null & echo \$!")
  for i in $(seq 1 150); do curl -sf "http://$HN:$PORT/health" >/dev/null && return 0; sleep 2; done
  echo "server $1 did not come up"; stop_server; return 1
}
F='^test |test result|panicked|FINDING|RIP|bi8 engine|pairs bitwise|left|right|row-invariant|assert'
P=${PARTS:-model served shared}
[[ " $P " == *" shared "* ]] && P="$P kern s35"  # rip2: shared = kern (rows_exact, row_invariance, plan show) + s35 (35B)
if [[ " $P " == *" model "* ]]; then
  echo "== probe (cold reference)"
  rsh "$RPATH $CT INTARSIA_ROW_INVARIANT_PREFILL=1 INTARSIA_BI8_PROBE=1 INTARSIA_QWEN3_DIR=$M8 cargo test --release -p intarsia-models --test batch_cuda_real real_qwen3_8b_prefix_reuse_probe -- --test-threads=1 --nocapture 2>&1 | grep -aE '^test |test result|panicked|COLD|SELFREUSE|SPLIT|CROSS'" > "$OUT/probe.log"
  python3 "$D/../bi8/cold-ref.py" "$OUT/probe.log" "$OUT/cold.json"
  echo "SELFREUSE equal: $(grep -c 'logits row0 EQUAL' "$OUT/probe.log") / $(grep -c SELFREUSE "$OUT/probe.log"); SPLIT with a differing split: $(grep SPLIT "$OUT/probe.log" | grep -c ':x'); CROSS differing: $(grep CROSS "$OUT/probe.log" | grep -vc 'None')"
  echo "== engine gates"
  rsh "$RPATH $CT INTARSIA_QWEN3_DIR=$M8 cargo test --release -p intarsia-models --test batch_cuda_real -- real_qwen3_8b_reuse_and_busy real_qwen3_8b_batched --test-threads=1 --nocapture 2>&1 | grep -aE '$F'" | tee "$OUT/engine.log" | grep -aE '^test |test result|RIP'
  echo "== decode-written rows vs prefill rows (bitwise), ordinary short pass deep in the context"
  rsh "$RPATH $CT INTARSIA_QWEN3_DIR=$M8 cargo test --release -p intarsia-models --test batch_cuda_real -- real_qwen3_8b_decode_written_rows real_qwen3_8b_ordinary_short_pass --test-threads=1 --nocapture 2>&1 | grep -aE '$F'" | tee "$OUT/decoderows.log" | grep -aE '^test |RIP|test result'
fi
if [[ " $P " == *" served "* ]]; then
  REF=${REF:-$OUT/cold.json}
  ARMS=("solo|" "mb16|--max-batch 16")
  if ! python3 -c "import json,sys; sys.exit(0 if len(json.load(open('$REF')))>0 else 1)" 2>/dev/null; then echo "no cold reference in $REF: served part skipped"; ARMS=(); fi
  echo "== served greedy vs cold ($REF)"
  for arm in "${ARMS[@]}"; do
    IFS='|' read -r name args <<< "$arm"
    start "$name" intarsia "$M8" "$args" || continue
    case $name in solo) ns=1,4,16 ;; mb16) ns=1,4,8,16 ;; esac
    python3 "$D/../bi8/greedy.py" run "http://$HN:$PORT/v1" $ns 3 "$OUT/greedy-$name.json" > /dev/null
    python3 "$D/../bi8/greedy.py" compare "$REF" "$OUT/greedy-$name.json" | tee "$OUT/greedy-$name.compare" | tail -1
    python3 "$D/../htk/equal.py" run --url "http://$HN:$PORT/v1" --out "$OUT/eq-$name.json" > /dev/null
    stop_server
  done
  python3 "$D/../htk/equal.py" compare "$OUT/eq-solo.json" "$OUT/eq-mb16.json" | tee "$OUT/eq-compare.log" | tail -1
fi
if [[ " $P " == *" kern "* ]]; then
  echo "== rows_exact + row_invariance + cuda unit tests"
  rsh "$RPATH $CT cargo test --release -p intarsia-backend-cuda --test rows_exact --test row_invariance 2>&1 | grep -aE '^test result|FAILED|panicked' | tail -6" | tee "$OUT/rows_exact.log"
  echo "== plan show base vs fix (fix default, and fix with INTARSIA_ROW_INVARIANT_PREFILL=0)"
  rsh "O=~/intarsiaml-rip2/out/plan; mkdir -p \$O; for mm in 8:$M8 35:$M35 g:$MG; do m=\${mm%%:*}; M=\${mm#*:}; for t in base intarsia off; do b=\$t; e=; [ \$t = off ] && { b=intarsia; e=INTARSIA_ROW_INVARIANT_PREFILL=0; }; for k in json txt; do j=; [ \$k = json ] && j=--json; env \$e $B/\$b plan show --backend cuda --model \$M --max-seq 8192 --mem-budget 40 \$j < /dev/null > \$O/\$m-\$t.\$k 2> \$O/\$m-\$t.\$k.err; done; done; for t in intarsia off; do for k in json txt; do if [ -s \$O/\$m-base.\$k ] && cmp -s \$O/\$m-base.\$k \$O/\$m-\$t.\$k; then echo \"SAME plan \$m \$t \$k byte-identical to base (\$(wc -c < \$O/\$m-\$t.\$k) B)\"; else echo \"DIFF plan \$m \$t \$k (\$(wc -c < \$O/\$m-base.\$k) -> \$(wc -c < \$O/\$m-\$t.\$k) B)\"; fi; done; done; grep -h row-invariant \$O/\$m-intarsia.txt.err | head -2; diff \$O/\$m-base.txt \$O/\$m-intarsia.txt | head -8; done" | tee "$OUT/plan.log"
  mkdir -p "$OUT/plan"; scp -q -o BatchMode=yes "$H:intarsiaml-rip2/out/plan/*" "$OUT/plan/" || true
fi
if [[ " $P " == *" s35 "* ]]; then
  echo "== batch_cuda_real 35B"
  rsh "$RPATH $CT INTARSIA_QWEN36_DIR=$M35 cargo test --release -p intarsia-models --test batch_cuda_real real_35b_batched -- --test-threads=1 --nocapture 2>&1 | grep -aE '^test |test result|panicked|pairs bitwise'" | tee "$OUT/batch35.log"
  echo "== 35B served identity, base vs fix, --max-batch 16"
  for bin in base intarsia; do
    start "ident35-$bin" "$bin" "$M35" "--max-batch 16" || continue
    python3 "$D/../htk/equal.py" run --url "http://$HN:$PORT/v1" --out "$OUT/eq35-$bin.json" > /dev/null
    stop_server
  done
  python3 "$D/../htk/equal.py" compare "$OUT/eq35-base.json" "$OUT/eq35-intarsia.json" | tee "$OUT/eq35-compare.log" | tail -1
fi
echo "== done"
