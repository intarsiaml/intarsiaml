#!/usr/bin/env bash
# Lane spark35, window 1 (one fleetq slot on $HOST, --quiet gpu,cpu): builds, gates, the KL arms (D, and C by emulation) and the serve arms
# (greedy ids, needle at 32k / 64k, MMLU, GSM8K). Run from the repo root of the lane worktree:
#   HOST=spark2 fleetq run --host spark2 --quiet gpu,cpu --mem 74 --expect 170 --label "spark35: ..." -- scripts/spark35num/w1.sh
# STAGES selects (default: build data gates lanes). BUDGET_MIN (default 165) bounds the window: an arm that does not fit is skipped.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
STAGES=${STAGES:-"build data gates lanes"}
export S35_DEADLINE=$(( $(date +%s) + ${BUDGET_MIN:-165} * 60 ))
TOK=$(ls ~/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/*/tokenizer.json | head -1)
echo "window 1 on $HOST, $(date +%T), head $(git -C "$REPO" rev-parse --short HEAD), MemAvailable $(free_gb) GB"
for st in $STAGES; do case $st in
  build)
    BASE=$(git -C "$REPO" merge-base HEAD origin/main); echo "main = $BASE" | tee "$RES/main-ref.txt"
    "$H/build.sh" "$BASE" main || { echo "MAIN BUILD FAILED"; exit 1; }
    if ! "$H/build.sh" HEAD dev; then
      echo "dev build from HEAD failed: falling back to the emulation-only commit ${FALLBACK_REF:-f7278f38b} (no real bf16 layout in this window)"
      "$H/build.sh" "${FALLBACK_REF:-f7278f38b}" dev || { echo "BUILD FAILED"; exit 1; }
    else
      rsh "cd $L/src && CARGO_BUILD_JOBS=16 cargo test -p intarsia-backend-cuda --release --test gdn_state16 --no-run 2>&1 | grep -E '^error|Finished|could not' | tail -4; cp \$(ls -t target/release/deps/gdn_state16-* | grep -v '\\.d\$' | head -1) $L/bin/gdn_state16 && echo built gdn_state16"
    fi ;;
  data)
    (cd "$REPO" && uv run --with numpy --with safetensors --with tokenizers scripts/spark35num/mkcarrier.py "$TOK" "$RES/data") > "$RES/data-carriers.log" 2>&1
    (cd "$REPO" && uv run --with tokenizers scripts/spark35num/mkneedle.py "$TOK" "$RES/data") > "$RES/data-needle.log" 2>&1
    cat "$RES/data-carriers.log" "$RES/data-needle.log" | tail -4
    rsh "mkdir -p $L/carriers"; scp -q -o BatchMode=yes "$RES/data/std.safetensors" "$RES/data/dec8k.safetensors" "$HOST:intarsiaml-s35/carriers/" ;;
  gates)
    O='$HOME/intarsiaml-s35/out'; fails=0
    for t in main dev; do
      for k in json txt; do
        rsh "$L/bin/$t plan show --backend cuda --model $M35 --max-seq 8192 --mem-budget 40 $([[ $k = json ]] && echo --json) < /dev/null > $O/plan-35-$t.$k 2> /dev/null"
      done
      rsh "$L/bin/$t kernels test --backend cuda --quick < /dev/null > $O/kt-$t.log 2>&1; echo kernels-test-$t exit \$?"
    done
    for k in json txt; do
      if rsh "cmp -s $O/plan-35-main.$k $O/plan-35-dev.$k && [ -s $O/plan-35-dev.$k ]"; then echo "PASS plan show 35B $k byte-identical"; else echo "FAIL plan show 35B $k"; fails=$((fails+1)); fi
    done
    rsh "grep -a -E 'ADMITTED|SKIPPED|FAIL|incomplete' $O/kt-main.log | tail -4; echo ---; grep -a -E 'ADMITTED|SKIPPED|FAIL|incomplete' $O/kt-dev.log | tail -4"
    if rsh "diff <(grep -a -E 'ADMITTED|SKIPPED|FAIL' $O/kt-main.log) <(grep -a -E 'ADMITTED|SKIPPED|FAIL' $O/kt-dev.log) > /dev/null"; then echo "PASS kernels test --quick main == dev (status lines)"; else echo "FAIL kernels test --quick differs"; fails=$((fails+1)); fi
    if rsh "test -x $L/bin/gdn_state16"; then
      rsh "$L/bin/gdn_state16 --nocapture --test-threads=1 > $O/gdn_state16.log 2>&1; echo gdn_state16 exit \$?; grep -a -E 'test result|gdn_state16:' $O/gdn_state16.log | tail -12"
    else echo "gdn_state16: no binary (emulation-only build)"; fi
    scp -q -o BatchMode=yes "$HOST:intarsiaml-s35/out/plan-35-*" "$HOST:intarsiaml-s35/out/gdn_state16.log" "$HOST:intarsiaml-s35/out/kt-*.log" "$RES/" 2>&1 | tail -1
    echo "gates fails=$fails" ;;
  lanes)
    fb=$(free_gb); echo "lanes: MemAvailable $fb GB"
    # priority order: the D / C evidence first, the noise reference and the combination next, the real-layout equality check last
    KL1="base cemu fp8"; SV1="main base fp8 cemu"; KL2="noise cdemu"; SV2="cdemu creal"; KL3="creal"
    if (( fb >= 100 )); then
      echo "vLLM is down: KL lane and serve lane in parallel (MEMB=40 each)"
      (for a in "$KL1" "$KL2" "$KL3"; do MEMB=40 ARMS="$a" "$H/kl.sh"; done) > "$RES/lane-kl.log" 2>&1 &
      (for a in "$SV1" "$SV2"; do MEMB=40 PORT=8196 ARMS="$a" "$H/serve.sh"; done) > "$RES/lane-serve.log" 2>&1 &
      wait
    else
      echo "vLLM is up: sequential, MEMB=45"
      { ARMS="$KL1" "$H/kl.sh"; ARMS="$SV1" "$H/serve.sh"; ARMS="$KL2" "$H/kl.sh"; ARMS="$SV2" "$H/serve.sh"; ARMS="$KL3" "$H/kl.sh"; } > "$RES/lane-all.log" 2>&1
    fi
    tail -3 "$RES"/lane-*.log ;;
esac; done
echo "window 1 done $(date +%T), $(left_min) min of the budget left"
