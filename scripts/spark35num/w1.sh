#!/usr/bin/env bash
# Lane spark35, window 1 (one fleetq slot on $HOST, --quiet gpu,cpu): builds, gates, the KL arms (D, and C by emulation) and the serve arms
# (greedy ids, needle at 32k / 64k, MMLU, GSM8K). Run from the repo root of the lane worktree:
#   HOST=spark2 fleetq run --host spark2 --quiet gpu,cpu --mem 60 --expect 170 --label "spark35: ..." -- scripts/spark35num/w1.sh
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
    "$H/build.sh" "$BASE" main && "$H/build.sh" WORKTREE dev || { echo "BUILD FAILED"; exit 1; } ;;
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
    scp -q -o BatchMode=yes "$HOST:intarsiaml-s35/out/plan-35-*" "$HOST:intarsiaml-s35/out/kt-*.log" "$RES/" 2>&1 | tail -1
    echo "gates fails=$fails" ;;
  lanes)
    fb=$(free_gb); echo "lanes: MemAvailable $fb GB"
    if (( fb >= 100 )); then
      echo "vLLM is down: KL lane and serve lane in parallel (MEMB=40 each)"
      MEMB=40 "$H/kl.sh" > "$RES/lane-kl.log" 2>&1 &
      MEMB=40 PORT=8196 "$H/serve.sh" > "$RES/lane-serve.log" 2>&1 &
      wait
    else
      echo "vLLM is up: sequential, MEMB=45"
      "$H/serve.sh" > "$RES/lane-serve.log" 2>&1
      "$H/kl.sh" > "$RES/lane-kl.log" 2>&1
    fi
    tail -3 "$RES/lane-kl.log" "$RES/lane-serve.log" ;;
esac; done
echo "window 1 done $(date +%T), $(left_min) min of the budget left"
