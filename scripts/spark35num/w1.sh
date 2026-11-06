#!/usr/bin/env bash
# Lane spark35, window 1 (one fleetq slot on $HOST, --quiet gpu,cpu): builds, gates, the KL arms (D, and C by emulation) and the serve arms
# (greedy ids, needle at 32k / 64k, MMLU, GSM8K). Run from the repo root of the lane worktree:
#   HOST=spark2 fleetq run --host spark2 --quiet gpu,cpu --mem 74 --expect 170 --label "spark35: ..." -- scripts/spark35num/w1.sh
# DEV_REF pins the commit the "dev" binary is built from (default HEAD; the D default-flip commit must never be the base arm). STAGES selects (default: build data gates lanes; build always runs first and is a no-op for a commit already built). KL1..KL3 / SV1 / SV2 (arm lists, "" skips) split the lanes across jobs. BUDGET_MIN (default 165) bounds the window: an arm that does not fit is skipped.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
STAGES=${STAGES:-"build data gates lanes"}
export S35_DEADLINE=$(( $(date +%s) + ${BUDGET_MIN:-165} * 60 ))
TOK=$(ls ~/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/*/tokenizer.json | head -1)
echo "window 1 on $HOST, $(date +%T), head $(git -C "$REPO" rev-parse --short HEAD), MemAvailable $(free_gb) GB"
[[ ${SKIP_DATA:-0} = 1 ]] || case " $STAGES " in *" data "*) ;; *) STAGES="data $STAGES" ;; esac   # SKIP_DATA=1: keep the committed carriers (the data stage hashes the CURRENT docs/ and crates/ tree)
case " $STAGES " in *" build "*) ;; *) STAGES="build $STAGES" ;; esac   # idempotent: build.sh keeps a binary already built from the commit
for st in $STAGES; do case $st in
  build)
    BASE=$(git -C "$REPO" merge-base "$DEV_SHA" origin/main); echo "main = $BASE, dev = $DEV_SHA" | tee "$RES/main-ref.txt"
    "$H/build.sh" "$BASE" main || { echo "MAIN BUILD FAILED"; exit 1; }
    ensure_dev || exit 1
    # test binaries: the GDN state test, and the existing bitwise guards of the kernels this lane rewrote
    for t in gdn_state16 gdn_split gdn_conv16 gdn_lanes rows_exact row_tables; do
      rsh "cd $L/src && CARGO_BUILD_JOBS=16 cargo test -p intarsia-backend-cuda --release --test $t --no-run > $L/out/build-test-$t.log 2>&1; f=\$(ls -t target/release/deps/$t-* 2>/dev/null | grep -v '\\.d\$' | head -1); [ -n \"\$f\" ] && cp \$f $L/bin/t_$t && echo built t_$t || { echo TEST BUILD FAILED $t; tail -15 $L/out/build-test-$t.log; }"
    done ;;
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
    for t in gdn_state16 gdn_split gdn_conv16 gdn_lanes rows_exact row_tables; do
      if rsh "test -x $L/bin/t_$t"; then
        rsh "cd $L/src && INTARSIA_REQUIRE_GPU=1 $L/bin/t_$t --nocapture --test-threads=1 > $O/test-$t.log 2>&1; echo \"test $t exit \$?\"; grep -a -E 'test result|gdn_state16:|SKIPPED' $O/test-$t.log | tail -4" | tee -a "$RES/gpu-tests.txt"
      else echo "FAIL test binary t_$t missing"; fails=$((fails+1)); fi
    done
    "$H/idgate.sh" || fails=$((fails+1))
    scp -q -o BatchMode=yes "$HOST:intarsiaml-s35/out/plan-35-*" "$HOST:intarsiaml-s35/out/test-*.log" "$HOST:intarsiaml-s35/out/kt-*.log" "$RES/" 2>&1 | tail -1
    echo "gates fails=$fails" ;;
  lanes)
    fb=$(free_gb); echo "lanes: MemAvailable $fb GB"
    # priority order: the D / C evidence first, the noise reference and the combination next, the real-layout equality check last
    KL1=${KL1-"base cemu fp8"}; SV1=${SV1-"main base fp8 cemu"}; KL2=${KL2-"noise cdemu"}; SV2=${SV2-"cdemu creal"}; KL3=${KL3-"creal"}
    kl() { [[ -n $1 ]] && MEMB=${2:-45} ARMS="$1" "$H/kl.sh"; return 0; }
    sv() { [[ -n $1 ]] && MEMB=${2:-45} PORT=8196 ARMS="$1" "$H/serve.sh"; return 0; }
    if (( fb >= 100 )); then
      echo "vLLM is down: KL lane and serve lane in parallel (MEMB=40 each)"
      (kl "$KL1" 40; kl "$KL2" 40; kl "$KL3" 40) > "$RES/lane-kl.log" 2>&1 &
      (sv "$SV1" 40; sv "$SV2" 40) > "$RES/lane-serve.log" 2>&1 &
      wait
    else
      echo "vLLM is up: sequential, MEMB=45"
      { kl "$KL1"; sv "$SV1"; kl "$KL2"; sv "$SV2"; kl "$KL3"; } > "$RES/lane-all.log" 2>&1
    fi
    tail -3 "$RES"/lane-*.log 2>/dev/null ;;
esac; done
echo "window 1 done $(date +%T), $(left_min) min of the budget left"
