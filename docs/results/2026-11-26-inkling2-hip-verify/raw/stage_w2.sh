#!/usr/bin/env bash
# Window 2 (lane ink-hip-verify), both cards; binary split1 (expert slices, default = the rule) and its test targets. Full logs in logs/w2/.
#   card 0: (a) verify cost probe at the real 1k prompt, 1..8 rows, INTARSIA_HIP_INK_MOE_SPLIT = 1 (main's grid) / unset (the rule) /
#           0:1 (gate_up sliced only) / 1 again; (b) tests/ink_moe.rs rows_exact + odd ids (the slices bitwise); (c) the bitwise verify
#           probe (rows 2..8 at 17 / 1024 / 4000 + 16 greedy windows) under the rule.
#   card 1: (d) the MoE bench with real-text unions (T 1, 2, 3, 4, 6, 8; slice counts); (e) draft-length screen under the rule, one
#           process each of k3 / k5 / k4 (cut 0.5), sweep.json, while time is left.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-verify
S=$REPO/.scratch
O=$REPO/logs/w2
rm -rf "$O"; mkdir -p "$O/sweep"
M=/models/Inkling-2-Mini
B=/work/.scratch/bin/intarsia-${BR:-split1}
T0=$(date +%s)
LIMIT=${LIMIT:-150}
now() { date '+%T'; }
r0() { HIP_VISIBLE_DEVICES=0 bash "$S/run.sh" "$@"; }
r1() { HIP_VISIBLE_DEVICES=1 INK_CONTAINER=ink-hip-verify-run2 bash "$S/run.sh" "$@"; }
TEXE=$(grep -aoE '/target/release/deps/verify_rows_real-[0-9a-f]+' "$REPO/logs/build-mtest-verify_rows_real.log" | tail -1)
MEXE=$(grep -aoE '/target/release/deps/ink_moe-[0-9a-f]+' "$REPO/logs/build-htest-ink_moe.log" | tail -1)
PENV="INTARSIA_VERIFY_DIR=$M INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_VERIFY_ROWS=8 INTARSIA_VERIFY_TIME_ONLY=1 INTARSIA_VERIFY_PROMPT=/work/.scratch/prompts/len1k.json"
card0() {
  echo "[c0] exes $TEXE $MEXE"
  for sp in 1 unset 0:1 1b; do
    echo "[c0] (a) probe 1k rows 1..8, split $sp $(now)"
    case $sp in
      unset) r0 env $PENV $TEXE --nocapture > $O/probe-$sp.log 2>&1 ;;
      1b) INTARSIA_HIP_INK_MOE_SPLIT=1 r0 env $PENV $TEXE --nocapture > $O/probe-$sp.log 2>&1 ;;
      *) INTARSIA_HIP_INK_MOE_SPLIT=$sp r0 env $PENV $TEXE --nocapture > $O/probe-$sp.log 2>&1 ;;
    esac
    echo "rc $?"; grep -aE "PROBE_TIME |PROBE_TIME_MIN|panicked" $O/probe-$sp.log | cut -c1-400
  done
  echo "[c0] (b) ink_moe rows_exact + odd ids $(now)"
  r0 $MEXE ink_moe_rows_exact ink_moe_wave_grouping_odd_ids --test-threads=1 > $O/moe-tests.log 2>&1
  echo "rc $?"; grep -aE "^test |test result|panicked|bitwise" $O/moe-tests.log | cut -c1-300
  echo "[c0] (c) bitwise verify probe, rows 2..8 at 17 / 1024 / 4000 + 16 greedy windows, the rule $(now)"
  r0 env INTARSIA_VERIFY_DIR=$M INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_VERIFY_ROWS=8 INTARSIA_VERIFY_LENS=17,1024,4000 $TEXE --nocapture > $O/probe-bitwise.log 2>&1
  echo "rc $?"; grep -aE "PROBE_TIME |PROBE_RESULT|PROBE_GRAPHS|DIFF|panicked|test result" $O/probe-bitwise.log | cut -c1-400 | head -12
  echo "[c0] done $(now)"
}
sweep_arm() {  # k
  local f=$O/sweep/k$1@w2
  local el=$(( $(date +%s) - T0 ))
  if [ $el -gt $LIMIT ]; then echo "[sweep] skip k$1 (t+${el}s)"; return; fi
  r1 env INTARSIA_DSPARK_TAU=0 INTARSIA_DSPARK_CUT=0.5 $B generate -m $M --backend hip --max-seq 20480 --profile-every 0 --show-ids \
    --max-tokens 128 --prompt-ids /work/.scratch/prompts/sweep.json --spec dflash --spec-draft $M --spec-k $1 > $f.out 2> $f.err
  echo "[sweep] k$1 rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aE '^decode overall' $f.err | head -1) | $(grep -aE '^spec: verify' $f.err | head -1)"
}
card1() {
  echo "[c1] (d) MoE bench, real-text unions $(now)"
  r1 env INK_MOE_BENCH=1 INK_MOE_BENCH_REAL=1 INK_MOE_BENCH_T=1,2,3,4,6,8 INK_MOE_BENCH_NOSWEEP=1 $MEXE ink_moe_bench --nocapture --test-threads=1 > $O/bench.log 2>&1
  echo "rc $?"; grep -aE "INKMOEBENCH" $O/bench.log | grep -E "default|split" | cut -c1-200
  echo "[c1] (e) k screen under the rule $(now)"
  for k in 3 5 4; do sweep_arm $k; done
  echo "[c1] done $(now)"
}
echo "[stage] start $(now)"
card0 > "$O/c0.log" 2>&1 &
card1 > "$O/c1.log" 2>&1 &
wait
echo "---- card 0"; cat "$O/c0.log"
echo "---- card 1"; cat "$O/c1.log"
echo "[stage] done $(now)"
