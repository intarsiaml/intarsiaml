#!/usr/bin/env bash
# Window 1 (lane ink-hip-verify), both cards; main 3226af2b9 (binary intarsia-main, probe test built from the branch = main + the
# probe's timing-only knobs). Full logs in logs/w1/. Nothing set in the environment unless named; verify graph = default (on).
#   card 0: (a) verify cost probe, 1..8 rows, real 1k prompt (len1k.json, 1021 ids) with its greedy continuation as the rows (real
#           routing), no profiler; (b) the same at the 4108-id prompt; (c) (a) under rocprofv3 --kernel-trace (8 reps, csv, gzip);
#           (d) the draft-length sweep rep c0a (k3 k4 k5 k7) while time is left.
#   card 1: the draft-length sweep reps c1a (k3 k4 k5 k7) and c1b (k7 k5 k4 k3).
# Sweep arms: --spec dflash --spec-k K, cut 0.5 (INTARSIA_DSPARK_TAU=0 INTARSIA_DSPARK_CUT=0.5, HIP's default rule, set explicitly),
# 128 tokens, greedy, EOS honoured, over sweep.json (warm-up + 6 short golden prompts + 1021 / 4108 / 16455 ids).
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-verify
S=$REPO/.scratch
O=$REPO/logs/w1
rm -rf "$O"; mkdir -p "$O/sweep"
M=/models/Inkling-2-Mini
B=/work/.scratch/bin/intarsia-${BR:-main}
T0=$(date +%s)
LIMIT=${LIMIT:-230}   # seconds after the start: no new sweep process starts later
now() { date '+%T'; }
r0() { HIP_VISIBLE_DEVICES=0 bash "$S/run.sh" "$@"; }
r1() { HIP_VISIBLE_DEVICES=1 INK_CONTAINER=ink-hip-verify-run2 bash "$S/run.sh" "$@"; }
TEXE=$(grep -aoE '/target/release/deps/verify_rows_real-[0-9a-f]+' "$REPO/logs/build-mtest-verify_rows_real.log" | tail -1)
PENV="INTARSIA_VERIFY_DIR=$M INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_VERIFY_ROWS=8 INTARSIA_VERIFY_TIME_ONLY=1"
sweep_arm() {  # card k rep
  local f=$O/sweep/k$2@$3
  local el=$(( $(date +%s) - T0 ))
  if [ $el -gt $LIMIT ]; then echo "[sweep] skip k$2@$3 (t+${el}s)"; return; fi
  r$1 env INTARSIA_DSPARK_TAU=0 INTARSIA_DSPARK_CUT=0.5 $B generate -m $M --backend hip --max-seq 20480 --profile-every 0 --show-ids \
    --max-tokens 128 --prompt-ids /work/.scratch/prompts/sweep.json --spec dflash --spec-draft $M --spec-k $2 > $f.out 2> $f.err
  echo "[sweep] k$2@$3 rc $? t+$(( $(date +%s) - T0 ))s | $(grep -aE '^decode overall' $f.err | head -1) | $(grep -aE '^spec: verify' $f.err | head -1)"
}
card0() {
  echo "[c0] probe test exe $TEXE"
  echo "[c0] (a) probe 1k, rows 1..8 $(now)"
  r0 env $PENV INTARSIA_VERIFY_PROMPT=/work/.scratch/prompts/len1k.json $TEXE --nocapture > $O/probe1k.log 2>&1
  echo "rc $?"; grep -aE "PROBE_TIME|PROBE_GRAPHS|panicked|test result" $O/probe1k.log | cut -c1-400
  echo "[c0] (b) probe 4k, rows 1..8 $(now)"
  r0 env $PENV INTARSIA_VERIFY_PROMPT=/work/.scratch/prompts/len4k.json $TEXE --nocapture > $O/probe4k.log 2>&1
  echo "rc $?"; grep -aE "PROBE_TIME|panicked|test result" $O/probe4k.log | cut -c1-400
  echo "[c0] (c) rocprofv3 kernel trace of (a), 8 reps $(now)"
  mkdir -p "$O/rp"
  INTARSIA_IMAGE=intarsia-hip-prof:rocm7.2.4-rust1.90.0 r0 env $PENV INTARSIA_VERIFY_PROMPT=/work/.scratch/prompts/len1k.json \
    INTARSIA_VERIFY_TIME_REPS=8 rocprofv3 --kernel-trace --output-format csv -d /work/logs/w1/rp -o run -- $TEXE --nocapture > $O/probe-rp.log 2>&1
  echo "rc $?"; grep -aE "PROBE_TIME|panicked|test result" $O/probe-rp.log | cut -c1-400
  csv=$(find "$O/rp" -name '*kernel_trace.csv' | head -1)
  echo "kernel trace: ${csv:-none} $(now)"
  if [ -n "$csv" ]; then
    python3 "$S/rowtrace.py" "$csv" 8 8 2 --kernels > $O/rowtrace.txt 2>&1; head -12 $O/rowtrace.txt
    gzip -f "$csv"
  fi
  echo "[c0] (d) sweep rep c0a $(now)"
  for k in 3 4 5 7; do sweep_arm 0 $k c0a; done
  echo "[c0] done $(now)"
}
card1() {
  for k in 3 4 5 7; do sweep_arm 1 $k c1a; done
  for k in 7 5 4 3; do sweep_arm 1 $k c1b; done
  echo "[c1] done $(now)"
}
echo "[stage] start $(now)"
card0 > "$O/c0.log" 2>&1 &
card1 > "$O/c1.log" 2>&1 &
wait
echo "---- card 0"; cat "$O/c0.log"
echo "---- card 1"; cat "$O/c1.log"
echo "[stage] done $(now)"
