#!/usr/bin/env bash
# Lane htk GPU window 1, ON tatooine inside a fleetq job (--takes-down radiance, <= 30 min): the HIP top-k kernel tests, the admission sweep,
# the mutants (every one must FAIL tests/topk.rs), and the 35B model-level gates (device draws == host draws; sampled step timing).
# Radiance is restarted by the exit trap whatever happens.
#   fleetq run --host tatooine --takes-down radiance --mem 30 --quiet gpu,cpu --expect 28 --label "HTK window 1" -- ssh tatooine 'bash -s' < scripts/htk/window1.sh
set -uo pipefail
R=$HOME/intarsiaml-htk; SRC=$R/src; L=$R/logs/w1; mkdir -p $L
export INTARSIA_REPO=$SRC INTARSIA_TARGET_VOL=intarsia-cargo-htk HIP_VISIBLE_DEVICES=0
D=$SRC/deploy/tatooine/dev.sh
T0=$(date +%s); el() { echo $(( $(date +%s) - T0 )); }
# HELD=1: the caller (window.sh) already stopped radiance and restores it
if [ -z "${HELD:-}" ]; then
  trap 'bash $SRC/scripts/htk/win.sh restore' EXIT
  echo "[w1] stopping radiance $(date +%T)"; systemctl --user stop radiance; sleep 5
fi
M=/models/Qwen3.6-35B-A3B-NVFP4
echo "== topk tests (t+$(el) s)"
$D timeout 400 /work/.bin/topk --nocapture > $L/topk.txt 2>&1 < /dev/null; echo "topk exit $?"; grep -E 'G-B|test result|FAILED|panicked' $L/topk.txt | head -20
echo "== mutants (t+$(el) s)"
for m in ${MUTANTS:-htk-tiebreak htk-slice-end htk-stage2-n htk-nanflag htk-nonstrict htk-exhausted htk-nomemset htk-kv htk-scratch}; do
  if [ ! -e $SRC/.bin/topk-$m ]; then echo "$m: NO BINARY"; continue; fi
  $D timeout 200 /work/.bin/topk-$m > $L/mut-$m.txt 2>&1 < /dev/null; rc=$?
  echo "$m: exit $rc $([ $rc -ne 0 ] && echo CAUGHT || echo SURVIVED) ($(grep -E '^test .*FAILED' $L/mut-$m.txt | sed -E 's/test (.*) \.\.\. FAILED/\1/' | tr '\n' ' '))"
done
echo "== admission (t+$(el) s)"
$D timeout 420 /work/.bin/admission --nocapture > $L/admission.txt 2>&1 < /dev/null; echo "admission exit $?"; grep -E 'test result|FAILED|panicked|passed|failed' $L/admission.txt | head
echo "== 35B: device draws == host draws (t+$(el) s)"
$D env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_QWEN36_DIR=$M INTARSIA_DEVICE_SAMPLING=1 INTARSIA_REQUIRE_GPU=1 timeout 480 /work/.bin/batch_cuda_real real_35b_device_sampling_draws_the_host_tokens --exact --nocapture > $L/draws.txt 2>&1 < /dev/null
echo "draws exit $?"; grep -E 'tokens drawn|test result|panicked|SKIPPED' $L/draws.txt | head
if [ "$(el)" -gt "${TIMING_BEFORE:-780}" ]; then echo "== 35B: sampled step timing SKIPPED (late, t+$(el) s)"; echo "[w1] done t+$(el) s"; exit 0; fi
echo "== 35B: sampled step timing (t+$(el) s)"
$D env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_QWEN36_DIR=$M INTARSIA_DEVICE_SAMPLING=1 INTARSIA_REQUIRE_GPU=1 timeout 400 /work/.bin/batch_cuda_real real_35b_batch_step_sampled_timing --exact --ignored --nocapture > $L/timing.txt 2>&1 < /dev/null
echo "timing exit $?"; grep -E 'TIMING|test result|panicked' $L/timing.txt
echo "[w1] done t+$(el) s"
