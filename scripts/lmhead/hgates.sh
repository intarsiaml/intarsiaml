#!/usr/bin/env bash
# Lane lmhead-fp8, ON tatooine inside a window (hdrive.sh `gates`): variable unset, `plan show` (text + JSON, tp2 and tp1, 27B and 35B) and
# `kernels test --backend hip` of the base (main 5390e65b4) and the head binary; the end of this script compares the files.
set -uo pipefail
M27=/models/radiance/hf-cache/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
M35=/models/intarsia/Qwen3.6-35B-A3B-NVFP4
mkdir -p ~/lmh-gates; rm -f ~/lmh-gates/*
d() { # d <bin dir> <args...>
  local b=$1; shift
  sleep 8   # the previous container gives its cards back
  timeout -s KILL 900 docker run --rm --network host --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" \
    --group-add video --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=0,1 \
    -v "$HOME/$b":/b:ro -v /data/models:/models:ro intarsia-hip-dev:rocm7.2.4-rust1.90.0 /b/intarsia "$@" < /dev/null
}
for t in base head; do
  b=lmhead-bin; [[ $t == base ]] && b=lmhbase-bin
  for m in 27 35; do
    M=$M27; [[ $m == 35 ]] && M=$M35
    for tp in 2 1; do
      x=""; [[ $tp == 2 ]] && x="--tp 2 --devices 0,1"
      d $b plan show --backend hip --model $M --max-seq 8192 --mem-budget 29 $x --json > ~/lmh-gates/plan-$m-tp$tp-$t.json 2> ~/lmh-gates/plan-$m-tp$tp-$t.json.err
      d $b plan show --backend hip --model $M --max-seq 8192 --mem-budget 29 $x > ~/lmh-gates/plan-$m-tp$tp-$t.txt 2> ~/lmh-gates/plan-$m-tp$tp-$t.txt.err
    done
  done
  d $b kernels test --backend hip > ~/lmh-gates/kernels-$t.out 2>&1; echo "GATES kernels $t exit $?"
done
for f in ~/lmh-gates/plan-*-base.*; do g=${f/-base./-head.}; if cmp -s "$f" "$g" && [ -s "$f" ]; then echo "GATES $(basename $f | sed 's/-base//'): byte-identical ($(wc -c < $f) B)"; else echo "GATES $(basename $f | sed 's/-base//'): DIFFER or empty"; fi; done
if diff <(grep -vE "[0-9]+(\.[0-9]+)? ?(ms|us|s)\b" ~/lmh-gates/kernels-base.out) <(grep -vE "[0-9]+(\.[0-9]+)? ?(ms|us|s)\b" ~/lmh-gates/kernels-head.out) > ~/lmh-gates/kernels.diff; then echo "GATES kernels test: identical output (timing lines excluded)"; else echo "GATES kernels test: DIFFERS"; fi
tail -n 3 ~/lmh-gates/kernels-base.out ~/lmh-gates/kernels-head.out
