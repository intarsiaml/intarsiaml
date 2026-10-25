#!/usr/bin/env bash
# Lane sp window 2, ON tatooine inside a fleetq job that takes radiance down (the stop is below, as in baseline.sh).
#   card 0: Qwen3.8-27B MXFP4 (tp1): spec off vs --spec mtp k=2,3 (captured) and k=2 eager, ids md5 + acceptance + tok/s
#   card 1: 35B k sweep 1/2/3 at 1k and 4k (captured), and an op table of eager MTP k=2 (where the time goes)
set -uo pipefail
trap 'bash $HOME/sp-win.sh restore' EXIT
systemctl --user stop radiance; sleep 3
S=$HOME/intarsiaml-sp/scripts/sp
H=/data/models/radiance/hf-cache/hub
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
mkdir -p $HOME/sp-ids
# the 27B's MXFP4 row-exact kernels have never run: prove them bitwise on the device before any 27B number is trusted
D=$HOME/intarsiaml-sp/deploy/tatooine/dev.sh
INTARSIA_REPO=$HOME/intarsiaml-sp INTARSIA_TARGET_VOL=intarsia-cargo-target-sp HIP_VISIBLE_DEVICES=0 $D env INTARSIA_REQUIRE_GPU=1 \
  cargo test --release -p intarsia-backend-hip --test rows_exact mxfp4 -- --test-threads=1 --nocapture < /dev/null 2>&1 \
  | grep -aE "^test |^test result|FAILED|panicked|G-B|^error" | tee $HOME/sp-ids/w2-rows-mxfp4.log
if grep -q "test result: ok" $HOME/sp-ids/w2-rows-mxfp4.log; then RUN27=1; else RUN27=0; echo "ROWS_EXACT MXFP4 FAILED: 27B runs skipped"; fi
( if [ "$RUN27" = 1 ]; then for a in "off:--spec off" "mtp2:--spec mtp --spec-k 2" "mtp3:--spec mtp --spec-k 3" "mtp2e:--spec mtp --spec-k 2"; do
    tag=${a%%:*}; args=${a#*:}
    if [ "$tag" = mtp2e ]; then export INTARSIA_SPEC_GRAPH=0; fi
    HIP_VISIBLE_DEVICES=0 MODELS=$H MODEL=$M27 bash $S/ids.sh sp t27-$tag 128,1024,4096,16384 $args
  done; fi ) > $HOME/sp-ids/w2-27.log 2>&1 &
( for k in 1 2 3; do HIP_VISIBLE_DEVICES=1 bash $S/ids.sh sp k$k 1024,4096 --spec mtp --spec-k $k; done
  HIP_VISIBLE_DEVICES=1 bash $S/ids.sh sp prof2 1024 --spec mtp --spec-k 2 --profile-every 1 --op-table ) > $HOME/sp-ids/w2-35.log 2>&1 &
wait
cat $HOME/sp-ids/w2-27.log $HOME/sp-ids/w2-35.log
