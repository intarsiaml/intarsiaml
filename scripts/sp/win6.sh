#!/usr/bin/env bash
# Lane sp window 6, ON endor inside a fleetq job (--takes-down radiance): localeval speed-spec (natural-v1 128/1k/4k/16k, 35B),
# three arms on one server each: main (eager MTP k=2), branch (captured k=2), branch + --mtp-draft-vocab 65536; then the extras
# (27B Chinese/code, no-op ids). Radiance is restored by the trap whatever happens.
S=/opt/code/wt-sp/scripts/sp
trap 'ssh tatooine "bash ~/sp-win.sh restore" 2>&1 | tail -2' EXIT
bash $S/arm.sh sp-base main-mtp2 spec "--spec mtp --spec-k 2" "" 1
bash $S/arm.sh sp br-mtp2 spec "--spec mtp --spec-k 2" "" 1
bash $S/arm.sh sp br-mtp2-v65536 spec "--spec mtp --spec-k 2 --mtp-draft-vocab 65536" "" 1
ssh tatooine "bash ~/intarsiaml-sp/scripts/sp/extras.sh"
# the 8B batched-decode gate (skipped without INTARSIA_QWEN3_DIR in the first gates window)
ssh tatooine 'INTARSIA_REPO=$HOME/intarsiaml-sp INTARSIA_TARGET_VOL=intarsia-cargo-target-sp HIP_VISIBLE_DEVICES=1 $HOME/intarsiaml-sp/deploy/tatooine/dev.sh env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN3_DIR=/models/Qwen3-8B cargo test --release -p intarsia-models --test batch_cuda_real real_qwen3_8b -- --test-threads=1 --nocapture < /dev/null 2>&1 | grep -aE "^test |^test result|FAILED|panicked|^error"' | tee /tmp/sp-arm/8b.log
