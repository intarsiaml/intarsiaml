#!/usr/bin/env bash
# Lane sp, ON tatooine inside a fleetq job that takes radiance down: micro-timing of the 27B verify GEMVs and lm_head (card 1).
D=$HOME/intarsiaml-sp/deploy/tatooine/dev.sh
trap 'bash $HOME/sp-win.sh restore' EXIT
systemctl --user stop radiance; sleep 3
mkdir -p $HOME/sp-ids
INTARSIA_REPO=$HOME/intarsiaml-sp INTARSIA_TARGET_VOL=intarsia-cargo-target-sp HIP_VISIBLE_DEVICES=1 $D env INTARSIA_REQUIRE_GPU=1 \
  cargo test --release -p intarsia-backend-hip --test rows_exact -- --ignored --nocapture timing_mxfp4_rows < /dev/null 2>&1 \
  | grep -aE "^TIME|^test |panicked|^error" | tee $HOME/sp-ids/timing-mxfp4.log
