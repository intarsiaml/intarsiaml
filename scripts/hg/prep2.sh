#!/usr/bin/env bash
# Lane hg, on tatooine inside a fleetq job (--load cpu): rebuild the dev and base serve binaries and compile (not run) the
# dev tests the HIP gate window runs, so the window itself only runs.
set -uo pipefail
cd ~
for t in hg hg-base; do
  vol=intarsia-cargo-target-${t/-base/base}
  echo "== build $t"
  INTARSIA_REPO=$HOME/intarsiaml-$t INTARSIA_TARGET_VOL=$vol \
    ~/intarsiaml-hg/deploy/tatooine/dev.sh cargo build --release --locked --features hip -p intarsia-cli 2>&1 < /dev/null | tail -3
done
echo "== compile dev tests"
INTARSIA_REPO=$HOME/intarsiaml-hg INTARSIA_TARGET_VOL=intarsia-cargo-target-hg \
  ~/intarsiaml-hg/deploy/tatooine/dev.sh cargo test --release --no-run -p intarsia-models --test batch_recurrent --test batch_cuda_real 2>&1 < /dev/null | tail -4
INTARSIA_REPO=$HOME/intarsiaml-hg INTARSIA_TARGET_VOL=intarsia-cargo-target-hg \
  ~/intarsiaml-hg/deploy/tatooine/dev.sh cargo test --release --no-run -p intarsia-core --lib 2>&1 < /dev/null | tail -2
INTARSIA_REPO=$HOME/intarsiaml-hg INTARSIA_TARGET_VOL=intarsia-cargo-target-hg \
  ~/intarsiaml-hg/deploy/tatooine/dev.sh cargo test --release --no-run -p intarsia-backend-hip --test packed_rows 2>&1 < /dev/null | tail -6
echo PREP2 DONE
