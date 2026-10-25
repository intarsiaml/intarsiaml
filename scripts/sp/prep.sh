#!/usr/bin/env bash
# Lane sp, ON tatooine inside a fleetq job (--load cpu,disk; radiance stays up): copy a warm cargo target volume pair
# and build the sp (branch) and sp-base (main) serve binaries in the HIP dev container.
#   prep.sh [sp|sp-base ...]   (default both)
set -euo pipefail
trees=("$@"); [[ ${#trees[@]} -eq 0 ]] && trees=(sp sp-base)
cd ~
for pair in "hg sp" "hgbase spbase"; do
  set -- $pair
  docker volume inspect intarsia-cargo-target-$2 >/dev/null 2>&1 && continue
  docker volume create intarsia-cargo-target-$2 >/dev/null
  docker run --rm -v intarsia-cargo-target-$1:/s:ro -v intarsia-cargo-target-$2:/d alpine sh -c 'cp -a /s/. /d/ && chmod 0777 /d'
done
for t in "${trees[@]}"; do
  vol=intarsia-cargo-target-${t/-base/base}
  echo "== build $t ($vol)"
  INTARSIA_REPO=$HOME/intarsiaml-$t INTARSIA_TARGET_VOL=$vol \
    ~/intarsiaml-sp/deploy/tatooine/dev.sh cargo build --release --locked --features hip -p intarsia-cli 2>&1 | tail -4
done
echo "== compile gate tests (sp tree)"
for spec in "-p intarsia-models --test batch_recurrent --test batch_cuda_real --test mtp" "-p intarsia-backend-hip --test rows_exact --test rows_registry"; do
  INTARSIA_REPO=$HOME/intarsiaml-sp INTARSIA_TARGET_VOL=intarsia-cargo-target-sp \
    ~/intarsiaml-sp/deploy/tatooine/dev.sh cargo test --release --no-run $spec 2>&1 < /dev/null | tail -4
done
echo PREP DONE
