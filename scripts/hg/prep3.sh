#!/usr/bin/env bash
# Lane hg, on tatooine inside a fleetq job (--load cpu,disk): the packed-prefill tree's cargo target volume (seeded from hg's) and build.
set -uo pipefail
cd ~
v=intarsia-cargo-target-hgpk
if ! docker volume inspect $v >/dev/null 2>&1; then
  docker volume create $v >/dev/null
  docker run --rm -v intarsia-cargo-target-hg:/s:ro -v $v:/d alpine sh -c 'cp -a /s/. /d/ && chmod 0777 /d'
fi
D="env INTARSIA_REPO=$HOME/intarsiaml-hgpk INTARSIA_TARGET_VOL=$v $HOME/intarsiaml-hgpk/deploy/tatooine/dev.sh"
echo "== build hgpk"; $D cargo build --release --locked --features hip -p intarsia-cli 2>&1 < /dev/null | tail -3
echo "== compile tests"
$D cargo test --release --no-run -p intarsia-models --test batch_recurrent --test batch_cuda_real 2>&1 < /dev/null | tail -3
$D cargo test --release --no-run -p intarsia-backend-hip --test packed_rows 2>&1 < /dev/null | tail -2
echo PREP3 DONE
