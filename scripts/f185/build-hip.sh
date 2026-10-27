#!/usr/bin/env bash
# Lane f185: CPU-slot build ON tatooine of the synced trees (scripts/f185/sync.sh) in the HIP dev image, one after the other in ONE work tree
# (~/intarsiaml-f185/work) and ONE target volume (intarsia-cargo-f185), so the second build is incremental. Release CLIs end up in work/.bin/<name>.
#   fleetq run --host tatooine --load cpu --expect 40 --label "f185 build" -- ssh tatooine "NAMES='base dev' bash -s" < scripts/f185/build-hip.sh
set -uo pipefail
R=$HOME/intarsiaml-f185
mkdir -p $R/work/.bin $R/logs
export INTARSIA_REPO=$R/work INTARSIA_TARGET_VOL=intarsia-cargo-f185
for name in ${NAMES:-base dev}; do
  echo "== $name $(date +%T)"
  rsync -a --delete --exclude .bin $R/src-$name/ $R/work/
  "$INTARSIA_REPO"/deploy/tatooine/dev.sh env CARGO_BUILD_JOBS=${JOBS:-20} cargo build --release -p intarsia-cli --features hip > $R/logs/build-$name.log 2>&1 < /dev/null
  echo "build $name exit $? $(date +%T)"; grep -E '^error|Finished' $R/logs/build-$name.log | tail -3
  "$INTARSIA_REPO"/deploy/tatooine/dev.sh cp /target/release/intarsia /work/.bin/$name < /dev/null
done
ls -la $R/work/.bin; md5sum $R/work/.bin/* | cut -c1-12,33-
