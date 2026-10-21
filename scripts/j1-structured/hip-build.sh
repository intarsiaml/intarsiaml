#!/usr/bin/env bash
# J1 HIP no-op, step 1 (CPU slot, radiance stays up): ship base and head trees to tatooine and build both in the HIP dev
# image, one shared cargo target volume, sequentially. The release CLI lands in each tree as j1-intarsia.
#   fleetq run --host tatooine --mem 0 --label "J1 HIP build" -- bash scripts/j1-structured/hip-build.sh BASE HEAD
set -uo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd)
BASE=${1:?base rev} HEAD=${2:?head rev}
for v in base:$BASE head:$HEAD; do
  n=${v%%:*}; rev=${v#*:}
  git -C "$R" archive "$rev" | ssh -o BatchMode=yes tatooine "rm -rf intarsiaml-j1-$n && mkdir -p intarsiaml-j1-$n && tar -x -C intarsiaml-j1-$n"
  ssh -o BatchMode=yes tatooine "set -o pipefail; export INTARSIA_REPO=\$HOME/intarsiaml-j1-$n INTARSIA_TARGET_VOL=intarsia-cargo-target-j1
    D=\$INTARSIA_REPO/deploy/tatooine/dev.sh
    \$D env INTARSIA_BUILD_STAMP='j1 $n $rev' CARGO_BUILD_JOBS=20 cargo build --release -p intarsia-cli --features hip > \$HOME/j1-build-$n.log 2>&1 \
      && \$D cp /target/release/intarsia /work/j1-intarsia && echo 'built $n $rev' || { tail -20 \$HOME/j1-build-$n.log; exit 2; }" \
    || exit 2
done
