#!/usr/bin/env bash
# Lane mtp-snap-gpu (ON tatooine, CPU only): the stamped release CLI of the synced tree, copied to /target/intarsia-<pr|main> in the cargo volume.
set -u
. "$(dirname "$0")/lib.sh"
which=${1:?pr|main}
"$D" env INTARSIA_BUILD_STAMP="$STAMP" cargo build --release -p intarsia-cli --features hip < /dev/null; rc=$?; echo "BUILD $which cli exit $rc"
[[ $rc == 0 ]] || exit "$rc"
"$D" cp /target/release/intarsia "/target/intarsia-$which" < /dev/null; echo "COPY $which exit $?"
