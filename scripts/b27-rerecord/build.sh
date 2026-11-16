#!/usr/bin/env bash
# Lane b27-rerecord (ON tatooine, CPU only): the stamped release CLI with HIP.
set -u
. "$(dirname "$0")/lib.sh"
"$D" env INTARSIA_BUILD_STAMP="$STAMP" cargo build --release -p intarsia-cli --features hip < /dev/null; echo "BUILD cli exit $?"
