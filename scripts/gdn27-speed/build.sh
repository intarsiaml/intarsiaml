#!/usr/bin/env bash
# Lane gdn27-speed (ON tatooine, CPU only): the stamped release CLI.
set -u
. "$(dirname "$0")/lib.sh"
"$D" env INTARSIA_BUILD_STAMP="$STAMP" cargo build --release -p intarsia-cli --features hip < /dev/null; echo "BUILD cli exit $?"
