#!/usr/bin/env bash
# Lane wire-probe (ON tatooine, CPU only: `fleetq run --host tatooine --load cpu --mem 0`): the stamped release CLI and the comm test binaries.
set -u
cd "$(dirname "$0")/../.." && . scripts/wire-probe/lib.sh
"$TREE/deploy/tatooine/dev.sh" env INTARSIA_BUILD_STAMP="$STAMP" cargo build --release -p intarsia-cli --features hip < /dev/null; echo "BUILD cli exit $?"
"$TREE/deploy/tatooine/dev.sh" cargo test --release -p intarsia-backend-hip --test ar_wht6 --test comm --test ar_push --test ar_norm --no-run < /dev/null; echo "BUILD tests exit $?"
