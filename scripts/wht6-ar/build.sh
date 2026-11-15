#!/usr/bin/env bash
# Lane wht6-ar (ON tatooine, CPU only: `fleetq run --host tatooine --load cpu --mem 0`): the release CLI and the comm test binaries.
set -u
cd "$(dirname "$0")/../.." && . scripts/wht6-ar/lib.sh
D="$TREE/deploy/tatooine/dev.sh"
"$D" cargo build --release -p intarsia-cli --features hip < /dev/null; echo "BUILD cli exit $?"
"$D" cargo test --release -p intarsia-backend-hip --test ar_wht6 --test comm --test ar_push --test ar_norm --no-run < /dev/null; echo "BUILD tests exit $?"
