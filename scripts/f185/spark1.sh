#!/usr/bin/env bash
# Lane f185 CPU-slot build ON spark1 (everything under ~/intarsiaml-f185): src = the synced dev tree, target-dev seeded from lane mpf's if it is still there (registry crates reused).
#   fleetq run --host spark1 --load cpu --expect 30 --label "f185 build" -- ssh spark1 'bash -s' < scripts/f185/spark1.sh
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-f185
mkdir -p $R/out
[ -d $R/target-dev ] || { [ -d $HOME/intarsiaml-mpf/target-dev ] && cp -a $HOME/intarsiaml-mpf/target-dev $R/target-dev; }
rm -rf $R/src; cp -a $R/src-dev $R/src
(cd $R/src && CARGO_TARGET_DIR=$R/target-dev CARGO_BUILD_JOBS=${JOBS:-12} cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3)
echo "build dev done $(date +%T) $(md5sum $R/target-dev/release/intarsia | cut -c1-12)"
