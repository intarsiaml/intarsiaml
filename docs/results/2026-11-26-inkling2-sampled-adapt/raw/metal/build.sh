#!/usr/bin/env bash
# endor --load cpu (lane ink-sampled-adapt): Metal CLI binaries: the branch (target/) and main aee06201c (.scratch/main-src,
# .scratch/main-target), copied to .scratch/bin/intarsia-<name>; and the lane's test binaries (release) with --no-run.
set -uo pipefail
W=/opt/code/intarsiaml/.worktrees/ink-sampled-adapt; S=$W/.scratch; mkdir -p $S/bin
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
cd $W
echo "[build] branch cli $(date +%T) head $(git rev-parse --short HEAD)"
CARGO_TARGET_DIR=$W/target cargo build --release -p intarsia-cli --features metal 2>&1 | tail -1
cp $W/target/release/intarsia $S/bin/intarsia-${BR:-ad3} && md5 $S/bin/intarsia-${BR:-ad3}
if [ -z "${NOTESTS:-}" ]; then
  echo "[build] tests $(date +%T)"
  CARGO_TARGET_DIR=$W/target cargo test --release -p intarsia-models --test inkling2_sampled_real --test inkling2_dspark_sampled_adapt --no-run 2>&1 | tail -3
fi
if [ -z "${NOMAIN:-}" ]; then
  echo "[build] main cli $(date +%T)"
  cd $S/main-src && CARGO_TARGET_DIR=$S/main-target cargo build --release -p intarsia-cli --features metal 2>&1 | tail -1
  cp $S/main-target/release/intarsia $S/bin/intarsia-main6 && md5 $S/bin/intarsia-main6
fi
echo "[build] done $(date +%T)"
