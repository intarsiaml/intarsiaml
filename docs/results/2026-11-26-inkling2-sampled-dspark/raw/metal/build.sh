#!/usr/bin/env bash
# endor --load cpu: lane ink-sampled Metal CLI binaries: the branch (target/) and main (.scratch/main-src, .scratch/main-target),
# copied to .scratch/bin/intarsia-<name>. Also the lane's test binaries (release, cpu + Metal backends) with --no-run.
set -uo pipefail
W=/opt/code/intarsiaml/.worktrees/ink-sampled; S=$W/.scratch; mkdir -p $S/bin
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
cd $W
echo "[build] branch cli $(date +%T)"
CARGO_TARGET_DIR=$W/target cargo build --release -p intarsia-cli --features metal 2>&1 | tail -2
cp $W/target/release/intarsia $S/bin/intarsia-${BR:-br1} && md5 $S/bin/intarsia-${BR:-br1}
if [ -z "${NOTESTS:-}" ]; then
  echo "[build] tests $(date +%T)"
  CARGO_TARGET_DIR=$W/target cargo test --release -p intarsia-models --test inkling2 --test inkling2_dspark --test inkling2_dspark_cut --test inkling2_verify_softcap --test inkling2_dspark_sampled --test inkling2_dspark_sampled_gumbel --test inkling2_sampled_real --no-run 2>&1 | tail -3
fi
if [ -z "${NOMAIN:-}" ]; then
  echo "[build] main cli $(date +%T)"
  cd $S/main-src && CARGO_TARGET_DIR=$S/main-target cargo build --release -p intarsia-cli --features metal 2>&1 | tail -2
  cp $S/main-target/release/intarsia $S/bin/intarsia-main && md5 $S/bin/intarsia-main
fi
echo "[build] done $(date +%T)"
