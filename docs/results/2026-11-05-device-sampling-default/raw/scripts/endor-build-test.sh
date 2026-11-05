#!/bin/zsh
# Lane lane-ds, endor, inside a fleetq --load cpu slot: Metal binaries of origin/main (602a1fcfe) and of this branch, then cargo test --workspace --no-fail-fast.
[ -n "${FLEETQ_JOB:-}" ] || exit 2
cd /opt/code/intarsiaml/.worktrees/device-sampling || exit 2
S=$PWD/.scratch; mkdir -p $S/logs
rm -rf $S/main-src; mkdir -p $S/main-src
git archive 602a1fcfe | tar -x -C $S/main-src
echo "== build main"; (cd $S/main-src && CARGO_TARGET_DIR=$S/target-metal-main cargo build --release --locked -p intarsia-cli --features metal 2>&1 | tail -3)
cp $S/target-metal-main/release/intarsia $S/intarsia.main
echo "== build branch"; CARGO_TARGET_DIR=$S/target-metal-ds cargo build --release --locked -p intarsia-cli --features metal 2>&1 | tail -3
cp $S/target-metal-ds/release/intarsia $S/intarsia.ds
shasum -a 256 $S/intarsia.main $S/intarsia.ds
echo "== cargo test --workspace --no-fail-fast"
CARGO_TARGET_DIR=$S/target-test cargo test --workspace --no-fail-fast > $S/logs/cargo-test-workspace.txt 2>&1
echo "cargo test exit $?"
grep -E "^test result|FAILED|failed" $S/logs/cargo-test-workspace.txt | awk '/^test result/{p+=$4; f+=$6; i+=$8} /FAILED|failed/ && !/^test result/{print} END{print "passed",p,"failed",f,"ignored",i}' | tail -15
