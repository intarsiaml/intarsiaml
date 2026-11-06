#!/usr/bin/env bash
# Lane lane-ds: is the rows_exact SIGSEGV (seen in the spark2 workspace test, #8655) caused by this branch, by GPU contention, or pre-existing?
# usage (inside a fleetq --quiet gpu,cpu slot on the box): rows-exact.sh <host> <branch-sha> <main-sha>
# Builds intarsia-backend-cuda's rows_exact test from `git archive` of both commits into ~/intarsiaml-ds-rx/{branch,main} (own target dirs), runs it twice each
# with the default harness and twice each with --test-threads=1, RUST_BACKTRACE=1; on a crash, gdb's backtrace when gdb exists. Removes the trees at the end.
set -uo pipefail
H=$1; BR=$2; MN=$3
WT=/opt/code/intarsiaml/.worktrees/device-sampling; OUT=$WT/docs/results/2026-11-05-device-sampling-default/raw/rows-exact-$H; mkdir -p "$OUT"
rs() { ssh -n -o BatchMode=yes "$H" "$@"; }
trap 'rs "rm -rf ~/intarsiaml-ds-rx"' EXIT
rs 'rm -rf ~/intarsiaml-ds-rx; mkdir -p ~/intarsiaml-ds-rx/branch ~/intarsiaml-ds-rx/main'
(cd $WT && git archive $BR) | ssh -o BatchMode=yes $H 'tar -x -C ~/intarsiaml-ds-rx/branch'
(cd $WT && git archive $MN) | ssh -o BatchMode=yes $H 'tar -x -C ~/intarsiaml-ds-rx/main'
rs 'free -g | sed -n 2p; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; which gdb' | tee "$OUT/box.txt"
for t in branch main; do
  echo "== build $t $(date +%T)"
  rs "cd ~/intarsiaml-ds-rx/$t && CARGO_BUILD_JOBS=8 ~/.cargo/bin/cargo test -p intarsia-backend-cuda --test rows_exact --no-run 2>&1 | tail -3" | tee "$OUT/build-$t.txt"
done
run() { # run <tree> <label> <extra harness args>
  local t=$1 lab=$2; shift 2
  rs "cd ~/intarsiaml-ds-rx/$t && RUST_BACKTRACE=1 ~/.cargo/bin/cargo test -p intarsia-backend-cuda --test rows_exact -- $* > ~/intarsiaml-ds-rx/$t.$lab.log 2>&1; echo \$?" > "$OUT/$t.$lab.exit"
  rs "cat ~/intarsiaml-ds-rx/$t.$lab.log" > "$OUT/$t.$lab.log"
  echo "$t $lab: exit $(cat $OUT/$t.$lab.exit) | $(grep -a 'test result' $OUT/$t.$lab.log | tail -1) | last test line: $(grep -a '^test ' $OUT/$t.$lab.log | tail -1)"
}
for r in 1 2; do for t in main branch; do run $t default-$r; done; done
for r in 1 2; do for t in main branch; do run $t single-$r --test-threads=1; done; done
# a crash: backtrace with gdb on the test binary
for t in main branch; do
  if grep -aq "SIGSEGV\|signal: 11" "$OUT"/$t.*.log 2>/dev/null; then
    bin=$(rs "ls -t ~/intarsiaml-ds-rx/$t/target/debug/deps/rows_exact-* | grep -v '\\.d$' | head -1")
    echo "== gdb $t $bin"
    rs "cd ~/intarsiaml-ds-rx/$t/crates/intarsia-backend-cuda && gdb -batch -ex run -ex bt --args $bin --test-threads=1 2>&1 | tail -40" | tee "$OUT/gdb-$t.txt"
  fi
done
