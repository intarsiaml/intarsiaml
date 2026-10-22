#!/bin/zsh
# so lane: build two Metal binaries in parallel inside ONE non-exclusive fleetq slot (separate target dirs):
#   base = origin/main (wt-so-base), head = branch tip (timers off unless INTARSIA_CONSTRAIN_PROFILE=1).
# fleetq run --host endor --mem 40 --expect 25 --label "so: build base/head metal" -- zsh build.sh <out dir>
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
OUT=${1:?out dir}
mkdir -p $OUT
cd /opt/code/wt-so
git -C /opt/code/wt-so rev-parse HEAD > $OUT/head-commit.txt
git -C /opt/code/wt-so-base rev-parse HEAD > $OUT/base-commit.txt
b() { # dir
  (cd $1 && CARGO_TARGET_DIR=$1/target-metal cargo build --release -p intarsia-cli --features metal > $OUT/build-${1:t}.log 2>&1; echo "exit $?" >> $OUT/build-${1:t}.log)
}
b /opt/code/wt-so-base & p1=$!
b /opt/code/wt-so & p3=$!
wait $p1 $p3
for d in wt-so-base wt-so; do
  f=/opt/code/$d/target-metal/release/intarsia
  echo "$d $(md5 -q $f 2>/dev/null) $(tail -1 $OUT/build-$d.log)" | tee -a $OUT/builds.txt
done
