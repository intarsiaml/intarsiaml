#!/bin/bash
# lane ink-decroof job B (--load cpu): build base (d8c3c7148), cand (branch tip), diag (cand + uncommitted arm patch), each with
# its own target dir; compile the Metal / models test binaries in cand; docs_env_vars (cpu). Results line per step.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
kill_tree() { local c; for c in $(pgrep -P $1 2>/dev/null); do kill_tree $c; done; kill $1 2>/dev/null; }
cleanup() { for c in $(pgrep -P $$ 2>/dev/null); do kill_tree $c; done; }
trap cleanup EXIT; trap 'echo "CUT (TERM) at step ${STEP:-?}"; cleanup; exit 143' INT TERM
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
W=/opt/code/intarsiaml/.worktrees/ink-decroof; S=$W/.scratch; O=$S/B; mkdir -p $O
echo "jobB start $(date +%T)"
b() { # tag dir
  STEP="build $1"; local t0=$(date +%s)
  (cd $2 && CARGO_TARGET_DIR=$2/target cargo build --release -p intarsia-cli --features metal) > $O/build_$1.log 2>&1
  local rc=$?
  echo "== build $1 exit $rc $(( $(date +%s) - t0 )) s tree $(git -C $2 rev-parse --short HEAD) dirty $(git -C $2 status --porcelain --untracked-files=no | wc -l | tr -d ' ') md5 $(md5 -q $2/target/release/intarsia 2>/dev/null)"
  [ $rc = 0 ] && { mkdir -p $S/bin; cp $2/target/release/intarsia $S/bin/intarsia-$1; } || tail -25 $O/build_$1.log
}
b base $S/base
b cand $W
b diag $S/diag
STEP="test build"; t0=$(date +%s)
(cd $W && cargo test --release --no-run -p intarsia-backend-metal --test inkling2_attn --test rows_exact --test inkmix --test inkctrl --test inkload \
  && cargo test --release --no-run -p intarsia-models --test inkling2 --test inkling2_batch) > $O/test_build.log 2>&1
echo "== test build exit $? $(( $(date +%s) - t0 )) s"; grep -E "^error" $O/test_build.log | head -5
STEP="docs_env_vars"
(cd $W && cargo test --release -p intarsia-cli --test docs_env_vars) > $O/docs_env_vars.log 2>&1
echo "== docs_env_vars exit $?: $(grep -h 'test result' $O/docs_env_vars.log)"
echo "binaries: $(md5 $S/bin/* | tr '\n' ' ')"
echo "jobB done $(date +%T)"
