#!/usr/bin/env bash
# One A/B window on spark1 (inside scripts/prefill-cuda-window.sh, vLLM down): optionally the kernel tests of the build under
# test first (cheap, no model), then the "ours" arms of one round.
#   ab-window.sh <round> <with-tests 0|1> <arm:bin:flags> ...
set -u
P=$HOME/intarsia-perf
RND=$1; WT=$2; shift 2
if [ "$WT" = 1 ]; then
  echo "== kernel tests of the build under test $(date +%T)"
  cd $P/dev2 && CARGO_TARGET_DIR=target/acc cargo test --release -p intarsia-backend-cuda --no-fail-fast --test gdn_lanes --test quantmoe --test admission -- --nocapture --test-threads=1 > $P/out/ab-tests.log 2>&1
  echo "exit $?"; grep -E "^test .*(ok|FAILED)$|test result|panicked" $P/out/ab-tests.log | cut -c1-170
fi
bash $P/scripts/ab-ours.sh $RND "$@"
