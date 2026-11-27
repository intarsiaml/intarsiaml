#!/usr/bin/env bash
# endor (Mac) cpu tests of the merged head (lane ink-hip-overlap), inside a fleetq --load cpu slot. Logs: .scratch/cpu-endor-<group>.log
set -uo pipefail
cd /opt/code/intarsiaml/.worktrees/ink-hip-overlap
L=.scratch
rc_all=0
run() {  # name cmd...
  local g=$1; shift
  env -u INTARSIA_INKLING2_BACKEND "$@" > "$L/cpu-endor-$g.log" 2>&1
  local rc=$?
  echo "== $g exit $rc ($(date '+%T'))"
  grep -E '^test result|FAILED|panicked|^error' "$L/cpu-endor-$g.log" | head -20
  [ $rc -ne 0 ] && rc_all=$rc
}
echo "head $(git rev-parse --short HEAD) start $(date '+%T')"
run core cargo test --release -p intarsia-core --lib
run sched cargo test --release -p intarsia-core --test scheduler
run models cargo test --release -p intarsia-models --test inkling2 --test inkling2_batch --test inkling2_packed --test batch_recurrent --test prefix_store --test inkling2_dspark_sampled --test inkling2_dspark_sampled_gumbel
run docs cargo test --release -p intarsia-cli --test docs_env_vars
run server cargo test --release -p intarsia-server --lib
echo "all exit $rc_all ($(date '+%T'))"
exit $rc_all
