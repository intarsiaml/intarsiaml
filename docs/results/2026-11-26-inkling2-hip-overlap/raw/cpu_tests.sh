#!/usr/bin/env bash
# ON tatooine inside a fleetq --load cpu job (lane ink-hip-overlap, in ink-hip-bpf's directory): the cpu-only checks, no GPU (INTARSIA_INKLING2_BACKEND unset =
# the cpu reference): core lib tests (scheduler, registry, ops), the HIP crate's lib tests (registration only, no device), and
# the tiny-fixture model tests on cpu. Args: which groups (core hiplib models); default all. Logs: logs/cpu-<group>.log.
set -uo pipefail
cd "$HOME/intarsiaml-ink-hip-bpf"
export INTARSIA_REPO=$PWD INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-bpf INTARSIA_CONTAINER_NAME=ink-hip-bpf-build
trap 'docker rm -f ink-hip-bpf-build >/dev/null 2>&1' EXIT
mkdir -p logs
groups=${*:-core hiplib models}
rc_all=0
for g in $groups; do
  case $g in
    core) cmd="cargo test --release -j8 -p intarsia-core --lib -- --test-threads 8" ;;
    hiplib) cmd="cargo test --release -j8 -p intarsia-backend-hip --lib -- inkling2 segment" ;;
    cpulib) cmd="cargo test --release -j8 -p intarsia-backend-cpu" ;;
    models) cmd="cargo test --release -j8 -p intarsia-models --test inkling2_packed --test inkling2_batch --test inkling2" ;;
    sched) cmd="cargo test --release -j8 -p intarsia-core --test scheduler" ;;
    packed) cmd="cargo test --release -j8 -p intarsia-models --test inkling2_packed -- --nocapture" ;;
    docs) cmd="cargo test --release -j8 -p intarsia-cli --test docs_env_vars" ;;
    server) cmd="cargo test --release -j8 -p intarsia-server --lib" ;;
    *) cmd="cargo test --release -j8 -p intarsia-models --test $g" ;;
  esac
  deploy/tatooine/dev.sh bash -c "env -u INTARSIA_INKLING2_BACKEND $cmd" < /dev/null > logs/cpu-$g.log 2>&1
  rc=$?
  echo "== $g exit $rc"
  grep -E '^test result|FAILED|panicked|^error' logs/cpu-$g.log | head -20
  [ $rc -ne 0 ] && rc_all=$rc
done
exit $rc_all
