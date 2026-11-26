#!/usr/bin/env bash
# ON tatooine inside a fleetq --load cpu job (lane ink-hip-bpf): main's (80719483a, .scratch/main-src = git archive) intarsia-models
# test binaries, in their own target volume, for main's baseline of the tiny hip suite.
set -uo pipefail
R=$HOME/intarsiaml-ink-hip-bpf/.scratch/main-src
cd "$R"
export INTARSIA_REPO=$R INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-bpf-main INTARSIA_CONTAINER_NAME=ink-hip-bpf-build-main
trap 'docker rm -f ink-hip-bpf-build-main >/dev/null 2>&1' EXIT
mkdir -p "$HOME/intarsiaml-ink-hip-bpf/logs"
for t in inkling2 inkling2_batch; do
  deploy/tatooine/dev.sh cargo test --release -j8 -p intarsia-models --test $t --no-run < /dev/null > "$HOME/intarsiaml-ink-hip-bpf/logs/build-main-test-$t.log" 2>&1
  rc=$?; tail -2 "$HOME/intarsiaml-ink-hip-bpf/logs/build-main-test-$t.log"; echo "main test $t build exit $rc"; [ $rc -ne 0 ] && exit $rc
done
exit 0
