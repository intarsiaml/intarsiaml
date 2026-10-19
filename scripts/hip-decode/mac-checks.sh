#!/usr/bin/env bash
# The Mac checks of the core-op fusion rounds (docs/results/2026-10-19-core-op-fusion-p2.md): workspace tests (the CUDA and
# Metal pin tests among them), workspace check, the CLI with the hip and metal features, the Metal backend build.
#   scripts/hip-decode/mac-checks.sh > docs/results/<round>/raw/logs/mac-checks.log 2>&1
set -uo pipefail
cd "$(dirname "$0")/../.."
echo "HEAD $(git rev-parse --short HEAD)$(git diff --quiet HEAD || echo ' (dirty)')"
cargo test --workspace > /tmp/mac-checks-test.$$ 2>&1; rc=$?
grep -E "test result" /tmp/mac-checks-test.$$ | awk '{p+=$4; f+=$6} END {print "cargo test --workspace: " NR " binaries, passed " p ", failed " f}'
grep -E "FAILED|panicked" /tmp/mac-checks-test.$$ | head -10
echo "cargo test exit $rc"
rm -f /tmp/mac-checks-test.$$
for c in "check --workspace" "check -p intarsia-cli --features hip" "check -p intarsia-cli --features metal" "build -p intarsia-backend-metal"; do
  out=$(cargo $c 2>&1); rc=$?
  echo "cargo $c: exit $rc $(echo "$out" | grep -E '^error' | head -3) $(echo "$out" | tail -1)"
done
