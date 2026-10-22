#!/usr/bin/env bash
# Lane FX5c, spark1 cpu side, small non-exclusive slot: unit tests of the oracle-relative continue gate and the hybrid tests.
#   fleetq run --host spark1 --mem 8 --expect 25 --label "fx5c cargo test intarsia-cli + hybrid (cpu)" -- scripts/fx5c/test.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
HOST=${HOST:-spark1}
RAW=${RAW:-$REPO/docs/results/2026-10-22-fx5c-oracle-gate/raw}
mkdir -p "$RAW"
R='~/intarsia-decfuse/fx5c-test'
RPATH='export PATH=$HOME/.cargo/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
rsh "mkdir -p $R; test -d $R/target || { test -d ~/intarsia-decfuse/fx5-test/target && cp -a ~/intarsia-decfuse/fx5-test/target $R/target; true; }"
rsync -a --delete --include golden/ --include "golden/ref_noise/***" --exclude "golden/*" --exclude target --exclude .git --exclude tools/.venv --exclude .claude "$REPO/" "$HOST:intarsia-decfuse/fx5c-test/"
T="cd $R && CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-8} CARGO_TARGET_DIR=$R/target cargo test"
echo "=== cargo check --workspace"
rsh "cd $R && CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-8} CARGO_TARGET_DIR=$R/target cargo check --workspace 2>&1" | tee "$RAW/test-check-workspace.log" | grep -a -e '^error' -e 'Finished' | cut -c1-200
echo "=== intarsia-cli (unit + docs_env_vars), intarsia-core"
rsh "$T -p intarsia-cli -p intarsia-core 2>&1" | tee "$RAW/test-cli-core.log" | grep -a -e 'test result' -e 'FAILED' -e '^error' -e 'oracle_relative\|continue_oracle\|kl_oracle' | cut -c1-200
echo "=== hybrid.rs, recipes_doc.rs"
rsh "$T -p intarsia-models --test hybrid --test recipes_doc 2>&1" | tee "$RAW/test-hybrid.log" | grep -a -e '^test ' -e 'test result' -e '^error' | cut -c1-200
