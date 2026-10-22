#!/usr/bin/env bash
# Lane FX5, spark1 cpu side, small non-exclusive slot (the CPU backend needs no GPU):
#   fleetq run --host spark1 --mem 8 --expect 25m --label "fx5 cargo test: hybrid reuse test + restorable_len mutants (cpu)" -- scripts/fx5/test.sh
# Syncs the tree to ~/intarsia-decfuse/fx5-test (own CARGO_TARGET_DIR), then
#   1. hybrid.rs (whole file) and the intarsia-cli accuracy / qualify unit tests on the clean tree
#   2. the hybrid reuse test against mutants of SequenceState::restorable_len (each must FAIL), reverting after each
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"; cd "$REPO"
HOST=${HOST:-spark1}
RAW=${RAW:-$REPO/docs/results/2026-10-22-fx5-hybrid-reuse-ci/raw}
mkdir -p "$RAW"
R='~/intarsia-decfuse/fx5-test'
RPATH='export PATH=$HOME/.cargo/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
rsh "mkdir -p $R; test -d $R/target || { test -d ~/intarsia-decfuse/fx5/target/test && cp -a ~/intarsia-decfuse/fx5/target/test $R/target; true; }"
rsync -a --delete --include golden/ --include "golden/ref_noise/***" --exclude "golden/*" --exclude target --exclude .git --exclude tools/.venv --exclude .claude "$REPO/" "$HOST:intarsia-decfuse/fx5-test/"
T="cd $R && CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-8} CARGO_TARGET_DIR=$R/target cargo test"
echo "=== cargo check --workspace"
rsh "cd $R && CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-8} CARGO_TARGET_DIR=$R/target cargo check --workspace 2>&1" | tee "$RAW/test-check-workspace.log" | grep -a -e '^error' -e 'Finished' | cut -c1-200
echo "=== clean: hybrid.rs, recipes_doc.rs"
rsh "$T -p intarsia-models --test hybrid --test recipes_doc 2>&1" | tee "$RAW/test-hybrid-clean.log" | grep -a -e '^test ' -e 'test result' -e '^error' | cut -c1-200
echo "=== clean: intarsia-cli unit tests (accuracy, qualify)"
rsh "$T -p intarsia-cli -p intarsia-core 2>&1" | tee "$RAW/test-cli-clean.log" | grep -a -e 'test result' -e 'FAILED' -e '^error' | cut -c1-200
F=crates/intarsia-core/src/state.rs
rsh "cp $R/$F $R/$F.orig"
mutant() { # name, sed expression
  echo "=== mutant $1"
  rsh "cp $R/$F.orig $R/$F && sed -i '$2' $R/$F && diff $R/$F.orig $R/$F | head -6"
  rsh "$T -p intarsia-models --test hybrid 2>&1" > "$RAW/test-hybrid-mutant-$1.log"
  grep -a -e '^test ' -e 'test result' -e 'panicked' -e '^error' "$RAW/test-hybrid-mutant-$1.log" | cut -c1-220
}
mutant le-plus3 's/filter(|&l| l <= want)/filter(|\&l| l <= want + 3)/'
mutant none-returns-want 's/\.max()\.unwrap_or(0)/.max().unwrap_or(want)/'
mutant no-snapshot-lookup 's/self\.snapshots\.iter()\.map(|s| s\.len)\.filter(|&l| l <= want)\.max()\.unwrap_or(0)/self.snapshots.iter().map(|s| s.len).max().unwrap_or(0).min(want)/'
# silent-corruption mutants: tokens of a random tiny model may survive them, logits do not
mutant restore-first-slot-only '228s/off += r.bytes;/off += r.bytes; break;/'
mutant skip-zero-on-reset '214s/backend.zero(&r.buf, 0)?;/let _ = \&r.buf;/'
rsh "cp $R/$F.orig $R/$F && diff $R/$F.orig $R/$F && echo reverted"
echo "=== clean again after revert"
rsh "$T -p intarsia-models --test hybrid 2>&1" | tee "$RAW/test-hybrid-reverted.log" | grep -a -e 'test result' -e '^error' | cut -c1-200
