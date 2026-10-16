#!/bin/zsh
# Phase C (inside one slot): metal crate tests after the selection-test fix, the G0 selection test on the merge base
# (ca18d27, before this branch's changes), and the resolved plan of the base / spec-off / spec-on binaries.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
[ -d "$DEVELOPER_DIR" ] || { echo "DEVELOPER_DIR missing" >&2; exit 3; }
R=/opt/code/intarsiaml-spec/docs/results/raw/2026-10-15-metal-spec-g1
M=$HOME/models/Qwen3-8B
cd /opt/code/intarsiaml-spec
cargo test -p intarsia-backend-metal --release > $R/cargo-test-metal-final.log 2>&1; echo "cargo test metal exit $?"
(cd /private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/mergebase && cargo test --release -p intarsia-backend-metal --test multirow serving_selection_unchanged > $R/mergebase-ca18d27-serving-selection.log 2>&1; echo "merge-base selection test exit $?")
mkdir -p $R/plan
/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/bin/intarsia-base-c556008 plan show --model $M --backend metal --max-seq 8192 > $R/plan/base.txt 2>&1; echo "plan base $?"
./target/release/intarsia plan show --model $M --backend metal --max-seq 8192 --spec off > $R/plan/off.txt 2>&1; echo "plan off $?"
./target/release/intarsia plan show --model $M --backend metal --max-seq 8192 --spec ngram > $R/plan/on4.txt 2>&1; echo "plan on4 $?"
