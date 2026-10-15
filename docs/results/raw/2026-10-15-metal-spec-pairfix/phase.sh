#!/bin/zsh
R=/opt/code/intarsiaml-spec/docs/results/raw/2026-10-15-metal-spec-pairfix
NEW=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/bin/intarsia-pairfix
cd /opt/code/intarsiaml-spec
cargo test -p intarsia-backend-metal --release > $R/cargo-test-metal.log 2>&1; echo "cargo test metal exit $?"
$NEW kernels test --backend metal -m $HOME/models/Qwen3-8B > $R/kernels-test-metal.log 2>&1; echo "kernels test exit $?"
$R/run_identity.sh $R/identity $R/cases.txt off on4 on6 > $R/identity.log 2>&1; echo "identity exit $?"
