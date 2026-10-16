#!/bin/zsh
# One slot: cargo test -p intarsia-backend-metal --release (includes keepwarm.rs), then a single 3 s-idle serve spot check at 4k.
R=/opt/code/intarsiaml-residency/docs/results/raw/2026-10-16-metal-weight-residency
cd /opt/code/intarsiaml-residency
cargo test -p intarsia-backend-metal --release -- --test-threads=1 > $R/review-fix-cargo-test.txt 2>&1; echo "cargo test exit $?"
grep -h "test result" $R/review-fix-cargo-test.txt | awk '{p+=$4; f+=$6} END {print "total passed", p, "failed", f}'
NOPROF=1 IDLES="3" $R/serve-idle2.sh $R/review-fix-spot 1 rs
