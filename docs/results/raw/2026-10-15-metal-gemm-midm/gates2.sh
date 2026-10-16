#!/bin/zsh
# final tile-table gates: cargo test (metal crate), then the engine A/B of the mid table at held-out M, one slot
cd /opt/code/intarsiaml-gemm-midm
D=docs/results/raw/2026-10-15-metal-gemm-midm
mkdir -p $D/gates2
cargo test -p intarsia-backend-metal --release -- --test-threads=1 > $D/gates2/cargo-test-metal.txt 2>&1; echo "cargo test metal exit $?"
$D/engine-ab4.sh 2 1000 585 663 745 830 930 1075 1130 1230 1330 1400 1470 1580 1650 1760 1900 2048 1024 640
