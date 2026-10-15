#!/bin/zsh
# attention bench again with the NAX flash candidates (the registry's actual AttnPrefill pick for T = 2..8)
set -u
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot"; exit 4; }
cd /opt/code/intarsiaml-g0b
O=docs/results/raw/2026-10-15-metal-g0b-attn/nax
mkdir -p $O
/Users/josef/bin/macmon pipe -i 100 > $O/macmon.jsonl 2>/dev/null &
MM=$!
sleep 2
G0B_OUT=$O/g0b.csv G0B_ROUNDS=30 ./target/release/examples/g0b_bench attn > $O/g0b.log 2>&1
echo "bench exit $?" | tee $O/slot.log
kill $MM 2>/dev/null
