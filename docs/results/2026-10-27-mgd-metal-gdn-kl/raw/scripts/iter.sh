#!/bin/bash
# build, gdn test, quick 4k profile; usage: iter.sh TAG
cd /opt/code/intarsiaml/.worktrees/mgd; TAG=$1
export CARGO_TARGET_DIR=$PWD/target
fleetq run --host endor --load cpu --expect 6 --label "mgd: build+test $TAG" -- bash -c 'cargo test --release -p intarsia-backend-metal --test gdn 2>&1 | grep -E "^error|test result|FAILED|panicked|differs"; cargo build --release -p intarsia-cli --features metal 2>&1 | grep -E "^error|Finished"' 
cp target/release/intarsia run/intarsia-$TAG
fleetq run --host endor --quiet gpu,cpu --mem 45 --expect 4 --label "mgd: profile $TAG" -- run/pq.sh $PWD/run/intarsia-$TAG $PWD/run/prof-$TAG > /dev/null 2>&1
python3 docs/results/2026-10-27-m35p-metal-35b-prefill-kl/raw/scripts/perop.py run/prof-$TAG/kernel-4k.jsonl | grep -E "gdn_delta|GPU sum"
grep -o "prefill [0-9]* ms ([0-9.]* tok/s)" run/prof-$TAG/warm.err | tail -1
