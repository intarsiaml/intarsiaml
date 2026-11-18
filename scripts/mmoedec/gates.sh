#!/bin/zsh
# mmoedec: the gates, each in its own fleetq slot (run from the worktree root, in the background):
#   plan show, ids (GDN fuse unset, then set), docs_env_vars, the whole Metal crate.
cd ${0:A:h:h:h}; R=docs/results/2026-11-18-mmoedec/raw; CT=$PWD/target-mmoedec
Q="--host endor --quiet gpu,cpu"
fleetq run ${=Q} --mem 40 --expect 6 --label "mmoedec: plan show" -- scripts/mmoedec/plan.sh > $R/plan-gate.txt 2>&1
fleetq run ${=Q} --mem 40 --expect 25 --label "mmoedec: ids gdn unset" -- env O=$PWD/$R/ids-gdnoff NTOK=500 ARMS="off mtp mtp+on" scripts/mmoedec/ids.sh chat-short code-gen reason copy3k > $R/ids-gate-gdnoff.txt 2>&1
fleetq run ${=Q} --mem 40 --expect 25 --label "mmoedec: ids gdn set" -- env O=$PWD/$R/ids-gdnon NTOK=500 XE=INTARSIA_METAL_GDN_VERIFY_FUSE=1 ARMS="off mtp mtp+on" scripts/mmoedec/ids.sh chat-short code-gen reason copy3k > $R/ids-gate-gdnon.txt 2>&1
fleetq run --host endor --load cpu --mem 8 --expect 8 --label "mmoedec: docs_env_vars" -- sh -c "CARGO_TARGET_DIR=$CT cargo test --release -p intarsia-cli --test docs_env_vars" > $R/docs-env-vars.txt 2>&1
fleetq run ${=Q} --mem 12 --expect 25 --label "mmoedec: metal crate tests" -- sh -c "CARGO_TARGET_DIR=$CT cargo test --release -p intarsia-backend-metal -- --test-threads 1" > $R/metal-crate-tests.txt 2>&1
echo GATES_DONE >> $R/metal-crate-tests.txt
