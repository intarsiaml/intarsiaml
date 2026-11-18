#!/bin/zsh
# mgdnv27 driver, second pass: BATCHED_MTP = concurrency, so drafting (batched verify) really runs. Each piece in its own fleetq slot.
R=docs/results/2026-11-18-mgdnv27/raw
for r in 1 2 3; do fleetq run --host endor --quiet gpu,cpu --mem 50 --expect 12 --label "mgdnv27: 35B mb4 bmtp4 round $r" -- env MB=4 N=4 R=$r scripts/mgdnv27/batch.sh > $R/batch-mb4-bmtp4-r$r.log 2>&1; done
for r in 1 2 3; do fleetq run --host endor --quiet gpu,cpu --mem 50 --expect 12 --label "mgdnv27: 35B mb16 bmtp8 round $r" -- env MB=16 N=8 R=$r scripts/mgdnv27/batch.sh > $R/batch-mb16-bmtp8-r$r.log 2>&1; done
fleetq run --host endor --quiet gpu,cpu --mem 50 --expect 25 --label "mgdnv27: 35B sampled ids" -- env MODEL=$HOME/models/Qwen3.6-35B-A3B-NVFP4 SAMPLED=1 SEED=42 ARMS="def def+on" TAGP=s42 O=$R/sampled scripts/mgdnv27/ids.sh chat-short code-gen reason json-xform > $R/sampled-run.log 2>&1
fleetq run --host endor --quiet gpu,cpu --mem 50 --expect 25 --label "mgdnv27: 35B sampled ids seed2" -- env MODEL=$HOME/models/Qwen3.6-35B-A3B-NVFP4 SAMPLED=1 SEED=7 ARMS="def def+on" TAGP=s7 O=$R/sampled scripts/mgdnv27/ids.sh chat-short code-gen reason copy3k > $R/sampled2-run.log 2>&1
fleetq run --host endor --quiet gpu,cpu --mem 50 --expect 12 --label "mgdnv27: 35B mb16 sampled" -- env MB=16 N=8 R=1 SAMPLED=1 scripts/mgdnv27/batch.sh > $R/batch-sampled-mb16.log 2>&1
echo ALLDONE > $R/alldone
