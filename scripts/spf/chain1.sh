#!/usr/bin/env bash
# Lane spf: queue the rope-fix evidence jobs one after another (each its own fleetq job, honest --expect).
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
RAW=docs/results/2026-10-27-spf-cuda-prefill/raw
fleetq run --host spark1 --quiet gpu,cpu --mem 45 --expect 8 --label "spf: 8B serve steady-state nsys, rope tree" -- scripts/spf/prof-serve.sh rope serve-rope4087 > $RAW/serve-rope4087.driver.log 2>&1
fleetq run --host spark1 --quiet gpu,cpu --mem 45 --expect 30 --label "spf: 8B prefill A/B base vs rope x3" -- scripts/spf/le-window.sh ab8b 3 base:base rope:rope > $RAW/ab8b.driver.log 2>&1
fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 12 --label "spf: plan show base vs rope" -- ssh spark1 'bash -s' < scripts/spf/plan.sh > $RAW/plan.log 2>&1
fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 30 --label "spf: ids 8B base vs rope vs rope0" -- ssh spark1 'MODELS="8" PROMPTS="short-0 4k-0 4k-1 16k-0" bash -s' < scripts/spf/ids.sh > $RAW/ids8.log 2>&1
fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 30 --label "spf: ids 35B base vs rope" -- ssh spark1 'MODELS="35" PROMPTS="short-0 4k-0 16k-0" ARMS="base:base rope:rope" bash -s' < scripts/spf/ids.sh > $RAW/ids35.log 2>&1
