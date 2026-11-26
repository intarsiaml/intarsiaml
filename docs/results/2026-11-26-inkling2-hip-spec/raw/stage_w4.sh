#!/usr/bin/env bash
# Window D (lane ink-hip-spec), card 0: the hip tiny suite and topk on the PR head (binary spec4's test targets; #526's harness
# set_flash(0) / set_wmma(0), the cut default on hip).
set -uo pipefail
S=$HOME/intarsiaml-ink-hip-spec/.scratch
now() { date '+%F %T'; }
r() { bash "$S/run.sh" "$@"; }
echo "[stage] tiny inkling2 + verify_softcap on hip $(now)"
r env INTARSIA_INKLING2_BACKEND=hip cargo test --release -p intarsia-models --test inkling2 --test inkling2_verify_softcap -- --test-threads 1 2>&1 | grep -aE "^test |test result|panicked" | head -40
echo "[stage] tiny dspark on hip $(now)"
r env INTARSIA_INKLING2_BACKEND=hip cargo test --release -p intarsia-models --test inkling2_dspark --test inkling2_dspark_cut -- --test-threads 1 2>&1 | grep -aE "^test |test result|panicked" | head -20
echo "[stage] hip topk $(now)"
r env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test topk -- --test-threads 1 2>&1 | grep -aE "^test |test result|panicked" | head -12
echo "[stage] done $(now)"
