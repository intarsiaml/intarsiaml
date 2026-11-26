#!/usr/bin/env bash
# Window E (lane ink-hip-spec), card 0: inkling2_verify_softcap on hip alone (in windows 1 / D cargo stopped after the inkling2 binary's
# two by-design screened-head failures, so this binary never ran).
set -uo pipefail
S=$HOME/intarsiaml-ink-hip-spec/.scratch
echo "[stage] verify_softcap on hip $(date '+%F %T')"
bash "$S/run.sh" env INTARSIA_INKLING2_BACKEND=hip cargo test --release -p intarsia-models --test inkling2_verify_softcap -- --test-threads 1 2>&1 | grep -aE "^test |test result|panicked|Running" | head -8
echo "[stage] done $(date '+%F %T')"
