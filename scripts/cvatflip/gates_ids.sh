#!/usr/bin/env bash
# cvatflip gate (a), run from endor under one fleetq slot: greedy ids knob unset (def, the new default) = knob 0 (zero) = `--spec off`, chat-short and code-gen; copy3k also with INTARSIA_SPEC_COPY=20 (defcopy / zerocopy).
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 45 --label "cvatflip ids gate" -- bash scripts/cvatflip/gates_ids.sh
D=$(dirname "${BASH_SOURCE[0]}")
ssh -n spark1 true
ssh spark1 'ARMS="off def zero" bash -s chat-short code-gen' < $D/ids.sh
ssh spark1 'ARMS="off def zero defcopy=20 zerocopy=20" bash -s copy3k' < $D/ids.sh
