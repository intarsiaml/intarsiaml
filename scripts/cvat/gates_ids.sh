#!/usr/bin/env bash
# cvat gate (a), run from endor under one fleetq slot: greedy ids knob on (vat) = knob off (mtp) = `--spec off`, chat-short and code-gen; copy3k also with INTARSIA_SPEC_COPY=20 (copy / vatcopy).
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 45 --label "cvat ids gate" -- bash scripts/cvat/gates_ids.sh
D=$(dirname "${BASH_SOURCE[0]}")
ssh -n spark1 true
ssh spark1 'ARMS="off mtp vat" bash -s chat-short code-gen' < $D/ids.sh
ssh spark1 'ARMS="off mtp vat copy=20 vatcopy=20" bash -s copy3k' < $D/ids.sh
