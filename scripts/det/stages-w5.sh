#!/usr/bin/env bash
# Lane det w5 (ON tatooine, radiance down): fresh-process rate of the early divergence, cheaply. Every event so far was in
# the first ~30 generated tokens of a fresh process, so: 48 tokens per process, all processes of an arm in ONE dev.sh container
# (a shell loop, no docker run per process). Arms: tp=2 x150, tp=2 with the capture and peer syncs x150, then tp=1 on card 0 and on card 1 at the same time x150 each.
set -uo pipefail
export O=${1:?out dir}
source "$(dirname "$0")/lib.sh"
clear w5 || exit 2
loop tp2 0,1 150 --tp 2 --devices 0,1 --spec off
# the peer rank is never host-synced, so its first capture can start while its prefill is queued: sync before capture, after a peer step
E="INTARSIA_GRAPH_SYNC_CAPTURE=1 INTARSIA_TP_PEER_SYNC=1" loop tp2sync 0,1 150 --tp 2 --devices 0,1 --spec off
loop tp1c0 0 150 --spec off &
loop tp1c1 1 150 --spec off &
wait
log "w5 done"
