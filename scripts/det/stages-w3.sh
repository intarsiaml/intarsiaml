#!/usr/bin/env bash
# Lane det w3 (ON tatooine, radiance down): the server-mode misses (vat w3: ho-chat chat-sleep differed from off in 1 of 5 fresh dflash
# rounds per arm; vrc: ho-recall recall-tls in one old-path dflash round) with generate: 512 forced greedy tokens as localeval asks
# (ignore_eos, so the on-device argmax), the same prompt several times per process and in several processes, per arm.
set -uo pipefail
export O=${1:?out dir}
source "$(dirname "$0")/lib.sh"
clear w3 || exit 2
TP2="--tp 2 --devices 0,1"
DFL="--spec dflash --spec-draft $DRAFT --spec-k 7"
export MT=512
for i in 1 2; do g off-sleep$i 0,1 chat-sleep 3 $TP2 --spec off; done
for i in 1 2 3 4 5 6; do E="INTARSIA_SPEC_TRACE=1" g dfl-sleep$i 0,1 chat-sleep 3 $TP2 $DFL; done
for i in 1 2; do g mtp-sleep$i 0,1 chat-sleep 3 $TP2 --spec mtp --spec-k 2; done
g off-tls1 0,1 recall-tls 3 $TP2 --spec off
for i in 1 2 3 4; do E="INTARSIA_SPEC_TRACE=1" g dfl-tls$i 0,1 recall-tls 3 $TP2 $DFL; done
for i in 1 2; do TAP=1 g dfltap-sleep$i 0,1 chat-sleep 3 $TP2 $DFL; done
log "w3 done"
