#!/usr/bin/env bash
# Lane sp window 6, ON endor inside a fleetq job (--takes-down radiance): localeval speed-spec (natural-v1 128/1k/4k/16k, 35B),
# three arms on one server each: main (eager MTP k=2), branch (captured k=2), branch + --mtp-draft-vocab 65536; then the extras
# (27B Chinese/code, no-op ids). Radiance is restored by the trap whatever happens.
S=/opt/code/wt-sp/scripts/sp
trap 'ssh tatooine "bash ~/sp-win.sh restore" 2>&1 | tail -2' EXIT
bash $S/arm.sh sp-base main-mtp2 spec "--spec mtp --spec-k 2" "" 1
bash $S/arm.sh sp br-mtp2 spec "--spec mtp --spec-k 2" "" 1
bash $S/arm.sh sp br-mtp2-v65536 spec "--spec mtp --spec-k 2 --mtp-draft-vocab 65536" "" 1
ssh tatooine "bash ~/intarsiaml-sp/scripts/sp/extras.sh"
