#!/bin/zsh
# mgdnvsoak: shared helpers; source inside a fleetq slot on endor. Same shape as scripts/mgdnv/lib.sh (this lane's own paths).
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mgdnvsoak; S=$W/scripts/mgdnvsoak; RAW=$W/docs/results/2026-11-18-mgdnvsoak/raw
NEW_BIN=$W/target-mgdnvsoak/intarsia-new; CELLS_DIR=$W/target-mgdnvsoak/cells
M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4; M27=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4
PORT=${PORT:-18797}; URL=http://127.0.0.1:$PORT
th() { notifyutil -g com.apple.system.thermalpressurelevel 2>&1 | awk '{print $2}'; }
cool() { local w=0; while [ "$(th)" != 0 ] && [ $w -lt ${COOL:-120} ]; do sleep 5; w=$((w+5)); done; echo "cooled ${w}s thermal $(th)"; }
mkdir -p $CELLS_DIR; [ -f $CELLS_DIR/copy3k.json ] || python3 $W/scripts/mcopy/prompts.py $CELLS_DIR > /dev/null
