#!/bin/zsh
# mmoedec (Metal, Qwen3.6-35B-A3B NVFP4 on endor): shared helpers; source inside a fleetq slot on endor.
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mmoedec; S=$W/scripts/mmoedec; RAW=$W/docs/results/2026-11-18-mmoedec/raw
# the base binary lives in its OWN target dir (cargo hashes workspace paths: a shared dir let one lane clobber its base)
BASE_BIN=$W/target-mmoedec-base/intarsia-base; NEW_BIN=$W/target-mmoedec/intarsia-new; CELLS_DIR=$W/target-mmoedec/cells
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4
PORT=${PORT:-18716}; URL=http://127.0.0.1:$PORT
th() { notifyutil -g com.apple.system.thermalpressurelevel 2>&1 | awk '{print $2}'; }
cool() { local w=0; while [ "$(th)" != 0 ] && [ $w -lt ${COOL:-120} ]; do sleep 5; w=$((w+5)); done; echo "cooled ${w}s thermal $(th)"; }
mkdir -p $CELLS_DIR; [ -f $CELLS_DIR/copy3k.json ] || python3 $W/scripts/mcopy/prompts.py $CELLS_DIR > /dev/null
