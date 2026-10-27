#!/usr/bin/env bash
# Lane H27P: ids gate then tail sweep in one window (win.sh chain.sh)
S="$(dirname "${BASH_SOURCE[0]}")"
REUSE_BASE=1 bash "$S/ids.sh"
bash "$S/tails.sh"
