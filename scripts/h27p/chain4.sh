#!/usr/bin/env bash
# Lane H27P window 9: G0 (plan show / ids of the 8B and the 35B, base vs dev), then the tp1 check
S="$(dirname "${BASH_SOURCE[0]}")"
bash "$S/g0.sh"
bash "$S/tp1.sh"
