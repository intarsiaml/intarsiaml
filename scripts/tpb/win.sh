#!/usr/bin/env bash
# Lane TPB generic window, ON tatooine inside ONE fleetq slot that takes radiance down: stop radiance, run the given script, bring radiance
# back and wait until it answers (trap). usage (under fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance ...):
#   win.sh <script in scripts/tpb> [args...]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
trap restore EXIT
systemctl --user stop radiance; sleep 3; echo "[tpb] radiance $(systemctl --user is-active radiance) $(now)"
s=$1; shift
bash "$(dirname "${BASH_SOURCE[0]}")/$s" "$@"
echo "[tpb] $s done $(now)"
