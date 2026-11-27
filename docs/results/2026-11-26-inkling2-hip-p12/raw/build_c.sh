#!/usr/bin/env bash
# ON tatooine inside one fleetq --load cpu job: the merged head p12c (cli, hip test ink_p12, CPU tests), the intarsia-models gate
# tests, then main2 (origin/main 80719483a) from .scratch/main-src.
set -uo pipefail
R=$HOME/intarsiaml-ink-hip-p12/docs/results/2026-11-26-inkling2-hip-p12/raw
bash "$R/cpu.sh" p12c ink_p12 || exit $?
bash "$R/build_mtests.sh" || exit $?
bash "$R/build_main.sh" main2
