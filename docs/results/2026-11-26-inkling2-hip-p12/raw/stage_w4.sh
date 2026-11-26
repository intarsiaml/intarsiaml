#!/usr/bin/env bash
# Window 4 (lane ink-hip-p12): plain decode 1k and 4k, spec-default short, 4 + 4 ABBA each.
set -uo pipefail
R=$HOME/intarsiaml-ink-hip-p12/docs/results/2026-11-26-inkling2-hip-p12/raw
BR=${BR:-p12b} SPEC=off PROMPTS="1k 4k" bash "$R/stage_speed.sh"
BR=${BR:-p12b} SPEC=default PROMPTS="short" bash "$R/stage_speed.sh"
