#!/usr/bin/env bash
# Window 5 (lane ink-hip-p12): plain decode 16k (prefill too), spec-default 4k, 4 + 4 ABBA each.
set -uo pipefail
R=$HOME/intarsiaml-ink-hip-p12/docs/results/2026-11-26-inkling2-hip-p12/raw
BR=${BR:-p12b} SPEC=off PROMPTS="16k" bash "$R/stage_speed.sh"
BR=${BR:-p12b} SPEC=default PROMPTS="4k" bash "$R/stage_speed.sh"
