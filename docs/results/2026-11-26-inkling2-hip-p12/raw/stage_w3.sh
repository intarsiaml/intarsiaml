#!/usr/bin/env bash
# Window 3 (lane ink-hip-p12): G2 short / long (main, branch), the tiny Inkling-2 suites on hip, plain decode short 4 + 4 ABBA.
set -uo pipefail
R=$HOME/intarsiaml-ink-hip-p12/docs/results/2026-11-26-inkling2-hip-p12/raw
BR=${BR:-p12b} PARTS="g2 tiny" bash "$R/stage_gate.sh"
BR=${BR:-p12b} SPEC=off PROMPTS="short" bash "$R/stage_speed.sh"
