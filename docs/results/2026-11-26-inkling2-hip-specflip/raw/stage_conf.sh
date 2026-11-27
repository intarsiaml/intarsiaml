#!/usr/bin/env bash
# Lane ink-hip-specflip: one confirmation window = the ABBA half (stage_ab_k.sh, K=4, ORDER / R1 / R2 from the caller) and then the KV
# loads (stage_kv.sh PART), both on card $CARD (default 1) with binary $BR (default main7).
RAW=$HOME/intarsiaml-ink-hip-specflip/docs/results/2026-11-26-inkling2-hip-specflip/raw
bash "$RAW/stage_ab_k.sh"
bash "$RAW/stage_kv.sh"
