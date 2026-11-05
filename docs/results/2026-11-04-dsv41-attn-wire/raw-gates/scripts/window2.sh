#!/usr/bin/env bash
# The dsv41/attn-wire + failpaths window: ONE job over both Sparks (vLLM down, always restored by run-full.sh's trap).
#   FLEETQ_OWNER=claude-c8e51313 fleetq run --host spark1,spark2 --exclusive --takes-down spark1-vllm,spark2-vllm --mem spark1=116,spark2=116 --expect 120 --label ... -- bash .scratch/window2.sh
# Binaries: bin/intarsia = merged head (attn-wire + failpaths), bin/intarsia.base = dbe5581ae (failpaths). Configs, fresh start of both ranks each:
#   ab0 base c2048 : golden               ab1 head c2048 : golden             ab2 head c2048 ROPE_INPLACE=0 : golden
#   ab3 base c512  : golden longp(32K)    ab4 head c512  : golden longp(32K)
#   ab5 head c2048 CAP 131072 : longp(126K)   ab6 base c2048 CAP 131072 : longp(126K)
# CAP 65536 for ab0-ab4 (as the last window), 131072 for ab5/ab6 (state per sequence at 131072: about 0.40 GiB, 0.20 more than at 65536).
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
export SPARK_DIR=intarsia-dsv41-attnwire
BASEBIN='$HOME/intarsia-dsv41-attnwire/bin/intarsia.base'
C2="INTARSIA_DSV41_CHUNK=2048"; C5="INTARSIA_DSV41_CHUNK=512"; BIG="INTARSIA_DSV41_CAP=131072"
export AB="$C2|$C2|$C2 INTARSIA_DSV41_ROPE_INPLACE=0|$C5|$C5|$C2 $BIG|$C2 $BIG"
for i in 0 3 6; do export BIN_AB$i="$BASEBIN"; done
export STEPS_AB0="golden" STEPS_AB1="golden" STEPS_AB2="golden" STEPS_AB3="golden longp" STEPS_AB4="golden longp" STEPS_AB5="longp" STEPS_AB6="longp"
export STEPS="golden"
L32=$PWD/docs/results/2026-11-02-dsv41-prefill/prompts/long32k.txt
export LONG_PROMPTS="$L32"
export LONG_AB5="$PWD/.scratch/long128k.txt" LONG_AB6="$PWD/.scratch/long128k.txt"
export CAP=65536 CHUNK=2048 MAX_MIN=${MAX_MIN:-110} EARLYOOM_OK="3,2 2,1"
export OUT=$PWD/.scratch/window2
exec scripts/dsv41-tp2/run-full.sh
