#!/usr/bin/env bash
# Second window, 126K only: the first window's 126K prompt ended in a question and both binaries answered with end-of-sequence at once (1 token, empty text).
# This prompt ends mid-document, and LONGP_IGNORE_EOS=1 makes sure the 16 tokens exist either way. Same binaries (STAMP ccb37aa94), same configs as ab5 / ab6 of window 2.
#   ab0 head c2048 CAP 131072 : longp(126K)     ab1 base c2048 CAP 131072 : longp(126K)
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
export SPARK_DIR=intarsia-dsv41-attnwire
BASEBIN='$HOME/intarsia-dsv41-attnwire/bin/intarsia.base'
C2="INTARSIA_DSV41_CHUNK=2048"; BIG="INTARSIA_DSV41_CAP=131072"
export AB="$C2 $BIG|$C2 $BIG"
export BIN_AB1="$BASEBIN"
export STEPS_AB0="longp" STEPS_AB1="longp" STEPS="longp"
export LONG_PROMPTS="$PWD/.scratch/long128k.txt" LONGP_IGNORE_EOS=1
export CAP=131072 CHUNK=2048 MAX_MIN=${MAX_MIN:-40} EARLYOOM_OK="3,2 2,1"
export OUT=$PWD/.scratch/window3
exec scripts/dsv41-tp2/run-full.sh
