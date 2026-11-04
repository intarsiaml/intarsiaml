#!/usr/bin/env bash
# The dsv41/attn-wire window: ONE job over both Sparks (vLLM down, always restored by run-full.sh's trap).
#   fleetq run --host spark1,spark2 --exclusive --takes-down spark1-vllm,spark2-vllm --mem spark1=116,spark2=116 --expect 150 --label ... -- bash .scratch/window.sh
# Configs (fresh start of both ranks each; BIN_ABi picks the binary, the cfg string the chunk). Alternating base / head:
#   ab0 base c2048: golden longp lspeed    ab1 head c2048: golden longp lspeed
#   ab2 base c2048: lspeed                 ab3 head c2048: lspeed
#   ab4 base c512 : lspeed                 ab5 head c512 : lspeed
#   ab6 base c512 : lspeed                 ab7 head c512 : lspeed
# CAP 65536 for all (the 65000-token prompt needs it); lspeed sizes 4096 and 32768.
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
export SPARK_DIR=intarsia-dsv41-attnwire
BASEBIN='$HOME/intarsia-dsv41-attnwire/bin/intarsia.base'
C2="INTARSIA_DSV41_CHUNK=2048"; C5="INTARSIA_DSV41_CHUNK=512"
export AB="$C2|$C2|$C2|$C2|$C5|$C5|$C5|$C5"
for i in 0 2 4 6; do export BIN_AB$i="$BASEBIN"; done
export STEPS_AB0="golden longp lspeed" STEPS_AB1="golden longp lspeed"
for i in 2 3 4 5 6 7; do export STEPS_AB$i="lspeed"; done
export STEPS="lspeed"
export LONG_PROMPTS="$PWD/docs/results/2026-11-02-dsv41-prefill/prompts/long32k.txt $PWD/docs/results/2026-11-03-dsv41-attn-fusion/prompts/long64k.txt"
export LSIZES=4096,32768 LREPS=3
export CAP=65536 CHUNK=2048 MAX_MIN=${MAX_MIN:-150} EARLYOOM_OK="3,2 2,1"
export OUT=$PWD/.scratch/window
exec scripts/dsv41-tp2/run-full.sh
