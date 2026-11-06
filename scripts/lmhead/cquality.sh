#!/usr/bin/env bash
# Lane lmhead-fp8-default, CUDA KL / top-1 check, FROM THE MAC inside a spark1 quiet slot (same harness as the HIP and Metal rows of
# docs/results/2026-11-05-lmhead-fp8-27b.md): the 4k ids-prompts 4k-0..2 + 128 greedy tokens of the bf16 head -> prompts-only golden -> bf16 head's
# top-64 self-golden -> the same binary with INTARSIA_LMHEAD_QUANT=fp8 (and =off as the control).
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 50 --label "..." -- scripts/lmhead/cquality.sh <out dir>
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$S/../.." && pwd)"; O=${1:?out dir}; mkdir -p "$O"
NAMES=${NAMES:-4k-0 4k-1 4k-2}; MLOCAL=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4
say() { echo "$(date +%T) $*" | tee -a "$O/steps.log"; }
R='~/lmh-cuda'; SSH="ssh -n -o BatchMode=yes spark1"
COMMON="--backend cuda --model $R/models/Qwen3.8-27B-Quark-AWQ-MXFP4 --mem-budget 45 --max-seq 8192 --mxfp4-act none"
specs=()
for n in $NAMES; do
  $SSH "cd $R && export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH && INTARSIA_LMHEAD_QUANT=off target-head/release/intarsia generate -m models/Qwen3.8-27B-Quark-AWQ-MXFP4 --backend cuda --mem-budget 45 --max-seq 8192 --mxfp4-act none --temperature 0 --profile-every 0 --ignore-eos --show-ids -p \"\$(cat src/scripts/cmtp/ids-prompts/$n.txt)\" --max-tokens 128 < /dev/null > out/qgen-$n.out 2> out/qgen-$n.err"
  $SSH "grep -a '^ids:' $R/out/qgen-$n.err" > "$O/qgen-$n.ids"; say "gen $n: $(tr ',' '\n' < "$O/qgen-$n.ids" | wc -l) ids"
  specs+=("$n=$REPO/scripts/cmtp/ids-prompts/$n.txt:$O/qgen-$n.ids")
done
( cd "$REPO" && uv run --with tokenizers --with numpy --with safetensors python scripts/lmhead/mkgolden.py $MLOCAL "$O/prompts.safetensors" "${specs[@]}" ) 2>&1 | tail -2 | tee -a "$O/steps.log"
scp -q "$O/prompts.safetensors" spark1:lmh-cuda/out/prompts.safetensors
POL="--allow-same-code --allow-dirty --allow-unstamped --pre-change $(git -C "$REPO" rev-parse --short=9 HEAD)"
ACC="cd $R && export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH && target-head/release/intarsia accuracy $COMMON --mode both --decode-positions 128 --self-topk 64 $POL"
$SSH "$ACC --golden out/prompts.safetensors --write-self-golden out/self-bf16.safetensors < /dev/null > out/q-write.out 2>&1"; say "write exit $?"
for arm in fp8 off; do
  $SSH "INTARSIA_LMHEAD_QUANT=$arm bash -c '$ACC --self-golden out/self-bf16.safetensors --self-max-kl 1 --self-max-pos-kl 100 --self-max-flip-pct 100 --self-max-tail 1 --self-prompt-factor 1000 < /dev/null' > $R/out/q-cmp-$arm.out 2>&1"; say "compare $arm exit $?"
done
scp -q "spark1:lmh-cuda/out/q-*.out" "$O/" ; say done
