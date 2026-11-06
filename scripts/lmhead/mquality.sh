#!/bin/zsh
# Lane lmhead-fp8, quality gate on Metal (endor), ONE slot: for the 27B (Qwen3.8-27B-Quark-AWQ-MXFP4, W4A16) KL / top-1 of the quantised head
# against the bf16 head, same binary, on the ids-prompts at 4k and 16k.
#   fleetq run --host endor --quiet gpu --mem 30 --expect 60 --label "..." -- zsh scripts/lmhead/mquality.sh <binary> <out dir>
# 1. per prompt: 128 greedy tokens with the bf16 head (raw text, no chat template) -> the continuation joins the prompt in a prompts-only
#    golden (mkgolden.py), so the teacher-forced decode window of `accuracy` lies on the model's own distribution;
# 2. `accuracy --write-self-golden` with the variable unset (the bf16 head's top-64 rows, prefill = every position, decode = the last
#    $DEC positions through the single-token kernels);
# 3. `accuracy --self-golden` with INTARSIA_LMHEAD_QUANT=fp8 and =mxfp4 (budgets set wide open: the verdict is the pass rule in the results doc).
set -u
BIN=${1:?binary}; O=${2:?out dir}; mkdir -p $O
R=${0:A:h:h:h}; M=${M:-/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4}; P=$R/scripts/cmtp/ids-prompts
NAMES=(${=NAMES:-4k-0 4k-1 4k-2 16k-0 16k-1 16k-2}); DEC=${DEC:-128}; NTOK=${NTOK:-128}
SHA=$(git -C $R rev-parse --short=9 HEAD)
[[ -n "${FLEETQ_JOB:-}" ]] || { echo "not inside a fleetq slot" >&2; exit 2; }
specs=()
for n in $NAMES; do
  ms=8192; [[ $n == 16k* ]] && ms=20480
  if [[ ! -s $O/gen-$n.ids ]]; then
    $BIN generate -m $M --backend metal --max-seq $ms --temperature 0 --profile-every 0 --ignore-eos --show-ids --mxfp4-act none \
      -p "$(cat $P/$n.txt)" --max-tokens $NTOK < /dev/null > $O/gen-$n.out 2> $O/gen-$n.err
    grep -a '^ids:' $O/gen-$n.err > $O/gen-$n.ids
  fi
  echo "gen $n: $(tr ',' '\n' < $O/gen-$n.ids | wc -l) ids"
  specs+=("$n=$P/$n.txt:$O/gen-$n.ids")
done
( cd $R && uv run --with tokenizers --with numpy --with safetensors python scripts/lmhead/mkgolden.py $M $O/prompts.safetensors $specs ) 2>&1 | tail -3
COMMON=(accuracy --backend metal --model $M --max-seq 20480 --mxfp4-act none --mode both --decode-positions $DEC --self-topk 64)
POL=(--pre-change $SHA --allow-same-code --allow-dirty --allow-unstamped)
env -u INTARSIA_LMHEAD_QUANT $BIN $COMMON --golden $O/prompts.safetensors --write-self-golden $O/self-bf16.safetensors $POL > $O/write.out 2> $O/write.err
echo "write exit $?"
for q in ${=ARMS:-fp8 mxfp4 off}; do
  INTARSIA_LMHEAD_QUANT=$q $BIN $COMMON --self-golden $O/self-bf16.safetensors --self-max-kl 1 --self-max-pos-kl 100 --self-max-flip-pct 100 --self-max-tail 1 \
    --self-prompt-factor 1000 $POL > $O/cmp-$q.out 2> $O/cmp-$q.err
  echo "compare $q exit $?"
done
