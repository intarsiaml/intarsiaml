#!/usr/bin/env bash
# M2 (EXL3) end to end on spark1 (CUDA): a real EXL3 Qwen3-8B (turboderp/Qwen3-8B-exl3 4.0bpw) through intarsia.
# Runs inside a fleetq slot (non-exclusive, vLLM stays up):
#   fleetq acquire --host spark1 --mem 25 --label "M2 e2e: ..." --expect 120 --wait ; FLEETQ_JOB=<id> scripts/m2-exl3-e2e/run.sh <steps>
# STEPS (comma separated): convert,build,smoke,qualify,qualify-bf16,golden,acc,kl
#   convert       tools/exl3_plain_to_bf16.py (in the dsv41-serve:v4 container, python3 -I): plain F16 -> bf16, EXL3 parts byte for byte
#   build         scripts/cuda-decode/build.sh (rsync this tree to spark1:~/intarsia-decfuse/$TREE, cargo build --features cuda)
#   smoke         intarsia generate --chat, greedy, 64 tokens
#   qualify       intarsia qualify --quick-only, the EXL3 model, bf16 golden as reference
#   qualify-bf16  the same on the bf16 Qwen3-8B (speed comparison, same build)
#   golden        tools/golden_exl3.py on the UNCONVERTED EXL3 dir (exllamav3's own log-probs, same token ids)
#   acc           intarsia accuracy: EXL3 model vs the exllamav3 golden and vs the bf16 golden
#   dump          intarsia accuracy --write-self-golden: our own full-vocab distribution
#   kl            golden_kl.py (container): exllamav3 vs bf16 golden, intarsia vs bf16, intarsia vs exllamav3 (needs dump)
# (the container runs as root: the converted dir and the golden are root-owned; the golden is chmod 644, and `clean` removes the root-owned
# converted dir with a container too)
# Logs: docs/results/2026-10-21-m2-exl3/raw/e2e-<step>.log. Intermediate files are deleted by the `clean` step.
set -euo pipefail
: "${FLEETQ_JOB:?run under a fleetq slot (FLEETQ_JOB)}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
HOST=${HOST:-spark1}
TREE=${TREE:-m2-exl3-e2e}
STEPS=${1:-${STEPS:-convert,build,smoke,qualify,golden,acc,kl}}
OUT="$REPO/docs/results/2026-10-21-m2-exl3/raw"
R='~/intarsia-decfuse'
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
IMG=dsv41-serve:v4
EXL3='$HOME/models-m2/qwen3-8b-exl3-4.0bpw'
CONV='$HOME/models-m2/qwen3-8b-exl3-4.0bpw-bf16plain'
BF16='$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/* | head -1)'
GOLD_BF16='$HOME/intarsiaml-golden/qwen3-8b-fp32.safetensors'
GOLD_EXL3='$HOME/intarsiaml-golden/tmp-m2e2e-qwen3-8b-exl3-exllamav3.safetensors'
SELF='$HOME/intarsiaml-golden/tmp-m2e2e-intarsia-self.safetensors'
REP='$HOME/intarsia-decfuse/'"$TREE"'/reports-m2e2e'
BIN="$R/$TREE/target/acc/release/intarsia"
mkdir -p "$OUT"
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
has() { [[ ",$STEPS," == *",$1,"* ]]; }
if has convert; then
  # LoadOptions.act_dtype exists but the CLI never sets it (common.rs engine_options: act_dtype: None), so the activation dtype is
  # the embedding's; F16 embedding -> f16 activations, which the CUDA path does not run. Hence the converted copy.
  scp -q "$REPO/tools/exl3_plain_to_bf16.py" "$HOST:/tmp/m2e2e_plain_to_bf16.py"
  rsh "test ! -e $CONV || { echo 'dest exists'; exit 1; }; mkdir -p $CONV;
       docker run --rm --name m2e2e-convert --entrypoint python3 -v /tmp/m2e2e_plain_to_bf16.py:/work/c.py:ro -v $EXL3:/src:ro -v $CONV:/dst:rw $IMG -I /work/c.py /src /dst; ls -la $CONV; rm -f /tmp/m2e2e_plain_to_bf16.py" 2>&1 | tee "$OUT/e2e-convert.log"
fi
if has build; then
  "$REPO/scripts/cuda-decode/build.sh" "$TREE" 2>&1 | tee "$OUT/e2e-build.log"
fi
if has smoke; then
  rsh "$BIN generate --model $CONV --backend cuda --chat --no-think --prompt 'Explain in two sentences why the sky is blue.' --max-tokens 64 --temperature 0" 2>&1 | tee "$OUT/e2e-smoke.log"
fi
if has qualify; then
  rsh "mkdir -p $REP; $BIN qualify --model $CONV --backend cuda --quick-only --golden $GOLD_BF16 --out $REP/exl3" 2>&1 | tee "$OUT/e2e-qualify-exl3.log" || true  # exit 1 = a gate failed, still wanted
  rsh "cat $REP/exl3/quick-report.md" > "$OUT/e2e-qualify-exl3-quick-report.md"
fi
if has qualify-bf16; then
  rsh "mkdir -p $REP; $BIN qualify --model $BF16 --backend cuda --quick-only --golden $GOLD_BF16 --out $REP/bf16" 2>&1 | tee "$OUT/e2e-qualify-bf16.log" || true
  rsh "cat $REP/bf16/quick-report.md" > "$OUT/e2e-qualify-bf16-quick-report.md"
fi
if has golden; then
  scp -q "$REPO/tools/golden_exl3.py" "$HOST:/tmp/m2e2e_golden_exl3.py"
  rsh "docker run --rm --name m2e2e-golden --gpus all --entrypoint python3 -v /tmp/m2e2e_golden_exl3.py:/work/g.py:ro -v $EXL3:/model:ro -v \$HOME/intarsiaml-golden:/gold:rw $IMG -I /work/g.py --model /model --ids-from /gold/qwen3-8b-fp32.safetensors --out /gold/tmp-m2e2e-qwen3-8b-exl3-exllamav3.safetensors; docker run --rm --name m2e2e-chmod --entrypoint chmod -v \$HOME/intarsiaml-golden:/gold:rw $IMG 644 /gold/tmp-m2e2e-qwen3-8b-exl3-exllamav3.safetensors; ls -la $GOLD_EXL3; rm -f /tmp/m2e2e_golden_exl3.py" 2>&1 | tee "$OUT/e2e-golden.log"
fi
if has acc; then
  { echo "== KL(exllamav3 golden || intarsia): the engine on identical weights"
    rsh "$BIN accuracy --model $CONV --backend cuda --golden $GOLD_EXL3 || true"
    echo; echo "== KL(bf16 golden || intarsia EXL3): quantisation loss + engine"
    rsh "$BIN accuracy --model $CONV --backend cuda --golden $GOLD_BF16 || true"; } 2>&1 | tee "$OUT/e2e-accuracy.log"
fi
if has dump; then
  # intarsia's own full-vocabulary distribution (prefill rows + decode window) on the golden's ids, for golden_kl.py (p99 and the
  # three-way table on one footing). Prefill rows are all that is compared; --mode prefill keeps it to those.
  rsh "$BIN accuracy --model $CONV --backend cuda --golden $GOLD_BF16 --mode prefill --write-self-golden $SELF --self-topk 151936 --allow-unstamped || true; ls -la $SELF" 2>&1 | tee "$OUT/e2e-dump.log"
fi
if has kl; then
  scp -q "$S/golden_kl.py" "$HOST:/tmp/m2e2e_golden_kl.py"
  K="docker run --rm --name m2e2e-kl --entrypoint python3 -v /tmp/m2e2e_golden_kl.py:/work/k.py:ro -v \$HOME/intarsiaml-golden:/gold:ro $IMG -I /work/k.py"
  G=/gold/qwen3-8b-fp32.safetensors; X=/gold/tmp-m2e2e-qwen3-8b-exl3-exllamav3.safetensors; I=/gold/tmp-m2e2e-intarsia-self.safetensors
  { echo "== KL(exllamav3 || bf16 golden): the format's own quantisation loss"; rsh "$K $G $X"
    echo; echo "== KL(intarsia EXL3 || bf16 golden): quantisation loss + engine (prefill rows)"; rsh "$K $G $I"
    echo; echo "== KL(exllamav3 golden || intarsia): the engine on identical weights (prefill rows)"; rsh "$K $X $I"; } 2>&1 | tee "$OUT/e2e-kl.log"
  rsh "rm -f /tmp/m2e2e_golden_kl.py"
fi
if has clean; then
  rsh "rm -f $GOLD_EXL3 $SELF; rm -rf $REP; rm -f /tmp/m2e2e_*" 2>&1 | tee "$OUT/e2e-clean.log"
fi
echo "steps done: $STEPS"
