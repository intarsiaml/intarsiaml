#!/usr/bin/env bash
# Gate stage (lane ink-hip-p12), parts chosen by PARTS (any of: md5 g2 adm tiny): greedy md5 at 64 tokens (short, 4k) for main and the
# branch with --spec off and the spec default, plus the branch at INTARSIA_INK_PACK12=0 (spec off); G2 short / long, main and
# branch; admission of the new / touched ops; the tiny Inkling-2 suites on hip. Card 0. Radiance down <= 5 min per window.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-p12
R=$REPO/docs/results/2026-11-26-inkling2-hip-p12/raw
OUT=$REPO/logs/gate
mkdir -p "$OUT"
M=/models/Inkling-2-Mini
BR=${BR:-p12a}
PARTS=${PARTS:-md5 g2 adm tiny}
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
now() { date '+%T'; }
has() { case " $PARTS " in *" $1 "*) return 0 ;; esac; return 1; }
gen() {  # arm binary env spec prompt-name prompt tokens
  local f=$OUT/$1-$4-$5-$7 sp=""
  [ "$4" = off ] && sp="--spec off"
  bash "$R/run.sh" env $3 /work/.scratch/bin/intarsia-$2 generate -m $M --backend hip --max-seq 8192 $sp --ignore-eos --profile-every 0 --max-tokens "$7" -p "$6" > "$f.out" 2> "$f.err"
  local md5; md5=$(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$f.out" | md5sum | cut -c1-32)
  echo "MD5 $1 $4-$5 $7 $md5 | $(cat "$f.out" "$f.err" | grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) | $(grep -acE 'P12 \(' "$f.err") P12 load lines [$(now)]"
}
if has md5; then
  for p in short 4k; do
    [ $p = short ] && PR="$PS" || PR="$P4"
    for sp in off default; do
      gen main main X=1 $sp $p "$PR" 64
      gen branch $BR X=1 $sp $p "$PR" 64
    done
    gen branch0 $BR INTARSIA_INK_PACK12=0 off $p "$PR" 64
  done
fi
if has g2; then
  for g in short long; do
    for a in main $BR; do
      echo "[stage] G2 $g $a $(now)"
      bash "$R/run.sh" /work/.scratch/bin/intarsia-$a accuracy -m $M --backend hip --max-seq 4096 --mode prefill --golden golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors 2>&1 | grep -aE "prefill floor|floor verdict|kl|KL|Error|panicked" | head -4 | cut -c1-400
    done
  done
fi
if has adm; then
  for op in matmul_p12 ink_in_proj; do
    echo "[stage] kernels test $op $(now)"
    bash "$R/run.sh" /work/.scratch/bin/intarsia-$BR kernels test --backend hip --op $op 2>&1 | grep -aE "ADMITTED|REJECT|FAIL|NO CASES|Error|panicked|cases" | head -12
  done
fi
if has tiny; then
  echo "[stage] tiny inkling2 $(now)"
  bash "$R/run.sh" env INTARSIA_INKLING2_BACKEND=hip cargo test --release -p intarsia-models --test inkling2 -- --test-threads 1 2>&1 | grep -aE "^test |test result|panicked" | head -30
  for t in inkling2_dspark inkling2_dspark_cut inkling2_verify_softcap; do
    echo "[stage] tiny $t $(now)"
    bash "$R/run.sh" env INTARSIA_INKLING2_BACKEND=hip cargo test --release -p intarsia-models --test $t -- --test-threads 1 2>&1 | grep -aE "^test |test result|panicked" | head -10
  done
fi
echo "[stage] done $(now)"
