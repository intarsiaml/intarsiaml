#!/usr/bin/env bash
# Lane G2b window 2 (ON tatooine, radiance down, called by window.sh): 35B tp=2 gates on both R9700s (in-process, HIP P2P
# all-reduce), dev tree (feat/35b-tp2, no HIP-specific change). Needs the window-1 tp=1 dumps copied to
# ~/intarsiaml-g2b-dev/g2b-out/S1-{short,long}.safetensors.
#   tp=2 long accuracy dump (KL21 vs vLLM offline, scripts/m3-accuracy/setkl.py), self-golden tp=2 vs tp=1 (G1b's 35B
#   budget) plus the 1 % partial-scale bug on rank 1, 256 greedy ids at ~16k (tp=1 and tp=2), prefix reuse (tp=1 and tp=2),
#   then the HIP crate's GPU tests on the dev tree.
set -uo pipefail
O=${1:?out dir}
R=$HOME/intarsiaml-g2b-dev
log() { echo "$(date +%T) $*" | tee -a "$O/steps.log"; }
M35=/models/Qwen3.6-35B-A3B-NVFP4; G=/models/golden
SG=$G/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors; LG=$G/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
TP2="--tp 2 --devices 0,1"
BS="--self-max-kl 1.8e-2 --self-max-pos-kl 2.3 --self-max-flip-pct 3.76 --self-max-tail 2.8e-3 --self-prompt-factor 6.5"
BL="--self-max-kl 5.7e-2 --self-max-pos-kl 54.6 --self-max-flip-pct 8.7 --self-max-tail 7.1e-3 --self-prompt-factor 5.9"
run() { # run <gpus> <timeout> <out-prefix> <intarsia args...>
  local gpus=$1 to=$2 pfx=$3; shift 3
  INTARSIA_REPO=$R INTARSIA_TARGET_VOL=intarsia-cargo-target-g2b-dev HIP_VISIBLE_DEVICES=$gpus timeout "$to" \
    "$R/deploy/tatooine/dev.sh" /work/g2b-intarsia "$@" > "$O/$pfx.out" 2> "$O/$pfx.err"
  local rc=$?; log "$pfx exit $rc: $(grep -ahE '^golden gate|self-golden gate|SELF-STATS|prefix-reuse gate|^Error' "$O/$pfx.out" "$O/$pfx.err" | cut -c1-200 | tr '\n' '|')"
  return $rc
}
gen16() { # gen16 <gpus> <tag> [args]
  local gpus=$1 tag=$2; shift 2
  run "$gpus" 600 "gen-$tag" generate -m $M35 --backend hip --mem-budget 29 --max-seq 20480 --chat --no-think --profile-every 0 \
    --ignore-eos --show-ids -p "$(for i in 1 2 3 4; do cat $R/scripts/hip-decode/prompt4k.txt; done)" --max-tokens 256 "$@"
  grep -a '^ids:' "$O/gen-$tag.err" > "$O/gen-$tag.ids"
}
A="accuracy --backend hip --model $M35 --mem-budget 29 --max-seq 17408"

run 0,1 900 acc-tp2-long $A --golden $LG --mode both --write-self-golden /work/g2b-out/tp2-long.safetensors $TP2
mv $R/g2b-out/tp2-long.safetensors "$O/" 2>/dev/null
run 0,1 900 self-tp2-short $A --self-golden /work/g2b-out/S1-short.safetensors --mode both --allow-config-mismatch $TP2 $BS
run 0,1 900 self-tp2-long $A --self-golden /work/g2b-out/S1-long.safetensors --mode both --allow-config-mismatch $TP2 $BL
INTARSIA_TP_TEST_BUG=scale:1.01 run 0,1 900 self-tp2bug-short $A --self-golden /work/g2b-out/S1-short.safetensors --mode both --allow-config-mismatch $TP2 $BS
INTARSIA_TP_TEST_BUG=scale:1.01 run 0,1 900 self-tp2bug-long $A --self-golden /work/g2b-out/S1-long.safetensors --mode both --allow-config-mismatch $TP2 $BL
gen16 0 tp1-16k
gen16 0,1 tp2-16k $TP2
run 0,1 900 reuse-tp2 $A --golden $LG --mode reuse $TP2
run 0 900 reuse-tp1 $A --golden $LG --mode reuse
python3 - "$O" <<'PY' | tee -a "$O/steps.log"
import sys, re
o = sys.argv[1]
def ids(t):
    try: return [int(v) for v in re.findall(r"-?\d+", open(f"{o}/gen-{t}.ids").read().split(":", 1)[1])]
    except Exception: return None
x, y = ids("tp1-16k"), ids("tp2-16k")
if x is None or y is None: print("IDS 16k: missing")
else: print(f"IDS 16k tp1 vs tp2: {len(x)} / {len(y)} ids, first divergence at {next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), 'none')}")
PY
# the HIP crate's GPU tests on the dev tree (test binaries built off-window)
INTARSIA_REPO=$R INTARSIA_TARGET_VOL=intarsia-cargo-target-g2b-dev timeout 600 "$R/deploy/tatooine/dev.sh" \
  cargo test --release -p intarsia-backend-hip --no-fail-fast > "$O/hiptest.log" 2>&1
log "hip crate tests exit $?: $(grep -E '^test result' "$O/hiptest.log" | awk '{p+=$4; f+=$6} END {print NR " suites, " p " passed, " f " failed"}')"
log "done"
