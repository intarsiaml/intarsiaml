#!/bin/zsh
# Lane msm192 gates (from scripts/msmallm/gates.sh and gates_extra.sh; not timing runs). Run under fleetq, one step per job:
#   gates.sh route | ids [default|off]   (ids: MTP default and --spec off; one of them with the argument)
# Arms: base = origin/main binary (target/base/intarsia), unset = this branch with INTARSIA_METAL_MXFP4_NAXD unset (per-shape limits on),
# zero = this branch with INTARSIA_METAL_MXFP4_NAXD=0 (the scratch path at every M).
# Prompts (scripts/msm192/prompts.json, exact token counts; the prefill runs tokens - 1 rows): 161 -> 160 rows, 193 -> 192 rows,
# 2209 -> a 2048-row chunk + 160 rows, and prompt-4096.txt (a 4k prompt).
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/msm192; R=$W/docs/results/2026-11-16-msm192/raw/gates; mkdir -p $R
BASE=$W/target/base/intarsia; BIN=$W/target/release/intarsia
M27=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4
PR=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
pj() { python3 -c "import json; print(json.load(open('$W/scripts/msm192/prompts.json'))['$1'][0], end='')"; }
P161="$(pj 161)"; P193="$(pj 193)"; TAIL="$(pj 2209)"; P4K="$(cat $PR/prompt-4096.txt)"

arm_env() { case $1 in base) B=$BASE; E=(X=1) ;; unset) B=$BIN; E=(X=1) ;; zero) B=$BIN; E=(INTARSIA_METAL_MXFP4_NAXD=0) ;; esac }

ids() { # pn prompt spec arms...
  local pn=$1 ptxt=$2 spec=$3; shift 3
  local SA=(); [ $spec = off ] && SA=(--spec off)
  local ref="" ok=1 line="" pt=""
  for arm in "$@"; do
    arm_env $arm
    local o=$R/ids-$pn-$spec-$arm
    env $E $B generate --model $M27 --backend metal --max-seq 40000 --profile-every 0 --temperature 0 --max-tokens 256 --ignore-eos --show-ids $SA \
      --prompt "$ptxt" > $o.out 2> $o.err
    local got="$(grep -a '^ids' $o.err | md5 -q)" n=$(grep -a '^ids' $o.err | grep -oE '[0-9]+' | wc -l | tr -d ' ')
    pt="$(grep -aoE '^prompt: [0-9]+ tokens' $o.err)"
    [ -z "$ref" ] && ref=$got; [ "$got" = "$ref" ] && [ $n -ge 200 ] || ok=0
    line="$line | $arm $got[1,8] n=$n $(grep -aoE 'prefill [0-9]+ ms' $o.err | tail -1)"
  done
  echo "$( [ $ok = 1 ] && echo SAME || echo DIFFER ) 27b $pn spec=$spec ($pt)$line"
}

case ${1:?step} in
  route)
    # per (op, rows): in-kernel calls have two dispatches (x prep, GEMM), scratch-path calls three (`matmul_mxfp4_nax#2`)
    for arm in unset zero; do arm_env $arm
      for pn in p160 p192 tail2208; do
        case $pn in p160) P="$P161" ;; p192) P="$P193" ;; tail2208) P="$TAIL" ;; esac
        env $E INTARSIA_METAL_PROFILE=cb:$R/route-$pn-$arm.jsonl INTARSIA_METAL_COMMIT_EVERY=1 $B generate --model $M27 --backend metal --max-seq 40000 \
          --profile-every 0 --temperature 0 --max-tokens 2 --ignore-eos --spec off --prompt "$P" > $R/route-$pn-$arm.out 2> $R/route-$pn-$arm.err
        echo "route $pn $arm: $(python3 -c "
import json,collections
c=collections.Counter()
for l in open('$R/route-$pn-$arm.jsonl'):
    if l.startswith('{'):
        r=json.loads(l)
        if r['op'].startswith('matmul_mxfp4_nax'): c[(r['op'],r['t'])]+=1
print(sorted(c.items()))")"
        gzip -f $R/route-$pn-$arm.jsonl
      done
    done
    ;;
  ids)
    for spec in ${2:-default off}; do
      ids p160 "$P161" $spec base unset zero
      ids p192 "$P193" $spec base unset zero
      ids tail2208 "$TAIL" $spec base unset zero
      ids 4k "$P4K" $spec base unset zero
    done
    ;;
esac
echo "## done $*"
