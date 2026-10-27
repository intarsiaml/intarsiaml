#!/usr/bin/env bash
# Lane cmx: greedy-ids gate ON spark1 (Qwen3.8-27B MXFP4, CUDA, tp=1), run inside one fleetq quiet slot:
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 40 --label "cmx: ids" -- ssh spark1 'bash -s short-0 mid-0' < scripts/cmx/ids.sh
# Needs the prompts at $R/trees/dev/scripts/cmx/ids-prompts (the dev tree rsync). Arms (256 greedy ids, --ignore-eos):
#   base-off  base binary, --spec off              dev-off   dev binary, --spec off
#   dev0-off  dev, INTARSIA_CUDA_MXFP4_MMA=0, --spec off     dev-mtp   dev, --spec mtp --spec-k 2    base-mtp  base, --spec mtp --spec-k 2
# Output: $O/gen-<prompt>-<arm>.{ids,err}; then the comparison table (prompt tokens, first divergence index of each arm against base-off).
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-cmx; M=${MODEL:-$HOME/intarsiaml-s27d/models/Qwen3.8-27B-MXFP4}; O=${O:-$R/out/ids}; mkdir -p "$O"
P=$R/trees/dev/scripts/cmx/ids-prompts
ARMS=${ARMS:-"base-off dev-off dev0-off dev-mtp"}
gen() { # gen <prompt> <arm>
  local n=$1 arm=$2 tree env spec ms
  case $n in short*|mid*) ms=4096 ;; 4k*) ms=8192 ;; *) ms=20480 ;; esac
  case $arm in
    base-off) tree=base; env=""; spec="--spec off" ;;
    dev-off) tree=dev; env=""; spec="--spec off" ;;
    dev0-off) tree=dev; env="INTARSIA_CUDA_MXFP4_MMA=0"; spec="--spec off" ;;
    dev-mtp) tree=dev; env=""; spec="--spec mtp --spec-k 2" ;;
    base-mtp) tree=base; env=""; spec="--spec mtp --spec-k 2" ;;
  esac
  env $env $R/trees/$tree/target/release/intarsia generate -m $M --backend cuda --mem-budget 45 --max-seq $ms $spec --chat --no-think --temperature 0 \
    --profile-every 0 --ignore-eos --show-ids -p "$(cat $P/$n.txt)" --max-tokens ${NTOK:-256} < /dev/null > $O/gen-$n-$arm.out 2> $O/gen-$n-$arm.err
  grep -a '^ids:' $O/gen-$n-$arm.err | head -1 > $O/gen-$n-$arm.ids
  echo "$n $arm: $(grep -a 'finish=' $O/gen-$n-$arm.err | tail -1 | cut -c1-150)"
}
for n in "$@"; do for arm in $ARMS; do gen $n $arm; done; done
python3 - "$O" "$@" <<'PY'
import sys, re
O = sys.argv[1]
arms = "base-off dev-off dev0-off dev-mtp base-mtp".split()
def ids(n, a):
    try: s = open(f"{O}/gen-{n}-{a}.ids").read()
    except OSError: return None
    m = re.search(r"\[(.*)\]", s)
    return [int(x) for x in m.group(1).split(",") if x.strip()] if m else None
for n in sys.argv[2:]:
    b = ids(n, "base-off")
    try: pt = re.search(r"prompt=(\d+)", open(f"{O}/gen-{n}-base-off.err").read()).group(1)
    except Exception: pt = "?"
    print(f"{n} (prompt tokens {pt}): base-off {len(b) if b else None} ids")
    for a in arms[1:]:
        x = ids(n, a)
        if x is None or b is None: print(f"  {a}: missing"); continue
        d = next((i for i, (p, q) in enumerate(zip(b, x)) if p != q), None)
        print(f"  {a} vs base-off: " + ("IDENTICAL" if d is None and len(x) == len(b) else f"first divergence at id {d} of {len(b)}"))
    x, y = ids(n, "dev-mtp"), ids(n, "dev-off")
    if x and y: print(f"  dev-mtp vs dev-off (spec-on == spec-off): " + ("IDENTICAL" if x == y else "DIFFERENT"))
PY
