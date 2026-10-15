#!/bin/zsh
# usage: acc.sh <label> <binary> [NAX variant name]   (run inside a fleetq slot; FLEETQ_JOB set by the caller)
# Runs `intarsia accuracy` on both fp32 goldens (bf16 gate class, default chunk 256, prefill + decode teacher-forced)
# and prints one summary line per golden: positions, prefill top-1 (count), prefill KL, decode top-1 / KL.
LABEL=$1; BIN=$2; VAR=$3
W=/opt/code/intarsiaml/.claude/worktrees/agent-afe24a73270ed0c31
R=$W/docs/results/raw/2026-10-15-metal-prefill-f1b/acc
M=/Users/josef/models/Qwen3-8B
G=/opt/code/intarsiaml/golden
mkdir -p $R
export INTARSIA_CPU_FREE_MEM_GB=60
if [ -n "$VAR" ]; then export INTARSIA_METAL_FLASH_NAX=$VAR; fi
$BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > $R/$LABEL-short.out 2> $R/$LABEL-short.err
$BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 > $R/$LABEL-long.out 2> $R/$LABEL-long.err
for g in short long; do
  python3 -I - $R/$LABEL-$g.out $LABEL $g <<'EOF'
import re, sys
txt = open(sys.argv[1]).read().splitlines()
ov = [l for l in txt if l.startswith("OVERALL")]
if len(ov) < 2:
    print(f"{sys.argv[2]} {sys.argv[3]}: NO RESULT"); sys.exit()
def parse(l):
    f = l.split()
    return int(f[1]), float(f[2]), float(f[3])
n, t, kl = parse(ov[0]); dn, dt, dkl = parse(ov[1])
print(f"{sys.argv[2]} {sys.argv[3]}: prefill {n} pos top-1 {t:.2f}% ({round(n*t/100)}/{n}) KL {kl:.3e} | decode {dn} pos top-1 {dt:.2f}% ({round(dn*dt/100)}/{dn}) KL {dkl:.3e}")
EOF
done
