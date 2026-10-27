#!/usr/bin/env bash
# Lane TPO final window, run FROM THE MAC inside ONE fleetq job that holds both R9700s and takes radiance down (<= 45 min):
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu --mem 30 --expect 45 \
#     --label "tpo w6 ..." -- scripts/tpo/w6.sh
# Steps (each skipped when its start would pass its deadline, in priority order; see the file for the order actually run):
#   1 sweep      generate-based timing + 48 greedy tokens (md5) of the arms, the candidate set below            (w3.sh T I)
#   2 mutants    the planted bugs of the overlap, md5 must differ from the candidate's                       (w3.sh I)
#   3 g0 + rounds `plan show` 8B / 35B, then 6 interleaved base / best 4k prefill rounds                    (w3.sh G R)
#   4 drive      texts (18 prompts, MTP on) base / best / best spec off, localeval speed rounds, mixids      (drive.sh)
# The best arm = the lowest median prefill among the arms whose 48 tokens equal the base's (picked after step 1).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
RAW=$REPO/docs/results/2026-10-27-tpo-hip-27b-prefill-next/raw/w6; mkdir -p "$RAW" "$REPO/tpo-out"
T0=$(date +%s); LIMIT=${LIMIT:-2400}   # 40 min: the 45 min cap includes the restore
el() { echo $(( $(date +%s) - T0 )); }
say() { echo "$(date +%T) [+$(el)s] $*" | tee -a "$RAW/steps.log"; }
SSH="ssh -o BatchMode=yes tatooine"
Q="GPU_MAX_HW_QUEUES=4"
C4="--prefill-chunk,4096"
MIN="INTARSIA_TP_OVERLAP_MIN=3072"
SWEEP="base:tpo-base:: \
base_spec:tpo-base::SPECON \
v2c4096:tpo-dev4:INTARSIA_TP_OVERLAP=2,$MIN,$Q:$C4 \
v2c4096_spec:tpo-dev4:INTARSIA_TP_OVERLAP=2,$MIN,$Q:--prefill-chunk,4096,SPECON \
v1c4096:tpo-dev4:INTARSIA_TP_OVERLAP=2,INTARSIA_TP_OVERLAP_FFN=0,$MIN,$Q:$C4 \
v3c4096:tpo-dev4:INTARSIA_TP_OVERLAP=3,$MIN,$Q:$C4 \
ref4096q4:tpo-base:$Q:$C4"
run3() { # run3 <sections> <arms> <outdir name> [extra env assignments...]
  local sec=$1 arms=$2 d=$3; shift 3
  $SSH "KEEPDOWN=1 ARMS='$arms' O=\$HOME/intarsiaml-tpo-logs/$d $* bash ~/intarsiaml-tpo/scripts/tpo/w3.sh $sec" < /dev/null
}
fetch() { rsync -a "tatooine:intarsiaml-tpo-logs/$1/" "$RAW/$1/"; }

restore_radiance() { $SSH 'source ~/intarsiaml-tpo/scripts/tpo/lib.sh; restore' < /dev/null 2>&1 | tee -a "$RAW/steps.log"; }
trap restore_radiance EXIT
say "step 1 sweep"
run3 "T I" "$SWEEP" w6-sweep REPS=3 > "$RAW/sweep.log" 2>&1; fetch w6-sweep; grep -aE "^[TI] " "$RAW/sweep.log" | cut -c1-140 | tee -a "$RAW/steps.log"
# pick the best arm: lowest median of the 3 prefill ms among arms with the base's md5 (48 tokens)
BEST=$(python3 - "$RAW/sweep.log" <<'EOF'
import re,sys,statistics
t={};m={}
for l in open(sys.argv[1],errors='replace'):
    a=re.match(r'T (\S+) rep\d: .*prefill (\d+) ms',l)
    if a: t.setdefault(a.group(1),[]).append(int(a.group(2)))
    b=re.match(r'I (\S+): md5 (\S+)',l)
    if b: m[b.group(1)]=b.group(2)
base=m.get('base')
c=[(statistics.median(v),k) for k,v in t.items() if k not in ('base','ref4096q4','base_spec','v2c4096_spec') and m.get(k)==base and v]
print(min(c)[1] if c else 'none')
EOF
)
say "best arm: $BEST"
case "$BEST" in
  v1c4096) BE="INTARSIA_TP_OVERLAP=2,INTARSIA_TP_OVERLAP_FFN=0,$MIN,$Q"; BA="$C4" ;;
  v3c4096) BE="INTARSIA_TP_OVERLAP=3,$MIN,$Q"; BA="$C4" ;;
  *) BEST=v2c4096; BE="INTARSIA_TP_OVERLAP=2,$MIN,$Q"; BA="$C4" ;;
esac


if (( $(el) < 1500 )); then
  say "step 2 mutants (of $BEST)"
  MUT=""; for mu in skip_last scale offset; do MUT="$MUT mut_$mu:tpo-dev4:$BE,INTARSIA_TP_OVERLAP_MUTANT=$mu:$BA"; done
  run3 "I" "$MUT" w6-mut > "$RAW/mut.log" 2>&1; fetch w6-mut; grep -aE "^I " "$RAW/mut.log" | cut -c1-120 | tee -a "$RAW/steps.log"
fi
if (( $(el) < 1700 )); then
  say "step 3 plan show g0"
  run3 "G" "" w6-g GBIN=tpo-dev4 "GENVS='${BE//,/ }'" > "$RAW/g0.log" 2>&1; fetch w6-g; grep -aE "^G " "$RAW/g0.log" | cut -c1-120 | tee -a "$RAW/steps.log"
fi

if (( $(el) < 1500 )); then
  say "step 4 drive (texts, speed, mixids)"
  CH=""; [[ -n "$BA" ]] && CH="+chunk4096"
  export ENV_best="${BE//,/ }"
  ITEMS="rec:texts best+bin=tpo-dev4$CH:texts best+bin=tpo-dev4$CH+spoff:texts"
  B="best+bin=tpo-dev4$CH:speed"
  ITEMS="$ITEMS rec:speed $B $B rec:speed rec:speed $B $B rec:speed rec:speed $B $B rec:speed"
  ITEMS="$ITEMS best+bin=tpo-dev4$CH+mb16:mixids best+bin=tpo-dev4$CH+spoff:mixids"
  W=w6d ITEMS="$ITEMS" "$HERE/drive.sh" > "$RAW/drive.log" 2>&1 &
  DP=$!
  ( while kill -0 $DP 2>/dev/null; do if (( $(el) > LIMIT )); then say "deadline: stopping drive.sh"; kill -TERM $DP; break; fi; sleep 10; done ) &
  wait $DP
  say "drive ended (exit $?)"
fi
say "w6 done (+$(el)s)"
