#!/usr/bin/env bash
# Lane T2 window 1 (ON tatooine, radiance down, called by window.sh): smoke before 00:00. Does the 27B MXFP4 load and decode
# at tp=2? 256 greedy ids on the short prompt: head tp=1 (P1's md5 2086b6a4), head tp=2 eager (--no-graphs), tp=2 graphs
# (default), tp=2 graphs + INTARSIA_TP_FUSION=1.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/t2/lib.sh"
ids head 0 27-tp1-short $M27 short
ids head 0,1 27-tp2ng-short $M27 short $TP2 --no-graphs
ids head 0,1 27-tp2g-short $M27 short $TP2
E="INTARSIA_TP_FUSION=1" ids head 0,1 27-tp2gf-short $M27 short $TP2
same ids-27-tp2ng-short.ids ids-27-tp2g-short.ids "27B tp=2 graphs vs eager"
same ids-27-tp2g-short.ids ids-27-tp2gf-short.ids "27B tp=2 graphs+fusion vs graphs"
firstdiff ids-27-tp1-short.ids ids-27-tp2g-short.ids "27B tp=2 vs tp=1 short"
grep -ah "MiB\|GiB\|weights\|kv cap\|tokens of context" "$O"/ids-27-tp2g-short.err | head -8 | tee -a "$O/steps.log"
log done
