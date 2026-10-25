#!/usr/bin/env bash
# Lane tp2, window stage W=vocab2 (ON tatooine, radiance down, called by window.sh): the rest of the vocab-parallel lm_head
# gates (after stages-vocab.sh passed): longer context, the dense bf16 head, tp=1 at 4k, and decode speed A/B. D, B and "A/B" as
# there. Budget ~35 min (model loads dominate).
#  1. tp=2 35B ids at 4k context, A/B (split head vs replicated);
#  2. tp=2 8B (dense bf16 head, the other head format) ids short A/B;
#  3. tp=1 4k ids, D vs B (GPU 0 = 35B, GPU 1 = 8B 4k, in parallel);
#  4. speed: decode tok/s of 4k generations, alternating A/B x3 (35B) and x2 (8B): the lines are in steps.log.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/tp2/lib.sh"
D=${D:-vocab}; B=${B:-base}
A_ON="INTARSIA_TP_VOCAB=1"; A_OFF="INTARSIA_TP_VOCAB=0"
for t in $D $B; do echo "$t: $(cat "$(T $t)/tp2-intarsia.stamp" 2>/dev/null || ls -la "$(T $t)/tp2-intarsia")"; done > "$O/versions.txt"
tok_s() { grep -ahoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' "$1" | tail -1; }

E="$A_OFF" gen $D 0,1 tp2-off-35b-4k $M35 4k $TP2
E="$A_ON" gen $D 0,1 tp2-on-35b-4k $M35 4k $TP2
same gen-tp2-off-35b-4k.ids gen-tp2-on-35b-4k.ids "35B tp=2 256 greedy ids at 4k context, split head vs replicated"
E="$A_OFF" gen $D 0,1 tp2-off-8b-short $M8 short $TP2
E="$A_ON" gen $D 0,1 tp2-on-8b-short $M8 short $TP2
same gen-tp2-off-8b-short.ids gen-tp2-on-8b-short.ids "8B tp=2 256 ids short (dense bf16 head), split vs replicated"
track35() { for t in $D $B; do gen $t 0 tp1-$t-35b-4k $M35 4k; done; }
track8() { for t in $D $B; do gen $t 1 tp1-$t-8b-4k $M8 4k; done; }
track35 & p1=$!
track8 & p2=$!
wait $p1 $p2
same gen-tp1-$D-35b-4k.ids gen-tp1-$B-35b-4k.ids "35B tp=1 256 ids 4k, D vs B"
same gen-tp1-$D-8b-4k.ids gen-tp1-$B-8b-4k.ids "8B tp=1 256 ids 4k, D vs B"
for i in 1 2 3; do
  E="$A_OFF" gen $D 0,1 tp2-speed-off-$i-35b-4k $M35 4k $TP2
  E="$A_ON" gen $D 0,1 tp2-speed-on-$i-35b-4k $M35 4k $TP2
  log "speed 35B $i: replicated $(tok_s "$O/gen-tp2-speed-off-$i-35b-4k.err") | split $(tok_s "$O/gen-tp2-speed-on-$i-35b-4k.err")"
done
for i in 1 2; do
  E="$A_OFF" gen $D 0,1 tp2-speed-off-$i-8b-4k $M8 4k $TP2
  E="$A_ON" gen $D 0,1 tp2-speed-on-$i-8b-4k $M8 4k $TP2
  log "speed 8B $i: replicated $(tok_s "$O/gen-tp2-speed-off-$i-8b-4k.err") | split $(tok_s "$O/gen-tp2-speed-on-$i-8b-4k.err")"
done
log "done"
