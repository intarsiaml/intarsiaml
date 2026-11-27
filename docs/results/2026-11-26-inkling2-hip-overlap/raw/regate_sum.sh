#!/usr/bin/env bash
# Lane ink-hip-overlap2: the regate summary of windows S6-S8 (args: the window dirs, default s6 s7 s8): the paired ratios and gate-4
# verdict (speed_sum.py), the server's own ITL / step histograms per real level and arm (srvitl.py), the client-path check
# (coalesce.py), B's validity (decode-lane line, no "overlap unavailable", overlap steps), and B = A replies per window.
cd "$(dirname "$0")"
W=${*:-s6 s7 s8}
python3 speed_sum.py $W
for w in $W; do
  for v in A B; do echo "== server $w $v"; python3 srvitl.py $w $v; done
  echo "== client path $w"; python3 coalesce.py $w/*-real*-n*.json
  echo "== B validity $w: $(grep -ac 'stream 1 (decode lane) at priority -1' $w/B.log) decode-lane line, $(grep -ac 'overlap unavailable' $w/B.log) unavailable, $(grep -E '^intarsia_batch_overlap_steps_total ' $w/B-metrics1.txt); A: $(grep -E '^intarsia_batch_overlap_steps_total ' $w/A-metrics1.txt); KV $(grep -hE '^intarsia_kv_tokens_capacity ' $w/A-metrics0.txt $w/B-metrics0.txt | tr '\n' ' ')"
  for n in 4 8 16; do echo "== B = A $w n$n: $(python3 serve_bitwise.py compare $w/A-real*-n$n.json $w/B-real*-n$n.json | tail -1)"; done
done
