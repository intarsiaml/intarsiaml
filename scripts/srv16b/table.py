#!/usr/bin/env python3
"""Lane srv16b: one row per arm and round of a window: localeval tok/s at 1/4/8/16 and its 4/8/16-stream TTFT median, then lat.py's
TTFT median / p90 and gap p99 / max per stream count and the packed passes.
  table.py <raw/wN dir>"""
import glob, json, os, re, sys
d = sys.argv[1]
idx = open(os.path.join(d, 'index.txt')).read()
for line in idx.splitlines():
    label = line.split(':')[0]
    m = re.search(r'results/(\S+?)/result\.json', line)
    if not m: continue
    r = json.load(open(f'/opt/code/deepeval/results/{m.group(1)}/result.json'))
    c = {x['streams']: x for x in r['speed']['concurrency']}
    tps = ' / '.join(f"{c[n]['aggregate_decode_tok_s']:.1f}" for n in (1, 4, 8, 16))
    tt = ' / '.join(f"{c[n]['ttft_s']['median']:.2f}" for n in (4, 8, 16))
    arm = label.split('-', 1)[1]
    lat = os.path.join(d, f"lat-{arm.rsplit('-r', 1)[0]}-r{arm.rsplit('-r', 1)[1]}.txt")
    ls = []
    for l in open(lat).read().splitlines() if os.path.exists(lat) else []:
        g = lambda p: re.search(p, l).group(1)
        v = [g(p) for p in (r'streams (\d+)', r'ttft med (\S+)', r'p90 (\S+)', r'gap p50 \S+ p99 (\S+)ms', r'ms max (\d+)ms', r'>0.5s (\d+)', r'packs (\d+)', r'chunks (\d+)')]
        ls.append('%s: ttft %s/%s gap p99 %s max %s >0.5s %s packs %s/%s' % tuple(v))
    print(f"| {arm} | {tps} | {tt} | " + ' ; '.join(ls) + ' |')
