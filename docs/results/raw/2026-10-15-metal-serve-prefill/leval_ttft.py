import json, glob
# Per-run 1k / 4k TTFT and the marginal prefill localeval derived, for the 03:08-03:20 remeasure runs.
print("run | 1k ttft min/med/max | 4k ttft min/med/max | marginal 1k->4k tok/s | 4k prompt tokens")
def find(o):
    if isinstance(o, dict):
        if 'rows' in o and isinstance(o['rows'], list) and o['rows'] and 'ttft_s' in o['rows'][0]: return o['rows']
        for v in o.values():
            r = find(v)
            if r: return r
    if isinstance(o, list):
        for v in o:
            r = find(v)
            if r: return r
for d in sorted(glob.glob('/opt/code/deepeval/results/20261015-03*-adhoc-remeasure-*')):
    j = json.load(open(d + '/result.json'))
    rows = {r['target_prompt_tokens']: r for r in find(j)}
    a, b = rows[1024], rows[4096]
    print("%-22s | %.2f/%.2f/%.2f | %.2f/%.2f/%.2f | %s | %s" % (d.split('remeasure-')[1], a['ttft_s']['min'], a['ttft_s']['median'], a['ttft_s']['max'],
          b['ttft_s']['min'], b['ttft_s']['median'], b['ttft_s']['max'], b.get('marginal_prefill_tok_s'), b.get('prompt_tokens')))
