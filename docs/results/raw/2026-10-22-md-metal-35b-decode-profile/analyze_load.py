import gzip,json
def load(p): return [json.loads(l) for l in gzip.open(p,'rt')]
