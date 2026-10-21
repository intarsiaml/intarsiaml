"""Mean tool-exact KL21 per 2048-row bin of the 12k prompts (position growth check). usage: bins.py POS.tsv"""
import sys
rows = [l.split('\t') for l in open(sys.argv[1])]
for name in ['long_docs_4k', 'long_code_4k', 'long_docs_12k', 'long_code_12k', 'chat_reuse_4k']:
    r = [float(x[5]) for x in rows if x[0] == 'prefill' and x[1] == name]
    print(f'{name:14s}', ' '.join('%.1e' % (sum(r[b:b + 2048]) / len(r[b:b + 2048])) for b in range(0, len(r), 2048)))
