import json,sys
base='docs/results/2026-10-27-m35p-metal-35b-prefill-kl/raw/accuracy'
for tag in sys.argv[1:]:
    print('==',tag)
    for n in ['short','long','4k']:
        for ptag,path in [('base',base),(tag,tag if tag.startswith('/') or tag.startswith('.') else tag)]:
            pass
for n in ['short','long','4k']:
    row=[]
    for path in [base]+sys.argv[1:]:
        d=json.load(open(f'{path}/{n}.verdict.json'))
        for v in d['verdicts']:
            row.append((v['mode'],round(v['kl'],6),v.get('kl_on_noise_rows') and round(v['kl_on_noise_rows'],6),v.get('pass')))
    print(n,row)
