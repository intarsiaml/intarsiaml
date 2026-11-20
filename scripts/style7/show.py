import sys,json
d=json.load(sys.stdin)
for k,v in d['files'].items(): print('FILE',k.split('/')[-1].replace('ft-','').replace('.json',''),{a:b for a,b in v.items() if a!='n'})
for k,v in d['pairs'].items():
    r,c=[x.split('/')[-1].replace('ft-','').replace('.json','') for x in k.split(' vs ')]
    print('PAIR',r,'->',c,{a:b for a,b in v.items() if a not in('n',)})
