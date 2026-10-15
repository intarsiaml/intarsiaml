import sys, math; sys.path.insert(0,'.')
import grid
cfgs=[('streak',p) for p in [(0,4,64,1),(0,4,64,2),(0,4,64,3),(4,4,32,2),(3,4,32,2),(4,2,32,2),(6,1,16,2),(4,4,32,1),(3,4,64,1),(5,4,32,2)]]
names=list(grid.cases.keys())
short=lambda k:k.replace('sweep-','s:').replace('grounded-','g:').replace('open-','o:')[:14]
print('| '+' | '.join(short(k) for k in names)+' |')
for kind,p in cfgs:
    r,gm,gmin,om,omin=grid.evaluate(kind,p)
    geo=math.exp(sum(math.log(v) for v in r.values())/len(r))
    print(f'{kind}{p} geo {geo:.3f} grmean {gm:.2f} grmin {gmin:.2f} openmean {om:.3f} openmin {omin:.3f}')
    print('   '+' '.join(f'{short(k)}={r[k]:.2f}' for k in names))
