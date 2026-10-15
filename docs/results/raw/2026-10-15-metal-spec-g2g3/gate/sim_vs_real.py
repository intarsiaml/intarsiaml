import sys, re, glob; sys.path.insert(0,'.')
import grid
cfgs={2:(0,4,64,1),3:(2,4,64,1),4:(3,4,64,1),5:(2,8,64,1),6:(0,4,64,2),7:(2,4,64,2),8:(1,4,64,1)}
cases=['open-256','grounded-256','golden-docs-256','golden-code-256','4k-256','short-256']
ok=bad=0
for i,p in cfgs.items():
    for c in cases:
        tag=glob.glob(f"{i}-*-{c}.err")[0]
        t=open(tag).read(); tr=re.findall(r"spec: token \d+ pos \d+ drafted (\d+) accepted (\d+)",t)
        real=(len(tr),sum(int(x[1]) for x in tr))
        pi,g=grid.cases['sweep-'+c]; _,sp,sa=grid.sim2(pi,g,4,'streak',p)
        if (sp,sa)==real: ok+=1
        else: bad+=1; print('MISMATCH',i,p,c,'sim',(sp,sa),'real',real)
print(f"simulator vs real runs: {ok} of {ok+bad} (config, case) pairs reproduce the measured verify-pass and accepted-draft counts exactly")
