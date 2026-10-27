import sys; sys.path.insert(0,'/opt/code/intarsiaml/.worktrees/mgd/run')
from sg import *
for n in ['short','long','4k']:
    a,_=load(f'kl/{n}.off.selfgolden.safetensors'); b,_=load(f'kl/{n}.on.selfgolden.safetensors')
    assert a.keys()==b.keys()
    bad=[k for k in a if not (a[k].tobytes()==b[k].tobytes())]
    print(n,len(a),'tensors, differing:',len(bad),'total bytes',sum(v.nbytes for v in a.values()))
