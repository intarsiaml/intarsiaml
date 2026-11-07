# Lane au1: Policy2 variants side by side on one window: cmp_params.py <raw/W> <real,mtp,dfl arm names> <PRIOR_M tok,ms> <PRIOR_D tok,ms> (run from the repo root)
import sys, os
raw=sys.argv[1]; arms=sys.argv[2]; pm=sys.argv[3]; pd=sys.argv[4]
os.environ['REPLAY_ARMS']=arms; os.environ['PRIOR_M']=pm; os.environ['PRIOR_D']=pd
sys.path.insert(0,'tools/asa'); sys.argv=['replay',raw]
import replay as R
ev, ref = R.run([raw])
C=R.CATS
def show(name, kw):
    res=ev(lambda: R.Policy2(**kw))
    vb=[res[c][0]/max(ref['mtp'][c][0],ref['dfl'][c][0])*100-100 for c,_ in C]
    vm=[res[c][0]/ref['mtp'][c][0]*100-100 for c,_ in C]
    print(f"{name:34}", ' '.join(f'{x:+5.1f}' for x in vb), '| vsMTP', ' '.join(f'{x:+5.1f}' for x in vm), '| dfl%', ' '.join(f"{res[c][1]*100:2.0f}" for c,_ in C))
cur=dict(theta=2.5, alpha_m=0.06, dwell_m=4, dwell_d=16, hyst=1.1, probe_mtp=8, alpha_d=0.1)
new=dict(theta=2.5, alpha_m=0.04, dwell_m=4, dwell_d=8, hyst=1.2, probe_mtp=8, alpha_d=0.1)
print('cats', ' '.join(c[:5].rjust(5) for c,_ in C))
show('current',cur); show('new (a.04 dd8 h1.2)',new)
for k,v in [('alpha_m',0.04),('dwell_d',8),('hyst',1.2)]: show(f'current + {k}={v}', dict(cur,**{k:v}))
show('new + theta2.4', dict(new,theta=2.4)); show('new + theta2.6', dict(new,theta=2.6)); show('new + pm12', dict(new,probe_mtp=12)); show('new + pm6',dict(new,probe_mtp=6))
