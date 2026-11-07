# Lane au1: grid over Policy2 with the sim scaled by the real/sim ratio of the current params per category (run from the repo root; window 1 paths hard-coded)
import sys, os, itertools, pickle, statistics as st
os.environ['REPLAY_ARMS']='auto,autotrmtp,autotrdfl'; os.environ['PRIOR_M']='2.2,35.4'; os.environ['PRIOR_D']='3.6,50.2'
sys.path.insert(0,'tools/asa')
sys.argv=['replay','docs/results/2026-11-06-spec-auto-tp1/raw/w1']
import replay as R
ev, ref = R.run(sys.argv[1:])
CATS=R.CATS
cur=dict(theta=2.5, alpha_m=0.06, dwell_m=4, dwell_d=16, hyst=1.1, probe_mtp=8, alpha_d=0.1)
simc=ev(lambda: R.Policy2(**cur))
f={c: ref['real'][c][0]/simc[c][0] for c,_ in CATS}
print('calibration factor real/sim (current params):', {c: round(v,3) for c,v in f.items()})
FLOOR=0.994
def score(res):
    v=[]
    for c,_ in CATS:
        m,d=ref['mtp'][c][0],ref['dfl'][c][0]
        a=res[c][0]*f[c]
        v.append(min(FLOOR*a/m/0.99, a/max(m,d)/0.97))
    return min(v), sum(v)/7, v
grid=dict(theta=(2.2,2.3,2.35,2.4,2.5,2.6), alpha_m=(0.04,0.06,0.1), dwell_m=(2,4,8), dwell_d=(4,8,16), hyst=(1.0,1.1,1.2), probe_mtp=(6,8,12,16), alpha_d=(0.1,0.15))
keys=list(grid); out=[]
for vals in itertools.product(*(grid[k] for k in keys)):
    kw=dict(zip(keys,vals)); s=score(ev(lambda kw=kw: R.Policy2(**kw))); out.append((s[0],s[1],kw,s[2]))
out.sort(key=lambda r:-(r[0]+0.5*r[1]))
print('current calibrated:', [round(x,3) for x in score(simc)[2]])
for o in out[:15]: print(f"min {o[0]:.3f} mean {o[1]:.3f}", o[2], ' '.join(f'{x:.3f}' for x in o[3]))
pickle.dump(out, open('.au1/calib.pkl','wb'))
