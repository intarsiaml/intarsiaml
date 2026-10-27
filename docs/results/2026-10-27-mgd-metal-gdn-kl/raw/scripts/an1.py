import collections, statistics as st, random
def load(f):
    pf={};dc={}
    for l in open(f):
        m,p,pos,tok,t1,kl,tr,to=l.rstrip('\n').split('\t')
        (pf if m.startswith('prefill') else dc)[(p,int(pos))]=float(kl)
    return pf,dc
on16=load('e1/pos.tsv'); off16=load('e3/pos16.tsv'); onF=load('e2/pos.tsv'); offF=load('e3/pos100000.tsv')
win=list(on16[1].keys())
def m(d,ks=win): return st.mean(d[k] for k in ks)
print('window positions',len(win))
for name,d in [('prefill NAXon',onF[0]),('prefill NAXoff',offF[0]),('decode-win NAXon',on16[1]),('decode-win NAXoff',off16[1]),('decode-full NAXon',onF[1]),('decode-full NAXoff',offF[1])]:
    print('%-20s %.3e'%(name,m(d)))
# full decode on/off identical?
print('full decode on vs off max abs diff',max(abs(onF[1][k]-offF[1][k]) for k in onF[1]))
# heavy tail: top positions contribution in window-decode NAXoff
v=sorted(((off16[1][k],k) for k in win),reverse=True)
tot=sum(x for x,_ in v)
print('top10 share window decode off %.2f'%(sum(x for x,_ in v[:10])/tot))
for x,k in v[:8]: print(k,'win_off %.3e win_on %.3e full %.3e prefill_off %.3e prefill_on %.3e'%(x,on16[1][k],offF[1][k],offF[0][k],onF[0][k]))
# same stats for all full-decode positions: top 1% share
allv=sorted(offF[1].values(),reverse=True); print('full decode top1%% share %.2f mean %.3e'%(sum(allv[:14])/sum(allv),st.mean(allv)))
# bootstrap CI of the mean for random 160-subsets (by prompt-block of 16 contiguous last positions)
random.seed(1)
vals=list(offF[1].values())
ms=[st.mean(random.sample(vals,160)) for _ in range(5000)]
ms.sort(); print('random 160-position subset mean of full decode: 2.5%% %.2e median %.2e 97.5%% %.2e'%(ms[125],ms[2500],ms[4875]))
