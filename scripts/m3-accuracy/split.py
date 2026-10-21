"""Offline KL21(ref || cand) per row, split into row classes and chat-turn regions.
usage: split.py REF.safetensors CAND.dump.safetensors [prompt_name ...]
ref: top-k golden (vLLM top-20, or a top-64 dump); cand: top-64 self-golden dump.
A candidate's mass on a ref id outside its own top-64 is approximated by its 64th probability (as xcompare.py does)."""
import sys, json
import numpy as np
from safetensors import safe_open
IM_END, IM_START = 248046, 248045

def load(path):
    f = safe_open(path, 'np')
    md = json.loads(f.metadata()['prompts'])
    out = {}
    for i, p in enumerate(md):
        out[p['name']] = (f.get_tensor(f'ids_{i}'), f.get_tensor(f'topk_ids_{i}'), f.get_tensor(f'topk_logprobs_{i}').astype(np.float64))
    return out

def kl21_rows(rid, rlp, cid, clp, K=20):
    n = rid.shape[0]
    kl = np.zeros(n); p_imend_r = np.zeros(n); p_imend_c = np.zeros(n); kl_imend = np.zeros(n)
    for t in range(n):
        pr = np.exp(rlp[t, :K]); pr_rest = max(1.0 - pr.sum(), 1e-12)
        cmap = dict(zip(cid[t].tolist(), np.exp(clp[t]).tolist()))
        floor = float(np.exp(clp[t, -1]))
        pc = np.array([cmap.get(int(x), floor) for x in rid[t, :K]])
        pc_rest = max(1.0 - pc.sum(), 1e-12)
        terms = pr * (np.log(pr) - np.log(np.maximum(pc, 1e-30)))
        kl[t] = terms.sum() + pr_rest * (np.log(pr_rest) - np.log(pc_rest))
        j = np.where(rid[t, :K] == IM_END)[0]
        if len(j):
            kl_imend[t] = terms[j[0]]
            p_imend_r[t] = pr[j[0]]
        p_imend_c[t] = cmap.get(IM_END, 0.0)
    return kl, p_imend_r, p_imend_c, kl_imend

def main():
    ref = load(sys.argv[1]); cand = load(sys.argv[2])
    names = sys.argv[3:] or list(ref)
    for name in names:
        ids, rid, rlp = ref[name]
        cids, cid, clp = cand[name]
        assert (ids == cids).all(), name
        n = len(ids)
        kl, pir, pic, kli = kl21_rows(rid, rlp, cid, clp)
        rows = np.arange(n - 1)
        actual = ids[1:]
        rtop = rid[:-1, 0]; ctop = cid[:-1, 0]
        print(f'== {name}  n={n}  mean KL21={kl.mean():.4e}  (sum {kl.sum():.1f})')
        # candidate's own quality on the actual text
        def nll(tid, tlp):
            v = []
            for t in range(n - 1):
                m = np.where(tid[t] == ids[t + 1])[0]
                v.append(-tlp[t, m[0]] if len(m) else -tlp[t, -1])
            return np.array(v)
        nr, nc = nll(rid, rlp), nll(cid, clp)
        print(f'   top1==actual: ref {np.mean(rtop == actual):.3f}  cand {np.mean(ctop == actual):.3f};  capped NLL of actual: ref {nr.mean():.3f}  cand {nc.mean():.3f}')
        # regions by chat special tokens
        sp = [t for t in range(n) if ids[t] in (IM_START, IM_END)]
        if sp:
            bounds = [0] + sp + [n]
            print('   regions (rows [a,b)): mean KL21 / sum / mean P(im_end) ref, cand on rows whose next token is not im_end / top1 agree')
            for a, b in zip(bounds[:-1], bounds[1:]):
                if b - a < 2:
                    continue
                r = np.arange(a, min(b, n - 1))
                if len(r) == 0:
                    continue
                nonend = r[actual[r] != IM_END]
                print(f'     [{a:5d},{b:5d}) n={len(r):5d}  KL {kl[r].mean():.3e}  sum {kl[r].sum():7.1f}  P(im_end) ref {pir[nonend].mean() if len(nonend) else 0:.3f} cand {pic[nonend].mean() if len(nonend) else 0:.3f}  top1agree {np.mean(rtop[r] == ctop[r]):.3f}')
        cls_a = (rtop == IM_END) & (actual != IM_END)
        cls_b = (rtop == actual)
        cls_c = ~cls_a & ~cls_b
        k = kl[:-1]
        for lab, m in [('ref top1 = im_end (not actual)', cls_a), ('ref top1 = actual', cls_b), ('other', cls_c)]:
            print(f'   class {lab:32s} rows {m.sum():5d}  KL sum {k[m].sum():7.1f}  mean {k[m].mean() if m.any() else 0:.3e}')
        print(f'   KL21 term on the im_end bucket alone: sum {kli.sum():.1f} of {kl.sum():.1f}')
        big = np.argsort(-kl)[:8]
        for t in big:
            print(f'     pos {t}: KL {kl[t]:.2f} ref top1 {rid[t,0]} p={np.exp(rlp[t,0]):.3f}  cand top1 {cid[t,0]} p={np.exp(clp[t,0]):.3f}  actual {ids[t+1] if t+1<n else -1}  P(im_end) ref {pir[t]:.3f} cand {pic[t]:.3f}')

if __name__ == '__main__':
    main()
