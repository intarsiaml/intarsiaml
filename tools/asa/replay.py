#!/usr/bin/env python3
"""Lane asa: replay `--spec auto` policies on per-pass logs (INTARSIA_SPEC_AUTO=pass,log), on the CPU.

Inputs: raw/<W>/pass-<arm>-r<N>.log for arms autotr (real auto), autotrmtp (pure MTP passes), autotrdfl (pure DFlash passes).
Requests are split on the `spec auto:` line and mapped to the seven category files in order (4 2 4 2 8 8 8 requests).
Counterfactual: greedy output is the same in every mode, so the pass of a drafter that covers generated position `pos`
in its pure run is what that drafter would do there. A switch pays the measured extra ms of a fresh pass.
Usage: replay.py <raw/W> [--grid]
"""
import sys, glob, re, itertools, statistics, os

NEAREST = os.environ.get('NEAREST', '1') == '1'
CATS = [("ho-chat", 4), ("ho-code", 2), ("ho-prose", 4), ("ho-recall", 2), ("rcp-code", 8), ("rcp-prose", 8), ("rcp-recall", 8)]


def load(path):
    reqs, cur = [], []
    for ln in open(path, errors="replace"):
        if ln.startswith("spec pass "):
            _, _, m, t, ms, fr = ln.split()[:6]
            cur.append((m, int(t), float(ms), int(fr)))
        elif ln.startswith("spec auto:"):
            reqs.append(cur)
            cur = []
    return reqs


def cat_of(i):
    for c, n in CATS:
        if i < n:
            return c
        i -= n
    return None


class Seq:
    """One request in one pure mode: passes with the cumulative position."""
    def __init__(self, passes):
        self.p = passes
        self.cum = []
        c = 0
        for _, t, ms, _ in passes:
            c += t
            self.cum.append(c)
        self.total = c

    def at(self, pos):
        # the pass covering generated position pos (first pass index with cum > pos)
        lo, hi = 0, len(self.cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.cum[mid] > pos:
                hi = mid
            else:
                lo = mid + 1
        if NEAREST and lo + 1 < len(self.p):
            start = self.cum[lo] - self.p[lo][1]
            if 2 * (pos - start) > self.p[lo][1]:
                lo += 1  # nearest pass start instead of the covering pass: the covering one is length-biased
        return self.p[lo]


class Policy:
    """Mirror of spec::AutoPolicy plus the candidate knobs. prior=(tokens, ms) per drafter with weight prior_w pseudo-passes."""
    def __init__(self, seed=(2, 2), alpha=0.2, margin=1.03, probe_every=48, far_ratio=1.3, probe_far=2,
                 prior=None, prior_w=0.0, margin_leave_mtp=None, backoff=False, dwell=0, fixed_ms=None, probe_w=0.5):
        self.seed, self.alpha, self.margin = seed, alpha, margin
        self.pe, self.fr, self.pf = probe_every, far_ratio, probe_far
        self.mlm = margin_leave_mtp if margin_leave_mtp is not None else margin
        self.backoff, self.dwell, self.fixed_ms, self.probe_w = backoff, dwell, fixed_ms, probe_w
        self.n = [0, 0]
        self.tok = [None, None]
        self.ms = [None, None]
        if prior:
            for d in (0, 1):
                self.tok[d], self.ms[d] = prior[d]
        self.prior_w = prior_w
        self.cur = 0
        self.since = 0
        self.probing = False
        self.confirm = False
        self.last = None
        self.fresh = False
        self.probe_pass = False
        self.in_cur = 0
        self.mult = 1

    def rate(self, d):
        t, ms = self.tok[d], (self.fixed_ms[d] if self.fixed_ms else self.ms[d])
        return None if t is None or ms is None or ms <= 0 else t / ms

    def pick(self):
        d = self._pick()
        self.fresh = self.last != d
        self.probe_pass = self.probing and d != self.cur
        self.last = d
        return d

    def _pick(self):
        for d in (0, 1):
            if self.n[d] < self.seed[d]:
                return d
        if self.probing:
            if self.confirm:
                self.confirm = False
                return 1 - self.cur
            self.probing = False
            return self.cur
        rc, ro = self.rate(self.cur), self.rate(1 - self.cur)
        self.in_cur += 1
        if rc is not None and ro is not None:
            m = self.mlm if self.cur == 0 else self.margin
            if ro > rc * m and self.in_cur >= self.dwell:
                self.cur = 1 - self.cur
                self.since = 0
                self.in_cur = 0
                self.mult = 1
        lead, idle = self.rate(self.cur), self.rate(1 - self.cur)
        every = self.pe
        if lead is not None and idle is not None and lead > idle * self.fr:
            every = self.pe * self.pf
        if self.backoff:
            every = every * self.mult
        self.since += 1
        if self.since >= every:
            self.since = 0
            self.probing = True
            return 1 - self.cur
        return self.cur

    def record(self, d, tokens, ms):
        self.n[d] += 1
        cnt = self.n[d] + self.prior_w
        w = max(1.0 / cnt, self.alpha)
        if self.probe_pass:
            w = max(w, self.probe_w)
        if self.probe_pass and not self.confirm:
            lead = self.rate(1 - d)
            ms_est = self.ms[d] if self.ms[d] is not None else ms
            self.confirm = lead is not None and tokens / ms_est > lead * self.margin
        else:
            self.confirm = False
        if self.probe_pass and self.backoff and not self.confirm:
            self.mult = min(self.mult * 2, 8)
        self.tok[d] = tokens if self.tok[d] is None else self.tok[d] + w * (tokens - self.tok[d])
        if not self.fresh:
            self.ms[d] = ms if self.ms[d] is None else self.ms[d] + w * (ms - self.ms[d])
        self.fresh = False
        self.probe_pass = False


def simulate(pol, mt, df, extra):
    """-> (tokens, ms, dflash passes, passes) for one request; mt / df are Seq."""
    pos, ms_tot, nd, np_ = 0, 0.0, 0, 0
    end = min(mt.total, df.total)
    last = None
    while pos < end:
        d = pol.pick()
        _, t, ms, _ = (mt if d == 0 else df).at(pos)
        c = ms + (extra[d] if last is not None and last != d else 0.0)
        pol.record(d, t, ms)  # fresh pass: record() skips its ms, the tokens count
        pos += t
        ms_tot += c
        nd += d
        np_ += 1
        last = d
    return min(pos, end), ms_tot, nd, np_


def run(args):
    raw = args[0]
    rounds = sorted({int(re.search(r"-r(\d+)\.log", f).group(1)) for f in glob.glob(f"{raw}/pass-autotr-r*.log")})
    global REAL, ROUNDS_G
    REAL = {}
    ROUNDS_G = rounds
    data = {}  # (round) -> {cat: [(mt, df)]}
    extra_acc = {0: [], 1: []}
    base_ms = {0: [], 1: []}
    for r in rounds:
        a, m, d = (load(f"{raw}/pass-{x}-r{r}.log") for x in ("autotr", "autotrmtp", "autotrdfl"))
        assert len(a) == len(m) == len(d) == 36, (len(a), len(m), len(d))
        data[r] = {}
        REAL[r] = {}
        for i in range(36):
            REAL[r].setdefault(cat_of(i), []).append(a[i])
        for i in range(36):
            data[r].setdefault(cat_of(i), []).append((Seq(m[i]), Seq(d[i])))
        for req in a:
            ms_non = {0: [], 1: []}
            for j, (mo, t, ms, fr) in enumerate(req):
                k = 0 if mo == "M" else 1
                (extra_acc if fr and j > 0 else ms_non)[k].append(ms)
            for k in (0, 1):
                base_ms[k] += ms_non[k]
    extra = {}
    for k in (0, 1):
        extra[k] = max(0.0, statistics.mean(extra_acc[k]) - statistics.mean(base_ms[k])) if extra_acc[k] else 0.0
    print(f"switch extra ms: to MTP {extra[0]:.1f}, to DFlash {extra[1]:.1f}; pass ms mtp {statistics.mean(base_ms[0]):.1f} dfl {statistics.mean(base_ms[1]):.1f}")

    def evaluate(mk, use=None, par=None):
        res = {}
        for c, _ in CATS:
            tk = ms = nd = npass = 0
            for r in (use or rounds):
                for qi, (mt, df) in enumerate(data[r][c]):
                    if par is not None and qi % 2 != par:
                        continue
                    t, m, d_, n_ = simulate(mk(), mt, df, extra)
                    tk += t; ms += m; nd += d_; npass += n_
            res[c] = (tk / ms * 1000, nd / npass)
        return res

    # pure references through the same machinery
    ref = {"mtp": evaluate(pure(0)), "dfl": evaluate(pure(1))}
    ref_by = {r: {"mtp": evaluate(pure(0), [r]), "dfl": evaluate(pure(1), [r])} for r in rounds}
    ref['by'] = ref_by
    # the real auto run, same units
    real = {}
    for c, _ in CATS:
        tk = ms = nd = npass = 0
        for r in rounds:
            for req in REAL[r][c]:
                for mo, t, m, fr in req:
                    tk += t; ms += m; nd += mo == "D"; npass += 1
        real[c] = (tk / ms * 1000, nd / npass)
    ref["real"] = real
    return evaluate, ref


def pure(d):
    class P(Policy):
        def _pick(self_): return d
    return lambda: P()


def table(name, res, ref):
    row = [name.ljust(34)]
    worst, tot = 9.0, 0
    for c, _ in CATS:
        best = max(ref["mtp"][c][0], ref["dfl"][c][0])
        v = res[c][0]
        worst = min(worst, v / ref["mtp"][c][0])
        row.append(f"{v / best * 100 - 100:+5.1f}")
    row.append(f"| minVsMTP {100 * worst - 100:+5.1f} dflshare " + " ".join(f"{res[c][1] * 100:2.0f}" for c, _ in CATS))
    print(" ".join(row))
    return worst


def main():
    evaluate, ref = run(sys.argv[1:])
    print("tok/s (sim)   " + "  ".join(c for c, _ in CATS))
    for k in ("mtp", "dfl"):
        print(k.ljust(5), " ".join(f"{ref[k][c][0]:7.1f}" for c, _ in CATS))
    print("rows: % vs per-category best of pure mtp/dfl")
    print(" " * 34 + " ".join(c[:6].rjust(5) for c, _ in CATS))
    BEST = dict(alpha=0.1, margin=1.03, margin_leave_mtp=1.08, probe_every=48, probe_far=4, prior=((2.4, 25.5), (3.0, 38.0)),
                prior_w=3, probe_w=0.2, dwell=8)
    cands = {
        "current": dict(),
        "P1 priors w3 a.1 leave1.08 dwell8": BEST,
        "P1 with probe_w .5": dict(BEST, probe_w=0.5),
        "P1 no dwell": dict(BEST, dwell=0),
        "P1 alpha .2": dict(BEST, alpha=0.2),
        "P1 no priors": dict(BEST, prior=None, prior_w=0),
        "P1 seed 4,2": dict(BEST, seed=(4, 2)),
        "P1 pe 96": dict(BEST, probe_every=96),
    }
    # split half: policies tuned on the other round are reported per round
    for r in sorted(ref["by"]):
        for n in ("current", "P1 priors w3 a.1 leave1.08 dwell8"):
            res = evaluate(lambda kw=cands[n]: Policy(**kw), [r])
            refr = {"mtp": ref["by"][r]["mtp"], "dfl": ref["by"][r]["dfl"]}
            table(f"{n} [round {r}]", res, refr)
    table("REAL auto (logged)", ref["real"], ref)
    if "--nbr" in sys.argv:
        for alpha, leave, dw, pw, seed, pwt in itertools.product((0.05, 0.1), (1.03, 1.08), (8, 16), (3, 6), ((2, 2), (4, 2)), (0.1, 0.2)):
            kw = dict(alpha=alpha, margin=1.03, margin_leave_mtp=leave, probe_every=48, probe_far=4, prior=((2.4, 25.5), (3.0, 38.0)),
                      prior_w=pw, probe_w=pwt, dwell=dw, seed=seed)
            table(f"a{alpha} l{leave} d{dw} pw{pw} s{seed[0]} w{pwt}", evaluate(lambda kw=kw: Policy(**kw)), ref)
        return
    for n, kw in cands.items():
        table(n, evaluate(lambda kw=kw: Policy(**kw)), ref)

    if "--c" in sys.argv:
        C = dict(theta=2.5, alpha_m=0.06, dwell_m=4, dwell_d=16, hyst=1.1, probe_mtp=8, alpha_d=0.1, probe_w=0.3)
        for n, kw in [("C", C), ("theta2.4", dict(C, theta=2.4)), ("theta2.6", dict(C, theta=2.6)), ("probe_w.15", dict(C, probe_w=0.15)),
                      ("probe_w.5", dict(C, probe_w=0.5)), ("pm12", dict(C, probe_mtp=12)), ("pm6", dict(C, probe_mtp=6)), ("am.1", dict(C, alpha_m=0.1)),
                      ("am.04", dict(C, alpha_m=0.04)), ("dd8", dict(C, dwell_d=8)), ("dd24", dict(C, dwell_d=24)), ("hyst1.0", dict(C, hyst=1.0)),
                      ("hyst1.2", dict(C, hyst=1.2)), ("bump.3 dec.005", dict(C, bump=0.3)), ("bump.3 dec.02", dict(C, bump=0.3, decay=0.02)),
                      ("bump.5 dec.01", dict(C, bump=0.5, decay=0.01)), ("bump.2 dec.002", dict(C, bump=0.2, decay=0.002)), ("bump.5 dec.003", dict(C, bump=0.5, decay=0.003)), ("dm8", dict(C, dwell_m=8)), ("ad.15", dict(C, alpha_d=0.15))]:
            table(n, evaluate(lambda kw=kw: Policy2(**kw)), ref)
        return
    if "--p2" in sys.argv:
        out = []
        for th, am, dwm, dwd, hy, pm, ad in itertools.product((2.5, 2.6, 2.7, 2.8), (0.05, 0.08, 0.12), (4, 8), (4, 8, 16), (1.0, 1.1, 1.2), (8, 16, 24), (0.08, 0.15)):
            kw = dict(theta=th, alpha_m=am, dwell_m=dwm, dwell_d=dwd, hyst=hy, probe_mtp=pm, alpha_d=ad)
            res = evaluate(lambda kw=kw: Policy2(**kw))
            vb = [res[c][0] / max(ref["mtp"][c][0], ref["dfl"][c][0]) * 100 - 100 for c, _ in CATS]
            vm = [res[c][0] / ref["mtp"][c][0] * 100 - 100 for c, _ in CATS]
            out.append((min(vb) + sum(vb) / 7, min(vb), sum(vb) / 7, min(vm), kw, vb))
        out.sort(key=lambda r: -r[0])
        for o in out[:12]:
            print(f"score {o[0]:+.2f} minVsBest {o[1]:+.1f} mean {o[2]:+.1f} minVsMTP {o[3]:+.1f}", o[4], " ".join(f"{x:+.1f}" for x in o[5]))
        return
    if "--grid" in sys.argv:
        rounds = ROUNDS_G
        PAR = int(os.environ.get("PAR", "-1"))
        PAR = None if PAR < 0 else PAR
        rp = {"mtp": evaluate(pure(0), [rounds[0]], PAR), "dfl": evaluate(pure(1), [rounds[0]], PAR)}
        rt = {"mtp": evaluate(pure(0), [rounds[0]], None if PAR is None else 1 - PAR), "dfl": evaluate(pure(1), [rounds[0]], None if PAR is None else 1 - PAR)}
        FX = {0: 25.5, 1: 38.0}
        rows = []
        for alpha, mg, ml, pe, pf, seed, pw, fx, pwt, dw in itertools.product(
                (0.05, 0.1, 0.2), (1.03, 1.08), (1.03, 1.08, 1.15), (48, 96), (2, 4), ((2, 2), (4, 2)), (0, 3), (False, True),
                (0.1, 0.2, 0.5), (0, 8)):
            pri = ((2.4, 25.5), (3.0, 38.0)) if pw else None
            kw = dict(alpha=alpha, margin=mg, margin_leave_mtp=ml, probe_every=pe, probe_far=pf, seed=seed, prior=pri, prior_w=pw,
                      fixed_ms=(FX if fx else None), probe_w=pwt, dwell=dw)
            res = evaluate(lambda kw=kw: Policy(**kw), [rounds[0]], PAR)
            vb = [res[c][0] / max(rp["mtp"][c][0], rp["dfl"][c][0]) for c, _ in CATS]
            vm = [res[c][0] / rp["mtp"][c][0] for c, _ in CATS]
            rows.append((min(vb), sum(vb) / 7, min(vm), kw))
        rows.sort(key=lambda r: -(r[0] + r[1]))
        for mn, mean, mm, kw in rows[:8]:
            tr = evaluate(lambda kw=kw: Policy(**kw), [rounds[0]], None if PAR is None else 1 - PAR)
            tv = [tr[c][0] / max(rt["mtp"][c][0], rt["dfl"][c][0]) for c, _ in CATS]
            tm = [tr[c][0] / rt["mtp"][c][0] for c, _ in CATS]
            print(f"  heldout: minVsBest {min(tv) * 100 - 100:+.1f} meanVsBest {sum(tv) / 7 * 100 - 100:+.1f} minVsMTP {min(tm) * 100 - 100:+.1f}")
            print(f"minVsBest {mn * 100 - 100:+.1f} meanVsBest {mean * 100 - 100:+.1f} minVsMTP {mm * 100 - 100:+.1f}", {k: v for k, v in kw.items() if v not in (None, False)})



class Policy2:
    """MTP acceptance as the free covariate: DFlash is entered when MTP's tokens per pass reach `theta` (DFlash wins 96% of the
    time above 2.6 and loses 87% below 2.4 per 96-token block), left when its measured rate falls under MTP's (last known) rate."""
    def __init__(self, theta=2.5, alpha_m=0.12, alpha_d=0.12, dwell_m=8, dwell_d=8, hyst=1.0, probe_mtp=24, probe_w=0.3,
                 m0=(2.2, 25.5), d0=(3.6, 38.0), prior_w=2.0, seed_m=2, probe_dfl=0, dfl_win=1.6, bump=0.0, decay=0.005):
        self.__dict__.update(locals())
        del self.__dict__["self"]
        self.n = [0, 0]
        self.tok = [m0[0], d0[0]]
        self.ms = [m0[1], d0[1]]
        self.cur = 0
        self.in_cur = 0
        self.since = 0
        self.fresh = False
        self.last = None
        self.probe_pass = False
        self.probing = False
        self.dpr = 0
        self.extra = 0.0

    def rate(self, d):
        return self.tok[d] / self.ms[d]

    def pick(self):
        d = self._pick()
        self.fresh = self.last != d
        self.probe_pass = self.probing and d != self.cur
        self.last = d
        return d

    def _pick(self):
        if self.n[0] < self.seed_m:
            return 0
        if self.probing:
            self.probing = False
            return self.cur
        self.in_cur += 1
        if self.cur == 0:
            self.extra = max(0.0, self.extra - self.decay)
            if self.tok[0] >= self.theta + self.extra and self.in_cur >= self.dwell_m:
                self.cur, self.in_cur, self.since = 1, 0, 0
                if self.n[1] == 0:
                    self.tok[1] = 1.5 * self.tok[0] * self.ms[1] / self.ms[0]  # enter neutral-ish: win needs real tokens
                return 1
            if self.probe_dfl:
                self.since += 1
                if self.since >= self.probe_dfl:
                    self.since = 0
                    self.probing = True
                    return 1
            return 0
        # on DFlash
        if self.rate(1) < self.rate(0) / self.hyst and self.in_cur >= self.dwell_d:
            self.cur, self.in_cur, self.since = 0, 0, 0
            self.extra = self.bump
            return 0
        self.since += 1
        if self.since >= self.probe_mtp:
            self.since = 0
            self.probing = True
            return 0
        return 1

    def record(self, d, tokens, ms):
        self.n[d] += 1
        a = self.alpha_m if d == 0 else self.alpha_d
        w = max(1.0 / (self.n[d] + self.prior_w), a)
        if self.probe_pass:
            w = max(w, self.probe_w)
        self.tok[d] += w * (tokens - self.tok[d])
        if not self.fresh:
            self.ms[d] += w * (ms - self.ms[d])
        self.fresh = False
        self.probe_pass = False


if __name__ == '__main__':
    main()
