#!/usr/bin/env python3
"""Question set 2 for the DSv4.1 chunk 512 vs 2048 long-context check (docs/results/2026-11-04-dsv41-chunk32k-quality-2.md).

Why a second set: set 1 (52 items) was half saturated (24 single-fact items right in both arms, 9 of 12 two-hop items wrong in
both) and tested lookups, not facts spread over distant blocks. Here every item needs SEVERAL facts from different places of a
~31K or ~64K token document, and the reply may show a few lines of working (thinking stays off) before a final line
`FINAL: <answer>`. The expected string is `FINAL: <answer>` (localeval "Contains expected": case-insensitive, whitespace
collapsed substring); answers never contain quotes (no quote artifact). Pilot lesson: with `FINAL: [answer]` the model dropped the
brackets in 30 of 30 replies, so the pilot's localeval score was 0 and was re-scored offline; brackets are not used. A wrong answer that
extends the right one ("FINAL: 14730" for 1473) would pass the substring check, so the analysis also reports an exact parse of the FINAL line.

Deterministic (fixed seeds, source text from git commit SOURCE_COMMIT). Every expected answer is re-derived from the finished
document text by an independent parser (`solve`), and entity words must not occur in the filler prose.

  uv run --no-project --with tokenizers python gen_items2.py --spec SPEC.json --out PATH-WITHOUT-EXTENSION

SPEC.json: {"seed": "final", "items": [{"kind": "hop2", "level": 2, "n": 10, "k": 32}, ...], "shard32": 25, "shard65": 10}
Kinds (levels 1..3 = harder):
  hop2     alias -> release -> rollback token. L1 facts 1-3 % of the document apart, 3 decoy aliases; L2 25-40 % apart, 6 decoys;
           L3 >= 60 % apart, 10 decoys. Order of the two facts is random.
  hop3     alias -> release -> owning team -> team token (3 facts). L1 within a 5 % window, 4 decoy chains; L2 anywhere, 8; L3 about 25 % or more apart, 12.
  agg_sum  total units of all shards of one cluster: k = 3 / 5 / 8 / 12 (L4) facts spread over the document, 10 / 20 / 30 / 40 decoy shards of other clusters
  agg_max  which release of a line had the most failed tests: k = 4 / 6 / 8, 10 / 20 / 30 decoy facts of other lines
  repo_sum sum of the values of integer constants named in k = 2 / 3 / 4 / 6 (L4) different Rust files of the document
"""
import argparse, hashlib, importlib.util, json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec_ = importlib.util.spec_from_file_location("gen_items", os.path.join(HERE, "../../2026-11-03-dsv41-chunk32k-quality/raw/gen_items.py"))
g1 = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(g1)   # load_pool, Tok, word, code (set 1's helpers, same pinned commit)
import subprocess
g1.REPO = subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True).stdout.strip()  # any checkout holding SOURCE_COMMIT

INTC = re.compile(r"^[ \t]*(?:pub(?:\([a-z]+\))? )?const ([A-Z][A-Z0-9_]+): (?:usize|u32|u64|i32|i64|u16|u8) = (\d[\d_]*);[ \t]*(?://.*)?$", re.M)
TARGET = {32: (30000, 30900), 65: (63500, 64300)}
LEAD = ["Release ledger", "Ops ledger", "Change log", "Tracker note", "Registry entry"]


def intconsts(text):
    d = {}
    for m in INTC.finditer(text):
        d.setdefault(m.group(1), []).append(int(m.group(2).replace("_", "")))
    return {k: v[0] for k, v in d.items() if len(v) == 1 and v[0] < 100000}


class Ctx:
    def __init__(self, tok, paras, rs, rng, k):
        self.tok, self.paras, self.rs, self.rng, self.k = tok, paras, rs, rng, k
        self.lo, self.hi = TARGET[k]
        cand = rng.sample(paras, min(len(paras), 1100 if k == 32 else 2400))
        cnt = tok.many(cand)
        self.fill, tot = [], 0
        for p, c in zip(cand, cnt):
            if tot >= self.hi + 400:
                break
            self.fill.append(p)
            tot += c + 1
        self.lower = "\n".join(self.fill).lower()
        self.usedw, self.usedc = set(), set()

    def word(self):
        while True:
            w = g1.word(self.rng, self.usedw)
            if w.lower() not in self.lower:
                return w
            # keep it used so that it is not drawn again

    def code(self):
        return g1.code(self.rng, self.usedc)

    def assemble(self, inserts):
        """Filler prose with the given [(depth, text)] inserted at paragraph boundaries, trimmed to the token window."""
        fill = list(self.fill)
        cum, acc = [], 0
        for p in fill:
            cum.append(acc)
            acc += len(p) + 2
        items = list(fill)
        marks = []
        for depth, text in sorted(inserts, key=lambda x: -x[0]):
            i = next((j for j, c in enumerate(cum) if c >= depth * acc), len(fill) - 1)
            items.insert(i, text)
            marks.append(text)
        while True:
            doc = "\n\n".join(items)
            n = self.tok.n(doc)
            if n <= self.hi:
                break
            j = len(items) - 1
            while items[j] in marks:
                j -= 1
            cut = max(40, int((n - self.hi) * 3.5))   # shave the tail paragraph instead of dropping it whole
            if len(items[j]) <= cut + 80:
                del items[j]
            else:
                items[j] = items[j][:-cut].rstrip()
        assert n >= self.lo, n
        return doc, n


def lead(rng):
    return rng.choice(LEAD)


def clamp(x):
    return min(0.96, max(0.04, x))


def mk_hop2(cx, level):
    r = cx.rng
    n_dec = {1: 3, 2: 6, 3: 10}[level]
    alias, name, tok_ = cx.word(), cx.word(), cx.code()
    gap = {1: r.uniform(0.01, 0.03), 2: r.uniform(0.25, 0.4), 3: r.uniform(0.6, 0.85)}[level]
    if r.random() < .5:
        dA = r.uniform(0.04, 0.96 - gap)
        dB = dA + gap
    else:
        dA = r.uniform(0.04 + gap, 0.96)
        dB = dA - gap
    A = f"{lead(r)}: the release nicknamed {alias} is tracked in this ledger as the {name} release."
    B = f"{lead(r)}: the rollback token of the {name} release was set to {tok_}."
    ins = [(dA, A), (dB, B)]
    for _ in range(n_dec):
        a2, n2 = cx.word(), cx.word()
        ins.append((r.uniform(0.03, 0.97), f"{lead(r)}: the release nicknamed {a2} is tracked in this ledger as the {n2} release."))
        ins.append((r.uniform(0.03, 0.97), f"{lead(r)}: the rollback token of the {n2} release was set to {cx.code()}."))
    q = f"What is the rollback token of the release nicknamed {alias}?"
    return ins, q, tok_, {"alias": alias, "facts": [A, B]}


def mk_hop3(cx, level):
    r = cx.rng
    n_dec = {1: 4, 2: 8, 3: 12}[level]
    alias, name, team, tok_ = cx.word(), cx.word(), cx.word(), cx.code()
    F = [f"{lead(r)}: the release nicknamed {alias} is tracked in this ledger as the {name} release.",
         f"{lead(r)}: the {name} release is owned by the {team} team.",
         f"{lead(r)}: the rollback token of the {team} team was set to {tok_}."]
    if level == 1:
        c = r.uniform(0.06, 0.9)
        ds = [clamp(c + r.uniform(0, 0.05)) for _ in F]
    elif level == 2:
        ds = [r.uniform(0.04, 0.96) for _ in F]
    else:
        base = sorted(r.uniform(0.04, 0.2) + i * 0.38 for i in range(3))
        ds = [clamp(b + r.uniform(0, 0.06)) for b in base]
        r.shuffle(ds)
    ins = list(zip(ds, F))
    for _ in range(n_dec):
        a2, n2, t2 = cx.word(), cx.word(), cx.word()
        full = r.random() < 0.5
        ins.append((r.uniform(0.03, 0.97), f"{lead(r)}: the release nicknamed {a2} is tracked in this ledger as the {n2} release."))
        ins.append((r.uniform(0.03, 0.97), f"{lead(r)}: the {n2} release is owned by the {t2} team."))
        if full:
            ins.append((r.uniform(0.03, 0.97), f"{lead(r)}: the rollback token of the {t2} team was set to {cx.code()}."))
    q = f"What is the rollback token of the team that owns the release nicknamed {alias}?"
    return ins, q, tok_, {"alias": alias, "facts": F}


def mk_agg_sum(cx, level):
    r = cx.rng
    k, n_dec = {1: (3, 10), 2: (5, 20), 3: (8, 30), 4: (12, 40)}[level]
    grp = cx.word()
    vals = [r.randrange(100, 1000) for _ in range(k)]
    ins, facts = [], []
    ids = set()
    for v in vals:
        sid = cx.code()
        t = f"{lead(r)}: storage shard {sid} of the {grp} cluster holds {v} units."
        facts.append(t)
        ins.append((r.uniform(0.04, 0.96), t))
    for _ in range(n_dec):
        ins.append((r.uniform(0.03, 0.97), f"{lead(r)}: storage shard {cx.code()} of the {cx.word()} cluster holds {r.randrange(100, 1000)} units."))
    q = f"What is the total number of units held by all storage shards of the {grp} cluster? Give the sum as digits only."
    return ins, q, str(sum(vals)), {"group": grp, "facts": facts}


def mk_agg_max(cx, level):
    r = cx.rng
    k, n_dec = {1: (4, 10), 2: (6, 20), 3: (8, 30)}[level]
    grp = cx.word()
    vals = r.sample(range(10, 400), k)
    names = [cx.word() for _ in range(k)]
    ins, facts = [], []
    for nm, v in zip(names, vals):
        t = f"{lead(r)}: release {nm} of the {grp} line recorded {v} failed tests."
        facts.append(t)
        ins.append((r.uniform(0.04, 0.96), t))
    for _ in range(n_dec):
        ins.append((r.uniform(0.03, 0.97), f"{lead(r)}: release {cx.word()} of the {cx.word()} line recorded {r.randrange(10, 400)} failed tests."))
    ans = names[vals.index(max(vals))]
    q = f"Which release of the {grp} line recorded the most failed tests? Give the release name."
    return ins, q, ans, {"group": grp, "facts": facts}


def mk_repo_sum(cx, level):
    """Integer constants of k different Rust files of the document; returns the document directly."""
    r, tok = cx.rng, cx.tok
    k = {1: 2, 2: 3, 3: 4, 4: 6}[level]
    order = r.sample(cx.rs, len(cx.rs))
    cnt = tok.many([f"=== FILE: {p} ===\n{t}" for p, t in order])
    ntok = {p: c for (p, _), c in zip(order, cnt)}
    text = dict(cx.rs)
    cands = [p for p, t in order if intconsts(t) and ntok[p] <= (4500 if k >= 6 else 7000)]
    tps = cands[:k]
    terms = []
    for p in tps:
        ic = intconsts(text[p])
        nm = r.choice(sorted(ic))
        terms.append((p, nm, ic[nm]))
    chosen, tot = list(tps), sum(ntok[p] for p in tps)
    for p, _ in order:
        if p in chosen:
            continue
        if tot + ntok[p] > cx.hi + 100:
            continue
        chosen.append(p)
        tot += ntok[p]
        if tot >= cx.hi - 100:
            break
    r.shuffle(chosen)
    doc = "\n\n".join(f"=== FILE: {p} ===\n{text[p]}" for p in chosen)
    n = tok.n(doc)
    assert cx.lo - 400 <= n <= cx.hi + 300, n
    ans = sum(v for _, _, v in terms)
    q = ("Add up the values of these integer constants: " + "; ".join(f"{nm} in the file {p}" for p, nm, _ in terms)
         + ". Give the sum as digits only.")
    return doc, n, q, str(ans), {"terms": [{"file": p, "const": nm, "value": v} for p, nm, v in terms]}


# ---- independent re-derivation of every answer from the finished document text ----
def solve(kind, doc, q, meta):
    if kind == "hop2":
        alias = meta["alias"]
        names = re.findall(rf"the release nicknamed {alias} is tracked in this ledger as the (\w+) release", doc)
        assert len(names) == 1
        toks = re.findall(rf"the rollback token of the {names[0]} release was set to ([A-Z]{{2}}-\d{{4}}-[A-Z]{{2}})", doc)
        assert len(toks) == 1
        return toks[0]
    if kind == "hop3":
        alias = meta["alias"]
        names = re.findall(rf"the release nicknamed {alias} is tracked in this ledger as the (\w+) release", doc)
        assert len(names) == 1
        teams = re.findall(rf"the {names[0]} release is owned by the (\w+) team", doc)
        assert len(teams) == 1
        toks = re.findall(rf"the rollback token of the {teams[0]} team was set to ([A-Z]{{2}}-\d{{4}}-[A-Z]{{2}})", doc)
        assert len(toks) == 1
        return toks[0]
    if kind == "agg_sum":
        vs = re.findall(rf"storage shard [A-Z]{{2}}-\d{{4}}-[A-Z]{{2}} of the {meta['group']} cluster holds (\d+) units", doc)
        return str(sum(int(v) for v in vs))
    if kind == "agg_max":
        fs = re.findall(rf"release (\w+) of the {meta['group']} line recorded (\d+) failed tests", doc)
        return max(fs, key=lambda x: int(x[1]))[0]
    if kind == "repo_sum":
        tot = 0
        for t in meta["terms"]:
            m = re.search(rf"=== FILE: {re.escape(t['file'])} ===\n(.*?)(?=\n\n=== FILE: |\Z)", doc, re.S)
            tot += intconsts(m.group(1))[t["const"]]
        return str(tot)
    raise KeyError(kind)


MK = {"hop2": mk_hop2, "hop3": mk_hop3, "agg_sum": mk_agg_sum, "agg_max": mk_agg_max}
INSTR = ("Use only the document. You may write up to four short lines of working (quote the facts you use). Then end with a last line "
         "of exactly this form, in plain text without markdown or brackets, with nothing after the answer: FINAL: <answer>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    tok = g1.Tok()
    paras, rs = g1.load_pool()
    paras = [p for p in paras if "FINAL:" not in p]
    # item list: per size, round robin over the (kind, level) groups so that every shard is balanced
    groups = {}
    for it in spec["items"]:
        groups.setdefault(it["k"], []).append(it)
    order = []
    for k in sorted(groups):
        queues = [[(it["kind"], it["level"], k, i) for i in range(it["n"])] for it in groups[k]]
        while any(queues):
            for qu in queues:
                if qu:
                    order.append(qu.pop(0))
    rows, man = [], []
    for idx, (kind, level, k, i) in enumerate(order):
        rng = random.Random(f"chunk32k2-{spec['seed']}-{kind}-{level}-{k}-{i}")
        cx = Ctx(tok, paras, rs, rng, k)
        if kind == "repo_sum":
            doc, n, q, ans, meta = mk_repo_sum(cx, level)
            facts_d = []
            for t in meta["terms"]:
                facts_d.append(doc.index(f"=== FILE: {t['file']} ===") / len(doc))
        else:
            ins, q, ans, meta = MK[kind](cx, level)
            doc, n = cx.assemble(ins)
            facts_d = [doc.index(t) / len(doc) for t in meta["facts"]]
            for t in meta["facts"]:
                assert doc.count(t) == 1
        assert solve(kind, doc, q, meta) == ans, (kind, level, i, ans)
        assert ans not in q
        facts_d.sort()
        prompt = (f"Run id chunk32k2-{spec['seed']}-{idx:03d}-{kind}-L{level}-{k}k.\n\nBelow is a long document. Read it carefully, then answer the question after it.\n\n"
                  f"<document>\n{doc}\n</document>\n\nQuestion: {q}\n\n{INSTR}")
        name = f"{kind}-L{level}-{k}k-{i:03d}"
        exp = f"FINAL: {ans}"
        assert prompt.count(exp) == 0
        rows.append({"name": name, "input": prompt, "expected_output": exp})
        man.append({"idx": idx, "name": name, "kind": kind, "level": level, "k": k, "nfacts": len(facts_d), "doc_tokens": n,
                    "depths": [round(d, 3) for d in facts_d], "span": round(facts_d[-1] - facts_d[0], 3), "depth": round(facts_d[0], 3),
                    "prompt_tokens_est": tok.n(prompt), "expected": exp, "question": q, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
        print(f"{idx:3d} {name:26s} {man[-1]['prompt_tokens_est']:6d} tok span {man[-1]['span']:.2f} ans={ans}", file=sys.stderr)
    assert len({r["name"] for r in rows}) == len(rows)
    assert len({r["input"][:400] for r in rows}) == len(rows)
    with open(a.out + ".jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    json.dump({"source_commit": g1.SOURCE_COMMIT, "spec": spec, "items": man}, open(a.out + ".manifest.json", "w"), indent=1, ensure_ascii=False)
    # shards: 32K items in shards of spec.shard32, 65K items in shards of spec.shard65
    sh = []
    for k, size in ((32, spec.get("shard32", 25)), (65, spec.get("shard65", 10))):
        sel = [r for r, m in zip(rows, man) if m["k"] == k]
        for s in range(0, len(sel), size):
            sh.append((f"s{k}-{s // size:02d}", sel[s:s + size]))
    for nm, sl in sh:
        with open(f"{a.out}.{nm}.jsonl", "w") as f:
            for r in sl:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    shas = {"set": hashlib.sha256(open(a.out + ".jsonl", "rb").read()).hexdigest()}
    for nm, _ in sh:
        shas[nm] = hashlib.sha256(open(f"{a.out}.{nm}.jsonl", "rb").read()).hexdigest()
    json.dump(shas, open(a.out + ".sha256.json", "w"), indent=1)
    for kx, v in shas.items():
        print(kx, v, file=sys.stderr)


if __name__ == "__main__":
    main()
