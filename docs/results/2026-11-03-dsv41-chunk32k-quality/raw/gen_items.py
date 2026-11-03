#!/usr/bin/env python3
"""Question set for the DSv4.1 chunk 512 vs 2048 long-context check (docs/results/2026-11-03-dsv41-chunk32k-quality.md).

Deterministic: fixed seeds, all source text read from git commit SOURCE_COMMIT (not the working tree), so the output
file has a fixed sha256. Writes
  <out>.jsonl     localeval custom-dataset rows {"name", "input", "expected_output"}  (about 6 MB, not committed)
  <out>.manifest.json   per item: kind, depths, tokens, expected answer, prompt sha256
Run:  uv run --no-project --with tokenizers python gen_items.py --out PATH-WITHOUT-EXTENSION
Needs the DSv4.1 tokenizer (Mia): TOKENIZER below.

Kinds (52 items; every prompt is a different ~31K-token document, scored by "expected answer appears in the reply"):
  needle      one unique fact ("rollback token of release X is CODE") inserted into repo prose, depth 4..96 %
  needle_dist twelve near-identical facts (different release names and codes), ask for one, target depth 4..96 %
  needle_hop  two hops: "release nicknamed A is tracked as N" and "rollback token of release N is CODE", plus three
              decoy aliases and six decoy tokens; ask for the token of the release nicknamed A
  qa_const    a long concatenation of this repo's Rust sources; ask for the value of a `const NAME` in a named file
              (names that are also declared in other files of the same document are preferred)
Items are ordered round robin over the kinds, so items 0..11 (the "pilot shard") hold 3 of each kind.
"""
import argparse, hashlib, json, random, re, subprocess, sys

SOURCE_COMMIT = "47af187778c025a2379bc3924f075db910afadd2"
REPO = "/opt/code/intarsiaml/.worktrees/chunk32k"
TOKENIZER = "/opt/code/intarsiaml/.worktrees/dsv41-official/mia/tokenizer.json"
TARGET_MAX = 30900   # document tokens; + question + chat template stays under 31.2K (engine CAP is 33000, max_tokens 48)
TARGET_MIN = 30000
N_PER_KIND = {"needle": 12, "needle_dist": 12, "needle_hop": 12, "qa_const": 16}
SYL = ["bar", "cal", "der", "fen", "gor", "hal", "jor", "kel", "lum", "mar", "nov", "orv", "pel", "quin", "ros", "sel",
       "tor", "ulm", "vex", "wyn", "yar", "zan", "bri", "cor", "dun", "eld", "fal", "gwen", "hox", "ilk"]


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], check=True, capture_output=True, text=True).stdout


def load_pool():
    docs = [p for p in git("ls-tree", "-r", "--name-only", SOURCE_COMMIT).splitlines()
            if p.startswith("docs/") and p.endswith(".md") and "/raw/" not in p]
    paras = []
    for p in sorted(docs):
        for blk in re.split(r"\n\s*\n", git("show", f"{SOURCE_COMMIT}:{p}")):
            blk = blk.strip()
            if 80 <= len(blk) <= 2500 and "rollback token" not in blk and "nicknamed" not in blk:
                paras.append(blk)
    paras = sorted(set(paras))
    rs = []
    for p in sorted(x for x in git("ls-tree", "-r", "--name-only", SOURCE_COMMIT).splitlines() if x.endswith(".rs")):
        t = git("show", f"{SOURCE_COMMIT}:{p}")
        if 3000 <= len(t) <= 60000 and p.startswith("crates/"):
            rs.append((p, t))
    return paras, rs


USED_NAMES = set()  # (file, constant) pairs asked so far; names not asked yet are preferred
CONST = re.compile(r"^[ \t]*(?:pub(?:\([a-z]+\))? )?const ([A-Z][A-Z0-9_]{3,}): [^=\n]+= ([^;\n]{4,40});[ \t]*(?://.*)?$", re.M)


def consts(text):
    d = {}
    for m in CONST.finditer(text):
        d.setdefault(m.group(1), []).append(m.group(2).strip())
    return {k: v[0] for k, v in d.items() if len(v) == 1}


class Tok:
    def __init__(self):
        from tokenizers import Tokenizer
        self.t = Tokenizer.from_file(TOKENIZER)

    def n(self, s):
        return len(self.t.encode(s, add_special_tokens=False).ids)

    def many(self, xs):
        return [len(e.ids) for e in self.t.encode_batch(xs, add_special_tokens=False)]


def word(rng, used):
    while True:
        w = "".join(rng.choice(SYL) for _ in range(2)).capitalize()
        if w.lower() not in used and not any(w.lower() in u or u in w.lower() for u in used):
            used.add(w.lower())
            return w


def code(rng, used):
    while True:
        c = f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}-{rng.randrange(1000, 10000)}-{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}"
        if c not in used:
            used.add(c)
            return c


def fact(name, c):
    return f"Release ledger: the rollback token of the {name} release was set to {c}."


def build_prose(tok, rng, paras, inserts):
    """~31K-token document of repo prose with `inserts` [(depth, text)] placed at paragraph boundaries."""
    cand = rng.sample(paras, 900)
    cnt = tok.many(cand)
    fill, tot = [], 0
    for p, c in zip(cand, cnt):
        if tot >= TARGET_MAX + 400:
            break
        fill.append(p)
        tot += c + 1
    # insert needles by character depth of the filler-only text (the real depth is measured afterwards)
    cum, acc = [], 0
    for p in fill:
        cum.append(acc)
        acc += len(p) + 2
    items = list(fill)
    marks = []
    for depth, text in sorted(inserts, key=lambda x: -x[0]):  # from the back so earlier indexes stay valid
        i = next((k for k, c in enumerate(cum) if c >= depth * acc), len(fill) - 1)
        items.insert(i, text)
        marks.append(text)
    # trim the tail (never a needle) until the document fits
    while True:
        doc = "\n\n".join(items)
        n = tok.n(doc)
        if n <= TARGET_MAX:
            break
        j = len(items) - 1
        while items[j] in marks:
            j -= 1
        del items[j]
    assert n >= TARGET_MIN, n
    return doc, n


def build_code(tok, rng, rs, depth):
    """Concatenated Rust files; returns (doc, tokens, path, name, value, actual depth, n_dups)."""
    order = rng.sample(rs, len(rs))
    cnt = tok.many([f"=== FILE: {p} ===\n{t}" for p, t in order])
    byname = {}
    for p, t in rs:
        for k, v in consts(t).items():
            byname.setdefault(k, {})[p] = v
    # target file: one that has a const whose name is declared in >= 2 files with different values
    cands = [(p, t) for p, t in order if any(len(byname[k]) >= 2 and len(set(byname[k].values())) >= 2 for k in consts(t))]
    ntok = {p: c for (p, _), c in zip(order, cnt)}

    def asknames(t, tp):  # files of at most 9K tokens (target and twin) so that the document can fill up to ~31K
        return sorted(k for k, v in consts(t).items() if len(v) >= 4 and (tp, k) not in USED_NAMES and ntok[tp] <= 9000
                      and any(p != tp and byname[k][p] != v and ntok[p] <= 9000 for p in byname[k]))
    def plainnames(t, tp):  # a name declared in this file only, a value that says something
        return sorted(k for k, v in consts(t).items() if len(v) >= 6 and (tp, k) not in USED_NAMES and len(byname[k]) == 1)
    text = dict(rs)
    hit = next(((p, t) for p, t in cands if asknames(t, p)), None)
    if hit:  # the same name is declared with another value in a second file, which goes into the document too
        tp, tt = hit
        an = asknames(tt, tp)
        name = rng.choice([k for k in an if k not in {n for _, n in USED_NAMES}] or an)  # prefer a name not asked yet
        value = consts(tt)[name]
        twin = rng.choice(sorted(p for p in byname[name] if p != tp and byname[name][p] != value and ntok[p] <= 9000))
    else:
        tp, tt = next((p, t) for p, t in order if ntok[p] <= 12000 and plainnames(t, p))
        name = rng.choice(plainnames(tt, tp))
        value = consts(tt)[name]
        twin = None
    USED_NAMES.add((tp, name))
    tcount = cnt[[p for p, _ in order].index(tp)]
    others = [(p, c) for (p, _), c in zip(order, cnt) if p not in (tp, twin)]
    chosen, tot = ([twin], tcount + ntok[twin]) if twin else ([], tcount)
    for p, c in others:
        if tot + c > TARGET_MAX + 250:
            continue
        chosen.append(p)
        tot += c
        if tot >= TARGET_MAX + 100:
            break
    rng.shuffle(chosen)
    pos = next((k for k in range(len(chosen) + 1) if sum(cnt_of(order, cnt, q) for q in chosen[:k]) >= depth * (tot - tcount)), len(chosen))
    seq = chosen[:pos] + [tp] + chosen[pos:]
    inc = {p: consts(text[p]) for p in seq}
    doc = "\n\n".join(f"=== FILE: {p} ===\n{text[p]}" for p in seq)
    n = tok.n(doc)
    assert TARGET_MIN - 300 <= n <= TARGET_MAX + 300 and n < 31200, n
    ndup = sum(1 for p in seq if p != tp and name in inc[p])
    d = doc.index(f"=== FILE: {tp} ===") / len(doc)
    return doc, n, tp, name, value, d, ndup, twin


def cnt_of(order, cnt, p):
    return cnt[[q for q, _ in order].index(p)]


def depth_of(doc, needle):
    assert doc.count(needle) == 1
    return doc.index(needle) / len(doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = Tok()
    paras, rs = load_pool()
    print(f"pool: {len(paras)} prose paragraphs, {len(rs)} rust files", file=sys.stderr)
    per_kind = {}
    for kind, n in N_PER_KIND.items():
        # target depths spread 4..96 %, shuffled by a fixed seed
        ds = [0.04 + 0.92 * (i + 0.5) / n for i in range(n)]
        random.Random(f"depths-{kind}").shuffle(ds)
        per_kind[kind] = ds
    order = []
    for i in range(max(N_PER_KIND.values())):
        for kind in N_PER_KIND:
            if i < N_PER_KIND[kind]:
                order.append((kind, i))
    rows, man = [], []
    for idx, (kind, i) in enumerate(order):
        rng = random.Random(f"chunk32k-v1-{kind}-{i}")
        d = per_kind[kind][i]
        used_w, used_c = set(), set()
        meta = {}
        if kind == "needle":
            nm, c = word(rng, used_w), code(rng, used_c)
            f = fact(nm, c)
            doc, n = build_prose(tok, rng, paras, [(d, f)])
            q, ans = f"What was the rollback token of the {nm} release? Answer with the token only.", c
            meta["depth"] = depth_of(doc, f)
        elif kind == "needle_dist":
            names = [word(rng, used_w) for _ in range(12)]
            codes = [code(rng, used_c) for _ in range(12)]
            tgt = rng.randrange(12)
            others = [rng.uniform(0.03, 0.97) for _ in range(11)]
            ins = []
            for k in range(12):
                ins.append((d if k == tgt else others.pop(), fact(names[k], codes[k])))
            doc, n = build_prose(tok, rng, paras, ins)
            q, ans = f"What was the rollback token of the {names[tgt]} release? Answer with the token only.", codes[tgt]
            meta["depth"] = depth_of(doc, fact(names[tgt], codes[tgt]))
        elif kind == "needle_hop":
            names = [word(rng, used_w) for _ in range(7)]
            codes = [code(rng, used_c) for _ in range(7)]
            aliases = [word(rng, used_w) for _ in range(4)]
            tgt = 0
            dA = rng.uniform(0.04, 0.96)
            # fact B (the token) sits at the spread depth d; fact A (alias) elsewhere
            ins = [(d, fact(names[tgt], codes[tgt])),
                   (dA, f"Release ledger: the release nicknamed {aliases[0]} is tracked in this ledger as the {names[tgt]} release.")]
            for k in range(1, 4):
                ins.append((rng.uniform(0.03, 0.97), f"Release ledger: the release nicknamed {aliases[k]} is tracked in this ledger as the {names[k]} release."))
            for k in range(1, 7):
                ins.append((rng.uniform(0.03, 0.97), fact(names[k], codes[k])))
            doc, n = build_prose(tok, rng, paras, ins)
            q, ans = f"What was the rollback token of the release nicknamed {aliases[0]}? Answer with the token only.", codes[tgt]
            meta["depth"] = depth_of(doc, fact(names[tgt], codes[tgt]))
            meta["depth_alias"] = depth_of(doc, ins[1][1])
        else:
            doc, n, path, cname, value, dd, ndup, twin = build_code(tok, rng, rs, d)
            q = f"In the file {path}, what value is assigned to the constant {cname}? Answer with the value only, exactly as written in the source."
            ans = value
            meta.update(depth=dd, file=path, const=cname, same_name_in_other_files=ndup, twin_file=twin)
        assert ans not in q, (kind, i)
        ask_word = re.search(r"(?:of the (?:release nicknamed )?)(\w+)", q).group(1) if kind != "qa_const" else None
        assert ask_word is None or doc.count(ask_word) <= (2 if kind == "needle_hop" else 1), (kind, i, ask_word, doc.count(ask_word))
        if kind == "qa_const":
            assert len(ans) >= 4
        else:
            assert doc.count(ans) == 1, (kind, i, doc.count(ans))
        prompt = (f"Run id chunk32k-v1-{idx:02d}-{kind}-{i:02d}.\n\nBelow is a long document. Read it carefully, then answer the question after it.\n\n"
                  f"<document>\n{doc}\n</document>\n\nQuestion: {q}")
        name = f"{kind}-{i:02d}-d{round(meta['depth'] * 100):02d}"
        rows.append({"name": name, "input": prompt, "expected_output": ans})
        man.append({"idx": idx, "name": name, "kind": kind, "shard": "pilot" if idx < 12 else "rest1" if idx < 32 else "rest2", "doc_tokens": n,
                    "prompt_tokens_est": tok.n(prompt), "expected": ans, "question": q,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in meta.items()}})
        print(f"{idx:2d} {name:24s} {tok.n(prompt):6d} tok  ans={ans!r}", file=sys.stderr)
    assert len({r["name"] for r in rows}) == len(rows) == sum(N_PER_KIND.values())
    assert len({r["input"][:300] for r in rows}) == len(rows)
    with open(a.out + ".jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    with open(a.out + ".manifest.json", "w") as f:
        json.dump({"source_commit": SOURCE_COMMIT, "tokenizer": TOKENIZER, "items": man}, f, indent=1, ensure_ascii=False)
    # shards of the same rows: pilot = first 12 (3 of each kind, read early), rest1 = 12..31, rest2 = 32..51
    for nm, sl in (("pilot", rows[:12]), ("rest1", rows[12:32]), ("rest2", rows[32:])):
        with open(f"{a.out}.{nm}.jsonl", "w") as f:
            for r in sl:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    for ext in ("jsonl", "pilot.jsonl", "rest1.jsonl", "rest2.jsonl"):
        print(ext, hashlib.sha256(open(f"{a.out}.{ext}", "rb").read()).hexdigest(), file=sys.stderr)


if __name__ == "__main__":
    main()
