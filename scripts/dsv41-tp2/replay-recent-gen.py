#!/usr/bin/env python3
"""Recent-window item set for step 7 of replay-test.sh (docs/results/2026-11-03-dsv41-replay.md, "Step 7").

Each item is a log of repo prose with many `Vault <Name> code: XX-9999-XX.` lines; the question asks for the code of one vault.
The target line is planted at a MEASURED token distance from the end of the prompt the server prefills (BOS, <User>, text,
<Assistant>, </think>: text tokens + 4), with the DSv4.1 tokenizer. Six distance buckets x two lengths (about 4K and 30K tokens),
N items each. Deterministic: fixed seeds per item, filler read from git commit SOURCE_COMMIT (set 1's pool), so the files have a
fixed sha256.

  uv run --no-project --with tokenizers python replay-recent-gen.py --out DIR [--n4 12] [--n32 12]   generate
  uv run --no-project --with tokenizers python replay-recent-gen.py --verify DIR                      re-measure DIR's items
  python replay-recent-gen.py --summary MANIFEST.json                                               chunk statistics (markdown)

Writes DIR/recent.4k.jsonl, DIR/recent.32k.jsonl (localeval rows {"name", "input", "expected_output"}), DIR/recent.manifest.json
(per item: the measured distances, the decoys, the replay chunk arithmetic at chunk 512 and 2048) and DIR/recent.sha256.json.

Distances (all counted in tokens of the whole rendered prompt, the 4 template tokens included, up to and including the end):
  d_start    from the first token of the fact line to the end; the bucket is chosen by it. The line is replayed whole iff d_start <= 128.
  d_lineend  from the last token of the fact line to the end; the line lies wholly below the replay floor iff d_lineend >= 129.
  d_code     from the first token of the code to the end.
No item straddles the floor (d_start > 128 and d_lineend <= 128).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True).stdout.strip()
_spec = importlib.util.spec_from_file_location("gen_items", os.path.join(REPO, "docs/results/2026-11-03-dsv41-chunk32k-quality/raw/gen_items.py"))
g1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g1)      # load_pool, word, code (set 1's helpers, pinned commit)
g1.REPO = REPO                    # any checkout holding SOURCE_COMMIT

TOKENIZER = os.environ.get("TOKENIZER", g1.TOKENIZER)
WINDOW = 128                      # config sliding_window; replay::floor(total, window) = total - window
CHUNKS = (512, 2048)
SEED = "final"
BOS, USER, TAIL_TPL = "<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜></think>"
CODE_RE = re.compile(r"(?<![A-Z0-9])[A-Z]{2}-\d{4}-[A-Z]{2}(?![A-Z0-9]|-[A-Z0-9])")
LINE_RE = re.compile(r"^Vault (\w+) code: ([A-Z]{2}-\d{4}-[A-Z]{2})\.$", re.M)
QUESTION_RE = re.compile(r"Question: what is the code of vault (\w+)\? Reply with only the code\.$")
BUCKETS = {1: "<=64", 2: "65-128", 3: "129-256", 4: "257-1024", 5: "1025-4096", 6: "far"}
# total prompt tokens (template included), a range per length; the floor residue mod 512 is cycled over RESIDUES in every cell so
# that every chunk case occurs: the replayed window [floor, total) spans two chunks of 512 when the residue is >= 385, the floor
# sits exactly on a chunk boundary at residue 0, one row past it at 1.
LENGTHS = {"4k": (3700, 4400), "32k": (30000, 30800)}
RESIDUES = [0, 1, 100, 200, 300, 384, 385, 420, 450, 480, 510, 511]
FIXED_HEAD = ('Below is a long log. Vault entries have the form "Vault <name> code: <code>." Read the log, then answer the question '
              'after it.\n\n<log>\n')


# ---- the replay arithmetic, a Python mirror of crates/intarsia-core/src/dsv41/replay.rs (floor / classify / decoder_rows)
def rfloor(total, window=WINDOW):
    return max(total - window, 0)


def rclassify(fl, pos0, t):
    if pos0 >= fl:
        return ("whole", 0)
    if pos0 + t <= fl:
        return ("encoder", 0)
    return ("cross", fl - pos0)


def chunk_plan(total, chunk, fact_first, fact_last):
    """The chunks of a prefill of `total` rows from 0 and what the replay does in each; asserts that the rows the decoder half
    runs on are exactly [floor, total), i.e. the rows a bucket calls 'recent' are the replayed ones."""
    fl = rfloor(total)
    chunks, dec_rows, pos = [], [], 0
    while pos < total:
        t = min(chunk, total - pos)
        kind, first = rclassify(fl, pos, t)
        lo = {"whole": pos, "encoder": pos + t, "cross": pos + first}[kind]
        dec_rows.extend(range(lo, pos + t))
        chunks.append((pos, t, kind, first))
        pos += t
    assert dec_rows == list(range(fl, total)), (total, chunk)
    fc = next(i for i, (p, t, k, f) in enumerate(chunks) if p <= fl < p + t)
    lc = len(chunks) - 1
    cf, cl = fact_first // chunk, fact_last // chunk
    return {"n_chunks": len(chunks), "floor_chunk": fc, "floor_kind": chunks[fc][2], "floor_in_chunk_row": fl - chunks[fc][0],
            "floor_on_boundary": fl % chunk == 0, "cross_first": chunks[fc][3] if chunks[fc][2] == "cross" else None,
            "window_spans_chunks": fc != lc, "decoder_rows": len(dec_rows), "fact_chunk": cf, "fact_split": cf != cl,
            "fact_in_floor_chunk": cf == fc or cl == fc, "encoder_only_chunks": sum(k == "encoder" for _, _, k, _ in chunks)}


class Tk:
    def __init__(self, path=None):
        from tokenizers import Tokenizer
        self.path = path or TOKENIZER
        self.t = Tokenizer.from_file(self.path)
        self.bos, self.user = self.t.token_to_id(BOS), self.t.token_to_id(USER)
        self.asst, self.think_end = self.t.token_to_id("<｜Assistant｜>"), self.t.token_to_id("</think>")
        assert None not in (self.bos, self.user, self.asst, self.think_end)

    def n(self, s):
        return len(self.t.encode(s, add_special_tokens=False).ids)

    def many(self, xs):
        return [len(e.ids) for e in self.t.encode_batch(xs, add_special_tokens=False)]

    def render(self, content):
        return BOS + USER + content + TAIL_TPL

    def encode_prompt(self, content):
        e = self.t.encode(self.render(content), add_special_tokens=False)
        ids = e.ids
        assert ids[:2] == [self.bos, self.user] and ids[-2:] == [self.asst, self.think_end], "template tokens not where expected"
        return e


def tok_at(offsets, char):
    """index of the token that contains the character at text position `char`"""
    for i, (s, e) in enumerate(offsets):
        if s <= char < e:
            return i
    raise ValueError(f"no token covers character {char}")


def measure(tk, content, spans):
    """spans: name -> (start_char, end_char) in `content` (end exclusive). Tokenizes the whole rendered prompt once and returns the
    total and, per span, the index of its first and last token and the distances d_first = n - idx_first, d_last = n - idx_last."""
    e = tk.encode_prompt(content)
    off = len(BOS) + len(USER)
    n = len(e.ids)
    out = {"total": n}
    for k, (a, b) in spans.items():
        i0, i1 = tok_at(e.offsets, off + a), tok_at(e.offsets, off + b - 1)
        # a token that began before the span would make the span's first token shared with the text before it
        out[k] = {"first": i0, "last": i1, "d_first": n - i0, "d_last": n - i1, "clean_start": e.offsets[i0][0] >= off + a}
    return out


def words_prefix(text, k):
    return " ".join(text.split(" ")[:k]) if k > 0 else ""


class Item:
    pass


def bucket_range(b, total, line_tokens):
    """[lo, hi] of d_start for bucket b (hi may be below lo when impossible)."""
    if b == 1:
        return None, 64
    if b == 2:
        return 65, 128
    if b == 3:
        return 129 + line_tokens - 1, 256          # the whole line below the floor: d_lineend >= 129
    if b == 4:
        return 257, 1024
    if b == 5:
        return 1025, min(4096, total // 2 - 1)
    return total // 2, int(total * 0.90)


def pick_total(rng, length, residue):
    lo, hi = LENGTHS[length]
    ms = [m for m in range(0, 80) if lo <= 512 * m + residue + WINDOW <= hi]
    return 512 * rng.choice(ms) + residue + WINDOW


def targets_for(rng, b, n, total, line_tokens, dmin):
    lo, hi = bucket_range(b, total, line_tokens)
    if b == 1:
        lo = dmin
    edges = {1: [lo], 2: [128, 127], 3: [lo, lo + 1]}.get(b, [])
    free = n - len(edges)
    ts = list(edges)
    for i in range(free):
        a = lo + (hi - lo) * i / free
        c = lo + (hi - lo) * (i + 1) / free
        ts.append(int(round(rng.uniform(a, c))))
    ts = [min(hi, max(lo, t)) for t in ts]
    return ts


class Pool:
    def __init__(self, tk):
        paras, rs = g1.load_pool()
        bad = re.compile(r"｜|▁|FINAL:|vault|Vault|<think|</think", re.S)
        paras = [p for p in paras if not bad.search(p) and not CODE_RE.search(p.upper())]
        self.paras = paras
        self.cnt = tk.many(paras)
        self.lower = {}


def tuner_search(pool, cur_cands, text_fn, count_fn, r, want):
    """Find a paragraph and a word count k whose text makes count_fn(text_fn(par, k)) == want exactly (else the closest).
    cur_cands: candidate paragraph indices (not consumed by the document)."""
    best = None
    tries = 0
    for j in cur_cands:
        if pool.cnt[j] < r + 15:
            continue
        tries += 1
        par = pool.paras[j]
        kmax = len(par.split(" "))
        lo, hi = 0, kmax
        while lo < hi:
            mid = (lo + hi) // 2
            if count_fn(text_fn(par, mid)) >= want:
                hi = mid
            else:
                lo = mid + 1
        for k in (lo - 1, lo, lo + 1):
            if 0 <= k <= kmax:
                d = count_fn(text_fn(par, k))
                if best is None or abs(d - want) < best[0]:
                    best = (abs(d - want), j, k)
        if best[0] == 0 or tries >= 12:
            break
    return best


def build_item(tk, pool, length, bucket, i, idx, total, target, line_tokens):
    """One item at a requested d_start (target); returns a dict, or None when the result does not satisfy the bucket / total."""
    rng = random.Random(f"recent-{SEED}-{length}-{bucket}-{i}-{target}")
    used_w, used_c = set(), set()
    name, code = g1.word(rng, used_w), g1.code(rng, used_c)
    n_dec = 6 if length == "4k" else 24
    decoys = [(g1.word(rng, used_w), g1.code(rng, used_c)) for _ in range(n_dec)]
    fact = f"Vault {name} code: {code}."
    dline = lambda d: f"Vault {d[0]} code: {d[1]}."
    question = f"Question: what is the code of vault {name}? Reply with only the code."
    tail = f"\n</log>\n\n{question}"
    run_id = f"Run id recent-{SEED}-{idx:03d}-{length}-b{bucket}."
    head = f"{run_id}\n\n{FIXED_HEAD}"
    order = list(range(len(pool.paras)))
    rng.shuffle(order)
    nm = [name] + [d[0] for d in decoys]                       # names must not occur in the filler prose
    order = [j for j in order if not any(w.lower() in pool.paras[j].lower() for w in nm)]
    cur = iter(order)

    # ---- the suffix: fact line, the after-part, the tail. d_start = its token count + 2 (<Assistant></think>)
    def suffix_text(units, tuner):
        parts = [fact] + units + ([tuner] if tuner else [])
        return "\n\n".join(parts) + tail

    d_of = lambda t: tk.n(t) + 2
    d0 = d_of(suffix_text([], ""))
    target = max(target, d0)
    near = decoys[0] if target - d0 >= 90 else None
    units = [dline(near)] if near else []
    # bulk paragraphs go BEFORE the near decoy (so the decoy is within the last tokens); the tuner paragraph comes last
    bulk = []
    r = target - d_of(suffix_text(bulk + units, ""))
    while r > 300:
        add, room = [], r - 150
        for j in cur:
            c = pool.cnt[j] + 1
            if c <= room:
                add.append(pool.paras[j])
                room -= c
            if room < 60 or len(add) >= 400:
                break
        if not add:
            return None
        bulk += add
        r = target - d_of(suffix_text(bulk + units, ""))
    while r < 0 and bulk:
        bulk.pop()
        r = target - d_of(suffix_text(bulk + units, ""))
    if r < 0:
        return None
    tb = tuner_search(pool, order[::-1], lambda par, k: suffix_text(bulk + units, words_prefix(par, k)), d_of, r, target) if r > 0 else (0, None, 0)
    if tb is None:
        return None
    suf = suffix_text(bulk + units, words_prefix(pool.paras[tb[1]], tb[2]) if tb[1] is not None else "")
    d_suffix = d_of(suf)
    dlo, dhi = bucket_range(bucket, total, line_tokens)
    if not ((dlo is None or d_suffix >= (0 if bucket == 1 else dlo)) and d_suffix <= dhi):
        return None
    # ---- the prefix: head, filler with the decoys, the decoy right before the fact; sized so that the total is exactly `total`
    want = total - d_suffix                                     # tokens before the fact line, BOS and <User> included
    before = decoys[1]
    far = decoys[2:2 + max(n_dec - 2 - (1 if near else 0), 0)]
    rng2 = random.Random(f"recent-{SEED}-{length}-{bucket}-{i}-{target}-far")

    def prefix_text(paras, tuner):
        body = ([tuner] if tuner else []) + paras + [dline(before)]
        return head + "\n\n".join(body) + "\n\n"

    count_pre = lambda t: tk.n(BOS + USER + t)
    paras = []
    r = want - count_pre(prefix_text(paras, ""))
    while r > 300:
        add, room = [], r - 150
        for j in cur:
            c = pool.cnt[j] + 1
            if c <= room:
                add.append(pool.paras[j])
                room -= c
            if room < 60 or len(add) >= 600:
                break
        if not add:
            return None
        paras += add
        r = want - count_pre(prefix_text(paras, ""))
    # the far decoys at paragraph boundaries (their tokens count: re-measure)
    for d in far:
        paras.insert(rng2.randrange(len(paras) + 1), dline(d))
    r = want - count_pre(prefix_text(paras, ""))
    while r < 0 and paras:
        k = max(i2 for i2, u in enumerate(paras) if not u.startswith("Vault ")) if any(not u.startswith("Vault ") for u in paras) else None
        if k is None:
            return None
        paras.pop(k)
        r = want - count_pre(prefix_text(paras, ""))
    while r > 300:                                              # the decoys were few tokens; top up again
        add, room = [], r - 150
        for j in cur:
            c = pool.cnt[j] + 1
            if c <= room:
                add.append(pool.paras[j])
                room -= c
            if room < 60:
                break
        if not add:
            return None
        paras = add + paras
        r = want - count_pre(prefix_text(paras, ""))
    if r < 0:
        return None
    tb2 = tuner_search(pool, order[::-1][60:], lambda par, k: prefix_text(paras, words_prefix(par, k)), count_pre, r, want) if r > 0 else (0, None, 0)
    if tb2 is None:
        return None
    pre = prefix_text(paras, words_prefix(pool.paras[tb2[1]], tb2[2]) if tb2[1] is not None else "")
    content = pre + suf
    fact_a = len(pre)
    spans = {"line": (fact_a, fact_a + len(fact)), "code": (fact_a + len(f"Vault {name} code: "), fact_a + len(fact) - 1)}
    placed = []
    for d in [decoys[0], decoys[1]] + list(far):
        a = content.find(dline(d))
        if a >= 0 and content.count(dline(d)) == 1:
            spans["dec:" + d[1]] = (a, a + len(dline(d)))
            placed.append(d)
    m = measure(tk, content, spans)
    return {"content": content, "name": name, "code": code, "placed": placed, "m": m, "question": question, "run_id": run_id}


def make_item(tk, pool, length, bucket, i, idx, total, target, line_tokens):
    lo, hi = bucket_range(bucket, total, line_tokens)
    lo = 0 if bucket == 1 else lo
    for dt in (0, -1, 1, -2, 2, -3, 3):
        t = target + dt
        if t > hi or (bucket != 1 and t < lo):
            continue
        r = build_item(tk, pool, length, bucket, i, idx, total, t, line_tokens)
        if r is None:
            continue
        m = r["m"]
        d_start, d_lineend = m["line"]["d_first"], m["line"]["d_last"]
        ok = m["total"] == total and m["line"]["clean_start"] and lo <= d_start <= hi
        ok = ok and not (d_start > WINDOW and d_lineend <= WINDOW) and (bucket in (1, 2) or d_lineend >= WINDOW + 1)
        if ok:
            return r
    return None


def solve(content):
    """Independent solver: the code of the vault the question names, from the item text alone."""
    q = QUESTION_RE.search(content.rstrip().split("\n")[-1])
    if not q:
        raise ValueError("no question at the end")
    hits = [c for n, c in LINE_RE.findall(content) if n == q.group(1)]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} lines for vault {q.group(1)}")
    return hits[0]


def generate(out, n4, n32, only=None):
    tk = Tk()
    pool = Pool(tk)
    # tokens of the fact line (they depend on the name and code, a few tokens either way): probe with typical ones
    probe = "Vault Torzan code: QK-4821-MT."
    ptail = "\n</log>\n\nQuestion: what is the code of vault Torzan? Reply with only the code."
    pm = measure(tk, "A.\n\n" + probe + ptail, {"line": (4, 4 + len(probe))})["line"]
    line_tokens = pm["first"] and (pm["d_first"] - pm["d_last"] + 1)      # tokens the fact line spans
    dmin = pm["d_first"]
    rows = {"4k": [], "32k": []}
    man = []
    rrng = random.Random(f"recent-{SEED}-residues")
    plan = []
    for length, n in (("4k", n4), ("32k", n32)):
        if only and length not in only:
            continue
        cells = {}
        for b in range(1, 7):
            res = RESIDUES[:]
            rrng.shuffle(res)
            rng = random.Random(f"recent-{SEED}-{length}-{b}")
            tots = [pick_total(rng, length, res[i % len(res)]) for i in range(n)]
            ts = targets_for(rng, b, n, int(sum(tots) / n), line_tokens, dmin)
            # buckets 5 and 6 bound by half of each item's own total: clamp per item
            cells[b] = [(i, tots[i], ts[i]) for i in range(n)]
        for i in range(n):                                     # round robin over buckets inside a length
            for b in range(1, 7):
                plan.append((length, b) + cells[b][i])
    idxs = {"4k": 0, "32k": 0}
    for length, b, i, total, target in plan:
        lo, hi = bucket_range(b, total, line_tokens)
        target = min(hi, max(lo if b != 1 else dmin, target))
        idx = idxs[length]
        idxs[length] += 1
        r = make_item(tk, pool, length, b, i, idx, total, target, line_tokens)
        if r is None:
            raise SystemExit(f"could not build {length} bucket {b} item {i} (total {total}, target {target})")
        content, name, code, m, question, run_id = r["content"], r["name"], r["code"], r["m"], r["question"], r["run_id"]
        assert solve(content) == code
        assert content.count(code) == 1 and code not in question
        assert set(CODE_RE.findall(content)) == {code} | {d[1] for d in r["placed"]}
        nm = f"rw-{length}-b{b}-{i:02d}"
        rows[length].append({"name": nm, "input": content, "expected_output": code})
        fl = rfloor(m["total"])
        item = {"name": nm, "idx": idx, "length": length, "bucket": b, "bucket_label": BUCKETS[b], "total_tokens": m["total"],
                "text_tokens": m["total"] - 4, "floor": fl, "d_start": m["line"]["d_first"], "d_lineend": m["line"]["d_last"],
                "d_code": m["code"]["d_first"], "line_first_token": m["line"]["first"], "line_last_token": m["line"]["last"],
                "in_replayed_rows": m["line"]["d_first"] <= WINDOW, "expected": code, "vault": name, "question": question,
                "decoys": [{"name": d[0], "code": d[1], "d_start": m["dec:" + d[1]]["d_first"]} for d in r["placed"]],
                "run_id": run_id, "prompt_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "chunks": {str(c): chunk_plan(m["total"], c, m["line"]["first"], m["line"]["last"]) for c in CHUNKS}}
        man.append(item)
        print(f"{nm:14s} T {m['total']:6d} d_start {item['d_start']:6d} lineend {item['d_lineend']:6d} floor%512 {fl % 512:3d}", file=sys.stderr)
    os.makedirs(out, exist_ok=True)
    shas = {}
    for length in rows:
        if not rows[length]:
            continue
        p = os.path.join(out, f"recent.{length}.jsonl")
        with open(p, "w") as f:
            for r in rows[length]:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        shas[length] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    man.sort(key=lambda x: (x["length"] != "4k", x["idx"]))
    for k, x in enumerate(man):
        x["idx"] = k
    meta = {"seed": SEED, "source_commit": g1.SOURCE_COMMIT, "tokenizer_sha256": hashlib.sha256(open(tk.path, "rb").read()).hexdigest(),
            "window": WINDOW, "template_tokens": 4, "n4": n4, "n32": n32, "items": man}
    json.dump(meta, open(os.path.join(out, "recent.manifest.json"), "w"), indent=1, ensure_ascii=False)
    json.dump(shas, open(os.path.join(out, "recent.sha256.json"), "w"), indent=1)
    for k, v in shas.items():
        print(k, v, file=sys.stderr)


def verify(d):
    """Re-measure every row of DIR/recent.*.jsonl from the text alone and compare with the manifest."""
    tk = Tk()
    man = {m["name"]: m for m in json.load(open(os.path.join(d, "recent.manifest.json")))["items"]}
    bad, n = [], 0
    for length in ("4k", "32k"):
        p = os.path.join(d, f"recent.{length}.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            n += 1
            m = man[r["name"]]
            c = r["input"]
            code = solve(c)
            q = QUESTION_RE.search(c.rstrip().split("\n")[-1]).group(1)
            a = c.index(f"Vault {q} code: ")
            fact = f"Vault {q} code: {code}."
            ms = measure(tk, c, {"line": (a, a + len(fact)), "code": (a + len(f"Vault {q} code: "), a + len(fact) - 1)})
            ok = (code == r["expected_output"] == m["expected"] and ms["total"] == m["total_tokens"] and ms["line"]["d_first"] == m["d_start"]
                  and ms["line"]["d_last"] == m["d_lineend"] and ms["code"]["d_first"] == m["d_code"]
                  and hashlib.sha256(c.encode()).hexdigest() == m["prompt_sha256"] and bucket_of(m["d_start"], m["d_lineend"], ms["total"]) == m["bucket"])
            if not ok:
                bad.append(r["name"])
    print(f"verified {n} items, {len(bad)} differ" + (": " + ", ".join(bad[:10]) if bad else ""))
    return 1 if bad else 0


def bucket_of(d_start, d_lineend, total):
    if d_start <= 64:
        return 1
    if d_start <= 128:
        return 2
    if d_start <= 256 and d_lineend >= 129:
        return 3
    if d_start <= 1024:
        return 4
    if d_start <= 4096 and d_start < total // 2:
        return 5
    return 6 if d_start >= total // 2 else 5


def summary(path):
    man = json.load(open(path))["items"]
    L = ["| length | bucket | items | d_start min..max | floor % 512 spans 2 chunks of 512 | of 2048 | line split by a chunk boundary (512 / 2048) | line in the floor's chunk (512 / 2048) | floor exactly on a boundary (512) |",
         "|---|---|---:|---|---:|---:|---|---|---:|"]
    for ln in ("4k", "32k"):
        for b in range(1, 7):
            xs = [x for x in man if x["length"] == ln and x["bucket"] == b]
            if not xs:
                continue
            c5, c2 = [x["chunks"]["512"] for x in xs], [x["chunks"]["2048"] for x in xs]
            L.append(f"| {ln} | {BUCKETS[b]} | {len(xs)} | {min(x['d_start'] for x in xs)}..{max(x['d_start'] for x in xs)} | "
                     f"{sum(c['window_spans_chunks'] for c in c5)} | {sum(c['window_spans_chunks'] for c in c2)} | "
                     f"{sum(c['fact_split'] for c in c5)} / {sum(c['fact_split'] for c in c2)} | "
                     f"{sum(c['fact_in_floor_chunk'] for c in c5)} / {sum(c['fact_in_floor_chunk'] for c in c2)} | {sum(c['floor_on_boundary'] for c in c5)} |")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--n4", type=int, default=12)
    ap.add_argument("--n32", type=int, default=12)
    ap.add_argument("--only", help="comma list of lengths (4k,32k)")
    ap.add_argument("--verify")
    ap.add_argument("--summary")
    a = ap.parse_args()
    if a.verify:
        return verify(a.verify)
    if a.summary:
        return summary(a.summary)
    if not a.out:
        ap.error("--out DIR, --verify DIR or --summary MANIFEST")
    generate(a.out, a.n4, a.n32, set(a.only.split(",")) if a.only else None)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
