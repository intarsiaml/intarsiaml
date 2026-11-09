"""Build a balanced regen prompt set from every downloaded source (chat, code, prose, recall) plus the personal prompts.

  python mix.py --raw ~/mtp-ft/data/raw --personal ~/mtp-ft/data/personal --out prompts-mix.jsonl \
      [--n 2000 --personal-frac 0.10 --seed 0 --eval <eval jsonl files or dirs>]

The personal prompts go first (category "personal", at most `--personal-frac` of `--n`); the rest is split evenly over the four
categories and, inside a category, evenly over its sources that exist under `--raw` (a source short of its share leaves the rest
to the others). Filters, eval exclusion and dedup are prompts.py's (`select`). Output lines are regen.py's input:
  {"id": "recall-000012", "category": "recall", "name": "<source>", "messages": [{"role": ..., "content": ...}, ...]}
The `name` (the source) lands in regen.py's meta.json. The dataset files are untrusted text: only parsed, never executed.
Each source reads at most `--pool` candidates (its first rows), so the big files are not read whole.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import random
import re
import sys

import prompts as P


def _parquet(path, cols, pool):
    import pyarrow.parquet as pq

    n = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2000, columns=cols):
        for r in batch.to_pylist():
            yield r
            n += 1
            if n >= pool:
                return


def _jsonl(path, pool):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for i, line in enumerate(f):
            if i >= pool:
                return
            if line.strip():
                yield json.loads(line)


def _first_user(msgs, keep_system=False):
    """[system?, first user] of a message list, or None."""
    out = []
    for m in msgs or []:
        role, c = m.get("role"), m.get("content")
        if not isinstance(c, str):
            continue
        if role == "system" and keep_system and c.strip():
            out.append({"role": "system", "content": c})
        elif role == "user":
            return out + [{"role": "user", "content": c}]
    return None


def _user(t):
    return [{"role": "user", "content": t}] if isinstance(t, str) else None


_HH = re.compile(r"\n\nHuman: (.*?)\n\nAssistant:", re.S)


def _hh(r):
    m = _HH.match(r.get("chosen") or "")
    return _user(m.group(1)) if m else None


def _wp(r):
    t = re.sub(r"^\s*\[\s*\w+\s*\]\s*", "", r.get("prompt") or "")
    t = t.replace(" ,", ",").replace(" .", ".").replace(" '", "'").replace(" n't", "n't").replace("``", '"').replace("''", '"')
    return _user("Write a short story for this prompt: " + t.strip()) if t.strip() else None


def _summ(r):
    post = (r.get("info") or {}).get("post")
    return _user("Summarize this post in a few sentences:\n\n" + post) if post else None


# source name -> (file glob, category, reader(path, pool) -> iterable of message lists)
def _pq(cols, fn):
    return lambda p, pool: (fn(r) for r in _parquet(p, cols, pool))


def _jl(fn):
    return lambda p, pool: (fn(r) for r in _jsonl(p, pool))


SOURCES = {
    # chat
    "ultrachat": ("ultrachat_train_sft_0.parquet", "chat", _pq(["prompt"], lambda r: _user(r["prompt"]))),
    "nemotron-chat": ("nemotron_v2_chat-*.parquet", "chat", _pq(["messages"], lambda r: _first_user(r["messages"]))),
    "oasst2": ("oasst2_train.parquet", "chat", _pq(["text", "role", "parent_id", "lang"],
               lambda r: _user(r["text"]) if r["role"] == "prompter" and r["parent_id"] is None and r["lang"] == "en" else None)),
    "wildchat": ("wildchat_train_0.parquet", "chat", _pq(["conversation", "language", "toxic", "redacted"],
                 lambda r: _first_user(r["conversation"]) if r["language"] == "English" and not r["toxic"] and not r["redacted"] else None)),
    "smol-everyday": ("smoltalk_everyday-conversations.parquet", "chat", _pq(["messages"], lambda r: _first_user(r["messages"]))),
    "smol-magpie": ("smoltalk_smol-magpie-ultra.parquet", "chat", _pq(["messages"], lambda r: _first_user(r["messages"]))),
    "prism": ("prism_conversations.jsonl", "chat", _jl(lambda r: _user(r.get("opening_prompt")))),
    "hh-helpful": ("hh_rlhf_helpful-*.jsonl.gz", "chat", _jl(_hh)),
    # code
    "magicoder": ("magicoder_oss_instruct.jsonl", "code", _jl(lambda r: _user(r.get("problem")))),
    "nemotron-code": ("nemotron_v2_code-*.parquet", "code", _pq(["messages"], lambda r: _first_user(r["messages"]))),
    "codefeedback": ("codefeedback.jsonl", "code", _jl(lambda r: _user(r.get("query")))),
    "self-oss-sc2": ("self_oss_instruct_sc2.parquet", "code", _pq(["instruction"], lambda r: _user(r["instruction"]))),
    # prose
    "writingprompts": ("writingprompts_train_0.parquet", "prose", _pq(["prompt"], _wp)),
    "no-robots": ("no_robots_train.parquet", "prose", _pq(["messages"], lambda r: _first_user(r["messages"], keep_system=True))),
    "smol-rewrite": ("smoltalk_smol-rewrite.parquet", "prose", _pq(["messages"], lambda r: _first_user(r["messages"], keep_system=True))),
    "tldr-summarize": ("summarize_comparisons_train.parquet", "prose", _pq(["info"], _summ)),
    # recall (explanations, facts, reasoning)
    "natural-reasoning": ("natural_reasoning_first200k.jsonl", "recall", _jl(lambda r: _user(r.get("question")))),
    "nq-open": ("nq_open_train.parquet", "recall", _pq(["question"], lambda r: _user(r["question"].strip().capitalize() + "?"))),
    "webgpt": ("webgpt_comparisons.parquet", "recall", _pq(["question"], lambda r: _user((r["question"] or {}).get("full_text")))),
    "gsm8k": ("gsm8k_main_train.parquet", "recall", _pq(["question"], lambda r: _user(r["question"]))),
    "nemotron-math": ("nemotron_v2_math-*.parquet", "recall", _pq(["messages"], lambda r: _first_user(r["messages"]))),
    "nemotron-stem": ("nemotron_v2_stem-*.parquet", "recall", _pq(["messages"], lambda r: _first_user(r["messages"]))),
}
CATS = ("chat", "code", "prose", "recall")


def load_source(raw, name, pool):
    pat, cat, reader = SOURCES[name]
    out = []
    for path in sorted(glob.glob(os.path.join(raw, pat))):
        out += [m for m in reader(path, pool) if m]
    return cat, out


def take(items, quota, evalf, seen_full, seen_prefix, a, rng, stats):
    """prompts.select over message lists, keyed on the last (user) message's text."""
    by_text = {}
    for m in items:
        by_text.setdefault(m[-1]["content"].strip(), m)
    kept = P.select(by_text.keys(), quota, evalf, seen_full, seen_prefix, a, rng, stats)
    return [by_text[t] for t in kept]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--personal", default=None, help="dir of personal-*.jsonl (personal_prompts.py output)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--personal-frac", type=float, default=0.10)
    ap.add_argument("--pool", type=int, default=20000, help="candidates read per source file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", nargs="*", default=None)
    ap.add_argument("--min-chars", type=int, default=16)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--prefix-dedup", type=int, default=0)
    ap.add_argument("--near", type=float, default=0.6)
    a = ap.parse_args()

    ev = a.eval if a.eval is not None else [os.path.join(P.REPO, "scripts", "dfh", "cats")]
    files = []
    for p in ev:
        files += sorted(glob.glob(os.path.join(p, "*.jsonl"))) if os.path.isdir(p) else [p]
    if not files:
        sys.exit("no eval prompt files found (--eval): refusing to build a set that cannot exclude them")
    evalf = P.EvalFilter(files)
    print(f"eval prompts: {len(evalf.texts)} from {len(files)} files", flush=True)

    rng = random.Random(a.seed)
    seen_full, seen_prefix = set(), set()
    rows, report = [], {}

    def stats():
        return dict(examined=0, dropped_length=0, dropped_eval=0, dropped_dup=0, dropped_dup_prefix=0)

    if a.personal:
        pers = []
        for f in sorted(glob.glob(os.path.join(a.personal, "personal-*.jsonl"))):
            pers += [r["messages"] for r in P_jsonl(f)]
        st = stats()
        kept = take(pers, int(a.n * a.personal_frac), evalf, seen_full, seen_prefix, a, rng, st)
        rows += [("personal", "personal", m) for m in kept]
        report["personal"] = dict(pool=len(pers), kept=len(kept), **st)

    pools = {}
    for name in SOURCES:
        cat, items = load_source(a.raw, name, a.pool)
        if items:
            pools.setdefault(cat, []).append((name, items))
        else:
            print(f"warning: no prompts from {name} ({SOURCES[name][0]})", file=sys.stderr)
    left = a.n - len(rows)
    for ci, cat in enumerate(CATS):
        cq = left // (len(CATS) - ci)
        srcs = pools.get(cat, [])
        got = 0
        for si, (name, items) in enumerate(srcs):
            q = (cq - got) // (len(srcs) - si)
            st = stats()
            kept = take(items, q, evalf, seen_full, seen_prefix, a, rng, st)
            rows += [(cat, name, m) for m in kept]
            got += len(kept)
            report[name] = dict(category=cat, pool=len(items), kept=len(kept), **st)
        left -= got

    rng.shuffle(rows)
    cnt = {}
    with open(a.out, "w") as f:
        for cat, name, m in rows:
            i = cnt.get(cat, 0)
            cnt[cat] = i + 1
            f.write(json.dumps({"id": f"{cat}-{i:06d}", "category": cat, "name": name, "messages": m}, ensure_ascii=False) + "\n")
    for k, v in report.items():
        print(k, json.dumps(v))
    print(json.dumps(dict(kept=len(rows), **cnt)), flush=True)
    if len(rows) < a.n:
        print(f"warning: {len(rows)} < --n {a.n}", file=sys.stderr)
    print(f"wrote {a.out}")


def P_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


if __name__ == "__main__":
    main()
