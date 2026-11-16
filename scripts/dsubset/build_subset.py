#!/usr/bin/env python3
"""Lane dsubset: build an `--mtp-draft-subset` file (crates/intarsia-core/src/draft_subset.rs) from a model's own outputs.

  build_subset.py --tokenizer <model>/tokenizer.json --corpus raw/collect.jsonl [--corpus ...] --rows 65536 \
                  [--head-rows 248320] [--min-count 1] --out recipes/data/mtp-draft-subset/<name>.txt [--report report.json]

Drafts pay only when they equal the target's argmax, so the rows worth keeping are the tokens the target emits. The subset is,
in order, until the budget (`--rows`, a multiple of 8) is full:
  1. every added / special token (chat markup, <think>, <tool_call>, ...) that the head has a row for;
  2. every token the corpus shows at least `--min-count` times, most frequent first;
  3. the rest by script: the remaining budget is split over the scripts (latin+code, han, cyrillic, arabic, hangul, kana,
     other) in proportion to each script's share of the corpus tokens (each script gets at least a floor of 1 % when it has
     any tokens), and inside a script ids are taken in id order. A byte-level BPE vocabulary assigns ids in merge order,
     which is roughly frequency order inside a script (the Qwen vocabulary's Chinese block starts with 的), so this is a
     smoothed frequency: frequent unseen tokens of a script come before rare ones.
The `# tokenizer` header is the fingerprint `tokenizer_fingerprint` (Rust) computes, so a model with another vocabulary refuses
the file at load.

The corpus: JSON lines with an `ids` list (scripts/dsubset/collect.py writes them).
"""
from __future__ import annotations

import argparse
import collections
import json
import unicodedata

SCRIPTS = ["latin", "han", "cyrillic", "arabic", "hangul", "kana", "other"]


def fingerprint(tok: dict) -> str:
    toks: dict[int, str] = {}
    for s, i in tok["model"]["vocab"].items():
        toks[i] = s
    for t in tok.get("added_tokens", []):
        toks[t["id"]] = t["content"]
    h = 0xCBF29CE484222325
    for i in range(max(toks) + 1):
        for b in toks.get(i, "").encode("utf-8") + b"\xff":
            h ^= b
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def byte_decoder() -> dict[str, int]:
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def script_of(s: str, u2b: dict[str, int]) -> str:
    try:
        text = bytes(u2b[c] for c in s).decode("utf-8", errors="ignore")
    except KeyError:
        text = s
    for ch in text:
        if ch.isascii():
            if ch.isalnum():
                return "latin"
            continue
        name = unicodedata.name(ch, "")
        if name.startswith("CJK") or "IDEOGRAPH" in name:
            return "han"
        for k, pre in (("cyrillic", "CYRILLIC"), ("arabic", "ARABIC"), ("hangul", "HANGUL"), ("kana", "HIRAGANA"), ("kana", "KATAKANA"), ("latin", "LATIN")):
            if name.startswith(pre):
                return k
        if ch.isalpha():
            return "other"
    # punctuation, digits, whitespace, partial UTF-8 bytes: code and Latin text use them most
    return "latin"


def ranges(ids: list[int]) -> list[str]:
    out, i = [], 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        out.append(str(ids[i]) if i == j else f"{ids[i]}-{ids[j]}")
        i = j + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus", action="append", required=True)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--head-rows", type=int, default=248320, help="the lm_head's row count (vocab_size in config.json)")
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--floor", type=float, default=0.01, help="minimum share of the script fill per script that occurs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="")
    ap.add_argument("--name", default="", help="a free-text line for the header (model, corpus)")
    a = ap.parse_args()
    assert a.rows % 8 == 0 and 0 < a.rows < a.head_rows, "rows: a positive multiple of 8 below the head's rows"

    tok = json.load(open(a.tokenizer))
    u2b = byte_decoder()
    vocab = {i: s for s, i in tok["model"]["vocab"].items()}
    added = sorted(t["id"] for t in tok.get("added_tokens", []) if t["id"] < a.head_rows)
    script = {i: script_of(s, u2b) for i, s in vocab.items() if i < a.head_rows}

    counts = collections.Counter()
    n_req = 0
    for path in a.corpus:
        for line in open(path):
            r = json.loads(line)
            ids = r.get("ids") or []
            n_req += 1 if ids else 0
            counts.update(i for i in ids if i < a.head_rows)
    total = sum(counts.values())

    chosen: list[int] = []
    seen: set[int] = set()

    def take(i):
        if i not in seen and len(chosen) < a.rows:
            seen.add(i)
            chosen.append(i)

    for i in added:
        take(i)
    n_special = len(chosen)
    for i, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if c >= a.min_count:
            take(i)
    n_observed = len(chosen) - n_special

    # script fill
    mass = collections.Counter()
    for i, c in counts.items():
        mass[script.get(i, "other")] += c
    left = a.rows - len(chosen)
    shares = {k: mass[k] / total for k in SCRIPTS if mass[k] > 0} if total else {"latin": 1.0}
    shares = {k: max(v, a.floor) for k, v in shares.items()}
    z = sum(shares.values())
    quota = {k: int(left * v / z) for k, v in shares.items()}
    order = sorted(script)
    pools = {k: iter([i for i in order if script[i] == k and i not in seen]) for k in SCRIPTS}
    fill = collections.Counter()
    for k, q in quota.items():
        for _ in range(q):
            i = next(pools[k], None)
            if i is None:
                break
            take(i)
            fill[k] += 1
    # rounding or an exhausted script: the rest in id order
    for i in range(a.head_rows):
        if len(chosen) >= a.rows:
            break
        if i in script:
            take(i)
    ids = sorted(chosen)
    assert len(ids) == a.rows, (len(ids), a.rows)

    fp = fingerprint(tok)
    with open(a.out, "w") as f:
        f.write("# intarsia mtp draft subset v1\n")
        f.write(f"# tokenizer {fp}\n")
        f.write(f"# rows {len(ids)}\n")
        if a.name:
            f.write(f"# {a.name}\n")
        f.write(f"# built by scripts/dsubset/build_subset.py: {n_special} special, {n_observed} seen in {n_req} generated replies "
                f"({total} tokens, min count {a.min_count}), {len(ids) - n_special - n_observed} by script share {dict(fill)}\n")
        for r in ranges(ids):
            f.write(r + "\n")
    inside = sum(c for i, c in counts.items() if i in seen)
    rep = {
        "rows": len(ids), "tokenizer": fp, "special": n_special, "observed": n_observed, "distinct_seen": len(counts),
        "corpus_tokens": total, "corpus_replies": n_req, "fill_by_script": dict(fill), "script_mass": dict(mass),
        "corpus_coverage": inside / total if total else None,
        "prefix_coverage": {n: sum(c for i, c in counts.items() if i < n) / total for n in (32768, 65536, 98304, 131072)} if total else None,
        "per_script_ids": dict(collections.Counter(script.get(i, "other") for i in ids)),
        "max_id": ids[-1], "ranges": len(ranges(ids)),
    }
    if a.report:
        json.dump(rep, open(a.report, "w"), indent=1)
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
