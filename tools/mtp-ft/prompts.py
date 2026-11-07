"""Build the prompt set for regen.py from UltraChat (chat) and Magicoder-OSS-Instruct (code).

  python prompts.py --ultrachat ultrachat_train_sft_0.parquet --magicoder magicoder_oss_instruct.jsonl \
      --out prompts.jsonl [--n 40000 --chat-frac 0.6 --seed 0 --eval <dir of eval category jsonl>]

The dataset files are untrusted text: they are only parsed (pyarrow / json) and their strings written back out, never executed.
UltraChat: the first user turn of `messages`. Magicoder: the `problem` field. Output lines:
  {"id": "chat-000123", "category": "chat", "messages": [{"role": "user", "content": "..."}]}
(regen.py turns `category` into the `<category>__<n>` ids of tokens.jsonl).

Selection is deterministic for a seed: each pool is shuffled, then walked until its quota is met; a candidate is dropped when
  * it is too short / too long (`--min-chars`, `--max-chars`; long prompts make long, slow regenerations),
  * eval overlap: its normalised text (case-folded, punctuation and whitespace squeezed) equals an eval prompt's, contains it or is
    contained in it (eval text >= 40 chars), or its word 5-gram Jaccard similarity to an eval prompt is >= `--near` (0.6),
  * dup: its normalised text equals a kept prompt's (`--prefix-dedup N` also drops a shared first N normalised characters; off by default:
    Magicoder / UltraChat prompts are templated and share long prefixes while being different tasks).
Eval prompts: every `input` / user message of `--eval` (default scripts/dfh/cats/*.jsonl, the files scripts/dt1/cat-window.sh runs).
The counts printed are over the candidates walked, not the whole pools. Needs pyarrow for the parquet; nothing else.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def norm(s: str) -> str:
    return " ".join(_PUNCT.sub(" ", s.casefold()).split())


def shingles(n: str, k: int = 5) -> set:
    w = n.split()
    if len(w) < k:
        return {" ".join(w)} if w else set()
    return {" ".join(w[i : i + k]) for i in range(len(w) - k + 1)}


class EvalFilter:
    def __init__(self, files):
        self.texts = []
        for f in files:
            for line in open(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("messages"):
                    self.texts += [m["content"] for m in r["messages"] if m.get("role") == "user" and isinstance(m.get("content"), str)]
                else:
                    self.texts += [r[k] for k in ("input", "prompt", "text") if isinstance(r.get(k), str)]
        self.norms = [norm(t) for t in self.texts]
        self.exact = set(self.norms)
        self.sh = [shingles(n) for n in self.norms]
        self.union = set().union(*self.sh) if self.sh else set()

    def overlaps(self, n: str, near: float) -> bool:
        if n in self.exact:
            return True
        for e in self.norms:
            if len(e) >= 40 and (e in n or (len(n) >= 40 and n in e)):
                return True
        s = shingles(n)
        if s & self.union:
            for es in self.sh:
                inter = len(s & es)
                if inter and inter / len(s | es) >= near:
                    return True
        return False


def read_ultrachat(path: str) -> list:
    import pyarrow.parquet as pq

    out = []
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=2000, columns=["messages"]):
        for msgs in batch.column("messages").to_pylist():
            for m in msgs or []:
                if m.get("role") == "user":
                    if isinstance(m.get("content"), str):
                        out.append(m["content"])
                    break  # the first user turn only
    return out


def read_magicoder(path: str) -> list:
    out = []
    for line in open(path):
        if line.strip():
            p = json.loads(line).get("problem")
            if isinstance(p, str):
                out.append(p)
    return out


def select(pool, quota, evalf, seen_full, seen_prefix, a, rng, stats):
    pool = list(pool)
    rng.shuffle(pool)
    keep = []
    for t in pool:
        if len(keep) >= quota:
            break
        stats["examined"] += 1
        t = t.strip()
        if not (a.min_chars <= len(t) <= a.max_chars):
            stats["dropped_length"] += 1
            continue
        n = norm(t)
        if evalf.overlaps(n, a.near):
            stats["dropped_eval"] += 1
            continue
        if n in seen_full:
            stats["dropped_dup"] += 1
            continue
        if a.prefix_dedup and n[: a.prefix_dedup] in seen_prefix:
            stats["dropped_dup_prefix"] += 1
            continue
        seen_full.add(n)
        seen_prefix.add(n[: a.prefix_dedup])
        keep.append(t)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ultrachat", required=True)
    ap.add_argument("--magicoder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--chat-frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", nargs="*", default=None, help="eval prompt JSONL files or directories (default scripts/dfh/cats)")
    ap.add_argument("--min-chars", type=int, default=16)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--prefix-dedup", type=int, default=0, help="also drop prompts sharing their first N normalised characters with a kept one (0 = off; templated prompts share long prefixes)")
    ap.add_argument("--near", type=float, default=0.6, help="word 5-gram Jaccard threshold for near-exact eval overlap")
    a = ap.parse_args()

    ev = a.eval if a.eval is not None else [os.path.join(REPO, "scripts", "dfh", "cats")]
    files = []
    for p in ev:
        files += sorted(glob.glob(os.path.join(p, "*.jsonl"))) if os.path.isdir(p) else [p]
    if not files:
        sys.exit("no eval prompt files found (--eval): refusing to build a set that cannot exclude them")
    evalf = EvalFilter(files)
    print(f"eval prompts: {len(evalf.texts)} from {len(files)} files", flush=True)

    chat, code = read_ultrachat(a.ultrachat), read_magicoder(a.magicoder)
    print(f"pools: {len(chat)} ultrachat first-user prompts, {len(code)} magicoder problems", flush=True)
    rng = random.Random(a.seed)
    seen_full, seen_prefix = set(), set()
    stats = dict(examined=0, dropped_length=0, dropped_eval=0, dropped_dup=0, dropped_dup_prefix=0)
    n_chat = round(a.n * a.chat_frac)
    kept_chat = select(chat, n_chat, evalf, seen_full, seen_prefix, a, rng, stats)
    kept_code = select(code, a.n - len(kept_chat), evalf, seen_full, seen_prefix, a, rng, stats)
    if len(kept_chat) < n_chat and len(kept_code) < a.n - len(kept_chat):
        print("warning: pools exhausted before --n", file=sys.stderr)
    rows = [("chat", t) for t in kept_chat] + [("code", t) for t in kept_code]
    rng.shuffle(rows)
    cnt = {"chat": 0, "code": 0}
    with open(a.out, "w") as f:
        for cat, t in rows:
            f.write(json.dumps({"id": f"{cat}-{cnt[cat]:06d}", "category": cat, "messages": [{"role": "user", "content": t}]}, ensure_ascii=False) + "\n")
            cnt[cat] += 1
    print(json.dumps(dict(kept=len(rows), chat=len(kept_chat), code=len(kept_code), **stats)), flush=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
