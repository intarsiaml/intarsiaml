"""Offline depth-1 acceptance of the head on dumps: the fraction of rows whose head argmax equals tokens[p+2] (the greedy
acceptance criterion), per file and per category, over the full vocab and the first `draft_vocab` lm_head rows.

Row p of a dump pairs hidden_p with tokens[p+1] and predicts tokens[p+2], so rows 0..n-3 have a target. Only rows whose target
lies in the generated RESPONSE count (tokens[p+2] is the target model's own greedy argmax there; inside the prompt it is user
text, and the engine only drafts on decode rows): pass regen.py's `--meta` sidecar (prompt_len per id). Without it every row
counts (use `--all-rows` to force that even with a sidecar).

  python accept.py --ckpt <model dir> --dumps <dir> --meta meta.json [--mtp-file mtp-tuned.safetensors] [--lm-head ...]
Also prints the stock acceptance recorded in the dump (head_top1 == tokens[p+2], head_top1_dv likewise) as a cross-check.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import torch

import head as H


class Tally:
    def __init__(self):
        self.rows = self.full = self.dv = self.stock = self.stock_dv = self.has_dv = 0

    def add(self, rows, full, dv=None, stock=None, stock_dv=None):
        self.rows += rows
        self.full += full
        if dv is not None:
            self.dv += dv
            self.has_dv += rows
        self.stock += stock or 0
        self.stock_dv += stock_dv or 0

    def row(self):
        r = dict(rows=self.rows, full=self.full / max(self.rows, 1), stock_full=self.stock / max(self.rows, 1))
        if self.has_dv:
            r.update(draft_vocab=self.dv / self.has_dv, stock_draft_vocab=self.stock_dv / self.has_dv)
        return r


def response_rows(n_tokens: int, prompt_len: int | None) -> slice:
    """Rows p (0 <= p <= n-3) whose target tokens[p+2] is a response token (index >= prompt_len)."""
    lo = max(0, (prompt_len or 0) - 2)
    return slice(lo, n_tokens - 2)


def evaluate(head, files, meta, device, dv_override=None, all_rows=False, max_files=0, per_file=None):
    """-> (overall Tally, {category: Tally}). `per_file`, when a list, collects one dict per file."""
    overall, cats = Tally(), defaultdict(Tally)
    for f in files[: max_files or None]:
        d = H.read_dump(f)
        fid = H.file_id(f)
        n = d["tokens"].shape[0]
        if n < 3:
            continue
        info = meta.get(fid, {})
        sl = slice(0, n - 2) if all_rows else response_rows(n, info.get("prompt_len"))
        if sl.start >= sl.stop:
            continue
        dv = H.draft_vocab_of(d, dv_override)
        g = H.dump_g(head, d, device, rows=n - 2)[sl]
        t1, t1dv = head.argmax(g, dv)
        tgt = d["tokens"][2:n][sl].to(device)
        full = int((t1 == tgt).sum())
        dvh = int((t1dv == tgt).sum()) if dv else None
        stock = int((d["head_top1"][: n - 2][sl].to(device) == tgt).sum()) if "head_top1" in d else None
        sdv = int((d["head_top1_dv"][: n - 2][sl].to(device) == tgt).sum()) if "head_top1_dv" in d else None
        rows = int(tgt.numel())
        cat = info.get("category") or H.category_of(fid)
        for t in (overall, cats[cat]):
            t.add(rows, full, dvh, stock, sdv)
        if per_file is not None:
            per_file.append(dict(file=fid, category=cat, rows=rows, full=full / rows, draft_vocab=(dvh / rows if dvh is not None else None)))
    return overall, cats


def fmt(name, r):
    s = f"{name:<24} rows {r['rows']:>8}  full {r['full']*100:6.2f}%  (stock {r['stock_full']*100:6.2f}%)"
    if "draft_vocab" in r:
        s += f"  draft-vocab {r['draft_vocab']*100:6.2f}%  (stock {r['stock_draft_vocab']*100:6.2f}%)"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--all-rows", action="store_true")
    ap.add_argument("--lm-head", default="auto")
    ap.add_argument("--mtp-file", default=None)
    ap.add_argument("--draft-vocab", type=int, default=None)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--per-file", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    head = H.load_head(a.ckpt, a.device, a.lm_head, a.mtp_file)
    files = H.dump_files(a.dumps)
    pf = [] if a.per_file or a.json else None
    overall, cats = evaluate(head, files, H.load_meta(a.meta), a.device, a.draft_vocab, a.all_rows, a.max_files, pf)
    for c in sorted(cats):
        print(fmt(c, cats[c].row()))
    print(fmt("ALL", overall.row()))
    if a.per_file:
        for r in pf:
            print(json.dumps(r))
    if a.json:
        json.dump(dict(overall=overall.row(), categories={c: t.row() for c, t in cats.items()}, files=pf), open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
