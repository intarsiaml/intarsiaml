#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""O5 analyzer: compare self-golden dumps (`intarsia accuracy --write-self-golden`, top-k rows) written under different
prefill chunkings of the same prompts, row by row and bit by bit.

    tailcmp.py DIR [--ref prefill-c4557] [--mode prefill]   # DIR holds <mode>-c<chunk>.safetensors files

For every other dump of the mode and every prompt: rows (positions) in the dump, rows that differ from the reference dump
(any top-k id or any logprob bit), the first differing row, whether the LAST row differs (the row a 1-row tail computes),
argmax flips, the largest per-row KL(ref || run) over the shared top-k ids (renormalised) and that KL at the last row.
The tail schedule of a chunk C for a prompt of L tokens is `C, C, .., L mod C` (a tail of L mod C rows).
"""
import argparse, json, math, pathlib, re, struct
import numpy as np

DT = {"I32": np.int32, "F32": np.float32, "F16": np.float16, "I64": np.int64}


def load(path):
    b = pathlib.Path(path).read_bytes()
    n = struct.unpack("<Q", b[:8])[0]
    h = json.loads(b[8:8 + n])
    meta = h.pop("__metadata__", {})
    base = 8 + n
    t = {}
    for k, v in h.items():
        s, e = v["data_offsets"]
        t[k] = np.frombuffer(b[base + s: base + e], dtype=DT[v["dtype"]]).reshape(v["shape"])
    return t, meta


def kl_rows(rl, ri, ql, qi):
    """per-row KL(ref||run) over the shared ids, renormalised; NaN where fewer than 2 ids are shared"""
    out = np.full(len(rl), np.nan)
    for r in range(len(rl)):
        common, ra, qa = np.intersect1d(ri[r], qi[r], return_indices=True)
        if len(common) < 2:
            continue
        p = np.exp(rl[r][ra].astype(np.float64)); p /= p.sum()
        q = np.exp(ql[r][qa].astype(np.float64)); q /= q.sum()
        out[r] = float(np.sum(p * (np.log(p) - np.log(q))))
    return out


def prompts(meta):
    try:
        return json.loads(meta["prompts"])
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--mode", default="prefill")
    ap.add_argument("--ref", default=None, help="reference stem, default <mode>-c4557 (prefill) / <mode>-c4493 (decode)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    d = pathlib.Path(a.dir)
    files = sorted(d.glob(f"{a.mode}-c*.safetensors"), key=lambda p: int(re.search(r"-c(\d+)", p.name).group(1)))
    ref_stem = a.ref or (f"{a.mode}-c4557" if a.mode == "prefill" else f"{a.mode}-c4493")
    rp = d / f"{ref_stem}.safetensors"
    if not rp.exists():
        raise SystemExit(f"reference {rp} missing")
    R, rmeta = load(rp)
    pm = prompts(rmeta)
    pre = "" if a.mode == "prefill" else "dec_"
    n_prompts = len([k for k in R if k.startswith("ids_")])
    report = []
    print(f"reference {rp.name}  (prompts: {', '.join(str(p.get('n_prompt', p.get('n_tokens'))) for p in pm)})")
    hdr = f"{'run':<16}{'prompt':>7}{'L':>6}{'tail':>5}{'rows':>6}{'rows!=':>7}{'first!=':>8}{'last!=':>7}{'flips':>6}{'maxKL':>10}{'lastKL':>10}"
    print(hdr)
    for f in files:
        if f == rp:
            continue
        X, xmeta = load(f)
        chunk = int(re.search(r"-c(\d+)", f.name).group(1))
        for i in range(n_prompts):
            ki, kl_ = f"{pre}topk_ids_{i}", f"{pre}topk_logprobs_{i}"
            if ki not in R or ki not in X:
                continue
            ri, rl, xi, xl = R[ki], R[kl_], X[ki], X[kl_]
            if ri.shape != xi.shape:
                print(f"{f.stem:<16}{i:>7}  shape differs {ri.shape} vs {xi.shape}")
                continue
            n = ri.shape[0]
            # prompt length the chunking applies to: the prefill length of this mode
            L = int(R[f"ids_{i}"].shape[0]) if a.mode == "prefill" else int(pm[i].get("n_prompt", 0)) if i < len(pm) else 0
            tail = (L % chunk) if (L and chunk) else 0
            if L and L <= chunk:
                tail = 0  # a single chunk: no tail
            neq = (ri != xi).any(axis=1) | (rl.view(np.int32) != xl.view(np.int32)).any(axis=1)
            rows_diff = int(neq.sum())
            first = int(np.argmax(neq)) if rows_diff else -1
            flips = int((ri[np.arange(n), rl.argmax(axis=1)] != xi[np.arange(n), xl.argmax(axis=1)]).sum())
            kl = kl_rows(rl, ri, xl, xi)
            mk = float(np.nanmax(kl)) if np.isfinite(kl).any() else float("nan")
            row = dict(run=f.stem, chunk=chunk, prompt=i, L=L, tail=tail, rows=n, rows_diff=rows_diff, first_diff=first,
                       last_diff=bool(neq[-1]), flips=flips, max_kl=mk, last_kl=float(kl[-1]),
                       diff_rows=[int(x) for x in np.nonzero(neq)[0][:12]])
            report.append(row)
            print(f"{f.stem:<16}{i:>7}{L:>6}{tail:>5}{n:>6}{rows_diff:>7}{first:>8}{str(row['last_diff']):>7}{flips:>6}{mk:>10.2e}{row['last_kl']:>10.2e}")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
